"""
因果多頭自注意力 —— 整個模型裡唯一「位置之間互相看得到」的地方

------------------------------------------------------------------
前向的七個步驟
------------------------------------------------------------------
    1. q = x @ Wq^T      k = x @ Wk^T      v = x @ Wv^T
    2. 拆成多個 head：(B,T,D) → (B, heads, T, head_dim)
    3. 對 q 和 k 套 RoPE（位置資訊在這裡才進來，v 不用轉）
    4. scores = q @ k^T / sqrt(head_dim)
    5. 因果遮罩：把「看到未來」的位置設成 −inf
    6. attn = softmax(scores)          每一列加起來等於 1
    7. out = attn @ v  → 合併 head → y = out @ Wo^T

------------------------------------------------------------------
為什麼要除以 sqrt(head_dim)
------------------------------------------------------------------
q 和 k 的每個維度大約是獨立的、變異數約 1，點積 d 個維度之後
變異數會變成 d，也就是標準差 sqrt(d)。head_dim=64 時分數就會散到 ±8 以上，
softmax 吃到這種輸入會變得極度尖銳（幾乎變成 one-hot），梯度就消失了。
除以 sqrt(d) 把變異數拉回 1，softmax 才處在有梯度的區間。

------------------------------------------------------------------
因果遮罩
------------------------------------------------------------------
語言模型只能用「已經看過的字」預測下一個字。所以位置 i 只能注意 j ≤ i。
作法是在 softmax 之前把 j > i 的分數設成 −inf，exp(−inf)=0，
那些位置的注意力權重就變成 0，等於不存在。

------------------------------------------------------------------
反向（逐步，注意 softmax 那一步）
------------------------------------------------------------------
    dout   = o_proj.backward(dy)  再拆回 head
    dattn  = dout @ v^T                    ← out = attn@v，對 attn 微分得到 v
    dv     = attn^T @ dout                 ← 對 v 微分得到 attn
    dscores= attn ⊙ ( dattn − Σ_j dattn⊙attn )     ← softmax 的反向，見下
    dq     = dscores @ k / sqrt(d)
    dk     = dscores^T @ q / sqrt(d)
    再各自過 RoPE 的反向，最後三條線性層反向，dx 三份相加。

softmax 的反向（對每一列獨立）：
    p_i = exp(s_i)/Σexp(s_k)
    ∂p_i/∂s_j = p_i(δ_ij − p_j)
    dL/ds_j = Σ_i dp_i · p_i(δ_ij − p_j)
            = p_j·( dp_j − Σ_i dp_i·p_i )
    括號裡那個 Σ 是「這一列的加權平均」，每一列算一次就好。

（在 loss.py 裡 softmax 和 cross-entropy 合併後簡化成 p−y；
  這裡沒有 cross-entropy 接在後面，所以要用完整的形式。）
"""
import math
import torch

from .linear import Linear
from .rope import RoPE
from .tracer import NULL


class CausalSelfAttention:
    def __init__(self, dim, n_heads, max_len, rope_base=10000.0,
                 device='cpu', dtype=None, name='attn'):
        assert dim % n_heads == 0, 'dim 必須能被 head 數整除'
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.name = name

        self.q_proj = Linear(dim, dim, device=device, dtype=dtype, name=name + '.q_proj')
        self.k_proj = Linear(dim, dim, device=device, dtype=dtype, name=name + '.k_proj')
        self.v_proj = Linear(dim, dim, device=device, dtype=dtype, name=name + '.v_proj')
        self.o_proj = Linear(dim, dim, device=device, dtype=dtype, name=name + '.o_proj')
        self.rope_q = RoPE(self.head_dim, max_len, rope_base, device, dtype)
        self.rope_k = RoPE(self.head_dim, max_len, rope_base, device, dtype)

        self._cache = {}

    # ---------------------------------------------------------------
    def _split(self, t, B, T):
        """(B,T,D) → (B, heads, T, head_dim)"""
        return t.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, t, B, T):
        """(B, heads, T, head_dim) → (B,T,D)"""
        return t.transpose(1, 2).reshape(B, T, self.dim)

    # ---------------------------------------------------------------
    def forward(self, x, tr=NULL):
        B, T, D = x.shape

        q = self._split(self.q_proj.forward(x), B, T)
        k = self._split(self.k_proj.forward(x), B, T)
        v = self._split(self.v_proj.forward(x), B, T)

        q = self.rope_q.forward(q, tr=tr)
        k = self.rope_k.forward(k, tr=NULL)

        scores = (q @ k.transpose(-1, -2)) * self.scale       # (B,h,T,T)

        # 因果遮罩：j > i 的地方設成 −inf
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

        # softmax（沿最後一維，也就是「這個位置分配給各個過去位置的注意力」）
        smax = scores.max(dim=-1, keepdim=True).values
        e = torch.exp(scores - smax)
        attn = e / e.sum(dim=-1, keepdim=True)

        out = attn @ v                                        # (B,h,T,hd)
        y = self.o_proj.forward(self._merge(out, B, T))

        self._cache = dict(q=q, k=k, v=v, attn=attn, B=B, T=T)

        if tr.enabled:
            tr.section('%s · forward' % self.name)
            tr.formula('scores = q @ k^T / sqrt(head_dim)', 'head_dim=%d, scale=%.6f' % (self.head_dim, self.scale))
            tr.formula('因果遮罩：j > i 的位置設成 −inf')
            tr.formula('attn = softmax(scores)   每一列和為 1')
            tr.formula('out  = attn @ v')
            tr.matrix('attn (第 0 個 head)', attn[0, 0], rows=5, cols=6,
                      note='下三角，每列和=1')
            tr.vector('每一列的和（驗證 softmax）', attn[0, 0].sum(-1), n=6)
            tr.note('上三角是 0，代表每個位置真的看不到未來。\n'
                    '這是唯一讓不同位置交換資訊的地方——FFN 是各算各的。')
        return y

    # ---------------------------------------------------------------
    def backward(self, dy, tr=NULL):
        c = self._cache
        B, T = c['B'], c['T']
        q, k, v, attn = c['q'], c['k'], c['v'], c['attn']

        dout = self._split(self.o_proj.backward(dy), B, T)     # (B,h,T,hd)

        # out = attn @ v
        dattn = dout @ v.transpose(-1, -2)                     # (B,h,T,T)
        dv = attn.transpose(-1, -2) @ dout                     # (B,h,T,hd)

        # softmax 反向：dscores = attn ⊙ (dattn − Σ dattn⊙attn)
        row = (dattn * attn).sum(dim=-1, keepdim=True)
        dscores = attn * (dattn - row)

        dscores = dscores * self.scale
        dq = dscores @ k                                       # (B,h,T,hd)
        dk = dscores.transpose(-1, -2) @ q

        dq = self.rope_q.backward(dq, tr=NULL)
        dk = self.rope_k.backward(dk, tr=NULL)

        dx = (self.q_proj.backward(self._merge(dq, B, T))
              + self.k_proj.backward(self._merge(dk, B, T))
              + self.v_proj.backward(self._merge(dv, B, T)))

        if tr.enabled:
            tr.section('%s · backward' % self.name)
            tr.formula('dattn   = dout @ v^T        ;  dv = attn^T @ dout')
            tr.formula('dscores = attn ⊙ ( dattn − Σ dattn⊙attn )', 'softmax 的反向')
            tr.formula('dq = dscores @ k / sqrt(d)  ;  dk = dscores^T @ q / sqrt(d)')
            tr.formula('dx = dq_path + dk_path + dv_path', 'x 走了三條路 → 梯度相加')
            tr.matrix('dscores (head 0)', dscores[0, 0], rows=4, cols=6)
            tr.vector('dx (往下傳)', dx.reshape(-1, self.dim)[0])
            tr.note('dscores 的上三角是 0——被遮罩的位置 attn=0，\n'
                    '公式 attn⊙(...) 自動讓它們的梯度也是 0，不需要另外處理。')
        return dx

    # ---------------------------------------------------------------
    def zero_grad(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            m.zero_grad()

    def params(self):
        out = []
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            out += m.params()
        return out

    def numel(self):
        return sum(m.numel() for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj))
