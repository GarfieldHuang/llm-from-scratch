"""
SwiGLU 前饋網路（FFN）—— 每一層裡「各個位置各自加工」的部分

------------------------------------------------------------------
它跟 Attention 的分工
------------------------------------------------------------------
    Attention  橫向：位置與位置之間交換資訊
    FFN        縱向：每個位置各自加工，位置之間完全不互通

FFN 對第 7 個位置做的事，跟它對第 13 個位置做的事完全獨立。

------------------------------------------------------------------
前向
------------------------------------------------------------------
    g = x @ W_gate^T          (dim → hidden)
    u = x @ W_up^T            (dim → hidden)
    s = SiLU(g)               SiLU(z) = z · σ(z)，σ 是 sigmoid
    m = s ⊙ u                 ← 「gating」：用一路的輸出去調節另一路
    y = m @ W_down^T          (hidden → dim)

比起原始 Transformer 的兩矩陣版 `W2(ReLU(W1 x))`，這裡多了一路 up。
多出來的那一路的作用是「開關」——s 決定 u 的每個維度要放行多少。

注意：三個矩陣中間夾了非線性，這才叫 FFN。
如果只有一個矩陣、沒有非線性，那叫「線性投影」（像 q_proj、embedding），
疊幾層都等價於一層，不會增加表達力。

------------------------------------------------------------------
反向
------------------------------------------------------------------
SiLU 的導數（用 σ' = σ(1−σ) 推）：
    SiLU(z)  = z·σ(z)
    SiLU'(z) = σ(z) + z·σ(z)(1−σ(z))
             = σ(z)·( 1 + z·(1−σ(z)) )

整條鏈（dy 從上游來）：
    dm = down.backward(dy)              ← 線性層的標準反向
    ds = dm ⊙ u                         ← m = s⊙u，對 s 微分得到 u
    du = dm ⊙ s                         ← 對 u 微分得到 s
    dg = ds ⊙ SiLU'(g)
    dx = gate.backward(dg) + up.backward(du)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         x 同時餵給兩條路，所以兩邊的梯度要「相加」。
         這是多分支結構的通則：前向分岔，反向就相加。
"""
import torch

from .linear import Linear
from .tracer import NULL


def silu(z):
    return z * torch.sigmoid(z)


def silu_grad(z):
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


class SwiGLU:
    def __init__(self, dim, hidden, device='cpu', dtype=None, name='ffn'):
        self.name = name
        self.gate = Linear(dim, hidden, device=device, dtype=dtype, name=name + '.gate')
        self.up = Linear(dim, hidden, device=device, dtype=dtype, name=name + '.up')
        self.down = Linear(hidden, dim, device=device, dtype=dtype, name=name + '.down')
        self._g = None
        self._s = None
        self._u = None

    # ---------------------------------------------------------------
    def forward(self, x, tr=NULL):
        g = self.gate.forward(x)
        u = self.up.forward(x)
        s = silu(g)
        m = s * u
        y = self.down.forward(m)

        self._g, self._s, self._u = g, s, u

        if tr.enabled:
            tr.section('%s · forward (SwiGLU)' % self.name)
            tr.formula('g = x @ W_gate^T ;  u = x @ W_up^T')
            tr.formula('s = SiLU(g) = g · σ(g)')
            tr.formula('m = s ⊙ u', 'gating：s 決定 u 的每個維度放行多少')
            tr.formula('y = m @ W_down^T')
            tr.vector('g  (gate 那一路)', g.reshape(-1, g.shape[-1])[0])
            tr.vector('s  = SiLU(g)', s.reshape(-1, s.shape[-1])[0])
            tr.vector('u  (up 那一路)', u.reshape(-1, u.shape[-1])[0])
            tr.vector('m  = s ⊙ u', m.reshape(-1, m.shape[-1])[0])
            tr.vector('y  (輸出)', y.reshape(-1, y.shape[-1])[0])
            tr.note('中間維度 %d 比模型維度 %d 大 %.1f 倍——先升維加工再降回來，\n'
                    'FFN 的參數量因此佔了每一層的三分之二以上。'
                    % (g.shape[-1], y.shape[-1], g.shape[-1] / y.shape[-1]))
        return y

    # ---------------------------------------------------------------
    def backward(self, dy, tr=NULL):
        dm = self.down.backward(dy)
        ds = dm * self._u
        du = dm * self._s
        dg = ds * silu_grad(self._g)

        dx_gate = self.gate.backward(dg)
        dx_up = self.up.backward(du)
        dx = dx_gate + dx_up          # 前向分岔 → 反向相加

        if tr.enabled:
            tr.section('%s · backward' % self.name)
            tr.formula('dm = down.backward(dy)')
            tr.formula('ds = dm ⊙ u   ;   du = dm ⊙ s', 'm = s⊙u，對一邊微分得到另一邊')
            tr.formula("dg = ds ⊙ SiLU'(g)", "SiLU'(z) = σ(z)·(1 + z·(1−σ(z)))")
            tr.formula('dx = gate.backward(dg) + up.backward(du)', 'x 走了兩條路 → 梯度相加')
            tr.vector('dm', dm.reshape(-1, dm.shape[-1])[0])
            tr.vector('dg', dg.reshape(-1, dg.shape[-1])[0])
            tr.vector('dx (往下傳)', dx.reshape(-1, dx.shape[-1])[0])
        return dx

    # ---------------------------------------------------------------
    def zero_grad(self):
        for m in (self.gate, self.up, self.down):
            m.zero_grad()

    def params(self):
        return self.gate.params() + self.up.params() + self.down.params()

    def numel(self):
        return sum(m.numel() for m in (self.gate, self.up, self.down))
