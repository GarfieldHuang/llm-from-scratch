"""
RoPE（旋轉位置編碼）—— 位置資訊在這裡才被注入

------------------------------------------------------------------
為什麼不是加在 embedding 上
------------------------------------------------------------------
原始 Transformer / BERT / GPT-2 的做法是 x = E[id] + PE[pos]，
位置資訊直接混進詞向量裡。

RoPE 不碰 embedding，改成在 Attention 內部「旋轉」Q 和 K。
好處是「這個 token 是什麼」跟「它在第幾個位置」保持解耦：
詞向量純粹編碼身份，位置只在算注意力分數時才起作用。

------------------------------------------------------------------
做法
------------------------------------------------------------------
把 head_dim 維的向量看成 d/2 組二維座標，每一組用一個不同的頻率旋轉，
旋轉角度正比於它所在的位置 m：

    θ_i   = base^(−2i/d)            第 i 組的頻率（i 越大轉越慢）
    角度  = m · θ_i                  位置 m 的第 i 組要轉這麼多

實作上採 LLaMA 的「前後半配對」寫法（跟逐對配對數學等價，只是排列不同）：

    x1 = x[..., :d/2]    x2 = x[..., d/2:]
    y1 = x1·cos − x2·sin
    y2 = x2·cos + x1·sin

寫成一行：
    y = x ⊙ cos + rotate_half(x) ⊙ sin
    其中 rotate_half(x) = concat(−x2, x1)

------------------------------------------------------------------
為什麼旋轉能表達「相對位置」
------------------------------------------------------------------
兩個向量旋轉後做點積：
    (R_m · q) · (R_n · k) = q^T R_m^T R_n k = q^T R_{n−m} k

旋轉矩陣正交，所以 R_m^T R_n 只跟 (n−m) 有關。
也就是說，注意力分數自動只依賴「相對距離」，跟絕對位置無關。
這就是 RoPE 能外推到訓練時沒見過的長度的原因。

------------------------------------------------------------------
反向
------------------------------------------------------------------
旋轉沒有參數，而且是正交變換，反向就是轉回去（角度取負）：

    y1 = x1·cos − x2·sin        dx1 = dy1·cos + dy2·sin
    y2 = x2·cos + x1·sin        dx2 = dy2·cos − dy1·sin

合起來剛好是：
    dx = dy ⊙ cos − rotate_half(dy) ⊙ sin       （只有 sin 前面變號）
"""
import torch

from .tracer import NULL


def rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def build_tables(head_dim, max_len, base=10000.0, device='cpu', dtype=None):
    """預先算好每個位置、每個頻率的 cos / sin。這是常數，不參與訓練。"""
    dtype = dtype or torch.get_default_dtype()
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(max_len, device=device, dtype=dtype)
    ang = pos[:, None] * inv[None, :]           # (max_len, half)
    ang = torch.cat((ang, ang), dim=-1)         # 前後半用同一組角度
    return ang.cos(), ang.sin()


class RoPE:
    name = 'rope'

    def __init__(self, head_dim, max_len, base=10000.0, device='cpu', dtype=None):
        self.head_dim = head_dim
        self.cos, self.sin = build_tables(head_dim, max_len, base, device, dtype)
        self._c = None
        self._s = None

    def forward(self, x, start=0, tr=NULL):
        """x : (B, heads, T, head_dim)"""
        T = x.shape[-2]
        c = self.cos[start:start + T]           # (T, head_dim)
        s = self.sin[start:start + T]
        self._c, self._s = c, s

        y = x * c + rotate_half(x) * s

        if tr.enabled:
            tr.section('RoPE · forward')
            tr.formula('y = x ⊙ cos + rotate_half(x) ⊙ sin')
            tr.formula('θ_i = base^(−2i/d) ,  角度 = 位置 × θ_i')
            tr.vector('位置 0 的 cos', c[0], n=6)
            tr.vector('位置 1 的 cos', c[1] if T > 1 else c[0], n=6)
            tr.vector('x  (旋轉前)', x.reshape(-1, self.head_dim)[0], n=6)
            tr.vector('y  (旋轉後)', y.reshape(-1, self.head_dim)[0], n=6)
            n0 = float(x.reshape(-1, self.head_dim)[0].norm())
            n1 = float(y.reshape(-1, self.head_dim)[0].norm())
            tr.note('旋轉是正交變換，長度不變：%.6f → %.6f\n'
                    '它只改變方向，不改變向量的大小。' % (n0, n1))
        return y

    def backward(self, dy, tr=NULL):
        dx = dy * self._c - rotate_half(dy) * self._s
        if tr.enabled:
            tr.section('RoPE · backward')
            tr.formula('dx = dy ⊙ cos − rotate_half(dy) ⊙ sin', '就是反方向轉回去')
            tr.vector('dx', dx.reshape(-1, self.head_dim)[0], n=6)
        return dx

    def zero_grad(self):
        pass

    def params(self):
        return []       # RoPE 沒有可學參數

    def numel(self):
        return 0
