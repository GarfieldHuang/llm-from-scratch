"""
RMSNorm —— 每一層的入口都會先過它

------------------------------------------------------------------
前向
------------------------------------------------------------------
    r     = sqrt( mean(x²) + eps )        ← 這一列數字的「均方根」
    x̂     = x / r                          ← 正規化，長度被拉到約 sqrt(n)
    y     = g ⊙ x̂                          ← g 是可學的縮放，每個維度一個

跟 LayerNorm 的差別：LayerNorm 會先減掉平均數，RMSNorm 不減。
少一個減法、少一個 bias，效果幾乎一樣，所以現在的 LLM 幾乎都用 RMSNorm。

------------------------------------------------------------------
反向（手推，逐步）
------------------------------------------------------------------
設 n = 維度數，r = rms，a_i = dy_i · g_i

先求 r 對 x 的偏導：
    r = ( (1/n)Σ x_k² + eps )^(1/2)
    ∂r/∂x_j = (1/2)·( ... )^(-1/2) · (2 x_j / n) = x_j / (n·r)

再求 y 對 x 的偏導（注意 y_i 透過兩條路依賴 x_j：
分子的 x_i 本身，以及分母 r 裡面的 x_j）：
    y_i = g_i · x_i / r
    ∂y_i/∂x_j = g_i · [ δ_ij / r  −  x_i·(∂r/∂x_j) / r² ]
              = g_i · [ δ_ij / r  −  x_i·x_j / (n·r³) ]

鏈鎖律加總：
    dx_j = Σ_i dy_i · ∂y_i/∂x_j
         = a_j / r  −  x_j·(Σ_i a_i·x_i) / (n·r³)

把 x = x̂·r 代進去化簡，得到最後要寫進程式的形式：

    dx = ( a − x̂ · mean(a ⊙ x̂) ) / r          a = dy ⊙ g

對 g 的梯度單純得多（y = g⊙x̂，g 只出現一次）：

    dg = Σ(所有位置) dy ⊙ x̂

------------------------------------------------------------------
順帶解釋一個實測到的現象
------------------------------------------------------------------
embedding 的向量很短（每個元素的 rms 大約 0.03），除以 r 等於乘上約 30 倍。
反向經過同一個除法時，梯度也會被放大差不多的倍數。
這就是為什麼梯度從第 1 層傳到 embedding 時會突然變大一個數量級。
（第二項 −x̂·mean(a⊙x̂) 會抵消掉一部分，所以實測倍率會比 1/rms 略小。）
"""
import torch

from .tracer import NULL


class RMSNorm:
    def __init__(self, dim, eps=1e-5, device='cpu', dtype=None, name='norm'):
        self.dim = dim
        self.eps = eps
        self.name = name
        self.g = torch.ones(dim, device=device, dtype=dtype)   # 從 1 開始 = 一開始不改變輸入
        self.dg = torch.zeros_like(self.g)
        self._xhat = None
        self._r = None

    # ---------------------------------------------------------------
    def forward(self, x, tr=NULL):
        # mean(x²) 沿最後一維（每個位置各自正規化，位置之間互不影響）
        ms = (x * x).mean(dim=-1, keepdim=True)
        r = torch.sqrt(ms + self.eps)
        xhat = x / r
        y = self.g * xhat

        self._xhat, self._r = xhat, r

        if tr.enabled:
            tr.section('%s · forward' % self.name)
            tr.formula('r = sqrt(mean(x²) + eps)')
            tr.formula('x̂ = x / r')
            tr.formula('y = g ⊙ x̂')
            tr.vector('x   (輸入)', x.reshape(-1, self.dim)[0])
            tr.scalar('r   (第 0 個位置的 rms)', float(r.reshape(-1)[0]))
            tr.vector('x̂   (正規化後)', xhat.reshape(-1, self.dim)[0])
            tr.vector('g   (可學縮放)', self.g)
            tr.vector('y   (輸出)', y.reshape(-1, self.dim)[0])
            tr.note('輸入的 rms = %.6f，所以這一步等於把數值放大約 %.1f 倍。\n'
                    '反向經過同一個除法時，梯度也會被放大差不多的倍數。'
                    % (float(r.reshape(-1)[0]), 1.0 / float(r.reshape(-1)[0])))
        return y

    # ---------------------------------------------------------------
    def backward(self, dy, tr=NULL):
        xhat, r = self._xhat, self._r

        # 對 g：y = g⊙x̂，所以 dg = Σ dy⊙x̂（把所有位置加起來）
        self.dg += (dy * xhat).reshape(-1, self.dim).sum(dim=0)

        # 對 x：dx = ( a − x̂·mean(a⊙x̂) ) / r
        a = dy * self.g
        corr = (a * xhat).mean(dim=-1, keepdim=True)     # mean(a ⊙ x̂)
        dx = (a - xhat * corr) / r

        if tr.enabled:
            tr.section('%s · backward' % self.name)
            tr.formula('a  = dy ⊙ g')
            tr.formula('dx = ( a − x̂ · mean(a ⊙ x̂) ) / r')
            tr.formula('dg = Σ dy ⊙ x̂')
            tr.vector('dy  (上游)', dy.reshape(-1, self.dim)[0])
            tr.vector('a', a.reshape(-1, self.dim)[0])
            tr.scalar('mean(a ⊙ x̂)', float(corr.reshape(-1)[0]))
            tr.vector('dx  (往下傳)', dx.reshape(-1, self.dim)[0])
            amp = float(dx.norm() / dy.norm()) if float(dy.norm()) > 0 else 0.0
            tr.note('第二項 −x̂·mean(a⊙x̂) 的作用是「把沿著 x̂ 方向的分量扣掉」。\n'
                    '因為 x̂ 的長度被 r 固定住了，沿它自己方向推是無效的。\n'
                    '這一層讓梯度範數變成原來的 %.2f 倍。' % amp)
        return dx

    # ---------------------------------------------------------------
    def zero_grad(self):
        self.dg.zero_()

    def params(self):
        return [(self.g, lambda: self.dg, '%s.g' % self.name)]

    def numel(self):
        return self.g.numel()
