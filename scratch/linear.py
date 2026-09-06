"""
線性層  y = x @ W^T + b        （最基礎的 Ax + B）

整個 Transformer 裡九成的參數都在這種層：q_proj / k_proj / v_proj / o_proj、
FFN 的三個矩陣、輸出層。把這一個搞懂，其他都是它的組合。

------------------------------------------------------------------
形狀約定（跟 PyTorch nn.Linear 一致，方便對照）
------------------------------------------------------------------
    x : (..., in_features)      輸入
    W : (out_features, in_features)
    b : (out_features,)          可選
    y : (..., out_features)      輸出

    y[i] = Σ_k  x[k] · W[i][k]  +  b[i]
           ^^^^^^^^^^^^^^^^^^^
           就是 A·x + B 裡的 A·x

------------------------------------------------------------------
反向傳播的三條公式（手推）
------------------------------------------------------------------
已知上游傳下來的 dy = ∂L/∂y，要求三樣東西：

1) ∂L/∂x  ——「我的輸入該怎麼動」，這個要繼續往下傳
       y[i] = Σ_k x[k]·W[i][k]
       所以 ∂y[i]/∂x[k] = W[i][k]
       鏈鎖律：∂L/∂x[k] = Σ_i dy[i]·W[i][k]
       寫成矩陣：      dx = dy @ W

2) ∂L/∂W  ——「我的權重該怎麼動」，這個交給 optimizer
       ∂y[i]/∂W[i][k] = x[k]
       ∂L/∂W[i][k] = dy[i]·x[k]           ← 外積
       批次要把所有樣本加起來：dW = dy^T @ x

3) ∂L/∂b
       ∂y[i]/∂b[i] = 1
       ∂L/∂b[i] = Σ(批次) dy[i]           ← 直接加總

注意 dx 用 W，dW 用 x：點積對其中一邊微分，得到的就是另一邊。
這條規律在整份程式碼裡會反覆出現。
"""
import math
import torch

from .tracer import NULL


class Linear:
    def __init__(self, in_features, out_features, bias=False, device='cpu', dtype=None, name='linear'):
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        # Kaiming uniform 初始化：範圍 ±1/sqrt(fan_in)
        # 目的是讓輸出的變異數跟輸入差不多，訊號不會逐層放大或消失
        bound = 1.0 / math.sqrt(in_features)
        self.W = (torch.rand(out_features, in_features, device=device, dtype=dtype) * 2 - 1) * bound
        self.b = torch.zeros(out_features, device=device, dtype=dtype) if bias else None

        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b) if bias else None
        self._x = None      # forward 時存下來，backward 要用

    # ---------------------------------------------------------------
    def forward(self, x, tr=NULL):
        """
        y = x @ W^T + b

        x 存起來的原因：算 dW 需要它（dW = dy^T @ x）。
        這就是「activation 記憶體」的來源——訓練比推論吃記憶體，
        主要就是因為每一層都要存自己的輸入等著反向用。
        """
        self._x = x

        # 這一行就是 Ax：對每個輸出維度 i，把輸入跟 W 的第 i 列做點積。
        # 等價的顯式寫法（慢很多，但意思一樣）：
        #     for i in range(out_features):
        #         y[..., i] = (x * W[i]).sum(-1)
        y = x @ self.W.t()

        if self.b is not None:
            y = y + self.b          # 這是 +B

        if tr.enabled:
            tr.section('%s · forward' % self.name)
            tr.formula('y = x @ W^T' + (' + b' if self.b is not None else ''),
                       'W shape (out=%d, in=%d)' % (self.out_features, self.in_features))
            tr.matrix('x  (輸入)', x)
            tr.matrix('W  (權重)', self.W)
            if self.b is not None:
                tr.vector('b  (偏置)', self.b)
            tr.matrix('y  (輸出)', y)
            # 手算第一個元素，讓讀者能對照
            xf = x.reshape(-1, self.in_features)[0]
            tr.note('驗算 y[0][0]：把 x 的第 0 列跟 W 的第 0 列做點積\n'
                    '  Σ x[k]·W[0][k] = %.6f\n'
                    '  程式算出來的   = %.6f'
                    % (float((xf * self.W[0]).sum()), float(y.reshape(-1, self.out_features)[0, 0])))
        return y

    # ---------------------------------------------------------------
    def backward(self, dy, tr=NULL):
        """
        輸入 dy = ∂L/∂y，輸出 dx = ∂L/∂x，並把 dW / db 累加起來。

        「累加」而不是「覆寫」是刻意的：梯度累積（gradient accumulation）
        跑好幾個 micro-batch 才更新一次，就靠這個累加。
        """
        x2 = self._x.reshape(-1, self.in_features)      # (N, in)
        dy2 = dy.reshape(-1, self.out_features)         # (N, out)

        # ① 對權重的梯度：外積後對批次加總
        #    dW[i][k] = Σ_n dy[n][i] · x[n][k]
        self.dW += dy2.t() @ x2

        # ② 對偏置的梯度：直接把批次加起來
        if self.b is not None:
            self.db += dy2.sum(dim=0)

        # ③ 對輸入的梯度：這個要往下游傳
        #    dx[n][k] = Σ_i dy[n][i] · W[i][k]
        dx = dy @ self.W

        if tr.enabled:
            tr.section('%s · backward' % self.name)
            tr.formula('dW = dy^T @ x        (權重的梯度，交給 optimizer)')
            tr.formula('dx = dy @ W          (輸入的梯度，繼續往下傳)')
            if self.b is not None:
                tr.formula('db = Σ dy            (偏置的梯度)')
            tr.matrix('dy (上游傳下來的)', dy)
            tr.matrix('dW', self.dW)
            tr.matrix('dx (往下傳)', dx)
            tr.note('注意 dx 用到 W、dW 用到 x —— 點積對一邊微分得到另一邊。\n'
                    'dW 是「+=」不是「=」，這樣才能做梯度累積。')
        return dx

    # ---------------------------------------------------------------
    def zero_grad(self):
        self.dW.zero_()
        if self.db is not None:
            self.db.zero_()

    def params(self):
        """回傳 (參數, 梯度) 配對，交給 optimizer。"""
        out = [(self.W, lambda: self.dW, '%s.W' % self.name)]
        if self.b is not None:
            out.append((self.b, lambda: self.db, '%s.b' % self.name))
        return out

    def numel(self):
        return self.W.numel() + (self.b.numel() if self.b is not None else 0)
