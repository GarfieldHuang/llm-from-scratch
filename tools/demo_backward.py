"""
兩條反向傳播規則，逐元素推導一遍

    python tools/demo_backward.py

用刻意挑過的數字（10、100、1000），讓每一項的來源都追得出來。
最後用有限差分驗證。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
L = '=' * 72


def fd(f, t, i, eps=1e-6):
    """有限差分：把 t 的第 i 個元素推一點點，看 f 變多少"""
    flat = t.reshape(-1)
    o = flat[i].item()
    flat[i] = o + eps; lp = f()
    flat[i] = o - eps; lm = f()
    flat[i] = o
    return (lp - lm) / (2 * eps)


# ==================================================================
print(L)
print('  規則一：y = a ⊙ b')
print(L)
a = torch.tensor([2., 3., 5.])
b = torch.tensor([7., 11., 13.])
y = a * b
dy = torch.tensor([10., 100., 1000.])      # 假裝上游傳了這些下來

print('  a  =', a.tolist())
print('  b  =', b.tolist())
print('  y  = a ⊙ b =', y.tolist())
print('  dy = ∂L/∂y =', dy.tolist(), '  ← 上游給的')
print()
print('  ---- 先看純量版本（一個維度）----')
print('     y₁ = a₁ × b₁ = 2 × 7 = 14')
print('     把 a₁ 加 1，y₁ 會加多少？加 b₁ = 7')
print('     所以 ∂y₁/∂a₁ = b₁ = 7')
print()
print('     鏈鎖律：∂L/∂a₁ = (∂L/∂y₁) × (∂y₁/∂a₁) = dy₁ × b₁')
print('                    = %g × %g = %g' % (dy[0], b[0], dy[0] * b[0]))
print()
print('  ---- 逐元素就是「純量版本重複 n 次」----')
print('     關鍵：a₁ 只影響 y₁，完全不影響 y₂、y₃。')
print('     每個維度是獨立的，所以直接照抄純量的結果。')
print()
print('     %-6s %-14s %-14s %s' % ('維度', 'dy_i', 'b_i', '∂L/∂a_i = dy_i × b_i'))
for i in range(3):
    print('     %-6d %-14g %-14g %g × %g = %g'
          % (i + 1, dy[i], b[i], dy[i], b[i], dy[i] * b[i]))
print()
print('     寫成向量：∂L/∂a = dy ⊙ b =', (dy * b).tolist())
print('               ∂L/∂b = dy ⊙ a =', (dy * a).tolist())
print()
print('  ---- 為什麼 ∂L/∂a 裡面出現的是 b ----')
print('     b 是「放大倍率」。b₂ = 11 代表 a₂ 動一點點，y₂ 會動 11 倍。')
print('     所以梯度傳回來的時候也要乘以 11。')

da = dy * b
db = dy * a
print()
print('  ---- 有限差分驗證 ----')
w1 = max(abs(fd(lambda: float(((a * b) * dy).sum()), a, i) - da[i].item()) for i in range(3))
w2 = max(abs(fd(lambda: float(((a * b) * dy).sum()), b, i) - db[i].item()) for i in range(3))
print('     ∂L/∂a  最大誤差 %.2e' % w1)
print('     ∂L/∂b  最大誤差 %.2e' % w2)


# ==================================================================
print()
print(L)
print('  規則二：y = x @ Wᵀ')
print(L)
x = torch.tensor([[2., 3., 5.]])                       # (1,3)
W = torch.tensor([[1., 0., -1.], [4., 2., 0.]])        # (2,3)
y = x @ W.t()                                          # (1,2)
dy = torch.tensor([[10., 100.]])                       # (1,2)

print('  x shape %s = %s' % (tuple(x.shape), x.tolist()))
print('  W shape %s = %s' % (tuple(W.shape), W.tolist()))
print('  y = x @ Wᵀ shape %s = %s' % (tuple(y.shape), y.tolist()))
print('  dy =', dy.tolist(), '  ← 上游給的')
print()
print('  ---- 把 y 完全展開 ----')
print('     y₁ = x₁·W₁₁ + x₂·W₁₂ + x₃·W₁₃ = %g·%g + %g·%g + %g·%g = %g'
      % (x[0, 0], W[0, 0], x[0, 1], W[0, 1], x[0, 2], W[0, 2], y[0, 0]))
print('     y₂ = x₁·W₂₁ + x₂·W₂₂ + x₃·W₂₃ = %g·%g + %g·%g + %g·%g = %g'
      % (x[0, 0], W[1, 0], x[0, 1], W[1, 1], x[0, 2], W[1, 2], y[0, 1]))
print()
print('  ---- ∂L/∂x：一個輸入影響「多個」輸出，所以要把路徑加起來 ----')
print('     注意 x₁ 同時出現在 y₁ 和 y₂ 裡面。這是跟 ⊙ 最大的差別。')
print()
print('     ∂L/∂x₁ = dy₁·(∂y₁/∂x₁) + dy₂·(∂y₂/∂x₁)')
print('            = dy₁·W₁₁      + dy₂·W₂₁')
for j in range(3):
    print('     ∂L/∂x%d = %g×%g + %g×%g = %g'
          % (j + 1, dy[0, 0], W[0, j], dy[0, 1], W[1, j],
             dy[0, 0] * W[0, j] + dy[0, 1] * W[1, j]))
print()
print('     x_j 配到的是 W 的「第 j 行」（W₁ⱼ 和 W₂ⱼ）。')
print('     矩陣乘法 dy @ W 取的剛好就是 W 的行 —— 所以是 @ W，不是 @ Wᵀ。')
print()
print('     ∂L/∂x = dy @ W =', (dy @ W).tolist())
print()
print('  ---- ∂L/∂W：每個權重只出現在「一個」輸出裡 ----')
print('     W_ij 只出現在 y_i 裡，而且乘的是 x_j。')
print('     所以 ∂L/∂W_ij = dy_i × x_j     （沒有加總，只有一項）')
print()
print('     %-8s %-12s %-12s %s' % ('W_ij', 'dy_i', 'x_j', '∂L/∂W_ij'))
for i in range(2):
    for j in range(3):
        print('     W%d%d      %-12g %-12g %g × %g = %g'
              % (i + 1, j + 1, dy[0, i], x[0, j], dy[0, i], x[0, j],
                 dy[0, i] * x[0, j]))
print()
print('     這是「外積」：每個 dy_i 乘上每個 x_j，排成一個矩陣。')
print('     ∂L/∂W = dyᵀ @ x =', (dy.t() @ x).tolist())

print()
print('  ---- 形狀檢查法（實務上最好用的記憶法）----')
print('     梯度的形狀，永遠跟被微分的東西一樣。光靠這點就能推出排列方式：')
print()
print('     ∂L/∂x 要是 %s  ->  dy%s @ W%s  只有這樣接得起來'
      % (tuple(x.shape), tuple(dy.shape), tuple(W.shape)))
print('     ∂L/∂W 要是 %s  ->  dyᵀ%s @ x%s'
      % (tuple(W.shape), tuple(dy.t().shape), tuple(x.shape)))
print()
print('     忘記公式的時候，把形狀寫出來，答案通常只有一種排法。')

dx = dy @ W
dW = dy.t() @ x
print()
print('  ---- 有限差分驗證 ----')
w3 = max(abs(fd(lambda: float(((x @ W.t()) * dy).sum()), x, i) - dx.reshape(-1)[i].item())
         for i in range(3))
w4 = max(abs(fd(lambda: float(((x @ W.t()) * dy).sum()), W, i) - dW.reshape(-1)[i].item())
         for i in range(6))
print('     ∂L/∂x  最大誤差 %.2e' % w3)
print('     ∂L/∂W  最大誤差 %.2e' % w4)


# ==================================================================
print()
print(L)
print('  兩條規則的差別，一句話')
print(L)
print('  ⊙   一個輸入只影響「一個」輸出   ->  不用加總，直接乘')
print('  @   一個輸入影響「多個」輸出     ->  要把所有路徑加起來')
print()
print('  這也是為什麼 @ 的反向會出現矩陣乘法：')
print('  矩陣乘法本身就是「乘完再加總」，正好就是在做路徑加總。')
