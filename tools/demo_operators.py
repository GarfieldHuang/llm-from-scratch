"""
⊙ 與 @ 的差別 —— 兩個都叫「乘」，但完全是兩回事

    python tools/demo_operators.py

這份文件裡（以及所有深度學習論文裡）最常混淆的就是這兩個符號。
用最小的矩陣把它們算出來，差別一眼就看得到。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_printoptions(precision=0, sci_mode=False)
L = '=' * 68


def show(name, t):
    rows = [' '.join('%6.0f' % float(v) for v in r) for r in t]
    print('  %-14s [ %s ]' % (name, (' ]\n%17s[ ' % '').join(rows)))


A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

print(L)
print('  兩個一樣的輸入')
print(L)
show('A', A)
print()
show('B', B)

print()
print(L)
print('  ⊙  逐元素相乘（Hadamard product）')
print(L)
print('  規則：形狀必須一樣，位置對位置相乘，形狀不變')
print()
print('       (A ⊙ B)[i][j] = A[i][j] × B[i][j]')
print()
show('A ⊙ B', A * B)
print()
print('  驗算左上角：A[0][0] × B[0][0] = %g × %g = %g'
      % (A[0, 0], B[0, 0], (A * B)[0, 0]))
print('  每一格只跟「同一格」的對手相乘，跟其他格完全無關。')
print('  → 維度之間不會互相混合')

print()
print(L)
print('  @  矩陣乘法（matrix multiplication）')
print(L)
print('  規則：(m,k) @ (k,n) -> (m,n)，左邊的列跟右邊的行做點積')
print()
print('       (A @ B)[i][j] = Σ_k  A[i][k] × B[k][j]')
print()
show('A @ B', A @ B)
print()
print('  驗算左上角：A 的第 0 列 · B 的第 0 行')
print('              = %g×%g + %g×%g = %g'
      % (A[0, 0], B[0, 0], A[0, 1], B[1, 0], (A @ B)[0, 0]))
print('  每一格都是「一整列」跟「一整行」的點積，用到了所有維度。')
print('  → 維度會互相混合')

print()
print(L)
print('  形狀規則的差別')
print(L)
x = torch.randn(4, 8)
W = torch.randn(3, 8)
g = torch.randn(8)
print('  x shape %s' % (tuple(x.shape),))
print()
print('  x @ W.T   %-12s  (4,8) @ (8,3) -> (4,3)   最後一維被換掉了'
      % (tuple((x @ W.t()).shape),))
print('  x * g     %-12s  (4,8) ⊙ (8,)  -> (4,8)   形狀完全不變'
      % (tuple((x * g).shape),))
print()
print('  x * g 用到了「廣播」：g 只有 8 個數字，會自動複製 4 份，')
print('  對 x 的每一列各做一次逐元素相乘。')

print()
print(L)
print('  在這個模型裡各自出現在哪')
print(L)
rows = [
    ('@  混合維度', 'Linear      y = x @ W.T', '把 d 維換成另外 d_out 維'),
    ('', 'Attention   Q @ K.T', '每個位置跟每個位置做點積'),
    ('', '輸出層      h @ E.T', '隱藏狀態跟每個詞向量做點積'),
    ('⊙  不混合維度', 'RMSNorm     y = g ⊙ x̂', '每個維度乘上自己的縮放係數'),
    ('', 'SwiGLU      m = s ⊙ u', '閘門：每個維度各自決定放行多少'),
    ('', 'RoPE        x ⊙ cos', '每個維度乘上自己的旋轉係數'),
]
for a, b, c in rows:
    print('  %-16s %-26s %s' % (a, b, c))

print()
print('  重點：⊙ 永遠不會把不同維度的資訊混在一起。')
print('        所以模型的「學習能力」幾乎全部來自 @，⊙ 只負責調整強度。')

print()
print(L)
print('  反向傳播的規則也不一樣')
print(L)
print('  y = a ⊙ b     ->   dL/da = dL/dy ⊙ b       dL/db = dL/dy ⊙ a')
print('  y = x @ W.T   ->   dL/dx = dL/dy @ W       dL/dW = dL/dy.T @ x')
print()
print('  兩個都符合「點積對其中一邊微分，得到的就是另一邊」。')

# 用有限差分驗證上面兩條規則
print()
print(L)
print('  驗證（有限差分，不用 autograd）')
print(L)
torch.set_default_dtype(torch.float64)
a = torch.randn(3, 4)
b = torch.randn(3, 4)
dy = torch.randn(3, 4)
da = dy * b                                    # 手推
eps, worst = 1e-5, 0.0
for i in range(3):
    for j in range(4):
        o = a[i, j].item()
        a[i, j] = o + eps; lp = float(((a * b) * dy).sum())
        a[i, j] = o - eps; lm = float(((a * b) * dy).sum())
        a[i, j] = o
        worst = max(worst, abs((lp - lm) / (2 * eps) - da[i, j].item()))
print('  ⊙ 的 dL/da   最大誤差 %.2e' % worst)

x2 = torch.randn(3, 5)
W2 = torch.randn(4, 5)
dy2 = torch.randn(3, 4)
dx2 = dy2 @ W2                                 # 手推
worst = 0.0
for i in range(3):
    for j in range(5):
        o = x2[i, j].item()
        x2[i, j] = o + eps; lp = float(((x2 @ W2.t()) * dy2).sum())
        x2[i, j] = o - eps; lm = float(((x2 @ W2.t()) * dy2).sum())
        x2[i, j] = o
        worst = max(worst, abs((lp - lm) / (2 * eps) - dx2[i, j].item()))
print('  @ 的 dL/dx   最大誤差 %.2e' % worst)
print()
print('  兩條規則都對。')
