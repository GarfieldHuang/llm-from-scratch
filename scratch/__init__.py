"""
scratch —— 從零手刻的 Transformer

規則：
  1. 不使用 torch.nn.Module、torch.autograd、loss.backward()
  2. 不使用 F.softmax / F.scaled_dot_product_attention / nn.Linear 這類包好的函式
  3. 每一層都自己寫 forward() 和 backward()，公式寫在註解裡
  4. 每一個 backward 都要通過有限差分檢查

torch 在這裡只被當成「會用 GPU 的 NumPy」——只用它的張量與基本運算
（加減乘除、矩陣乘法、索引），不用任何自動微分功能。
"""
from .tracer import Tracer, NULL
from .linear import Linear
from .embedding import Embedding, TiedOutputHead
from .rmsnorm import RMSNorm

__all__ = ['Tracer', 'NULL', 'Linear', 'Embedding', 'TiedOutputHead', 'RMSNorm']
