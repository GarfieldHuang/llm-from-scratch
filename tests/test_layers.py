"""
逐層梯度檢查。

跑法：
    python tests/test_layers.py

每一層都用「有限差分」驗證手推的 backward 公式：
    ∂L/∂w ≈ ( L(w+ε) − L(w−ε) ) / 2ε
完全不碰 PyTorch 的 autograd。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.linear import Linear                       # noqa: E402
from scratch.embedding import Embedding, TiedOutputHead  # noqa: E402
from scratch.rmsnorm import RMSNorm                      # noqa: E402
from scratch.swiglu import SwiGLU                        # noqa: E402
from scratch.rope import RoPE                            # noqa: E402
from scratch.attention import CausalSelfAttention        # noqa: E402
from scratch.loss import SoftmaxCrossEntropy             # noqa: E402
from scratch import gradcheck                            # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)      # 用 float64，讓差分誤差不被浮點精度蓋過

results = []
print('梯度檢查（有限差分 vs 手推公式）')
print('=' * 66)


def run(name, layer, x):
    worst, _ = gradcheck.check(layer, x)
    results.append((name, worst))


class Wrap:
    """把介面不一樣的層包成 gradcheck 認得的形狀。"""

    def __init__(self, name, fwd, bwd, params, zero):
        self.name = name
        self._f, self._b, self._p, self._z = fwd, bwd, params, zero

    def forward(self, x):
        return self._f(x)

    def backward(self, dy):
        return self._b(dy)

    def params(self):
        return self._p()

    def zero_grad(self):
        self._z()


# ---------------- 基礎層 ----------------
run('Linear (無 bias)', Linear(6, 4, bias=False, name='Linear 無bias'), torch.randn(3, 6))
run('Linear (有 bias)', Linear(6, 4, bias=True, name='Linear 有bias'), torch.randn(3, 6))
run('RMSNorm', RMSNorm(8, name='RMSNorm'), torch.randn(4, 8))

# ---------------- Embedding（輸入是整數，只檢查參數） ----------------
emb = Embedding(10, 5, name='embed')
ids = torch.tensor([[3, 1, 3, 7]])
run('Embedding (scatter-add)',
    Wrap('Embedding (scatter-add)',
         lambda _: emb.forward(ids),
         lambda dy: (emb.backward(dy), None)[1],
         emb.params, emb.zero_grad),
    torch.zeros(1))

emb2 = Embedding(10, 5, name='embed2')
run('TiedOutputHead', TiedOutputHead(emb2, name='TiedOutputHead'), torch.randn(2, 3, 5))

# ---------------- FFN ----------------
run('SwiGLU (FFN)', SwiGLU(6, 16, name='SwiGLU'), torch.randn(2, 4, 6))

# ---------------- RoPE（無參數，只檢查對輸入的梯度） ----------------
run('RoPE', RoPE(8, max_len=16), torch.randn(1, 2, 5, 8))

# ---------------- Attention ----------------
run('CausalSelfAttention', CausalSelfAttention(12, 3, max_len=16, name='Attention'),
    torch.randn(2, 5, 12))

# ---------------- Softmax + CrossEntropy ----------------
ce = SoftmaxCrossEntropy()
tgt = torch.tensor([[2, 5, -100, 1]])       # 故意放一個 ignore_index


class CEWrap:
    name = 'Softmax+CrossEntropy'

    def forward(self, logits):
        # gradcheck 需要「輸出張量」，這裡把純量 loss 包成 shape (1,)
        return ce.forward(logits, tgt).reshape(1)

    def backward(self, dy):
        # dy 是對 loss 的梯度（純量），乘進去即可
        return ce.backward() * float(dy.reshape(-1)[0])

    def params(self):
        return []

    def zero_grad(self):
        pass


run('Softmax+CrossEntropy', CEWrap(), torch.randn(1, 4, 7))

print('=' * 66)
os.makedirs('traces', exist_ok=True)
p = gradcheck.report(results, os.path.join('traces', 'gradcheck.txt'))
worst = max(w for _, w in results)
print('最差相對誤差 %.2e   ->  %s' % (worst, '全部通過' if worst < 2e-3 else '有問題'))
print('明細寫入', p)
sys.exit(0 if worst < 2e-3 else 1)
