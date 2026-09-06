"""
AdamW —— 唯一真正改動權重的地方

反向傳播只負責「算出該往哪修」，它從不修改任何參數。
真正動手的是這裡，而且方向相反、步伐很小。

------------------------------------------------------------------
更新公式（逐步）
------------------------------------------------------------------
    t ← t + 1                                   步數

    p ← p − lr·wd·p                             ① decoupled weight decay
                                                   跟梯度無關，每步都把權重往 0 拉一點

    m ← β₁·m + (1−β₁)·g                         ② 一階動量：梯度的移動平均
    v ← β₂·v + (1−β₂)·g²                        ③ 二階動量：梯度平方的移動平均

    m̂ ← m / (1 − β₁ᵗ)                           ④ bias correction
    v̂ ← v / (1 − β₂ᵗ)                              m、v 從 0 開始，前幾步會偏小，除掉這個係數補回來

    p ← p − lr · m̂ / (√v̂ + ε)                   ⑤ 真正的更新

------------------------------------------------------------------
為什麼位移量幾乎跟梯度大小無關
------------------------------------------------------------------
看 ⑤ 的 m̂/(√v̂+ε)：分子是梯度的平均，分母是梯度大小的平均。
兩者量級相同，相除之後大約落在 ±1 附近，於是每一步的位移 ≈ lr。

第一步時這件事是精確成立的：
    m₁ = (1−β₁)g          m̂₁ = m₁/(1−β₁) = g
    v₁ = (1−β₂)g²         v̂₁ = v₁/(1−β₂) = g²
    m̂/√v̂ = g/|g| = sign(g)         → 位移恰好是 ±lr

推論：跑 T 步之後，任何一個參數的總位移上限大約是 T × lr，
不管它的梯度是大是小。差別只在「方向有沒有一致」——
方向一致的走得遠，方向互相抵消的原地打轉。

------------------------------------------------------------------
decoupled weight decay 是什麼意思
------------------------------------------------------------------
舊的 Adam 是把 wd·p 加進梯度裡（p ← p − lr·(g + wd·p)），
但這樣一來 weight decay 也會被 √v̂ 正規化，強度變得跟梯度大小有關，很不直覺。

AdamW 把它拆出來獨立做（上面的 ①），強度固定是 lr·wd。這就是 W 的由來。

實務上要注意：預設會對「所有」參數施加 weight decay，包括 embedding 和 norm 的 gain。
生產級訓練通常把這兩類排除掉，因為很少拿到梯度的列會被 wd 慢慢磨向 0。
這份程式用 `no_decay` 參數控制。
"""
import math
import torch

from .tracer import NULL


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, no_decay=('.g', '.E')):
        """
        params    : [(tensor, grad_getter, name), ...]，由 model.params() 提供
        no_decay  : 名稱結尾符合這些字串的參數不做 weight decay
                    預設排除 norm 的 g 和 embedding 的 E
        """
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.no_decay = tuple(no_decay)
        self.t = 0
        self.m = {n: torch.zeros_like(p) for p, _, n in params}
        self.v = {n: torch.zeros_like(p) for p, _, n in params}

    def _decays(self, name):
        return self.wd > 0 and not name.endswith(self.no_decay)

    # ---------------------------------------------------------------
    def clip_grad_norm(self, max_norm):
        """
        全域梯度裁剪：把所有梯度看成一個超長向量，如果它的長度超過門檻，
        整體按比例縮小。注意是「整體」縮放，各參數之間的相對比例不變。
        """
        total = 0.0
        for _, dget, _ in self.params:
            g = dget()
            total += float((g * g).sum())
        total = math.sqrt(total)
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for _, dget, _ in self.params:
                dget().mul_(scale)
        return total

    # ---------------------------------------------------------------
    def step(self, tr=NULL, watch=None):
        """
        watch : 參數名稱，會把那一顆的每一步中間值印進追蹤檔
        """
        self.t += 1
        t = self.t
        bc1 = 1 - self.b1 ** t
        bc2 = 1 - self.b2 ** t

        if tr.enabled:
            tr.title('AdamW 更新（第 %d 步）' % t,
                     'lr=%.2e  β1=%.3f  β2=%.3f  wd=%.3f' % (self.lr, self.b1, self.b2, self.wd))
            tr.formula('m ← β₁·m + (1−β₁)·g')
            tr.formula('v ← β₂·v + (1−β₂)·g²')
            tr.formula('m̂ = m/(1−β₁ᵗ)   v̂ = v/(1−β₂ᵗ)')
            tr.formula('p ← p − lr·wd·p − lr·m̂/(√v̂+ε)')
            tr.scalar('bias correction 1−β₁ᵗ', bc1)
            tr.scalar('bias correction 1−β₂ᵗ', bc2)

        for p, dget, name in self.params:
            g = dget()

            if tr.enabled and watch and name == watch:
                before = p.reshape(-1)[:4].clone()

            # ① decoupled weight decay
            if self._decays(name):
                p.mul_(1 - self.lr * self.wd)

            # ②③ 動量
            m, v = self.m[name], self.v[name]
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            # ④ bias correction
            mh = m / bc1
            vh = v / bc2

            # ⑤ 更新
            upd = mh / (vh.sqrt() + self.eps)
            p.add_(upd, alpha=-self.lr)

            if tr.enabled and watch and name == watch:
                tr.section('追蹤參數：%s（前 4 個數字）' % name)
                tr.vector('更新前 p', before)
                tr.vector('梯度   g', g.reshape(-1)[:4])
                tr.vector('m̂', mh.reshape(-1)[:4])
                tr.vector('√v̂', vh.sqrt().reshape(-1)[:4])
                tr.vector('m̂/(√v̂+ε)', upd.reshape(-1)[:4])
                tr.vector('位移 Δ', (p.reshape(-1)[:4] - before))
                tr.scalar('lr', self.lr)
                r = [abs(float(x)) / self.lr for x in (p.reshape(-1)[:4] - before)]
                tr.note('|位移| / lr = %s\n'
                        '第 1 步時這幾個數字必定是 1.000——因為 m̂/√v̂ = sign(g)，\n'
                        '位移恰好是 ±lr，跟梯度多大完全無關。'
                        % '  '.join('%.4f' % x for x in r))

    # ---------------------------------------------------------------
    def set_lr(self, lr):
        self.lr = lr

    def stats(self):
        """回傳每一顆參數的梯度範數，用來觀察訓練是否健康。"""
        return [(n, float(dget().norm())) for _, dget, n in self.params]


def cosine_lr(step, total, base_lr, min_ratio=0.1, warmup=0):
    """
    cosine 衰減，前面可選 warmup。
        warmup 期間：線性從 0 升到 base_lr
        之後      ：從 base_lr 平滑降到 base_lr × min_ratio
    """
    if warmup and step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
