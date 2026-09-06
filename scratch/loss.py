"""
Softmax + Cross-Entropy —— 整條鏈唯一知道「正確答案」的地方

------------------------------------------------------------------
前向
------------------------------------------------------------------
    p_v = exp(z_v) / Σ_k exp(z_k)              softmax
    L   = − (1/N) Σ_t  log p_t[ y_t ]          cross-entropy（取平均）

實作上有一個必要的技巧：先減掉最大值。
    exp(z_v) 當 z_v = 90 時就溢位成 inf 了（float32 上限約 3.4e38）。
    但 softmax 對「整排一起平移」是不變的：
        exp(z_v − c) / Σ exp(z_k − c) = exp(z_v)/Σ exp(z_k)
    取 c = max(z)，指數最大就是 exp(0) = 1，永遠不會爆。
    這叫 log-sum-exp trick，不是近似，是恆等變換。

------------------------------------------------------------------
反向：softmax 和 cross-entropy 合起來推，會出現驚人的簡化
------------------------------------------------------------------
單獨對 softmax 微分很醜（會出現一個 V×V 的 Jacobian）：
    ∂p_i/∂z_j = p_i(δ_ij − p_j)

但如果把後面的 cross-entropy 接上去一起推：
    L = −log p_y
    ∂L/∂z_j = Σ_i (∂L/∂p_i)(∂p_i/∂z_j)
            = −(1/p_y)·p_y(δ_yj − p_j)
            = p_j − δ_yj

整個 Jacobian 消失了，只剩下：

        ∂L/∂z = p − y            （y 是正解的 one-hot）

這句話白話講就是：**正解那一格的分數要拉高，其他全部壓低。**
數值上：正解那格是 p−1（負的，代表要往上推），其他格是 +p（要往下壓）。

這就是整個訓練過程中，唯一含有「對錯」資訊的地方。
往下每一層都只是在做微分，不知道任務是什麼、不知道答案是什麼。

------------------------------------------------------------------
ignore_index
------------------------------------------------------------------
padding 的位置不該算進 loss。標記成 −100 的位置：
  · 不進 loss 的平均（分母 N 也不算它）
  · 梯度直接設 0
所以那些位置對整個模型完全沒有影響。
"""
import torch

from .tracer import NULL

IGNORE = -100


class SoftmaxCrossEntropy:
    name = 'softmax_ce'

    def __init__(self):
        self._p = None
        self._targets = None
        self._valid = None
        self._N = 0

    # ---------------------------------------------------------------
    def forward(self, logits, targets, tr=NULL):
        """
        logits  : (B, T, V)
        targets : (B, T)   整數，-100 代表忽略
        回傳    : 純量 loss
        """
        B, T, V = logits.shape
        z = logits.reshape(-1, V)
        y = targets.reshape(-1)

        valid = (y != IGNORE)
        N = int(valid.sum())

        # --- softmax（含減最大值）---
        zmax = z.max(dim=-1, keepdim=True).values
        e = torch.exp(z - zmax)
        p = e / e.sum(dim=-1, keepdim=True)

        # --- 取出正解那一格的機率，再取 −log ---
        safe_y = torch.where(valid, y, torch.zeros_like(y))
        p_correct = p.gather(1, safe_y.unsqueeze(1)).squeeze(1)
        nll = -torch.log(p_correct.clamp_min(1e-12))
        loss = (nll * valid).sum() / max(N, 1)

        self._p, self._targets, self._valid, self._N = p, safe_y, valid, max(N, 1)
        self._shape = (B, T, V)

        if tr.enabled:
            tr.section('softmax + cross-entropy · forward')
            tr.formula('p = exp(z − max z) / Σ exp(z − max z)', '減最大值只為了數值安全，結果不變')
            tr.formula('L = −(1/N) Σ log p[正解]')
            tr.scalar('有效位置數 N', N, '（%d 個位置被 ignore_index 排除）' % (len(y) - N))
            i = int(valid.nonzero()[0])
            row = p[i]
            top = row.topk(5)
            tr.text('    以第 %d 個有效位置為例，正解 id = %d：' % (i, int(y[i])))
            tr.table(['排名', 'token id', '機率'],
                     [[str(k + 1), str(int(top.indices[k])), float(top.values[k])] for k in range(5)],
                     [8, 12, 14])
            tr.scalar('正解的機率 p[y]', float(row[y[i]]))
            tr.scalar('這個位置的 −log p', float(nll[i]))
            tr.scalar('平均後的 loss', float(loss))
        return loss

    # ---------------------------------------------------------------
    def backward(self, tr=NULL):
        """
        回傳 dlogits，shape 跟 logits 一樣。

        公式就一行：dz = (p − y) / N
        除以 N 是因為 forward 取了平均。
        """
        p = self._p.clone()

        # p − y：把正解那一格減 1
        p.scatter_add_(1, self._targets.unsqueeze(1),
                       -torch.ones_like(self._targets, dtype=p.dtype).unsqueeze(1))

        # 被忽略的位置梯度歸零
        p = p * self._valid.unsqueeze(1).to(p.dtype)

        dz = p / self._N

        if tr.enabled:
            B, T, V = self._shape
            tr.section('softmax + cross-entropy · backward')
            tr.formula('dz = ( p − y ) / N', 'y 是正解的 one-hot。整條鏈只有這裡知道答案。')
            i = int(self._valid.nonzero()[0])
            yi = int(self._targets[i])
            row = self._p[i]
            top = row.topk(4)
            rows = []
            for k in range(4):
                v = int(top.indices[k])
                rows.append([str(v), float(row[v]), 1.0 if v == yi else 0.0,
                             float(dz[i, v]),
                             '拉上來' if v == yi else '壓下去'])
            if yi not in top.indices.tolist():
                rows.append([str(yi), float(row[yi]), 1.0, float(dz[i, yi]), '拉上來（正解）'])
            tr.table(['token id', 'p', 'y', 'dz = (p−y)/N', '作用'], rows, [12, 12, 8, 16, 16])
            tr.note('正解那一格 dz 是負的 → 更新時 E ← E − lr·dz，負負得正，分數被推高。\n'
                    '其他格 dz 是正的 → 分數被壓低。\n'
                    '整個訓練的「監督訊號」就只有這一行，往下全部是機械式的微分。')
        return dz.reshape(self._shape)

    # ---------------------------------------------------------------
    def zero_grad(self):
        pass

    def params(self):
        return []
