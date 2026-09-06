"""
Embedding 層：查表，以及它的反向 scatter-add

------------------------------------------------------------------
它其實是一個「輸入是 one-hot 的線性層」
------------------------------------------------------------------
    x = one_hot(id) @ E          E shape (vocab, dim)

one_hot(id) 是一個只有一個 1、其餘全 0 的列向量。乘出來的結果，
就是把 E 的第 id 列原封不動抄出來。

既然 99.98% 的乘法都是「乘以 0」，實作上當然直接用索引：
    x = E[id]

兩者結果逐位元相同，但查表快 20 倍以上、省 8 倍記憶體。

------------------------------------------------------------------
反向：為什麼是 scatter-add
------------------------------------------------------------------
把前向寫成矩陣形式 x = O·E（O 是常數 one-hot 矩陣），
套用線性層的標準結果 ∂L/∂E = O^T · G：

    O^T 的第 v 列，在「第 t 個位置的 id 等於 v」的地方是 1，其餘是 0。
    所以 (O^T·G)[v] = Σ_{t : id_t = v} G[t]

翻成白話：**把每個位置的梯度，加回它當初查表的那一列；同一列有多筆就相加。**

由此直接得到兩個常被誤會成「特殊規則」的性質，其實都只是算出來的結果：
  · 同一個 token 出現 k 次 → 那一列收到 k 筆梯度相加（O^T 那一列有 k 個 1）
  · 沒出現過的 token       → 梯度是 0        （O^T 那一列全是 0）

------------------------------------------------------------------
關於「離散的東西怎麼微分」
------------------------------------------------------------------
不對 id 微分。id 是資料不是參數，它只決定「哪些數字參與計算」。
微分是對 E 裡面的每一個實數做的，那些數字是連續的，完全可微。
沒參與計算的數字，斜率算出來自然是 0。
"""
import torch

from .tracer import NULL


class Embedding:
    def __init__(self, vocab_size, dim, device='cpu', dtype=None, name='embed'):
        self.vocab_size = vocab_size
        self.dim = dim
        self.name = name

        # 初始化：常態分佈乘上 0.02。這個數字是 GPT-2 以來的慣例，
        # 用意是讓初始向量夠小，模型不會一開始就對某些 token 有強烈偏好。
        self.E = torch.randn(vocab_size, dim, device=device, dtype=dtype) * 0.02
        self.dE = torch.zeros_like(self.E)
        self._ids = None

    # ---------------------------------------------------------------
    def forward(self, ids, tr=NULL):
        """
        ids : (B, T) 整數張量
        回傳: (B, T, dim)

        整個前向就是一次索引。沒有乘法、沒有加法、沒有非線性。
        """
        self._ids = ids
        x = self.E[ids]          # 這一行就是全部

        if tr.enabled:
            tr.section('%s · forward（查表）' % self.name)
            tr.formula('x = E[ids]', '等價於 x = one_hot(ids) @ E，只是跳過了乘以 0 的部分')
            tr.shape('ids', ids)
            tr.shape('x', x)
            flat = ids.reshape(-1)
            rows = []
            for t in range(min(6, len(flat))):
                v = int(flat[t])
                rows.append(['t=%d' % t, 'id=%d' % v,
                             '  '.join('%+.5f' % float(z) for z in self.E[v, :4])])
            tr.table(['位置', '查哪一列', 'E[id] 的前 4 維'], rows, [8, 12, 46])
            uniq = torch.unique(flat)
            tr.note('這一批用到 %d 個不同的 token，詞表共 %d 個。\n'
                    '剩下 %d 列這一步完全沒被碰到——反向時它們的梯度會是 0。'
                    % (len(uniq), self.vocab_size, self.vocab_size - len(uniq)))
        return x

    # ---------------------------------------------------------------
    def backward(self, dx, tr=NULL):
        """
        dx : (B, T, dim)  上游傳下來的梯度，也就是「修正單」

        embedding 是整條鏈的最底層，沒有東西要再往下傳，所以不回傳 dx。
        它唯一要做的事情是把修正單送回正確的列。
        """
        flat_ids = self._ids.reshape(-1)                 # (N,)
        flat_dx = dx.reshape(-1, self.dim)               # (N, dim)

        # 記下 scatter-add 之前的狀態。
        # 如果有 weight tying，這裡面已經裝著「輸出側」的梯度了
        # （TiedOutputHead.backward 比這一步先跑）。
        before = self.dE.clone() if tr.enabled else None

        # scatter-add：等價於 dE += O^T @ dx
        # index_add_ 的語意就是「把 flat_dx 的第 n 列，加到 dE 的第 flat_ids[n] 列」
        # 重複的索引會自動累加——這正是 O^T 那一列有多個 1 的效果。
        self.dE.index_add_(0, flat_ids, flat_dx)

        if tr.enabled:
            tr.section('%s · backward（scatter-add，輸入側）' % self.name)
            tr.formula('dE[v] += Σ_{t : id_t = v}  dx[t]',
                       '等價於 dE += O^T @ dx，O 是 one-hot 矩陣')
            cnt = torch.bincount(flat_ids, minlength=self.vocab_size)
            rep = int(cnt.argmax())
            pos = (flat_ids == rep).nonzero().reshape(-1)
            tr.text('    「id %d」（%s）在這一批出現 %d 次'
                    % (rep, repr(rep), len(pos)))
            tr.text('    看第 0 維，這 %d 筆修正單相加：' % len(pos))
            for p in pos[:8].tolist():
                tr.text('        位置 %-5d  dx[0] = %+.8e' % (p, float(flat_dx[p, 0])))
            if len(pos) > 8:
                tr.text('        ...(還有 %d 筆，全部列出會太長)' % (len(pos) - 8))
            ssum = float(flat_dx[pos, 0].sum())
            tr.text('        %-24s %+.8e' % ('這 %d 筆相加 =' % len(pos), ssum))
            tr.text('')

            nz_in = int((self.dE - before).abs().sum(1).gt(0).sum())
            nz_out = int(before.abs().sum(1).gt(0).sum())
            nz_all = int(self.dE.abs().sum(1).gt(0).sum())
            tr.text('    這一列的梯度其實是兩條路加起來的：')
            tr.table(['來源', 'dE[%d][0]' % rep, '有幾列非零'],
                     [['輸入側（本步 scatter-add）', ssum, '%d / %d' % (nz_in, self.vocab_size)],
                      ['輸出側（weight tying）', float(before[rep, 0]), '%d / %d' % (nz_out, self.vocab_size)],
                      ['合計 dE[%d][0]' % rep, float(self.dE[rep, 0]), '%d / %d' % (nz_all, self.vocab_size)]],
                     [30, 20, 16])

            tr.note('這裡最容易搞混，特別注意：\n'
                    '\n'
                    '  · 輸入側（查表這條路）確實是稀疏的——只有 %d 列拿到梯度，\n'
                    '    因為沒被查到的列根本沒參與計算，動它們 loss 不會變。\n'
                    '\n'
                    '  · 但輸出側（因為 E 同時是分類器權重）是稠密的——softmax 要對\n'
                    '    全部 %d 個 token 算分數，所以每一列每一步都會被碰到。\n'
                    '\n'
                    '  · 兩條相加，結果 %d / %d 列非零。\n'
                    '\n'
                    '所以「embedding 的梯度是稀疏的」這句話，在有 weight tying 的模型上\n'
                    '並不成立。把 tie_weights 設成 False 再跑一次，就會看到只剩 %d 列。'
                    % (nz_in, self.vocab_size, nz_all, self.vocab_size, nz_in))
        return None

    # ---------------------------------------------------------------
    def zero_grad(self):
        self.dE.zero_()

    def params(self):
        return [(self.E, lambda: self.dE, '%s.E' % self.name)]

    def numel(self):
        return self.E.numel()


class TiedOutputHead:
    """
    輸出層，跟 Embedding 共用同一個 E（weight tying）。

        logits = h @ E^T          logits[t][v] = h[t] · E[v]

    注意這裡的點積是「隱藏狀態 · 詞向量」，不是「詞向量 · 詞向量」。
    訓練過程中從來沒有計算過任何兩個詞向量之間的相似度。

    因為共用權重，E 會收到「第二條」梯度：
        · 輸入側（查表）：稀疏，只有出現過的 token
        · 輸出側（這裡）：稠密，softmax 碰到全部 vocab 列，每一步每一列都有

    兩條加起來才是 E 的完整梯度。這也是為什麼「embedding 梯度是稀疏的」
    這句話在有 tying 的模型上並不成立。
    """

    def __init__(self, embedding, name='lm_head'):
        self.emb = embedding
        self.name = name
        self._h = None

    def forward(self, h, tr=NULL):
        self._h = h
        logits = h @ self.emb.E.t()
        if tr.enabled:
            tr.section('%s · forward（與 embedding 共用權重）' % self.name)
            tr.formula('logits = h @ E^T', 'logits[t][v] = h[t] · E[v]  ← 隱藏狀態 · 詞向量')
            tr.shape('h', h)
            tr.shape('logits', logits)
            tr.matrix('logits', logits, rows=3, cols=8)
        return logits

    def backward(self, dlogits, tr=NULL):
        """
        dlogits : (B, T, vocab)

        回傳 dh 給下游，同時把輸出側的梯度累加進同一個 dE。
        """
        h2 = self._h.reshape(-1, self.emb.dim)                    # (N, dim)
        dl2 = dlogits.reshape(-1, self.emb.vocab_size)            # (N, vocab)

        # 對 E 的梯度（輸出側）：dE[v] += Σ_t dlogits[t][v] · h[t]
        self.emb.dE += dl2.t() @ h2

        # 對 h 的梯度：dh[t] = Σ_v dlogits[t][v] · E[v]
        dh = dlogits @ self.emb.E

        if tr.enabled:
            before_nz = int((dl2.t() @ h2).abs().sum(1).gt(0).sum())
            tr.section('%s · backward' % self.name)
            tr.formula('dE[v] += Σ_t dlogits[t][v] · h[t]', '輸出側梯度：稠密')
            tr.formula('dh[t]  = Σ_v dlogits[t][v] · E[v]', '往下游傳')
            tr.matrix('dh', dh)
            tr.note('這一步讓 dE 有 %d / %d 列拿到梯度。\n'
                    'softmax 要對全部 %d 個 token 算分數，所以每一列都被碰到——\n'
                    '連完全沒出現在這批資料裡的 token 也一樣。'
                    % (before_nz, self.emb.vocab_size, self.emb.vocab_size))
        return dh

    def zero_grad(self):
        pass          # 參數由 Embedding 持有，不重複清除

    def params(self):
        return []     # 同上，避免 optimizer 拿到兩份

    def numel(self):
        return 0
