"""
字元級 tokenizer —— 刻意選最簡單的做法

------------------------------------------------------------------
為什麼不用 BPE
------------------------------------------------------------------
真實的 LLM 用 BPE（把常見的字元組合併成一個 token），好處是序列變短。
但 BPE 會帶來一堆干擾理解的東西：
  · 一個中文詞可能被拆成 2~3 個沒有意義的 byte 碎片
  · 詞表裡會出現「作為AI語言模型」這種怪東西
  · 你在追蹤檔裡看到的 token 不見得對應到你認得的字

對「看懂 Transformer 怎麼運作」這個目的來說，這些全是雜訊。

字元級的做法是：**一個中文字 = 一個 token**。於是：
  · 追蹤檔裡每個 id 都對應到一個你認得的字
  · 詞表大小 = 語料裡出現過的相異字元數（繁中約 4000~6000）
  · 沒有 OOV 問題（沒見過的字用 <unk>）

代價是序列比 BPE 長約 1.5 倍。對教學規模的模型完全可以接受。

------------------------------------------------------------------
特殊 token
------------------------------------------------------------------
    <pad>  0   補齊用，不算進 loss
    <bos>  1   句子開始
    <eos>  2   句子結束
    <unk>  3   詞表裡沒有的字元
"""
import json
import os
from collections import Counter

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']


class CharTokenizer:
    def __init__(self, chars=None):
        self.itos = list(SPECIALS) + list(chars or [])
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    # ---------------------------------------------------------------
    @classmethod
    def train(cls, texts, max_vocab=6000, min_count=2, verbose=True):
        """
        「訓練」字元級 tokenizer 其實就是數字元出現次數，取最常見的前 N 個。
        沒有任何學習，純粹統計。
        """
        cnt = Counter()
        n = 0
        for t in texts:
            cnt.update(t)
            n += len(t)
        keep = [c for c, k in cnt.most_common() if k >= min_count][:max_vocab - len(SPECIALS)]
        tk = cls(keep)
        if verbose:
            cov = sum(cnt[c] for c in keep) / max(n, 1)
            print('  掃過 %s 個字元，相異字元 %s 個' % ('{:,}'.format(n), '{:,}'.format(len(cnt))))
            print('  詞表取 %s 個（含 %d 個特殊 token），覆蓋率 %.3f%%'
                  % ('{:,}'.format(len(tk)), len(SPECIALS), 100 * cov))
        return tk

    def __len__(self):
        return len(self.itos)

    # ---------------------------------------------------------------
    def encode(self, text, bos=False, eos=False):
        ids = [self.stoi.get(c, UNK) for c in text]
        if bos:
            ids = [BOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if skip_special and i < len(SPECIALS):
                continue
            out.append(self.itos[i] if i < len(self.itos) else '<unk>')
        return ''.join(out)

    # ---------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'itos': self.itos}, f, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        tk = cls()
        tk.itos = d['itos']
        tk.stoi = {c: i for i, c in enumerate(tk.itos)}
        return tk
