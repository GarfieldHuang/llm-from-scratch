"""
BPE（Byte Pair Encoding）—— 詞表大小是怎麼決定的

------------------------------------------------------------------
先回答那個問題：「詞表 6400 是哪來的？」
------------------------------------------------------------------
    6400  =  起始符號數  +  合併次數

BPE 的訓練就是「重複合併最高頻的相鄰配對」，**每合併一次，詞表就多一個**。
所以詞表大小不是算出來的，是你自己挑的停止條件：
想要 6400 個 token，就合併到總數變成 6400 為止。

    起始：語料裡所有相異字元，例如 3600 個
    合併 1 次 → 3601      合併 2 次 → 3602      ...
    合併 2800 次 → 6400   ← 停

------------------------------------------------------------------
訓練演算法
------------------------------------------------------------------
    1. 把語料切成一個個「詞塊」，統計每個詞塊出現幾次
    2. 每個詞塊先拆成單一字元的序列
    3. 統計所有「相鄰配對」的加權出現次數
    4. 取次數最高的那一組，合併成一個新符號，記錄這次合併
    5. 回到 3，直到詞表達到目標大小

範例（假設語料裡「機器學習」很常出現）：
    起始      機 器 學 習
    合併 #1   ('機','器') → '機器'        機器 學 習
    合併 #2   ('學','習') → '學習'        機器 學習
    合併 #3   ('機器','學習') → '機器學習'  機器學習

高頻的組合會被合併成單一 token，序列因此變短。
這就是 BPE 的全部——沒有任何學習、沒有梯度，純粹是貪婪的頻率統計。

------------------------------------------------------------------
編碼（推論時）
------------------------------------------------------------------
把訓練時記錄的合併**照原順序**再套用一次。順序不能亂：
'機器學習' 這個 token 的存在，前提是 '機器' 和 '學習' 已經先被合併出來了。

------------------------------------------------------------------
這份實作跟生產級的差別
------------------------------------------------------------------
生產級（GPT-2 / Qwen / minimind）用的是 **byte-level** BPE：
起始符號是 256 個 byte，而不是字元。好處是永遠沒有 OOV——
任何字元都能拆成 byte。代價是沒見過的中文字會變成 2~3 個看不懂的碎片。

這裡用**字元級**當起點，因為追蹤檔裡看到的每個符號都是你認得的字，
對理解機制幫助大很多。OOV 用 <unk> 處理。
"""
import json
import os
import re
from collections import Counter, defaultdict

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']

# 預切分：先把文字切成詞塊，BPE 不會跨越詞塊邊界合併。
# 這樣可以避免把「。」跟後面的字黏成一個 token。
#   - 連續的中日韓文字（每 6 個切一段，避免詞塊過長拖慢訓練）
#   - 連續的英文字母 / 數字
#   - 其他單一符號各自成塊
_PRE = re.compile(
    r'[一-鿿㐀-䶿]{1,6}'
    r'|[A-Za-z]+'
    r'|[0-9]+'
    r'|\s+'
    r'|.',
    re.UNICODE)


def pre_tokenize(text):
    return _PRE.findall(text)


class BPETokenizer:
    def __init__(self):
        self.itos = list(SPECIALS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.merges = []          # [(a, b), ...] 依學到的順序
        self.rank = {}            # (a, b) -> 第幾次合併
        self._cache = {}

    def __len__(self):
        return len(self.itos)

    # 訓練、編碼、解碼的實作寫在下面的模組層級函式，最後再綁回這個 class。
    # 這樣做的用意是讓演算法本身可以從上到下連貫地讀，不被 class 的縮排打斷。


def train_bpe(texts, vocab_size=6400, min_freq=2, min_char_freq=1, tracer=None, verbose=True):
    """
    回傳一個訓練好的 BPETokenizer。

    min_freq      合併的門檻：配對出現次數低於這個值就不再合併
    min_char_freq 起始詞表的門檻。預設 1 = 收錄語料裡的每一個字元，
                  這樣對訓練語料才能保證 encode/decode 完全可逆（無 OOV）。
                  設成 >1 會把罕見字排除成 <unk>，能縮小詞表但會失去可逆性。
    tracer        傳入 scratch.tracer.Tracer 就會把每一次合併寫進追蹤檔
    """
    tk = BPETokenizer()

    # ---------- 步驟 1：統計詞塊頻率 ----------
    wf = Counter()
    n_char = 0
    for t in texts:
        for w in pre_tokenize(t):
            wf[w] += 1
            n_char += len(w)

    # ---------- 步驟 2：建立起始符號表（所有相異字元）----------
    cf = Counter()
    for w, k in wf.items():
        for c in w:
            cf[c] += k
    base = [c for c, k in cf.most_common() if k >= min_char_freq]
    tk.itos = list(SPECIALS) + base
    tk.stoi = {c: i for i, c in enumerate(tk.itos)}
    n_base = len(tk.itos)
    n_merges = max(0, vocab_size - n_base)

    if verbose:
        print('  詞塊 %s 個（相異 %s 個），字元 %s 個'
              % ('{:,}'.format(sum(wf.values())), '{:,}'.format(len(wf)), '{:,}'.format(n_char)))
        print('  起始詞表 %s（%d 特殊 + %s 字元）'
              % ('{:,}'.format(n_base), len(SPECIALS), '{:,}'.format(len(base))))
        print('  目標 %s → 需要合併 %s 次' % ('{:,}'.format(vocab_size), '{:,}'.format(n_merges)))

    if tracer is not None and tracer.enabled:
        tracer.title('BPE 訓練 · 詞表是怎麼長出來的',
                     '目標詞表 %d' % vocab_size)
        tracer.formula('詞表大小 = 起始符號數 + 合併次數')
        tracer.table(['項目', '數量'],
                     [['特殊 token', len(SPECIALS)],
                      ['起始字元（相異且出現 ≥ %d 次）' % min_char_freq, len(base)],
                      ['起始詞表小計', n_base],
                      ['要做的合併次數', n_merges],
                      ['最終詞表', n_base + n_merges]], [40, 14])
        tracer.note('這就是「6400 個詞哪來的」的答案：\n'
                    '不是算出來的，是你自己挑的停止條件。\n'
                    '每合併一次詞表就 +1，合併到目標數字就停。')

    # ---------- 把每個詞塊拆成符號序列 ----------
    words = []       # [(符號 tuple, 出現次數)]
    for w, k in wf.items():
        syms = tuple(c if c in tk.stoi else '<unk>' for c in w)
        words.append([list(syms), k])

    # ---------- 建立配對計數，以及「哪些詞塊含有這個配對」 ----------
    pair_cnt = Counter()
    pair_where = defaultdict(set)

    def index_word(i):
        s, k = words[i]
        for j in range(len(s) - 1):
            p = (s[j], s[j + 1])
            pair_cnt[p] += k
            pair_where[p].add(i)

    def deindex_word(i):
        s, k = words[i]
        for j in range(len(s) - 1):
            p = (s[j], s[j + 1])
            pair_cnt[p] -= k
            if pair_cnt[p] <= 0:
                pair_cnt.pop(p, None)
            pair_where[p].discard(i)

    for i in range(len(words)):
        index_word(i)

    # ---------- 步驟 3~5：反覆合併 ----------
    for step in range(n_merges):
        if not pair_cnt:
            break
        pair, cnt = max(pair_cnt.items(), key=lambda kv: kv[1])
        if cnt < min_freq:
            break

        new_sym = pair[0] + pair[1]
        affected = list(pair_where[pair])

        for i in affected:
            deindex_word(i)
            s, k = words[i]
            out, j = [], 0
            while j < len(s):
                if j < len(s) - 1 and s[j] == pair[0] and s[j + 1] == pair[1]:
                    out.append(new_sym)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            words[i][0] = out
            index_word(i)

        tk.rank[pair] = len(tk.merges)
        tk.merges.append(pair)
        tk.stoi[new_sym] = len(tk.itos)
        tk.itos.append(new_sym)

        if tracer is not None and tracer.enabled and (step < 25 or (step + 1) % 500 == 0):
            tracer.text('    合併 #%-5d  (%r, %r) 出現 %s 次  ->  新 token %r   詞表 %d'
                        % (step + 1, pair[0], pair[1], '{:,}'.format(cnt), new_sym, len(tk.itos)))

        if verbose and ((step + 1) % 1000 == 0 or step == 0):
            print('    合併 #%-6d %r + %r -> %r  (%s 次)   詞表 %s'
                  % (step + 1, pair[0], pair[1], new_sym,
                     '{:,}'.format(cnt), '{:,}'.format(len(tk.itos))))

    if verbose:
        print('  完成，最終詞表 %s' % '{:,}'.format(len(tk)))

    if tracer is not None and tracer.enabled:
        tracer.section('前 25 次合併之後，最長的幾個 token')
        longs = sorted([s for s in tk.itos if len(s) > 1], key=len, reverse=True)[:20]
        tracer.text('    ' + '  '.join(repr(s) for s in longs))
        tracer.note('高頻的字組被合併成單一 token，序列因此變短。\n'
                    '整個過程沒有任何學習，純粹是貪婪的頻率統計。')

    return tk


# 把 train 綁回 class（寫在外面是為了讓演算法本身讀起來連貫）
BPETokenizer.train = staticmethod(train_bpe)


# ==============================================================
# 編碼 / 解碼
# ==============================================================
def _encode_word(tk, word):
    """對單一詞塊套用學到的合併，順序不能亂。"""
    if word in tk._cache:
        return tk._cache[word]
    syms = [c if c in tk.stoi else '<unk>' for c in word]
    while len(syms) > 1:
        # 找出目前序列中「最早被學到」的那個配對（rank 最小）
        best, best_rank, best_i = None, None, -1
        for i in range(len(syms) - 1):
            p = (syms[i], syms[i + 1])
            r = tk.rank.get(p)
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank, best_i = p, r, i
        if best is None:
            break
        syms = syms[:best_i] + [best[0] + best[1]] + syms[best_i + 2:]
    tk._cache[word] = syms
    return syms


def _encode(self, text, bos=False, eos=False):
    ids = []
    if bos:
        ids.append(BOS)
    for w in pre_tokenize(text):
        for s in _encode_word(self, w):
            ids.append(self.stoi.get(s, UNK))
    if eos:
        ids.append(EOS)
    return ids


def _decode(self, ids, skip_special=True):
    out = []
    for i in ids:
        i = int(i)
        if skip_special and i < len(SPECIALS):
            continue
        out.append(self.itos[i] if i < len(self.itos) else '<unk>')
    return ''.join(out)


def _save(self, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'bpe', 'itos': self.itos,
                   'merges': [list(m) for m in self.merges]}, f, ensure_ascii=False)
    return path


def _load(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    tk = BPETokenizer()
    tk.itos = d['itos']
    tk.stoi = {c: i for i, c in enumerate(tk.itos)}
    tk.merges = [tuple(m) for m in d['merges']]
    tk.rank = {p: i for i, p in enumerate(tk.merges)}
    return tk


BPETokenizer.encode = _encode
BPETokenizer.decode = _decode
BPETokenizer.save = _save
BPETokenizer.load = staticmethod(_load)
