"""
從 checkpoint 找出「當初訓練時用的那個 tokenizer」。

為什麼需要這個檔案：

權重跟 tokenizer 是綁死的（見 docs/07-權重與tokenizer綁定.md）。
但在這個檔案出現之前，程式是用「猜檔名」的方式找 tokenizer：

    tp = 'out/tokenizer_bpe.json' if os.path.exists(...) else 'out/tokenizer_char.json'

只要 out/ 底下同時存在兩份詞表（跑過 tests/test_bpe.py 就會），
猜出來的那個就跟權重對不上——這正是這份專案花一整章在警告的錯誤。

正確的做法是讓 checkpoint 自己記得：train.py 存檔時把 tokenizer_path 與
tokenizer_kind 一起寫進去，這裡直接讀出來，不用猜。

舊的 checkpoint 沒有這兩個欄位，就退回「比對指紋」：把候選詞表逐一算指紋，
挑跟 checkpoint 裡 tokenizer_fingerprint 對得上的那個。指紋涵蓋 id 的順序，
所以「大小一樣但 id 對應不同」也會被抓出來。
"""
import json
import os

import torch

from tokenizer.bpe import BPETokenizer
from tokenizer.char_tokenizer import CharTokenizer

# 猜不到時的候選清單。順序不重要——底下是用指紋比對，不是用順序決定。
CANDIDATES = ('out/tokenizer_char.json', 'out/tokenizer_bpe.json', 'out/tokenizer.json')


def load_tokenizer(path):
    """讀一份詞表檔，回傳 (tokenizer, kind)。kind 是 'char' 或 'bpe'。"""
    with open(path, encoding='utf-8') as f:
        kind = json.load(f).get('type', 'char')
    return (BPETokenizer if kind == 'bpe' else CharTokenizer).load(path), kind


class TokenizerNotFound(Exception):
    """找不到跟權重相符的 tokenizer。訊息裡會說明試過哪些、為什麼不合。"""


def resolve(model_path, explicit=None):
    """
    找出跟 model_path 這份權重相符的 tokenizer。

    回傳 (tokenizer, path, kind, how)。how 說明「是怎麼找到的」，
    印出來讓讀者看得見決策過程——這份專案的重點就是不留黑箱。

    explicit 有值就直接用（使用者自己指定 --tokenizer，尊重他的選擇；
    對不對得上交給 model.load 的指紋檢查去擋）。
    """
    if explicit:
        tok, kind = load_tokenizer(explicit)
        return tok, explicit, kind, '使用者指定'

    ckpt = torch.load(model_path, map_location='cpu')
    saved_path = ckpt.get('tokenizer_path')
    saved_fp = ckpt.get('tokenizer_fingerprint')

    # ① checkpoint 自己記得（train.py 存檔時寫進去的）
    if saved_path and os.path.exists(saved_path):
        tok, kind = load_tokenizer(saved_path)
        if not saved_fp or tok.fingerprint() == saved_fp:
            return tok, saved_path, kind, 'checkpoint 記錄的路徑'
        # 路徑還在但內容被換過了（例如用同一個檔名重訓詞表）——往下走指紋比對

    # ② 舊 checkpoint 沒記路徑，或記的那份已經對不上：用指紋去比對候選
    tried = []
    for cand in CANDIDATES:
        if not os.path.exists(cand):
            continue
        tok, kind = load_tokenizer(cand)
        fp = tok.fingerprint()
        if saved_fp and fp == saved_fp:
            return tok, cand, kind, '指紋比對（checkpoint 沒記路徑）'
        tried.append((cand, kind, len(tok), fp))

    # ③ checkpoint 根本沒存指紋（更舊的版本）：只剩一個候選就用它
    if not saved_fp and len(tried) == 1:
        cand, kind, _, _ = tried[0]
        tok, _ = load_tokenizer(cand)
        return tok, cand, kind, '唯一候選（checkpoint 沒有指紋可比對）'

    raise TokenizerNotFound(_explain(model_path, saved_fp, saved_path, tried))


def _explain(model_path, saved_fp, saved_path, tried):
    """把「為什麼找不到」講清楚，而不是丟一個 IndexError 讓人猜。"""
    nl = chr(10)
    lines = ['找不到跟 %s 相符的 tokenizer。' % model_path,
             '  權重需要的指紋 : %s' % (saved_fp or '（這份 checkpoint 沒有存指紋）')]
    if saved_path:
        lines.append('  checkpoint 記的路徑 : %s（檔案不存在或內容已變）' % saved_path)
    if tried:
        lines.append('  試過這些候選：')
        for cand, kind, n, fp in tried:
            lines.append('    %-28s %-5s 詞表 %-7s 指紋 %s' % (cand, kind, '{:,}'.format(n), fp))
    else:
        lines.append('  out/ 底下沒有任何詞表檔——請先跑一次 train.py。')
    lines.append('  用 --tokenizer 指定路徑可以略過這個檢查。')
    return nl.join(lines)
