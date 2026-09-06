"""
BPE 測試：訓練一個小詞表，驗證 encode/decode 是可逆的，並輸出合併過程。

    python tests/test_bpe.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.bpe import BPETokenizer, pre_tokenize     # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer       # noqa: E402
from scratch.tracer import Tracer                        # noqa: E402

CORPUS = 'data/corpus_sample.jsonl'
if not os.path.exists(CORPUS):
    CORPUS = 'data/corpus_zhtw.jsonl'

texts = []
with open(CORPUS, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        try:
            texts.append(json.loads(line)['text'])
        except Exception:
            pass

print('語料 %s 段' % '{:,}'.format(len(texts)))
print()
print('預切分示範（BPE 不會跨越詞塊邊界合併）')
s = texts[0][:40]
print('  原文  ', s)
print('  切成  ', pre_tokenize(s))
print()

os.makedirs('traces', exist_ok=True)
tr = Tracer('traces', 'bpe_training')

print('訓練 BPE，目標詞表 6400')
tk = BPETokenizer.train(texts, vocab_size=6400, tracer=tr, verbose=True)
tr.close()
print('  合併過程寫入', tr.path)
print()

# ---- 可逆性 ----
print('可逆性檢查（encode 之後 decode 要拿回原文）')
bad = 0
for t in texts[:300]:
    if tk.decode(tk.encode(t)) != t:
        bad += 1
print('  300 段文字，還原失敗 %d 段  %s' % (bad, 'PASS' if bad == 0 else 'FAIL'))
print()

# ---- 壓縮率：BPE vs 字元級 ----
ch = CharTokenizer.train(texts, max_vocab=6400, verbose=False)
sample = texts[:500]
n_char = sum(len(t) for t in sample)
n_bpe = sum(len(tk.encode(t)) for t in sample)
n_ch = sum(len(ch.encode(t)) for t in sample)
print('壓縮效果（500 段，共 %s 字元）' % '{:,}'.format(n_char))
print('  字元級  %9s tokens   %.3f token/字' % ('{:,}'.format(n_ch), n_ch / n_char))
print('  BPE     %9s tokens   %.3f token/字' % ('{:,}'.format(n_bpe), n_bpe / n_char))
print('  BPE 讓序列短了 %.1f%%' % (100 * (1 - n_bpe / n_ch)))
print()

# ---- 看幾個例子 ----
print('切分對照')
for s in ['機器學習是人工智慧的一個分支', '今天天氣很好，我想去公園散步']:
    print('  原文    ', s)
    print('  字元級  ', [ch.decode([i]) for i in ch.encode(s)])
    print('  BPE     ', [tk.decode([i]) for i in tk.encode(s)])
    print()

tk.save('out/tokenizer_bpe.json')
print('詞表寫入 out/tokenizer_bpe.json，大小 %s' % '{:,}'.format(len(tk)))
sys.exit(0 if bad == 0 else 1)
