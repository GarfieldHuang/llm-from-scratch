"""
權重與 tokenizer 是綁死的 —— 用實驗證明

情境：別人拿到你的權重，想做 continual pre-training。
如果他用了不同的 tokenizer 會怎樣？

    python tests/test_tokenizer_binding.py

這個腳本會示範三件事：
  1. 詞表大小不同 → 直接載不進去（會報錯，至少你知道出事了）
  2. 詞表大小相同但 id 對應不同 → **載得進去、跑得動、輸出是垃圾**（最危險）
  3. 正確的做法：擴充詞表時保留原有 id，只在後面接新的列
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.model import TinyLM                       # noqa: E402
from scratch.loss import SoftmaxCrossEntropy           # noqa: E402
from tokenizer.resolve import resolve                  # noqa: E402

MODEL = 'out/model.pt'
if not os.path.exists(MODEL):
    print('請先跑一次 train.py')
    sys.exit(1)

# ---------- 載入原本的模型與 tokenizer ----------
# 這裡曾經是「哪個檔案存在就用哪個」，結果跑過 tests/test_bpe.py 之後
# out/ 底下會多一份 bpe 詞表，這支測試就會拿 6,400 的詞表去餵 3,195 的權重。
# 諷刺的是那正是這支測試要證明的事——只是以 IndexError 的形式。
tok, tp, kind, how = resolve(MODEL)

ckpt = torch.load(MODEL, map_location='cpu')
cfg = ckpt['cfg']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyLM(vocab_size=cfg['vocab_size'], dim=cfg['dim'], n_layers=cfg['n_layers'],
               n_heads=cfg['n_heads'], hidden=cfg['hidden'], max_len=cfg['max_len'], device=dev)
model.load(MODEL, dev, tokenizer_fingerprint=tok.fingerprint())
ce = SoftmaxCrossEntropy()

TEXT = '貓是一種可愛的動物，牠喜歡睡覺。狗也是動物，狗喜歡跑步。今天天氣很好。'


def loss_of(ids):
    x = torch.tensor([ids[:-1]], device=dev)
    y = torch.tensor([ids[1:]], device=dev)
    return float(ce.forward(model.forward(x), y))


print('模型  %s   詞表 %s   %s（tokenizer 來源：%s）'
      % (MODEL, '{:,}'.format(cfg['vocab_size']), kind, how))
print('測試句：%s' % TEXT)
print()

# ============================================================
print('=' * 70)
print('① 正確的 tokenizer')
print('=' * 70)
ids_ok = tok.encode(TEXT)
l_ok = loss_of(ids_ok)
print('  切成 %d 個 token：%s' % (len(ids_ok), [tok.decode([i]) for i in ids_ok[:10]]))
print('  loss = %.4f' % l_ok)

# ============================================================
print()
print('=' * 70)
print('② 詞表大小相同，但 id 對應被打亂')
print('=' * 70)
print('  （模擬「同一份語料、但用不同亂數種子或不同實作重訓的 tokenizer」）')

rng = random.Random(0)
perm = list(range(4, cfg['vocab_size']))       # 特殊 token 保持原位
rng.shuffle(perm)
remap = {i: i for i in range(4)}
for a, b in zip(range(4, cfg['vocab_size']), perm):
    remap[a] = b

ids_bad = [remap[i] for i in ids_ok]
l_bad = loss_of(ids_bad)
print('  同一句話，換成打亂後的 id：%s' % ids_bad[:10])
print('  權重載得進去嗎？  是（形狀完全一樣，不會報錯）')
print('  模型跑得動嗎？    是')
print('  loss = %.4f   （正確的是 %.4f，變成 %.1f 倍）' % (l_bad, l_ok, l_bad / l_ok))

import math
print()
print('  換算成「等效於在幾個選項中亂猜」：')
print('    正確 tokenizer : %8.1f  （詞表 %s，所以越接近 1 越好）'
      % (math.exp(l_ok), '{:,}'.format(cfg['vocab_size'])))
print('    打亂之後       : %8.1f' % math.exp(l_bad))
print('    完全隨機的模型 : %8.1f' % cfg['vocab_size'])

# ============================================================
print()
print('=' * 70)
print('③ 詞表大小不同')
print('=' * 70)
try:
    m2 = TinyLM(vocab_size=cfg['vocab_size'] + 100, dim=cfg['dim'], n_layers=cfg['n_layers'],
                n_heads=cfg['n_heads'], hidden=cfg['hidden'], max_len=cfg['max_len'], device=dev)
    cur = m2.state()
    d = torch.load(MODEL, map_location=dev)
    cur['embed.E'].copy_(d['params']['embed.E'].to(dev))
    print('  沒有報錯 —— 這是個問題')
except Exception as e:
    print('  載入失敗：%s' % type(e).__name__)
    print('    %s' % str(e).split('\n')[0][:100])
    print('  這種情況反而安全：至少你馬上知道出事了。')

# ============================================================
print()
print('=' * 70)
print('④ 正確的做法：擴充而不是重建')
print('=' * 70)
NEW = 50
big = TinyLM(vocab_size=cfg['vocab_size'] + NEW, dim=cfg['dim'], n_layers=cfg['n_layers'],
             n_heads=cfg['n_heads'], hidden=cfg['hidden'], max_len=cfg['max_len'], device=dev)
old = torch.load(MODEL, map_location=dev)['params']
with torch.no_grad():
    for n, p in big.state().items():
        if n == 'embed.E':
            p[:cfg['vocab_size']].copy_(old[n].to(dev))          # 原有 id 原封不動
            p[cfg['vocab_size']:] = old[n].to(dev).mean(0)       # 新列用平均值起步
        elif n in old:
            p.copy_(old[n].to(dev))

big_ce = SoftmaxCrossEntropy()
x = torch.tensor([ids_ok[:-1]], device=dev)
y = torch.tensor([ids_ok[1:]], device=dev)
l_big = float(big_ce.forward(big.forward(x), y))
print('  詞表 %s → %s，前 %s 列完全保留，新增 %d 列用既有向量的平均值初始化'
      % ('{:,}'.format(cfg['vocab_size']), '{:,}'.format(cfg['vocab_size'] + NEW),
         '{:,}'.format(cfg['vocab_size']), NEW))
print('  同一句話的 loss = %.4f   （原本 %.4f，差 %.4f）' % (l_big, l_ok, abs(l_big - l_ok)))
print('  幾乎沒變 —— 因為原有的 id 對應沒被動到。')

print()
print('=' * 70)
print('結論')
print('=' * 70)
print('  詞表大小不同  → 載不進去，會報錯（安全）')
print('  id 對應不同    → 載得進去、跑得動、輸出是垃圾（危險，而且不會有任何警告）')
print('  只能「擴充」，不能「重建」：原有的 id 必須原封不動地保留在前面。')
