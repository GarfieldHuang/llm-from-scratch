"""
生成（推論）

    python generate.py --prompt "貓是一種"
    python generate.py --prompt "台灣的" --trace     # 把完整前向寫進 traces/inference.txt

推論跟訓練的差別只有一個：**沒有反向傳播**。
前向完全一樣，算完 logits 之後取樣一個 token，接到後面再算一次。

這也是為什麼推論比訓練省記憶體那麼多——不用存每一層的中間值等著反向用。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch.model import TinyLM                       # noqa: E402
from scratch.tracer import Tracer, NULL                # noqa: E402
from tokenizer.char_tokenizer import EOS                 # noqa: E402
from tokenizer.resolve import resolve                    # noqa: E402


def sample_next(logits, temperature=0.9, top_k=40, top_p=0.9, seen=None, rep_penalty=1.1):
    """
    從 logits 取樣下一個 token。

    重複懲罰要注意正負號：logit 是負的時候要「乘上」懲罰才會變得更負。
    寫成 logit / penalty 的話，負的 logit 反而會變大，等於在獎勵重複。
    """
    z = logits.float().clone()

    if seen and rep_penalty != 1.0:
        for t in set(seen):
            z[t] = z[t] / rep_penalty if z[t] > 0 else z[t] * rep_penalty

    if top_k:
        v, _ = z.topk(min(top_k, z.numel()))
        z[z < v[-1]] = float('-inf')

    p = torch.softmax(z / max(temperature, 1e-6), dim=-1)

    if top_p and top_p < 1.0:
        sp, si = p.sort(descending=True)
        cut = int((sp.cumsum(0) > top_p).nonzero()[0, 0]) + 1
        keep = si[:cut]
        p2 = torch.zeros_like(p)
        p2[keep] = p[keep]
        p = p2 / p2.sum()

    return int(torch.multinomial(p, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='貓是一種')
    ap.add_argument('--model', default='out/model.pt')
    ap.add_argument('--tokenizer', default=None,
                    help='不指定就從 checkpoint 記錄的路徑找，找不到再用指紋比對')
    ap.add_argument('--max-new', type=int, default=80)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-k', type=int, default=40)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--rep-penalty', type=float, default=1.1)
    ap.add_argument('--trace', action='store_true', help='把第一次前向的完整數值寫進檔案')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # 不猜檔名——問 checkpoint 它自己是用哪個 tokenizer 訓的。
    tok, tp, kind, how = resolve(args.model, args.tokenizer)
    print('tokenizer: %s（%s，詞表 %s，來源：%s）'
          % (tp, kind, '{:,}'.format(len(tok)), how))
    ckpt = torch.load(args.model, map_location=args.device)
    cfg = ckpt['cfg']

    model = TinyLM(vocab_size=cfg['vocab_size'], dim=cfg['dim'], n_layers=cfg['n_layers'],
                   n_heads=cfg['n_heads'], hidden=cfg['hidden'], max_len=cfg['max_len'],
                   device=args.device)
    model.load(args.model, args.device, tokenizer_fingerprint=tok.fingerprint())

    ids = tok.encode(args.prompt)
    print('提示詞：%s' % args.prompt)
    print('切成 %d 個 token：%s' % (len(ids), [tok.decode([i]) for i in ids]))
    print()

    # ---- 第一次前向，可選擇完整追蹤 ----
    tr = NULL
    if args.trace:
        os.makedirs('traces', exist_ok=True)
        tr = Tracer('traces', 'inference')
        tr.title('推論 · 完整前向軌跡', '提示詞「%s」，%d 層，dim=%d'
                 % (args.prompt, cfg['n_layers'], cfg['dim']))
        tr.note('推論只有前向，沒有反向傳播。\n'
                '所以不需要把每一層的中間值存起來——這就是推論比訓練省記憶體的原因。')

    x = torch.tensor([ids], device=args.device)
    logits = model.forward(x, tr)

    if args.trace:
        p = torch.softmax(logits[0, -1].float(), -1)
        top = p.topk(8)
        tr.section('最後一個位置的預測')
        tr.formula('p = softmax(logits[-1])')
        tr.table(['排名', 'token', 'id', '機率'],
                 [[str(i + 1), tok.decode([int(top.indices[i])]), str(int(top.indices[i])),
                   float(top.values[i])] for i in range(8)], [8, 10, 10, 14])
        tr.close()
        print('完整前向軌跡已寫入 %s' % tr.path)
        print()

    # ---- 逐字生成 ----
    out = list(ids)
    for _ in range(args.max_new):
        window = out[-cfg['max_len']:]
        lg = model.forward(torch.tensor([window], device=args.device))[0, -1]
        nxt = sample_next(lg, args.temperature, args.top_k, args.top_p, out, args.rep_penalty)
        if nxt == EOS:
            break
        out.append(nxt)

    print('生成結果：')
    print(tok.decode(out))


if __name__ == '__main__':
    main()
