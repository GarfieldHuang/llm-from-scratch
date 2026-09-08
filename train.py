"""
訓練腳本

    python train.py                        # 用預設值跑
    python train.py --steps 2000 --dim 256
    python train.py --trace-steps 1,500    # 指定要完整追蹤哪幾步

輸出：
    traces/train_step_0001.txt   第 1 步的完整前向 + 反向 + 更新（每個數字都有）
    traces/train_log.txt         每一步的 loss / lr / 梯度範數
    traces/model_summary.txt     模型結構與參數量
    out/model.pt                 訓練好的權重
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch.model import TinyLM                       # noqa: E402
from scratch.loss import SoftmaxCrossEntropy           # noqa: E402
from scratch.adamw import AdamW, cosine_lr             # noqa: E402
from scratch.tracer import Tracer, NULL                # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer     # noqa: E402
from tokenizer.bpe import BPETokenizer                 # noqa: E402


def load_corpus(path, max_lines):
    texts = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                texts.append(json.loads(line)['text'])
            except Exception:
                continue
    return texts


def main():
    ap = argparse.ArgumentParser()
    # 語料由大到小，挑第一個存在的：
    #   corpus_zhtw.jsonl      完整版 107 MB，未隨 repo 發布（見 README）
    #   corpus_zhtw_20k.jsonl  前 20,000 段 14 MB，隨 repo 發布，docs/11 用的就是這個規模
    #   corpus_sample.jsonl    前 2,000 段 1.4 MB，最小的可跑範例
    _corpus = next((p for p in ('data/corpus_zhtw.jsonl',
                                'data/corpus_zhtw_20k.jsonl',
                                'data/corpus_sample.jsonl')
                    if os.path.exists(p)), 'data/corpus_sample.jsonl')
    ap.add_argument('--corpus', default=_corpus)
    ap.add_argument('--max-lines', type=int, default=20000)
    ap.add_argument('--vocab', type=int, default=5000)
    ap.add_argument('--tokenizer', choices=['char', 'bpe'], default='char',
                    help='char = 一個字一個 token（好懂）；bpe = 合併高頻字組（序列短約 30%%）')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--trace-steps', default='1',
                    help='要完整追蹤的步數，逗號分隔。例如 1,500,2000')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs('traces', exist_ok=True)
    os.makedirs('out', exist_ok=True)
    trace_at = {int(s) for s in args.trace_steps.split(',') if s.strip()}

    # ---------- 語料與 tokenizer ----------
    print('讀語料 %s（前 %s 行）' % (args.corpus, '{:,}'.format(args.max_lines)))
    texts = load_corpus(args.corpus, args.max_lines)
    print('  %s 段文字' % '{:,}'.format(len(texts)))

    Tok = CharTokenizer if args.tokenizer == 'char' else BPETokenizer
    tok_path = 'out/tokenizer_%s.json' % args.tokenizer
    if os.path.exists(tok_path):
        tok = Tok.load(tok_path)
        print('載入既有 %s tokenizer，詞表 %s' % (args.tokenizer, '{:,}'.format(len(tok))))
    elif args.tokenizer == 'char':
        print('建立字元級 tokenizer')
        tok = CharTokenizer.train(texts, max_vocab=args.vocab)
        tok.save(tok_path)
    else:
        print('訓練 BPE tokenizer（目標詞表 %s）' % '{:,}'.format(args.vocab))
        btr = Tracer('traces', 'bpe_training')
        tok = BPETokenizer.train(texts, vocab_size=args.vocab, tracer=btr)
        btr.close()
        tok.save(tok_path)
        print('  合併過程寫入 %s' % btr.path)

    # ---------- 把整份語料打包成一條長串 ----------
    # 用「打包」而不是「每段補 padding」：沒有浪費的 pad 位置，
    # 也不需要 ignore_index。（loss.py 仍然支援 padding，見那裡的說明。）
    print('編碼中...')
    stream = []
    for t in texts:
        stream.extend(tok.encode(t, eos=True))
    data = torch.tensor(stream, dtype=torch.long, device=args.device)
    print('  總共 %s 個 token' % '{:,}'.format(len(data)))

    # ---------- 模型 ----------
    model = TinyLM(vocab_size=len(tok), dim=args.dim, n_layers=args.layers,
                   n_heads=args.heads, max_len=args.seq + 8, device=args.device)
    print()
    print(model.summary())
    with open('traces/model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(model.summary() + '\n\n')
        f.write('超參數\n' + '=' * 62 + '\n')
        for k, v in vars(args).items():
            f.write('  %-16s %s\n' % (k, v))

    ce = SoftmaxCrossEntropy()
    opt = AdamW(model.params(), lr=args.lr, weight_decay=args.wd)
    print('\noptimizer 拿到 %d 個參數張量' % len(model.params()))

    # ---------- 訓練 ----------
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    log = open('traces/train_log.txt', 'w', encoding='utf-8')
    log.write('%6s %10s %12s %12s %12s\n' % ('step', 'loss', 'lr', '梯度範數', '秒'))
    t0 = time.time()

    print('\n開始訓練 %s 步\n' % '{:,}'.format(args.steps))
    for step in range(1, args.steps + 1):
        tracing = step in trace_at
        tr = Tracer('traces', 'train_step_%04d' % step) if tracing else NULL
        if tracing:
            tr.title('訓練第 %d 步 · 完整數值軌跡' % step,
                     '模型 dim=%d layers=%d heads=%d，batch=%d seq=%d'
                     % (args.dim, args.layers, args.heads, args.batch, args.seq))

        # 隨機取一批連續片段
        idx = torch.randint(0, len(data) - args.seq - 1, (args.batch,), generator=g)
        x = torch.stack([data[i:i + args.seq] for i in idx])
        y = torch.stack([data[i + 1:i + args.seq + 1] for i in idx])

        if tracing:
            tr.section('這一批的資料')
            tr.text('    第 0 筆的前 40 個字：')
            tr.text('        ' + tok.decode(x[0, :40].tolist()))
            tr.text('    它要預測的下一個字（往右位移一格）：')
            tr.text('        ' + tok.decode(y[0, :40].tolist()))
            tr.note('語言模型的任務就這一句：看位置 0..t，猜位置 t+1。\n'
                    '整個訓練的監督訊號只有這個，沒有任何人標註「貓是動物」。')

        lr = cosine_lr(step, args.steps, args.lr, warmup=args.warmup)
        opt.set_lr(lr)

        model.zero_grad()
        logits = model.forward(x, tr)
        loss = ce.forward(logits, y, tr)
        dlogits = ce.backward(tr)
        model.backward(dlogits, tr)

        gnorm = opt.clip_grad_norm(args.clip)
        opt.step(tr, watch='embed.E' if tracing else None)

        if tracing:
            tr.section('這一步結束')
            tr.scalar('loss', float(loss))
            tr.scalar('梯度全域範數（裁剪前）', gnorm)
            nz = int((model.embed.dE.abs().sum(1) > 0).sum())
            tr.scalar('embedding 梯度非零列', nz)
            tr.scalar('詞表大小', len(tok))
            tr.note('非零列 = %d / %d。因為輸出層與 embedding 共用權重，\n'
                    'softmax 會碰到每一列，所以梯度是稠密的。' % (nz, len(tok)))
            tr.close()
            print('  [step %d 的完整軌跡已寫入 traces/train_step_%04d.txt]' % (step, step))

        if step % 20 == 0 or step == 1:
            log.write('%6d %10.4f %12.3e %12.4f %12.1f\n'
                      % (step, float(loss), lr, gnorm, time.time() - t0))
            log.flush()
        if step % 100 == 0 or step == 1:
            print('  step %5d/%d   loss %7.4f   lr %.2e   |g| %7.3f   %5.0fs'
                  % (step, args.steps, float(loss), lr, gnorm, time.time() - t0))

    log.close()
    model.save('out/model.pt', tokenizer_fingerprint=tok.fingerprint(),
               tokenizer_path=tok_path, tokenizer_kind=args.tokenizer)
    print('\n完成，耗時 %.0f 秒' % (time.time() - t0))
    print('  權重      out/model.pt')
    print('  訓練紀錄  traces/train_log.txt')
    print('  逐步軌跡  traces/train_step_*.txt')


if __name__ == '__main__':
    main()
