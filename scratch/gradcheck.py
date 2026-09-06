"""
有限差分梯度檢查 —— 這份專案的品質保證

我們手推了每一層的 backward 公式。怎麼確定沒推錯？
不靠 PyTorch 的 autograd，而是回到梯度的「定義」直接量：

    ∂L/∂w  ≈  ( L(w + ε) − L(w − ε) ) / 2ε

只要手推的公式跟這個「把數字推一點點、量 loss 變多少」的結果吻合，
公式就是對的。這也正好示範了梯度到底是什麼——它就是斜率，沒有別的。

用的是中央差分（兩邊各推一次）而不是單邊差分，因為它的誤差是 O(ε²)
而不是 O(ε)，同樣的 ε 準確度高很多。

ε 不能太小：float32 只有約 7 位有效數字，L 的值本身若是 5.0 左右，
兩次相減得到 1e-8 這種量級時就全是捨入誤差了。實務上 1e-3 附近最準。
"""
import torch


def _proxy_loss(y, dy):
    """
    人工造一個純量 loss：L = Σ (y ⊙ dy)
    這樣 ∂L/∂y 剛好等於 dy，我們就能用任意的 dy 去測 backward 對不對。
    """
    return float((y * dy).sum())


def check(layer, x, n_samples=12, eps=1e-3, seed=0, check_params=True, verbose=True, tr=None):
    """
    對一個層做完整梯度檢查。

    layer  必須有 forward(x) / backward(dy) / params() / zero_grad()
    回傳   (最大相對誤差, 明細列表)
    """
    from .tracer import NULL
    tr = tr or NULL
    g = torch.Generator(device='cpu').manual_seed(seed)

    x = x.clone()
    y0 = layer.forward(x)
    dy = torch.randn(y0.shape, generator=g).to(y0.device, y0.dtype)

    layer.zero_grad()
    dx = layer.backward(dy)

    rows = []
    worst = 0.0

    # ---------- 檢查對輸入的梯度 ----------
    if dx is not None:
        flat = x.reshape(-1)
        idx = torch.randperm(flat.numel(), generator=g)[:n_samples]
        for i in idx.tolist():
            orig = float(flat[i])
            flat[i] = orig + eps
            lp = _proxy_loss(layer.forward(x), dy)
            flat[i] = orig - eps
            lm = _proxy_loss(layer.forward(x), dy)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(dx.reshape(-1)[i])
            rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            worst = max(worst, rel)
            rows.append(('dx[%d]' % i, num, ana, rel))
        layer.forward(x)          # 還原內部快取

    # ---------- 檢查對參數的梯度 ----------
    if check_params:
        for p, dget, pname in layer.params():
            layer.zero_grad()
            layer.forward(x)
            layer.backward(dy)
            dp = dget()
            flat = p.reshape(-1)
            idx = torch.randperm(flat.numel(), generator=g)[:n_samples]
            for i in idx.tolist():
                orig = float(flat[i])
                flat[i] = orig + eps
                lp = _proxy_loss(layer.forward(x), dy)
                flat[i] = orig - eps
                lm = _proxy_loss(layer.forward(x), dy)
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                ana = float(dp.reshape(-1)[i])
                rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
                worst = max(worst, rel)
                rows.append(('%s[%d]' % (pname, i), num, ana, rel))

    if verbose:
        name = getattr(layer, 'name', layer.__class__.__name__)
        ok = 'PASS' if worst < 2e-3 else 'FAIL'
        print('  %-28s 最大相對誤差 %.2e   %s' % (name, worst, ok))

    if tr.enabled:
        tr.section('梯度檢查 · %s' % getattr(layer, 'name', ''))
        tr.formula('∂L/∂w ≈ ( L(w+ε) − L(w−ε) ) / 2ε', 'ε = %g' % eps)
        tr.table(['項目', '有限差分', '手推公式', '相對誤差'],
                 [[r[0], r[1], r[2], r[3]] for r in rows[:20]],
                 [22, 16, 16, 14])
        tr.note('兩欄吻合就代表手推的 backward 公式沒錯。\n'
                '殘留的 1e-5 等級誤差來自 float32 精度與有限差分本身的截斷誤差。')

    return worst, rows


def report(results, path):
    """把多個層的檢查結果寫成一份總表。"""
    import io
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write('梯度檢查總表\n')
        f.write('=' * 72 + '\n')
        f.write('用有限差分 ( L(w+ε) − L(w−ε) ) / 2ε 驗證每一層手推的 backward。\n')
        f.write('完全不使用 PyTorch autograd。\n\n')
        f.write('  %-30s %16s   %s\n' % ('層', '最大相對誤差', '結果'))
        f.write('  ' + '-' * 62 + '\n')
        allok = True
        for name, worst in results:
            ok = worst < 2e-3
            allok = allok and ok
            f.write('  %-30s %16.2e   %s\n' % (name, worst, 'PASS' if ok else 'FAIL'))
        f.write('\n  總結：%s\n' % ('全部通過' if allok else '有項目未通過，需要檢查'))
    return path
