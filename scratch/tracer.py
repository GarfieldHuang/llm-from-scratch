"""
追蹤器：把每一步計算的公式與實際數值寫成人看得懂的文字檔。

這個檔案本身沒有任何數學，它只負責「把算式和數字排版出來」。
每一個層（layer）在 forward / backward 的時候都會呼叫這裡的方法，
跑完之後 traces/ 底下就會出現完整的數值軌跡。
"""
import os
import io


def _fmt(v, w=10, p=5):
    """把一個純量排成固定寬度，正負號對齊。"""
    if v is None:
        return ' ' * w
    if isinstance(v, bool):
        return ('True' if v else 'False').rjust(w)
    if isinstance(v, int):
        return ('{:,}'.format(v)).rjust(w)
    av = abs(v)
    if av != 0 and (av < 1e-4 or av >= 1e6):
        return ('%+.*e' % (p - 2, v)).rjust(w)
    return ('%+.*f' % (p, v)).rjust(w)


class Tracer:
    """
    用法：
        tr = Tracer('traces/inference')
        tr.section('第 1 層 · Attention')
        tr.formula('scores = Q @ K^T / sqrt(d_head)')
        tr.matrix('Q', q, note='查詢向量')
        tr.close()

    enabled=False 的時候所有方法都是空操作，訓練時不會拖慢速度。
    """

    def __init__(self, outdir=None, name='trace', enabled=True, width=100):
        self.enabled = enabled and outdir is not None
        self.width = width
        self.f = None
        self.path = None
        if self.enabled:
            os.makedirs(outdir, exist_ok=True)
            self.path = os.path.join(outdir, name + '.txt')
            self.f = io.open(self.path, 'w', encoding='utf-8')

    # ---------- 排版 ----------

    def _w(self, s=''):
        if self.enabled:
            self.f.write(s + '\n')

    def title(self, t, sub=''):
        if not self.enabled:
            return
        self._w('=' * self.width)
        self._w('  ' + t)
        if sub:
            self._w('  ' + sub)
        self._w('=' * self.width)
        self._w()

    def section(self, t):
        if not self.enabled:
            return
        self._w()
        self._w('-' * self.width)
        self._w('  ' + t)
        self._w('-' * self.width)

    def text(self, t=''):
        self._w(t)

    def formula(self, f, note=''):
        """印一條公式。這是這份追蹤檔的靈魂——數字旁邊一定要有算式。"""
        if not self.enabled:
            return
        self._w('    公式:  ' + f)
        if note:
            self._w('           ' + note)

    def shape(self, name, t):
        if not self.enabled:
            return
        self._w('    %-22s shape %s' % (name, tuple(t.shape)))

    # ---------- 印數值 ----------

    def scalar(self, name, v, note=''):
        if not self.enabled:
            return
        self._w('    %-26s = %s   %s' % (name, _fmt(float(v), 14, 6), note))

    def vector(self, name, v, n=8, note=''):
        """印一個向量的前 n 個元素。"""
        if not self.enabled:
            return
        v = v.detach().reshape(-1).float().cpu()
        head = '  '.join(_fmt(float(x), 10, 5) for x in v[:n])
        tail = '  ...(共 %d 個)' % len(v) if len(v) > n else ''
        self._w('    %-22s %s%s' % (name, head, tail))
        if note:
            self._w('    %-22s %s' % ('', note))

    def matrix(self, name, m, rows=4, cols=6, note=''):
        """印一個矩陣的左上角。"""
        if not self.enabled:
            return
        m = m.detach().float().cpu()
        while m.dim() > 2:
            m = m[0]
        if m.dim() == 1:
            m = m.reshape(1, -1)
        R, C = m.shape
        self._w('    %s  shape (%d, %d)%s' % (name, R, C, ('   ' + note) if note else ''))
        for i in range(min(rows, R)):
            line = '  '.join(_fmt(float(x), 10, 5) for x in m[i, :cols])
            more = '  ...' if C > cols else ''
            self._w('        [%3d] %s%s' % (i, line, more))
        if R > rows:
            self._w('        ...(共 %d 列)' % R)

    def table(self, headers, rows, widths=None):
        if not self.enabled:
            return
        widths = widths or [max(12, len(str(h)) + 2) for h in headers]
        self._w('    ' + ''.join(str(h).rjust(w) for h, w in zip(headers, widths)))
        self._w('    ' + ''.join('-' * (w - 1) + ' ' for w in widths))
        for r in rows:
            cells = []
            for v, w in zip(r, widths):
                cells.append(v.rjust(w) if isinstance(v, str) else _fmt(v, w, 5))
            self._w('    ' + ''.join(cells))

    def note(self, t):
        """給讀者的白話說明，跟數字區分開。"""
        if not self.enabled:
            return
        self._w()
        for line in t.strip().split('\n'):
            self._w('    >> ' + line.strip())
        self._w()

    def close(self):
        if self.enabled and self.f:
            self.f.close()
            self.f = None


NULL = Tracer(None, enabled=False)   # 不追蹤時用這個，所有呼叫都是空操作
