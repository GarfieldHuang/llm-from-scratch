"""
把 docs/*.md 合併轉成一份 PDF。

    python tools/build_pdf.py
    python tools/build_pdf.py --split      # 每份文件各自輸出一個 PDF
    python tools/build_pdf.py --keep-html  # 保留中間產物，方便除錯

流程：markdown -> HTML（KaTeX + Mermaid）-> headless Chrome 列印成 PDF

------------------------------------------------------------------
最關鍵的一步：先把數學從 markdown 解析器手上保護起來
------------------------------------------------------------------
如果直接把含 $...$ 的 markdown 丟給解析器，公式會被毀掉：

    x_1 x_2        底線被當成斜體標記
    \\alpha         反斜線被吃掉
    a * b          星號被當成粗體

所以順序必須是：
    1. 把 code fence 抽出來換成佔位符
    2. 把行內 code 抽出來換成佔位符
    3. 把 $$...$$ 和 $...$ 抽出來換成佔位符
    4. 這時候才跑 markdown 轉換
    5. 最後把佔位符換回原文

佔位符用 \\x00 \\x01 這種控制字元包起來，確保不會跟正文撞名。
"""
import argparse
import glob
import html
import io
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
# 放 pdf/ 而不是 out/：out/ 有 gitignore（裡面是模型權重），
# 但 PDF 是要跟著 repo 一起發布的成品。
OUT = os.path.join(ROOT, 'pdf')

CHROME_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
]


def find_chrome():
    for p in CHROME_CANDIDATES:
        if os.path.exists(p):
            return p
    raise RuntimeError('找不到 Chrome 或 Edge，無法列印 PDF')


# ==================================================================
# 保護 / 還原
# ==================================================================
class Vault:
    """把不能讓 markdown 碰的片段存起來，換成佔位符。"""

    def __init__(self):
        self.items = []

    def stash(self, kind, text):
        self.items.append((kind, text))
        return '\x00P%dZ\x01' % (len(self.items) - 1)

    def restore(self, s):
        for i, (kind, text) in enumerate(self.items):
            ph = '\x00P%dZ\x01' % i
            if kind == 'fence':
                lang, code = text
                if lang == 'mermaid':
                    rep = '<pre class="mermaid">%s</pre>' % html.escape(code)
                else:
                    rep = '<pre class="code"><code>%s</code></pre>' % html.escape(code)
            elif kind == 'icode':
                rep = '<code>%s</code>' % html.escape(text)
            elif kind == 'dmath':
                rep = '<div class="kx-display">%s</div>' % html.escape(text)
            else:
                rep = '<span class="kx-inline">%s</span>' % html.escape(text)
            # 佔位符可能被 markdown 包進 <p>，用 replace 即可
            s = s.replace(ph, rep)
        return s


FENCE = re.compile(r'^```([a-zA-Z0-9_-]*)\n(.*?)^```\s*$', re.S | re.M)
ICODE = re.compile(r'`([^`\n]+)`')
DMATH = re.compile(r'\$\$(.+?)\$\$', re.S)
IMATH = re.compile(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)')


def protect(md, vault):
    md = FENCE.sub(lambda m: vault.stash('fence', (m.group(1), m.group(2))), md)
    md = ICODE.sub(lambda m: vault.stash('icode', m.group(1)), md)
    md = DMATH.sub(lambda m: vault.stash('dmath', m.group(1).strip()), md)
    md = IMATH.sub(lambda m: vault.stash('imath', m.group(1).strip()), md)
    return md


# ==================================================================
# HTML 樣板
# ==================================================================
CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Noto Sans TC","Microsoft JhengHei","PingFang TC",system-ui,sans-serif;
  font-size: 10.5pt; line-height: 1.75; color: #16202a; margin: 0;
}
h1 { font-size: 20pt; margin: 0 0 14px; padding-bottom: 8px;
     border-bottom: 2px solid #2c5f7c; color: #16202a; break-after: avoid; }
h2 { font-size: 14pt; margin: 26px 0 10px; color: #1f4e79; break-after: avoid;
     border-left: 4px solid #2c5f7c; padding-left: 10px; }
h3 { font-size: 11.5pt; margin: 18px 0 8px; color: #2c5f7c; break-after: avoid; }
p  { margin: 0 0 10px; }
strong { color: #0b2b3d; }
hr { border: 0; border-top: 1px solid #d6dee6; margin: 20px 0; }
a { color: #1f4e79; text-decoration: none; }

pre.code { background: #f6f8fa; border: 1px solid #dfe5ec; border-radius: 4px;
  padding: 10px 12px; overflow-x: auto; break-inside: avoid; margin: 12px 0; }
pre.code code { font-family: "Consolas","JetBrains Mono",monospace;
  font-size: 8.6pt; line-height: 1.55; white-space: pre; color: #223; }
code { font-family: "Consolas","JetBrains Mono",monospace; font-size: 9pt;
  background: #eef2f6; padding: 1px 4px; border-radius: 3px; color: #1f4e79; }

table { border-collapse: collapse; width: 100%; margin: 12px 0;
        font-size: 9.4pt; break-inside: avoid; }
th, td { border: 1px solid #d6dee6; padding: 5px 8px; text-align: left;
         vertical-align: top; }
th { background: #eef3f8; font-weight: 600; color: #1f4e79; }
tr:nth-child(even) td { background: #fbfcfd; }

blockquote { border-left: 3px solid #f0a500; background: #fffdf5;
  margin: 12px 0; padding: 8px 14px; break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }

ul, ol { margin: 0 0 10px; padding-left: 22px; }
li { margin: 3px 0; }

.kx-display { margin: 14px 0; text-align: center; break-inside: avoid; }
.kx-inline { white-space: nowrap; }
.katex { font-size: 1.02em; }
.katex-display { margin: 0; }

pre.mermaid { background: #fff; text-align: center; margin: 16px 0;
  break-inside: avoid; border: none; }
pre.mermaid svg { max-width: 100%; height: auto; }

.doc { break-before: page; }
.doc:first-of-type { break-before: auto; }

#cover { text-align: center; padding-top: 70mm; break-after: page; }
#cover .t { font-size: 26pt; font-weight: 700; color: #16202a; letter-spacing: 1px; }
#cover .s { font-size: 12pt; color: #5b6b7a; margin-top: 14px; line-height: 1.9; }
#cover .m { font-size: 9.5pt; color: #8494a3; margin-top: 40px;
            font-family: "Consolas",monospace; }
#toc { break-after: page; }
#toc ol { font-size: 11pt; line-height: 2.1; }
"""

PAGE = """<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<style>__CSS__</style>
</head><body>
__BODY__
<script>
(function () {
  document.querySelectorAll('.kx-display').forEach(function (el) {
    try { katex.render(el.textContent, el, {displayMode: true, throwOnError: false}); }
    catch (e) { el.style.color = 'red'; }
  });
  document.querySelectorAll('.kx-inline').forEach(function (el) {
    try { katex.render(el.textContent, el, {displayMode: false, throwOnError: false}); }
    catch (e) { el.style.color = 'red'; }
  });
  mermaid.initialize({startOnLoad: false, theme: 'neutral',
                      themeVariables: {fontFamily: 'Noto Sans TC, Microsoft JhengHei, sans-serif',
                                       fontSize: '14px'}});
  mermaid.run({querySelector: 'pre.mermaid'}).then(function () {
    window.__ready = true;
  }).catch(function () { window.__ready = true; });
})();
</script>
</body></html>
"""


def md_to_html(md_text):
    import markdown
    vault = Vault()
    protected = protect(md_text, vault)
    body = markdown.markdown(protected, extensions=['tables', 'sane_lists', 'attr_list'])
    return vault.restore(body)


def build_html(files, title, with_cover=True):
    import markdown  # noqa: F401  (提早失敗比較好除錯)
    parts = []
    if with_cover:
        parts.append(
            '<div id="cover"><div class="t">%s</div>'
            '<div class="s">從零手刻 Transformer<br>每一條公式與反向傳播都自己寫</div>'
            '<div class="m">github.com/GarfieldHuang/llm-from-scratch<br>%s</div></div>'
            % (html.escape(title), time.strftime('%Y-%m-%d')))
        items = []
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            if name.lower() == 'readme':
                continue
            items.append('<li>%s</li>' % html.escape(re.sub(r'^\d+-', '', name)))
        parts.append('<div id="toc"><h1>目錄</h1><ol>%s</ol></div>' % ''.join(items))

    for f in files:
        parts.append('<div class="doc">%s</div>'
                     % md_to_html(io.open(f, encoding='utf-8').read()))

    return (PAGE.replace('__CSS__', CSS)
                .replace('__TITLE__', html.escape(title))
                .replace('__BODY__', '\n'.join(parts)))


def print_pdf(chrome, html_path, pdf_path, budget=25000):
    cmd = [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
           '--no-pdf-header-footer', '--run-all-compositor-stages-before-draw',
           '--virtual-time-budget=%d' % budget,
           '--print-to-pdf=%s' % pdf_path,
           'file:///' + html_path.replace('\\', '/')]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    if not os.path.exists(pdf_path):
        raise RuntimeError('列印失敗：%s' % r.stderr.decode('utf-8', 'ignore')[:400])
    return os.path.getsize(pdf_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', action='store_true', help='每份文件各自一個 PDF')
    ap.add_argument('--keep-html', action='store_true')
    ap.add_argument('--title', default='llm-from-scratch 教學文件')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    chrome = find_chrome()
    print('用 %s 列印' % os.path.basename(chrome))

    files = sorted(glob.glob(os.path.join(DOCS, '*.md')))
    files = [f for f in files if os.path.basename(f).lower() != 'readme.md'] + \
            [f for f in files if os.path.basename(f).lower() == 'readme.md']
    print('找到 %d 份文件' % len(files))

    made = []
    if args.split:
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            hp = os.path.join(OUT, stem + '.html')
            pp = os.path.join(OUT, stem + '.pdf')
            io.open(hp, 'w', encoding='utf-8').write(
                build_html([f], stem, with_cover=False))
            size = print_pdf(chrome, hp, pp)
            made.append((pp, size))
            print('  %-34s %7.1f KB' % (os.path.basename(pp), size / 1024))
            if not args.keep_html:
                os.remove(hp)
    else:
        hp = os.path.join(OUT, 'docs.html')
        pp = os.path.join(OUT, 'llm-from-scratch-教學文件.pdf')
        io.open(hp, 'w', encoding='utf-8').write(build_html(files, args.title))
        size = print_pdf(chrome, hp, pp)
        made.append((pp, size))
        print('  %-34s %7.1f KB' % (os.path.basename(pp), size / 1024))
        if not args.keep_html:
            os.remove(hp)

    print()
    print('完成，共 %d 個檔案，都在 out/ 底下' % len(made))
    return 0


if __name__ == '__main__':
    sys.exit(main())
