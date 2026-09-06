"""
檢查 docs/ 底下的 markdown 公式是否安全。

    python tools/check_latex.py

------------------------------------------------------------------
① 不准用行內 LaTeX（$...$）
------------------------------------------------------------------
很多 markdown 檢視器只支援「區塊公式」$$...$$，不支援「行內公式」$...$。
在那些環境裡，行內公式會原封不動露出來給讀者看：

    $\\partial L / \\partial W$ 為什麼要轉置        ← 讀者看到的是這串原始碼

而區塊公式照樣正常渲染。所以這份專案的規則是：

    行內  ->  Unicode 純文字（∂L/∂W、x₁、σ、⊙、Wᵢⱼ、10⁻⁶）
    區塊  ->  維持 LaTeX（$$...$$）

Unicode 在 markdown、PDF、純文字終端機裡都能顯示，不依賴任何渲染器。
要批次轉換用 `python tools/delatex_inline.py`。

------------------------------------------------------------------
② 控制字元
------------------------------------------------------------------
用文字工具批次改檔案時，反斜線很容易被吃掉一層，變成控制字元：

    \\times  ->  <TAB>imes
    \\right  ->  <CR>ight

損壞後的檔案看起來幾乎正常，轉成 PDF 才會發現公式壞掉。
可靠的偵測特徵就是控制字元——正常的 markdown 裡不該有 TAB、FF、VT。

（不要用「找 imes、ight 這種殘骸」的方式偵測——\\times 本身就含有 imes。）

------------------------------------------------------------------
③ 表格裡沒有跳脫的 |
------------------------------------------------------------------
|V|、|Δθ| 這種絕對值符號放進表格會被當成欄位分隔符，整列排版就爛掉。
必須寫成 \\|V\\|。
"""
import glob
import io
import os
import re
import sys

CTRL = {
    9: r'TAB（\t 被吃掉）',
    11: r'VT（\v 被吃掉）',
    12: r'FF（\f 被吃掉）',
    8: r'BS（\b 被吃掉）',
    7: r'BEL（\a 被吃掉）',
}

TEXT_CMD = re.compile(r'\\(?:text|mathrm|textrm|mbox)\{[^{}]*\}')
CJK_PUNCT = re.compile(r'[，。；：、？！]')
INLINE = re.compile(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)')


def strip_code(s):
    s = re.sub(r'```.*?```', '', s, flags=re.S)
    return re.sub(r'`[^`\n]*`', '', s)


def check_file(path):
    raw = io.open(path, encoding='utf-8').read()
    body = strip_code(raw)
    issues = []

    # ---- ① 行內 LaTeX ----
    masked = re.sub(r'\$\$.*?\$\$', '', body, flags=re.S)
    inline = INLINE.findall(masked)
    if inline:
        issues.append('用了 %d 處行內公式 $...$，很多檢視器不支援。'
                      '跑 tools/delatex_inline.py 轉成 Unicode。' % len(inline))
        for x in inline[:3]:
            issues.append('    例如：$%s$' % x[:50])

    # ---- ② 控制字元 ----
    seen = {}
    for lineno, line in enumerate(raw.split('\n'), 1):
        for ch in line:
            o = ord(ch)
            if o in CTRL:
                seen.setdefault(o, []).append(lineno)
    for o, lines in sorted(seen.items()):
        more = '' if len(lines) <= 6 else ' 等 %d 處' % len(lines)
        issues.append('第 %s 行有 %s%s' % (','.join(map(str, lines[:6])), CTRL[o], more))

    # ---- ③ 表格裡沒跳脫的 | ----
    # 不能用「有空儲存格」判斷——空儲存格是合法的（例如因果遮罩那張表，
    # 空白代表被遮住）。可靠的訊號是「某一列的欄位數跟表頭對不上」，
    # 那才代表有 | 把儲存格切爛了。
    ncol, in_table = None, False
    for lineno, line in enumerate(raw.split('\n'), 1):
        s = line.strip()
        if not (s.startswith('|') and s.endswith('|')):
            ncol, in_table = None, False
            continue
        cells = len(re.split(r'(?<!\\)\|', s[1:-1]))
        if set(s) <= set('|-: '):          # 分隔線，用它來定錨欄位數
            ncol, in_table = cells, True
            continue
        if in_table and ncol and cells != ncol:
            issues.append('第 %d 行表格有 %d 欄，表頭是 %d 欄'
                          '——通常是 | 沒跳脫（要寫成 \\|）' % (lineno, cells, ncol))
        elif ncol is None:
            ncol = cells                   # 表頭列

    # ---- ④ 區塊公式裡的中文標點 ----
    for m in re.finditer(r'\$\$(.+?)\$\$', body, flags=re.S):
        if CJK_PUNCT.search(TEXT_CMD.sub('', m.group(1))):
            issues.append('$$...$$ 內有 \\text{} 以外的中文標點，可能誤貼')
            break

    n_block = body.count('$$')
    if n_block % 2:
        issues.append('$$ 共 %d 個（奇數，沒有成對）' % n_block)

    return issues, n_block // 2


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    files = sorted(glob.glob(os.path.join(root, '*.md')))
    bad = 0
    print('檢查 %d 份文件' % len(files))
    print('=' * 76)

    for f in files:
        issues, n_block = check_file(f)
        name = os.path.basename(f)
        if issues:
            bad += 1
            print('  %-30s FAIL' % name)
            for i in issues:
                print('      - %s' % i)
        else:
            print('  %-30s OK    區塊公式 %2d 組，無行內公式' % (name, n_block))

    print('=' * 76)
    print('全部通過' if not bad else '%d 份有問題' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
