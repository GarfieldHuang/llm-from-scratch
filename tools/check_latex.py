"""
檢查 docs/ 底下的 LaTeX 是否完好。

    python tools/check_latex.py

------------------------------------------------------------------
為什麼需要這個
------------------------------------------------------------------
用文字工具批次修改 markdown 時，如果經過 shell heredoc 之類的管道，
反斜線很容易被吃掉一層，變成控制字元：

    \\times  ->  <TAB>imes
    \\right  ->  <CR>ight
    \\frac   ->  <FF>rac
    \\text   ->  <TAB>ext

損壞後的檔案看起來幾乎正常（缺的字母混在字裡不明顯），
在 GitHub 上勉強能看，轉成 PDF 才會發現公式整段壞掉。

可靠的偵測特徵是**控制字元**：正常的 markdown 公式裡不該有 TAB、FF、VT。

不要用「搜尋 imes、ight 這類殘骸」的方式偵測——
\\times 本身就含有 imes，會產生大量誤判。
"""
import glob
import io
import os
import re
import sys

# 只認這幾個「絕不該出現在 markdown 公式裡」的控制字元。
# 刻意不檢查 CR（13）：git 在 Windows 上會把行尾轉成 CRLF，那是正常的，
# 而且下面用通用換行模式讀檔，CRLF 本來就會被正規化掉。
CTRL = {
    9: r'TAB（\t 被吃掉）',
    11: r'VT（\v 被吃掉）',
    12: r'FF（\f 被吃掉）',
    8: r'BS（\b 被吃掉）',
    7: r'BEL（\a 被吃掉）',
}

# \text{...} 與 \mathrm{...} 裡面本來就可以放中文，檢查前先拿掉
TEXT_CMD = re.compile(r'\\(?:text|mathrm|textrm|mbox)\{[^{}]*\}')
CJK_PUNCT = re.compile(r'[，。；：、？！]')


def strip_code(s):
    """把 code fence 與行內 code 拿掉，只留下散文與公式。"""
    s = re.sub(r'```.*?```', '', s, flags=re.S)
    return re.sub(r'`[^`\n]*`', '', s)


def check_file(path):
    raw = io.open(path, encoding='utf-8').read()      # 通用換行，CRLF -> LF
    body = strip_code(raw)
    issues = []

    # ---- ① 控制字元：反斜線被吃掉的鐵證 ----
    seen = {}
    for lineno, line in enumerate(raw.split('\n'), 1):
        for ch in line:
            o = ord(ch)
            if o in CTRL:
                seen.setdefault(o, []).append(lineno)
    for o, lines in sorted(seen.items()):
        more = '' if len(lines) <= 6 else ' 等 %d 處' % len(lines)
        issues.append('第 %s 行有 %s%s' % (','.join(map(str, lines[:6])), CTRL[o], more))

    # ---- ② $ 與 $$ 是否成對 ----
    disp = body.count('$$')
    if disp % 2:
        issues.append('$$ 共 %d 個（奇數，沒有成對）' % disp)
    inline = len(re.findall(r'(?<!\$)\$(?!\$)', body.replace('$$', '\x00')))
    if inline % 2:
        issues.append('$ 共 %d 個（奇數，沒有成對）' % inline)

    # ---- ③ 區塊公式裡出現裸露的中文標點（通常是誤貼）----
    for m in re.finditer(r'\$\$(.+?)\$\$', body, flags=re.S):
        inner = TEXT_CMD.sub('', m.group(1))
        if CJK_PUNCT.search(inner):
            issues.append('$$...$$ 內有 \\text{} 以外的中文標點，可能誤貼')
            break

    return issues, inline // 2, disp // 2


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    files = sorted(glob.glob(os.path.join(root, '*.md')))
    bad = 0
    print('檢查 %d 份文件' % len(files))
    print('=' * 76)

    for f in files:
        issues, n_inline, n_disp = check_file(f)
        name = os.path.basename(f)
        if issues:
            bad += 1
            print('  %-30s FAIL' % name)
            for i in issues:
                print('      - %s' % i)
        else:
            print('  %-30s OK    行內 %2d 組 / 區塊 %2d 組' % (name, n_inline, n_disp))

    print('=' * 76)
    print('全部通過' if not bad else '%d 份有問題' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
