"""
把行內 LaTeX（$...$）轉成 Unicode 純文字。

    python tools/delatex_inline.py --dry     # 只看會改成什麼，不寫檔
    python tools/delatex_inline.py           # 實際轉換

------------------------------------------------------------------
為什麼要做這件事
------------------------------------------------------------------
很多 markdown 檢視器只支援「區塊公式」$$...$$，不支援「行內公式」$...$。
在那些環境裡，行內公式會原封不動露出來：

    $\\partial L / \\partial W$ 為什麼要轉置        ← 讀者看到的是這串原始碼

而區塊公式照樣正常。所以最保險的作法是：

    行內  ->  Unicode 純文字（∂L/∂W、x₁、σ、⊙）
    區塊  ->  維持 LaTeX（$$...$$）

Unicode 在 markdown、PDF、純文字終端機裡都能正常顯示，不依賴任何渲染器。

------------------------------------------------------------------
轉不掉的怎麼辦
------------------------------------------------------------------
太複雜、沒有對應 Unicode 的（例如多層分式、求和符號帶上下標），
腳本會列出來讓你自己決定——通常最好的作法是把它「升級成區塊公式」。
"""
import argparse
import glob
import io
import os
import re
import sys

# ---- LaTeX 指令 -> Unicode ----
CMD = {
    r'\partial': '∂', r'\odot': '⊙', r'\otimes': '⊗', r'\times': '×',
    r'\cdot': '·', r'\approx': '≈', r'\ne': '≠', r'\le': '≤', r'\ge': '≥',
    r'\to': '→', r'\rightarrow': '→', r'\leftarrow': '←', r'\Longrightarrow': '⟹',
    r'\sum': 'Σ', r'\prod': 'Π', r'\sqrt': '√', r'\infty': '∞', r'\pm': '±',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', r'\Delta': 'Δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\eta': 'η', r'\theta': 'θ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\sigma': 'σ', r'\Sigma': 'Σ', r'\tau': 'τ',
    r'\phi': 'φ', r'\rho': 'ρ', r'\pi': 'π', r'\nabla': '∇',
    r'\top': 'ᵀ', r'\in': '∈', r'\lvert': '|', r'\rvert': '|', r'\mid': '|',
    r'\quad': ' ', r'\qquad': '  ', r'\,': '', r'\;': ' ', r'\!': '',
    r'\left': '', r'\right': '', r'\displaystyle': '',
    r'\operatorname': '', r'\mathrm': '', r'\mathbf': '', r'\mathcal': '',
    r'\text': '', r'\boxed': '', r'\bigg': '', r'\big': '',
    # 函數名與雜項
    r'\sim': '~', r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
    r'\arg\max': 'argmax', r'\arg\min': 'argmin',
    r'\max': 'max', r'\min': 'min', r'\cos': 'cos', r'\sin': 'sin',
    r'\tanh': 'tanh', r'\checkmark': '✓', r'\%': '%', r'\|': '‖',
    r'\lesssim': '≲', r'\gg': '≫', r'\ll': '≪', r'\equiv': '≡',
    r'\lVert': '‖', r'\rVert': '‖', r'\mathbin': '', r'\limits': '',
}

SUB = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅',
       '6': '₆', '7': '₇', '8': '₈', '9': '₉', 'i': 'ᵢ', 'j': 'ⱼ',
       'k': 'ₖ', 'n': 'ₙ', 't': 'ₜ', 'v': 'ᵥ', 'a': 'ₐ', 'x': 'ₓ'}
SUP = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵',
       '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻', '+': '⁺',
       'T': 'ᵀ', 'n': 'ⁿ', 'ᵀ': 'ᵀ', 't': 'ᵗ'}


def convert(m):
    """
    把一段 LaTeX 內容轉成 Unicode。轉不乾淨就回傳 None。

    順序很重要：一定要先把 \\mathbf{x} 這類「只是換字體」的包裝拆掉，
    否則 \\dfrac{\\partial L}{\\partial \\mathbf{x}} 的分母含有巢狀大括號，
    分式的正則會比對不到。
    """
    s = m

    # ⓪ \partial L 中間的空白先併掉。否則分式判斷「分子含空白 = 含運算子」
    #    會把 ∂L/∂x 加上多餘的括號，變成 (∂L)/(∂x)。
    s = re.sub(r'\\partial\s+', r'\\partial', s)

    # ① 先拆掉純粹改變外觀的包裝（可能有巢狀，跑幾輪）
    for _ in range(4):
        s = re.sub(r'\\(?:text|mathrm|mathbf|mathcal|mathbin|operatorname|'
                   r'textrm|mbox|limits)\s*\{([^{}]*)\}', r'\1', s)
    s = s.replace(r'\limits', '')

    # ② 矩陣：\begin{bmatrix} 2 & 3 & 5 \end{bmatrix}  ->  [2 3 5]
    def mat(x):
        body = x.group(1).replace(r'\\', ' ; ').replace('&', ' ')
        return '[' + re.sub(r'\s+', ' ', body).strip() + ']'
    s = re.sub(r'\\begin\{[bpv]?matrix\}(.*?)\\end\{[bpv]?matrix\}', mat, s, flags=re.S)

    # ③ \hat{x} -> x̂     \bar{x} -> x̄
    s = re.sub(r'\\hat\s*\{([^{}])\}', lambda x: x.group(1) + '\u0302', s)
    s = re.sub(r'\\bar\s*\{([^{}])\}', lambda x: x.group(1) + '\u0304', s)

    # ④ \sqrt{v} -> √v
    for _ in range(3):
        s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'√\1', s)

    # ⑤ 上下標。一定要在分式之前做——因為 e^{-z} 這種帶大括號的上標
    #    會讓分式的分子分母出現巢狀大括號，分式的正則就比對不到了。
    for _ in range(3):
        s = re.sub(r'\^\{([^{}]+)\}', lambda x: sup(x.group(1)), s)
        s = re.sub(r'_\{([^{}]+)\}', lambda x: sub(x.group(1)), s)
    s = re.sub(r'\^(\S)', lambda x: sup(x.group(1)), s)
    s = re.sub(r'_(\w)', lambda x: sub(x.group(1)), s)

    # ⑥ \frac{a}{b} -> a/b。只有在分子或分母本身含運算子時才加括號，
    #    否則 ∂L/∂x 這種單純的會變成醜醜的 (∂L)/(∂x)。
    def frac(x):
        wrap = lambda t: ('(%s)' % t) if re.search(r'[+\-−*/ ]', t.strip()) else t
        return wrap(x.group(1).strip()) + '/' + wrap(x.group(2).strip())
    for _ in range(4):
        s = re.sub(r'\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', frac, s)

    # ⑦ 指令替換（長的先換，避免 \sigma 被 \s 之類截斷）
    for k in sorted(CMD, key=len, reverse=True):
        s = s.replace(k, CMD[k])

    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\s+', ' ', s).strip()

    # 後處理排版
    s = re.sub(r'([∂√Σ∇]) +', r'\1', s)      # \partial L -> ∂L，不要「∂ L」
    s = s.replace('^ᵀ', 'ᵀ')                  # W^{\top} -> Wᵀ，不要「W^ᵀ」
    s = re.sub(r'‖ +', '‖', s)                # ‖ dy ‖ -> ‖dy‖
    s = re.sub(r' +‖', '‖', s)

    if '\\' in s:            # 還有轉不掉的指令
        return None
    return s


def sup(t):
    return ''.join(SUP.get(c, c) for c in t) if all(c in SUP for c in t) else '^' + t


def sub(t):
    return ''.join(SUB.get(c, c) for c in t) if all(c in SUB for c in t) else '_' + t


INLINE = re.compile(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)')

# 絕對值符號在 markdown 表格裡會被當成欄位分隔符，必須跳脫。
# 例如 |V|（詞表大小）、|Δθ|（位移量）轉出來之後就會撞到這個問題。
ABS = re.compile(r'\|([A-Za-zͰ-Ͽ∂Δ][^|\n]{0,12}?)\|')


def escape_table_pipes(text):
    """把表格列裡「不是欄位分隔符」的 | 跳脫成 \\|。"""
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if r'\|' in s:                       # 已經跳脫過了，不要再跳一次
            out.append(line)
            continue
        if s.startswith('|') and s.endswith('|') and not set(s) <= set('|-: '):
            # 拆成儲存格，只在儲存格內部做跳脫
            cells = s[1:-1].split('|')
            merged, i = [], 0
            while i < len(cells):
                c = cells[i]
                # 「空儲存格 + 內容 + 空儲存格」通常是 |V| 被誤拆的結果
                if (c.strip() == '' and i + 2 < len(cells)
                        and cells[i + 2].strip() == '' and cells[i + 1].strip()):
                    merged.append(r'\|' + cells[i + 1] + r'\|')
                    i += 3
                    continue
                merged.append(c)
                i += 1
            line = '|' + '|'.join(merged) + '|'
            line = ABS.sub(lambda m: r'\|' + m.group(1) + r'\|', line)
        out.append(line)
    return '\n'.join(out)


def process(path, dry=False):
    raw = io.open(path, encoding='utf-8').read()

    # 把區塊公式與 code 藏起來，避免誤傷
    holes = []

    def hide(m):
        holes.append(m.group(0))
        return '\x00H%dZ\x01' % (len(holes) - 1)

    body = re.sub(r'\$\$.*?\$\$', hide, raw, flags=re.S)
    body = re.sub(r'```.*?```', hide, body, flags=re.S)
    body = re.sub(r'`[^`\n]*`', hide, body)

    changed, failed = [], []

    def repl(m):
        out = convert(m.group(1))
        if out is None:
            failed.append(m.group(1))
            return m.group(0)
        changed.append((m.group(1), out))
        return out

    body = INLINE.sub(repl, body)
    body = escape_table_pipes(body)
    for i, h in enumerate(holes):
        body = body.replace('\x00H%dZ\x01' % i, h)

    # 用「內容有沒有變」判斷要不要寫檔，不要用「有沒有轉換」。
    # 表格 | 的跳脫是獨立的一步，就算這次沒有任何行內公式要轉，也可能需要寫檔。
    if not dry and body != raw:
        io.open(path, 'w', encoding='utf-8').write(body)
    return changed, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--show', type=int, default=6, help='每份檔案顯示幾個轉換範例')
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    total, allfail = 0, 0
    for f in sorted(glob.glob(os.path.join(root, '*.md'))):
        changed, failed = process(f, args.dry)
        total += len(changed)
        allfail += len(failed)
        if not changed and not failed:
            continue
        print('%-30s 轉換 %3d 處，轉不掉 %d 處' % (os.path.basename(f), len(changed), len(failed)))
        for a, b in changed[:args.show]:
            print('    %-42s ->  %s' % ('$' + a + '$', b))
        for x in failed[:args.show]:
            print('    轉不掉：%s' % x)
    print()
    print('合計轉換 %d 處' % total + ('，%d 處需要手動處理' % allfail if allfail else '，全部成功'))
    print('（--dry 模式，沒有寫檔）' if args.dry else '（已寫檔）')
    return 1 if allfail else 0


if __name__ == '__main__':
    sys.exit(main())
