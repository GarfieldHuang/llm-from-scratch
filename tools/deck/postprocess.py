"""
pptxgenjs 產出後的修正：
  ① 字型缺字 —— 把含缺字的 run 拆開，只讓那幾個字元改用 Cambria Math
  ② PowerPoint 要求修復 —— 重新打包 ZIP（[Content_Types].xml 排第一、
     不含資料夾項目、用 DEFLATE 壓縮）
"""
import glob, logging, os, re, shutil, sys, tempfile, warnings, zipfile
from lxml import etree
from fontTools.ttLib import TTFont, TTCollection

warnings.filterwarnings('ignore')
logging.getLogger('fontTools').setLevel(logging.ERROR)
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS = {'a': A}
FALLBACK = 'Cambria Math'
DEFAULT_LATIN = 'Calibri'


def font_db():
    dirs = ['/System/Library/Fonts/Supplemental', '/Library/Fonts', '/Library/Fonts/Microsoft',
            os.path.expanduser('~/Library/Fonts'), '/System/Library/Fonts']
    dirs += glob.glob('/Applications/Microsoft *.app/Contents/Resources/DFonts')
    dirs += glob.glob('/Volumes/*/Applications/Microsoft *.app/Contents/Resources/DFonts')
    paths = []
    for d in dirs:
        for ext in ('*.ttf', '*.ttc', '*.otf'):
            paths += glob.glob(os.path.join(d, ext))
    db = {}
    for p in sorted(set(paths)):
        try:
            fonts = TTCollection(p).fonts if p.endswith('.ttc') else [TTFont(p, fontNumber=0)]
        except Exception:
            continue
        for f in fonts:
            try:
                n = f['name'].getDebugName(4) or ''
                cm = set()
                for t in f['cmap'].tables:
                    cm |= set(t.cmap.keys())
            except Exception:
                continue
            if n and n not in db:
                db[n] = cm
    return db


DB = font_db()
if FALLBACK not in DB:
    sys.exit(f'找不到後援字型 {FALLBACK}')


# 這些區段由 OOXML 的 East Asian / 複雜文字槽負責，不歸 latin 字型管
_EA_RANGES = ((0x2E80, 0x9FFF), (0x3000, 0x303F), (0xAC00, 0xD7AF),
              (0x3040, 0x30FF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))


def latin_slot(ch):
    """這個字元會不會用到 latin 字型槽。"""
    o = ord(ch)
    if o < 0x80:
        return True
    return not any(lo <= o <= hi for lo, hi in _EA_RANGES)


def covers(font, ch):
    """這個字型有沒有這個字。非 latin 槽的字元一律當作「有」（不歸它管）。"""
    if not latin_slot(ch):
        return True
    cm = DB.get(font)
    return True if cm is None else ord(ch) in cm


def resolve(rPr):
    """算出這個 run 實際會用哪個字型檔（含粗體變體）。"""
    face, bold = DEFAULT_LATIN, False
    if rPr is not None:
        el = rPr.find('a:latin', NS)
        if el is not None and el.get('typeface'):
            face = el.get('typeface')
        bold = rPr.get('b') == '1'
    if bold and f'{face} Bold' in DB:
        return f'{face} Bold'
    return face


def set_face(rPr, name):
    for tag in ('a:latin', 'a:cs'):
        el = rPr.find(tag, NS)
        if el is None:
            el = etree.SubElement(rPr, f'{{{A}}}' + tag.split(':')[1])
        el.set('typeface', name)


def fix_runs(tree):
    """拆開含缺字的 run。回傳 (修好的字元數, 樣本)。"""
    fixed, samples = 0, set()
    for r in list(tree.iter(f'{{{A}}}r')):
        t = r.find('a:t', NS)
        if t is None or not t.text:
            continue
        rPr = r.find('a:rPr', NS)
        face = resolve(rPr)
        text = t.text
        bad = [c for c in text if not covers(face, c)]
        if not bad:
            continue
        # 依「這個字元是否缺」把文字切成連續區段
        chunks, cur, cur_bad = [], '', None
        for c in text:
            b = not covers(face, c)
            if cur_bad is None or b == cur_bad:
                cur += c; cur_bad = b
            else:
                chunks.append((cur, cur_bad)); cur, cur_bad = c, b
        if cur:
            chunks.append((cur, cur_bad))
        parent = r.getparent()
        idx = list(parent).index(r)
        parent.remove(r)
        for off, (chunk, is_bad) in enumerate(chunks):
            nr = etree.fromstring(etree.tostring(r))
            nt = nr.find('a:t', NS)
            nt.text = chunk
            if chunk != chunk.strip():
                nt.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            if is_bad:
                npr = nr.find('a:rPr', NS)
                if npr is None:
                    npr = etree.Element(f'{{{A}}}rPr')
                    nr.insert(0, npr)
                set_face(npr, FALLBACK)
                fixed += len(chunk)
                samples.update(chunk)
            parent.insert(idx + off, nr)
    return fixed, samples


C = 'http://schemas.openxmlformats.org/drawingml/2006/chart'
CNS = {'c': C}
_PLOT = ('lineChart', 'barChart', 'pieChart', 'doughnutChart', 'areaChart',
         'scatterChart', 'bubbleChart', 'radarChart', 'ofPieChart', 'surfaceChart',
         'line3DChart', 'bar3DChart', 'area3DChart', 'pie3DChart')


def fix_charts(workdir):
    """pptxgenjs 會寫出自己沒宣告的座標軸 id，PowerPoint 會判定檔案損毀。
    把懸空的 <c:axId> 移除。"""
    removed = 0
    for xml in glob.glob(os.path.join(workdir, 'ppt', 'charts', 'chart*.xml')):
        tree = etree.parse(xml)
        root = tree.getroot()
        declared = set()
        for ax in ('catAx', 'valAx', 'serAx', 'dateAx'):
            for el in root.iter(f'{{{C}}}{ax}'):
                a = el.find('c:axId', CNS)
                if a is not None and a.get('val'):
                    declared.add(a.get('val'))
        changed = False
        for plot in _PLOT:
            for el in root.iter(f'{{{C}}}{plot}'):
                for a in list(el.findall('c:axId', CNS)):
                    if a.get('val') not in declared:
                        el.remove(a)
                        removed += 1
                        changed = True
        if changed:
            tree.write(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    return removed


P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
R = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
THEME_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme'


def fix_presentation_order(workdir):
    """CT_Presentation 是嚴格序列：notesMasterIdLst 必須排在 sldIdLst 之前。
    pptxgenjs 寫反了，PowerPoint 因此要求修復。"""
    f = os.path.join(workdir, 'ppt', 'presentation.xml')
    tree = etree.parse(f)
    root = tree.getroot()
    kids = list(root)
    names = [etree.QName(k).localname for k in kids]
    if 'notesMasterIdLst' not in names or 'sldIdLst' not in names:
        return False
    if names.index('notesMasterIdLst') < names.index('sldIdLst'):
        return False
    nm = kids[names.index('notesMasterIdLst')]
    sl = kids[names.index('sldIdLst')]
    root.remove(nm)
    sl.addprevious(nm)
    tree.write(f, xml_declaration=True, encoding='UTF-8', standalone=True)
    return True


def fix_notes_theme(workdir):
    """slideMaster 與 notesMaster 不能共用同一個 theme part。
    複製一份 theme2 給 notesMaster。"""
    nrel = os.path.join(workdir, 'ppt', 'notesMasters', '_rels', 'notesMaster1.xml.rels')
    srel = os.path.join(workdir, 'ppt', 'slideMasters', '_rels', 'slideMaster1.xml.rels')
    if not (os.path.exists(nrel) and os.path.exists(srel)):
        return False

    def theme_of(path):
        t = etree.parse(path)
        for e in t.getroot():
            if e.get('Type') == THEME_REL:
                return t, e
        return t, None

    ntree, nel = theme_of(nrel)
    stree, sel = theme_of(srel)
    if nel is None or sel is None:
        return False
    if nel.get('Target') != sel.get('Target'):
        return False                      # 已經各自獨立

    # 找一個沒被用過的 themeN.xml
    tdir = os.path.join(workdir, 'ppt', 'theme')
    n = 2
    while os.path.exists(os.path.join(tdir, f'theme{n}.xml')):
        n += 1
    src_theme = os.path.join(tdir, os.path.basename(nel.get('Target')))
    new_theme = os.path.join(tdir, f'theme{n}.xml')
    shutil.copyfile(src_theme, new_theme)

    nel.set('Target', f'../theme/theme{n}.xml')
    ntree.write(nrel, xml_declaration=True, encoding='UTF-8', standalone=True)

    # [Content_Types].xml 補上覆寫
    ctf = os.path.join(workdir, '[Content_Types].xml')
    ctree = etree.parse(ctf)
    croot = ctree.getroot()
    ct_type = None
    for e in croot:
        if e.tag == f'{{{CT}}}Override' and e.get('PartName') == '/ppt/theme/theme1.xml':
            ct_type = e.get('ContentType')
    ov = etree.SubElement(croot, f'{{{CT}}}Override')
    ov.set('PartName', f'/ppt/theme/theme{n}.xml')
    ov.set('ContentType', ct_type or 'application/vnd.openxmlformats-officedocument.theme+xml')
    ctree.write(ctf, xml_declaration=True, encoding='UTF-8', standalone=True)
    return n


def fix_shape_ids(workdir):
    """pptxgenjs 的表格(graphicFrame)與一般圖形(sp)用不同的 id 計數器，
    有表格的投影片會撞號。OOXML 要求同一部分內 cNvPr/@id 唯一。"""
    total, slides = 0, 0
    pats = ['ppt/slides/slide*.xml', 'ppt/slideLayouts/slideLayout*.xml',
            'ppt/slideMasters/slideMaster*.xml', 'ppt/notesSlides/notesSlide*.xml',
            'ppt/notesMasters/notesMaster*.xml']
    for pat in pats:
        for xml in glob.glob(os.path.join(workdir, *pat.split('/'))):
            tree = etree.parse(xml)
            els = [e for e in tree.getroot().iter() if e.tag.endswith('}cNvPr')]
            used, fixed = set(), 0
            nxt = max([int(e.get('id')) for e in els if (e.get('id') or '').isdigit()] or [1]) + 1
            for e in els:
                v = e.get('id')
                if not v or not v.isdigit():
                    continue
                if v in used:
                    e.set('id', str(nxt))
                    used.add(str(nxt))
                    nxt += 1
                    fixed += 1
                else:
                    used.add(v)
            if fixed:
                tree.write(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
                total += fixed
                slides += 1
    return total, slides


def fix_content_types(workdir):
    """pptxgenjs 每張投影片都寫一筆 slideMasterN.xml 的 Override，但那些 part 根本不存在。
    OPC 不允許宣告不存在的 part —— 這是 PowerPoint 要求修復的主因。
    同時移除套件裡沒有用到的 Default 副檔名。"""
    ctf = os.path.join(workdir, '[Content_Types].xml')
    tree = etree.parse(ctf)
    root = tree.getroot()

    present, exts = set(), set()
    for r_, _, fs in os.walk(workdir):
        for fn in fs:
            rel = '/' + os.path.relpath(os.path.join(r_, fn), workdir).replace(os.sep, '/')
            present.add(rel)
            if '.' in fn:
                exts.add(fn.rsplit('.', 1)[-1].lower())

    ghosts, unused = 0, 0
    for e in list(root):
        tag = etree.QName(e).localname
        if tag == 'Override':
            if e.get('PartName') not in present:
                root.remove(e); ghosts += 1
        elif tag == 'Default':
            if (e.get('Extension') or '').lower() not in exts:
                root.remove(e); unused += 1
    if ghosts or unused:
        tree.write(ctf, xml_declaration=True, encoding='UTF-8', standalone=True)
    return ghosts, unused


_FILL = ('noFill', 'solidFill', 'gradFill', 'blipFill', 'pattFill', 'grpFill')


def fix_backgrounds(workdir):
    """CT_BackgroundProperties 的序列是 EG_FillProperties 接 EG_EffectProperties，
    後者 minOccurs=1 —— <a:effectLst/> 是必要元素。pptxgenjs 沒寫，
    每一張有背景色的投影片都違規。順便補上 slideMaster 缺的 <p:bg>。"""
    n_eff, n_bg = 0, 0
    pats = ['ppt/slides/slide*.xml', 'ppt/slideLayouts/slideLayout*.xml',
            'ppt/slideMasters/slideMaster*.xml', 'ppt/notesSlides/notesSlide*.xml',
            'ppt/notesMasters/notesMaster*.xml']
    for pat in pats:
        for xml in glob.glob(os.path.join(workdir, *pat.split('/'))):
            tree = etree.parse(xml)
            root = tree.getroot()
            changed = False

            for bgPr in root.iter(f'{{{P}}}bgPr'):
                if bgPr.find(f'{{{A}}}effectLst') is not None:
                    continue
                idx = 0
                for i, ch in enumerate(bgPr):
                    if etree.QName(ch).localname in _FILL:
                        idx = i + 1
                bgPr.insert(idx, etree.Element(f'{{{A}}}effectLst'))
                n_eff += 1
                changed = True

            # slideMaster 的 cSld 需要 <p:bg>，且必須排在 spTree 之前
            if os.path.basename(xml).startswith('slideMaster'):
                cSld = root.find(f'{{{P}}}cSld')
                if cSld is not None and cSld.find(f'{{{P}}}bg') is None:
                    bg = etree.Element(f'{{{P}}}bg')
                    ref = etree.SubElement(bg, f'{{{P}}}bgRef')
                    ref.set('idx', '1001')
                    clr = etree.SubElement(ref, f'{{{A}}}schemeClr')
                    clr.set('val', 'bg1')
                    cSld.insert(0, bg)
                    n_bg += 1
                    changed = True

            if changed:
                tree.write(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    return n_eff, n_bg


_NUMRE = re.compile(r'^val\s+(-?\d+)$')


def fix_bad_numbers(workdir):
    """pptxgenjs 遇到零尺寸的 roundRect 會把 Infinity / NaN 直接寫進
    <a:gd fmla="val ...">。該處必須是整數，寫成字串會讓 PowerPoint 判定損毀。"""
    fixed = 0
    for pat in ('ppt/slides/slide*.xml', 'ppt/slideLayouts/slideLayout*.xml',
                'ppt/slideMasters/slideMaster*.xml', 'ppt/notesSlides/notesSlide*.xml',
                'ppt/notesMasters/notesMaster*.xml', 'ppt/charts/chart*.xml'):
        for xml in glob.glob(os.path.join(workdir, *pat.split('/'))):
            tree = etree.parse(xml)
            changed = False
            for gd in tree.getroot().iter(f'{{{A}}}gd'):
                f = gd.get('fmla') or ''
                if not _NUMRE.match(f):
                    gd.set('fmla', 'val 0')
                    fixed += 1
                    changed = True
            if changed:
                tree.write(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    return fixed


def repack(src, dst, workdir):
    """[Content_Types].xml 排第一、不含資料夾項目、DEFLATE。"""
    names = []
    for root, _, files in os.walk(workdir):
        for fn in files:
            full = os.path.join(root, fn)
            names.append(os.path.relpath(full, workdir))
    names.sort(key=lambda n: (n != '[Content_Types].xml', n))
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in names:
            z.write(os.path.join(workdir, n), n.replace(os.sep, '/'))


def main(path):
    work = tempfile.mkdtemp()
    with zipfile.ZipFile(path) as z:
        for i in z.infolist():
            if i.filename.endswith('/'):
                continue
            tgt = os.path.join(work, i.filename)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            with open(tgt, 'wb') as f:
                f.write(z.read(i.filename))

    total, allsamples, touched = 0, set(), 0
    for xml in glob.glob(os.path.join(work, 'ppt', 'slides', '*.xml')) + \
               glob.glob(os.path.join(work, 'ppt', 'notesSlides', '*.xml')):
        tree = etree.parse(xml)
        n, s = fix_runs(tree)
        if n:
            tree.write(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
            total += n; allsamples |= s; touched += 1

    if fix_presentation_order(work):
        print('  結構修正：notesMasterIdLst 移到 sldIdLst 之前（OOXML 要求的順序）')
    tn = fix_notes_theme(work)
    if tn:
        print(f'  結構修正：notesMaster 改用獨立的 theme{tn}.xml（原本與 slideMaster 共用）')

    nbad = fix_bad_numbers(work)
    if nbad:
        print(f'  結構修正：{nbad} 個 <a:gd fmla> 的值不是數字（NaN / Infinity），已改為 0')

    ne, nb = fix_backgrounds(work)
    if ne or nb:
        print(f'  結構修正：補上 {ne} 個必要的 <a:effectLst/>' + (f'、{nb} 個 slideMaster 背景' if nb else ''))

    ng, nu = fix_content_types(work)
    if ng or nu:
        print(f'  結構修正：[Content_Types] 移除 {ng} 個不存在的 part 宣告、{nu} 個未使用的副檔名')

    nid, nsl = fix_shape_ids(work)
    if nid:
        print(f'  結構修正：{nsl} 個部分有重複的 shape id，重新編號 {nid} 個')

    nax = fix_charts(work)
    if nax:
        print(f'  圖表修正：移除 {nax} 個懸空的座標軸 id（PowerPoint 要求修復的元凶）')

    repack(path, path, work)
    shutil.rmtree(work)
    print(f'  字型修正：{total} 個字元，橫跨 {touched} 個 XML')
    if allsamples:
        print('  改用 Cambria Math 的字元：' + ' '.join(sorted(allsamples)))
    z = zipfile.ZipFile(path)
    dirs = len([i for i in z.infolist() if i.filename.endswith('/')])
    print(f'  重新打包：第一個項目 = {z.infolist()[0].filename}，資料夾項目 {dirs} 個，'
          f'檔案 {os.path.getsize(path)/1024:.0f} KB')


if __name__ == '__main__':
    main(sys.argv[1])
