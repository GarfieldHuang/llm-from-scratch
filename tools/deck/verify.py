"""交付前驗證：懸空軸 id、字型缺字、封裝結構。"""
import glob, logging, os, re, sys, warnings, zipfile
from lxml import etree
from fontTools.ttLib import TTFont, TTCollection
warnings.filterwarnings('ignore'); logging.getLogger('fontTools').setLevel(logging.ERROR)
A='http://schemas.openxmlformats.org/drawingml/2006/main'; NS={'a':A}
EA=((0x2E80,0x9FFF),(0x3000,0x303F),(0xAC00,0xD7AF),(0x3040,0x30FF),(0xF900,0xFAFF),(0xFF00,0xFFEF))
def latin(ch):
    o=ord(ch)
    return o<0x80 or not any(lo<=o<=hi for lo,hi in EA)
dirs=['/System/Library/Fonts/Supplemental','/Library/Fonts','/Library/Fonts/Microsoft',
      os.path.expanduser('~/Library/Fonts'),'/System/Library/Fonts']
dirs+=glob.glob('/Applications/Microsoft *.app/Contents/Resources/DFonts')
dirs+=glob.glob('/Volumes/*/Applications/Microsoft *.app/Contents/Resources/DFonts')
paths=[]
for d in dirs:
    for e in ('*.ttf','*.ttc','*.otf'): paths+=glob.glob(os.path.join(d,e))
DB={}
for p in sorted(set(paths)):
    try: fs=TTCollection(p).fonts if p.endswith('.ttc') else [TTFont(p,fontNumber=0)]
    except Exception: continue
    for f in fs:
        try:
            n=f['name'].getDebugName(4) or ''; cm=set()
            for t in f['cmap'].tables: cm|=set(t.cmap.keys())
        except Exception: continue
        if n and n not in DB: DB[n]=cm

path=sys.argv[1]; z=zipfile.ZipFile(path); fails=[]
# ① 封裝
if z.infolist()[0].filename!='[Content_Types].xml': fails.append('[Content_Types].xml 不是第一個項目')
nd=len([i for i in z.infolist() if i.filename.endswith('/')])
if nd: fails.append(f'含 {nd} 個資料夾項目')
ns=len([i for i in z.infolist() if i.compress_type==0])
if ns: fails.append(f'{ns} 個項目未壓縮')
# ② 圖表軸
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
for n in [x for x in z.namelist() if re.match(r'ppt/charts/chart\d+\.xml$',x)]:
    r=etree.fromstring(z.read(n)); decl=set()
    for ax in ('catAx','valAx','serAx','dateAx'):
        for el in r.iter(f'{{{C}}}{ax}'):
            a=el.find(f'{{{C}}}axId')
            if a is not None: decl.add(a.get('val'))
    for el in r.iter():
        if not el.tag.endswith('Chart'): continue
        for a in el.findall(f'{{{C}}}axId'):
            if a.get('val') not in decl: fails.append(f'{n}: 懸空軸 id {a.get("val")}')
# ③ 字型缺字
gaps={}
for n in z.namelist():
    if not re.match(r'ppt/(slides|notesSlides)/\w+\d+\.xml$',n): continue
    r=etree.fromstring(z.read(n))
    for run in r.iter(f'{{{A}}}r'):
        t=run.find('a:t',NS)
        if t is None or not t.text: continue
        rPr=run.find('a:rPr',NS); face='Calibri'; bold=False
        if rPr is not None:
            el=rPr.find('a:latin',NS)
            if el is not None and el.get('typeface'): face=el.get('typeface')
            bold=rPr.get('b')=='1'
        if bold and f'{face} Bold' in DB: face=f'{face} Bold'
        cm=DB.get(face)
        if cm is None: continue
        for ch in t.text:
            if latin(ch) and ord(ch) not in cm: gaps.setdefault(ch,set()).add(face)
if gaps:
    for ch,fs in sorted(gaps.items()): fails.append(f'字元 {ch!r} 在 {sorted(fs)} 裡缺字')
# ③0 XML 裡不得出現 NaN / Infinity / undefined 這類非數值
_junk=0; _where=None
for n in z.namelist():
    if not n.endswith('.xml'): continue
    for el in etree.fromstring(z.read(n)).iter():
        for k, val in el.attrib.items():
            if any(t in val for t in ('NaN','Infinity','undefined')):
                _junk+=1
                if _where is None:
                    _where=f'{n} 的 {etree.QName(el).localname}@{k.rsplit("}",1)[-1]}="{val}"'
if _junk: fails.append(f'{_junk} 個屬性值含 NaN / Infinity / undefined（{_where}）')

# ③a 每個 <p:bgPr> 都必須有 <a:effectLst>（schema 必要元素）
PNS='{http://schemas.openxmlformats.org/presentationml/2006/main}'
_nobg=0
for n in z.namelist():
    if not re.match(r'ppt/(slides|slideLayouts|slideMasters|notesSlides|notesMasters)/\w+\d+\.xml$', n): continue
    for bgPr in etree.fromstring(z.read(n)).iter(PNS+'bgPr'):
        if bgPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst') is None:
            _nobg+=1
if _nobg: fails.append(f'{_nobg} 個 <p:bgPr> 缺少必要的 <a:effectLst>')

# ③b Content_Types 不可宣告不存在的 part
CTNS='{http://schemas.openxmlformats.org/package/2006/content-types}'
_names={'/'+n for n in z.namelist()}
_ct=etree.fromstring(z.read('[Content_Types].xml'))
_ghost=[e.get('PartName') for e in _ct if e.tag==CTNS+'Override' and e.get('PartName') not in _names]
if _ghost:
    fails.append(f'[Content_Types] 宣告了 {len(_ghost)} 個不存在的 part（例：{_ghost[0]}）')

# ④ presentation.xml 子元素順序（CT_Presentation 是嚴格序列）
P='{http://schemas.openxmlformats.org/presentationml/2006/main}'
pres=etree.fromstring(z.read('ppt/presentation.xml'))
order=[etree.QName(c).localname for c in pres]
if 'notesMasterIdLst' in order and 'sldIdLst' in order:
    if order.index('notesMasterIdLst') > order.index('sldIdLst'):
        fails.append('presentation.xml: notesMasterIdLst 必須排在 sldIdLst 之前')

# ⑤ slideMaster 與 notesMaster 不可共用 theme
TH='http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme'
def theme_target(part):
    if part not in z.namelist(): return None
    for e in etree.fromstring(z.read(part)):
        if e.get('Type')==TH: return e.get('Target')
    return None
t1=theme_target('ppt/slideMasters/_rels/slideMaster1.xml.rels')
t2=theme_target('ppt/notesMasters/_rels/notesMaster1.xml.rels')
if t1 and t2 and t1==t2:
    fails.append(f'slideMaster 與 notesMaster 共用同一個 theme ({t1})')

# ⑥ 同一部分內 shape id 必須唯一
import collections, posixpath
for n in z.namelist():
    if not re.match(r'ppt/(slides|slideLayouts|slideMasters|notesSlides|notesMasters)/\w+\d+\.xml$', n):
        continue
    ids=[e.get('id') for e in etree.fromstring(z.read(n)).iter() if e.tag.endswith('}cNvPr') and e.get('id')]
    dup=[i for i,c in collections.Counter(ids).items() if c>1]
    if dup: fails.append(f'{n}: 重複的 shape id {dup[:4]}')

# ⑦ r:id 參照完整性、rels 的 Id 唯一與 Target 存在
RID='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
names=set(z.namelist())
for n in z.namelist():
    if not n.endswith('.xml') or n=='[Content_Types].xml': continue
    d,b=n.rsplit('/',1); rp=f'{d}/_rels/{b}.rels'
    have={e.get('Id') for e in etree.fromstring(z.read(rp))} if rp in names else set()
    used={el.get(RID) for el in etree.fromstring(z.read(n)).iter() if el.get(RID)}
    miss=used-have
    if miss: fails.append(f'{n}: 參照了不存在的關聯 {sorted(miss)}')
for n in [x for x in z.namelist() if x.endswith('.rels')]:
    root=etree.fromstring(z.read(n))
    ids=[e.get('Id') for e in root]
    dup=[i for i,c in collections.Counter(ids).items() if c>1]
    if dup: fails.append(f'{n}: 重複的 Id {dup}')
    base=posixpath.dirname(posixpath.dirname(n))
    for e in root:
        if e.get('TargetMode')=='External': continue
        tgt=e.get('Target')
        full = tgt.lstrip('/') if tgt.startswith('/') else posixpath.normpath(posixpath.join(base, tgt))
        if full not in names: fails.append(f'{n}: Target 不存在 {tgt}')

print(f'投影片 {len([n for n in z.namelist() if re.match(r"ppt/slides/slide\d+.xml$",n)])} 張、'
      f'{os.path.getsize(path)/1024:.0f} KB')
if fails:
    print('發現問題：')
    for f in fails: print('  ✕', f)
    sys.exit(1)
print('封裝、圖表軸、字型、元素順序、theme、shape id、關聯完整性 —— 全部通過')
