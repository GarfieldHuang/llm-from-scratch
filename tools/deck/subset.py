"""從 pptx 取出指定的投影片，產生一個結構完整的新檔。"""
import os, re, shutil, sys, tempfile, zipfile
from lxml import etree
P='http://schemas.openxmlformats.org/presentationml/2006/main'
R='http://schemas.openxmlformats.org/package/2006/relationships'
RID='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
CT='http://schemas.openxmlformats.org/package/2006/content-types'

def main(src, dst, keep):
    work=tempfile.mkdtemp()
    with zipfile.ZipFile(src) as z:
        for i in z.infolist():
            if i.filename.endswith('/'): continue
            t=os.path.join(work,i.filename); os.makedirs(os.path.dirname(t),exist_ok=True)
            open(t,'wb').write(z.read(i.filename))

    # presentation.xml：找出要保留 / 刪除的 slide
    pf=os.path.join(work,'ppt','presentation.xml')
    ptree=etree.parse(pf); lst=ptree.getroot().find(f'{{{P}}}sldIdLst')
    rf=os.path.join(work,'ppt','_rels','presentation.xml.rels')
    rtree=etree.parse(rf)
    rid2tgt={e.get('Id'):e.get('Target') for e in rtree.getroot()}
    drop_rids, drop_files = [], []
    for i, e in enumerate(list(lst), start=1):
        if i in keep: continue
        rid=e.get(RID); lst.remove(e); drop_rids.append(rid)
        drop_files.append(os.path.normpath(os.path.join(work,'ppt',rid2tgt[rid])))
    for e in list(rtree.getroot()):
        if e.get('Id') in drop_rids: rtree.getroot().remove(e)
    ptree.write(pf, xml_declaration=True, encoding='UTF-8', standalone=True)
    rtree.write(rf, xml_declaration=True, encoding='UTF-8', standalone=True)

    # 刪除投影片本體、其 rels，以及它獨佔的 notesSlide
    for sf in drop_files:
        srel=os.path.join(os.path.dirname(sf),'_rels',os.path.basename(sf)+'.rels')
        if os.path.exists(srel):
            for e in etree.parse(srel).getroot():
                t=e.get('Target') or ''
                if 'notesSlide' in t:
                    nf=os.path.normpath(os.path.join(os.path.dirname(sf),t))
                    for p_ in (nf, os.path.join(os.path.dirname(nf),'_rels',os.path.basename(nf)+'.rels')):
                        if os.path.exists(p_): os.remove(p_)
            os.remove(srel)
        if os.path.exists(sf): os.remove(sf)

    # Content_Types：移除已不存在的 part
    ctf=os.path.join(work,'[Content_Types].xml')
    ctree=etree.parse(ctf); present=set()
    for r_,_,fs in os.walk(work):
        for fn in fs: present.add('/'+os.path.relpath(os.path.join(r_,fn),work).replace(os.sep,'/'))
    for e in list(ctree.getroot()):
        if etree.QName(e).localname=='Override' and e.get('PartName') not in present:
            ctree.getroot().remove(e)
    ctree.write(ctf, xml_declaration=True, encoding='UTF-8', standalone=True)

    names=[]
    for r_,_,fs in os.walk(work):
        for fn in fs: names.append(os.path.relpath(os.path.join(r_,fn),work))
    names.sort(key=lambda n:(n!='[Content_Types].xml',n))
    with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n in names: z.write(os.path.join(work,n), n.replace(os.sep,'/'))
    shutil.rmtree(work)
    print(f'  {os.path.basename(dst)}：保留 {len(keep)} 張')

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2], set(int(x) for x in sys.argv[3].split(',')))
