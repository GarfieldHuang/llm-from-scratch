// ── 設計系統 ───────────────────────────────────────────────────────
const C = {
  INK:    '141A21',   // 深墨：封面 / 章節頁背景
  INK2:   '243140',   // 次深
  PAPER:  'FFFFFF',
  TXT:    '1B2430',
  MUTED:  '626D7A',
  RULE:   'DCE1E7',
  WASH:   'F4F6F8',   // 極淡底
  AMBER:  'E09B2D',   // 主色：前向 / 重點
  AMBER_L:'FAEFDA',
  AMBER_D:'A86F14',
  TEAL:   '1C7293',   // 次色：反向 / 對照
  TEAL_L: 'E1EDF3',
  RED:    'B23A32',   // 誤解 / 危險
  RED_L:  'F9E7E5',
  GREEN:  '3F7A52',   // 驗證通過
  GREEN_L:'E5F0E9',
};

const F = { TITLE: 'Calibri', BODY: 'Calibri', MATH: 'Cambria', CODE: 'Courier New' };
const SW = 13.333, SH = 7.5, M = 0.62, CW = SW - 2 * M;

// ── 公式排版：把 "E_{v}" / "r^{2}" 拆成原生 subscript / superscript run ──
function math(str, base) {
  const lines = String(str).split('\n');
  const out = [];
  lines.forEach((line, li) => {
    const runs = [];
    let buf = '', i = 0;
    while (i < line.length) {
      if ((line[i] === '_' || line[i] === '^') && line[i + 1] === '{') {
        const close = line.indexOf('}', i + 2);
        if (close > -1) {
          if (buf) { runs.push({ text: buf, options: { ...base } }); buf = ''; }
          const key = line[i] === '_' ? 'subscript' : 'superscript';
          runs.push({ text: line.slice(i + 2, close), options: { ...base, [key]: true } });
          i = close + 1; continue;
        }
      }
      buf += line[i]; i++;
    }
    if (buf) runs.push({ text: buf, options: { ...base } });
    if (runs.length === 0) runs.push({ text: ' ', options: { ...base } });
    if (li < lines.length - 1) runs[runs.length - 1].options.breakLine = true;
    out.push(...runs);
  });
  return out;
}

// ── 版面元件 ───────────────────────────────────────────────────────
function contentSlide(p, sec, title, kicker) {
  const s = p.addSlide();
  s.background = { color: C.PAPER };
  s.addShape(p.ShapeType.roundRect, {
    x: M, y: 0.42, w: 0.46, h: 0.46, rectRadius: 0.09,
    fill: { color: C.AMBER }, line: { type: 'none' },
  });
  s.addText(String(sec), {
    x: M, y: 0.42, w: 0.46, h: 0.46, align: 'center', valign: 'middle',
    fontSize: 17, bold: true, color: C.INK, fontFace: F.MATH, isTextBox: true, margin: 0,
  });
  s.addText(title, {
    x: M + 0.66, y: kicker ? 0.36 : 0.42, w: CW - 0.66, h: 0.56,
    fontSize: 30, bold: true, color: C.TXT, fontFace: F.TITLE,
    isTextBox: true, margin: 0, valign: 'middle',
  });
  if (kicker) s.addText(kicker, {
    x: M + 0.66, y: 0.94, w: CW - 0.66, h: 0.34,
    fontSize: 13.5, color: C.MUTED, fontFace: F.BODY, isTextBox: true, margin: 0, valign: 'middle',
  });
  s._top = kicker ? 1.52 : 1.28;
  return s;
}

function sectionSlide(p, num, title, sub, bullets) {
  const s = p.addSlide();
  s.background = { color: C.INK };
  s.addShape(p.ShapeType.roundRect, {
    x: M + 0.1, y: 2.28, w: 1.62, h: 1.62, rectRadius: 0.16,
    fill: { color: C.AMBER }, line: { type: 'none' },
  });
  s.addText(String(num), {
    x: M + 0.1, y: 2.28, w: 1.62, h: 1.62, align: 'center', valign: 'middle',
    fontSize: 64, bold: true, color: C.INK, fontFace: F.MATH, isTextBox: true, margin: 0,
  });
  s.addText(title, {
    x: M + 2.1, y: 2.34, w: 6.4, h: 0.86,
    fontSize: 38, bold: true, color: C.PAPER, fontFace: F.TITLE, isTextBox: true, margin: 0, valign: 'middle',
  });
  s.addText(sub, {
    x: M + 2.1, y: 3.24, w: 6.4, h: 0.66,
    fontSize: 15, color: C.AMBER, fontFace: F.BODY, isTextBox: true, margin: 0, valign: 'top',
  });
  if (bullets && bullets.length) {
    s.addText(bullets.map((b, i) => ({
      text: b, options: { bullet: { code: '2022' }, breakLine: i < bullets.length - 1 },
    })), {
      x: 9.35, y: 2.34, w: 3.35, h: 2.6, fontSize: 13, color: 'A9B4C0',
      fontFace: F.BODY, isTextBox: true, margin: 0, paraSpaceAfter: 8, valign: 'top',
    });
  }
  return s;
}

// 圓角卡片
function card(s, p, x, y, w, h, fill, opts) {
  opts = opts || {};
  s.addShape(p.ShapeType.roundRect, {
    x, y, w, h, rectRadius: opts.radius || 0.07,
    fill: { color: fill },
    line: opts.line ? { color: opts.line, width: opts.lineW || 1 } : { type: 'none' },
    ...(opts.shadow ? { shadow: { type: 'outer', color: '9AA5B1', blur: 8, offset: 2, angle: 90, opacity: 0.28 } } : {}),
  });
}

// 公式面板
function formula(s, p, x, y, w, h, str, o) {
  o = o || {};
  card(s, p, x, y, w, h, o.bg || C.AMBER_L, { radius: 0.07 });
  s.addText(math(str, { fontSize: o.size || 21, color: o.color || C.TXT, fontFace: F.MATH, bold: o.bold !== false }), {
    x: x + 0.14, y, w: w - 0.28, h, align: o.align || 'center', valign: 'middle',
    isTextBox: true, margin: 0, lineSpacing: o.lineSpacing || undefined,
  });
}

// 純文字（省去每次重打 isTextBox / margin）
function txt(s, str, o) {
  s.addText(str, { fontFace: F.BODY, color: C.TXT, isTextBox: true, margin: 0, ...o });
}
function rich(s, runs, o) {
  s.addText(runs, { fontFace: F.BODY, color: C.TXT, isTextBox: true, margin: 0, ...o });
}

// 流程方塊
function box(s, p, x, y, w, h, label, o) {
  o = o || {};
  card(s, p, x, y, w, h, o.fill || C.WASH, { line: o.line, lineW: o.lineW, radius: o.radius || 0.06 });
  s.addText(math(label, { fontSize: o.size || 11.5, color: o.color || C.TXT, fontFace: o.face || F.BODY, bold: o.bold !== false }), {
    x: x + 0.04, y, w: w - 0.08, h, align: 'center', valign: 'middle', isTextBox: true, margin: 0, lineSpacing: o.ls || undefined,
  });
}

// 箭頭
function arrow(s, p, x, y, w, h, o) {
  o = o || {};
  s.addShape(p.ShapeType.line, {
    x, y, w, h,
    line: { color: o.color || C.MUTED, width: o.width || 1.5, endArrowType: o.head === false ? 'none' : 'triangle', ...(o.dash ? { dashType: 'dash' } : {}) },
    ...(o.flipH ? { flipH: true } : {}), ...(o.flipV ? { flipV: true } : {}),
  });
}

// 大數字
function stat(s, big, label, o) {
  o = o || {};
  s.addText(big, {
    x: o.x, y: o.y, w: o.w, h: o.bh || 0.78, align: o.align || 'left', valign: 'middle',
    fontSize: o.size || 42, bold: true, color: o.color || C.AMBER_D, fontFace: F.MATH, isTextBox: true, margin: 0,
  });
  s.addText(label, {
    x: o.x, y: o.y + (o.bh || 0.78) - 0.04, w: o.w, h: o.lh || 0.4, align: o.align || 'left', valign: 'top',
    fontSize: o.lsize || 11.5, color: C.MUTED, fontFace: F.BODY, isTextBox: true, margin: 0,
  });
}

// 表格（統一樣式）
function table(s, rows, o) {
  s.addTable(rows, {
    x: o.x, y: o.y, w: o.w, colW: o.colW,
    border: { type: 'solid', pt: 0.75, color: C.RULE },
    fontFace: F.BODY, fontSize: o.fontSize || 12, color: C.TXT,
    valign: 'middle', rowH: o.rowH || 0.3, autoPage: false,
  });
}
function hdr(t) { return { text: t, options: { bold: true, color: C.PAPER, fill: { color: C.INK2 }, fontSize: 11.5 } }; }
function cell(t, o) { return { text: t, options: { ...(o || {}) } }; }

module.exports = { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell };
