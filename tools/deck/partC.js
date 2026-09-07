const L = require('./lib.js');
const { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell } = L;

module.exports = function partC(p) {
  let s;

  // ══ 章節 4 ═══════════════════════════════════════════════════
  sectionSlide(p, 4, '模型的骨架', '一個 Block 長什麼樣，以及為什麼那兩個「＋」是命脈',
    ['Norm → Attention → Norm → FFN', '殘差：梯度的直達路徑', 'Attention 橫向、FFN 縱向', '資料的形狀怎麼變']);

  // ══ 一個 Block 的內部 ════════════════════════════════════════
  s = contentSlide(p, 4, '一個 Block 的內部', '順序是 Norm → Attention → Norm → FFN。不是 FFN 先。');
  box(s, p, 0.62, 2.60, 0.80, 0.85, 'x', { fill: C.WASH, line: C.RULE, size: 16, face: F.MATH });
  arrow(s, p, 1.42, 3.02, 0.33, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 1.75, 2.60, 1.35, 0.85, 'RMSNorm', { fill: C.WASH, line: C.RULE, size: 11.5 });
  arrow(s, p, 3.10, 3.02, 0.35, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 3.45, 2.60, 2.20, 0.85, 'Attention\n橫向：位置之間交換資訊', { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 11, ls: 14 });
  arrow(s, p, 5.65, 3.02, 0.35, 0, { color: C.MUTED, width: 1.4 });
  s.addShape(p.ShapeType.ellipse, { x: 6.00, y: 2.77, w: 0.5, h: 0.5, fill: { color: C.AMBER }, line: { type: 'none' } });
  txt(s, '＋', { x: 6.00, y: 2.77, w: 0.5, h: 0.5, fontSize: 17, bold: true, color: C.INK, align: 'center', valign: 'middle' });
  arrow(s, p, 6.50, 3.02, 0.35, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 6.85, 2.60, 1.35, 0.85, 'RMSNorm', { fill: C.WASH, line: C.RULE, size: 11.5 });
  arrow(s, p, 8.20, 3.02, 0.35, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 8.55, 2.60, 2.20, 0.85, 'SwiGLU FFN\n縱向：每個位置各自加工', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 11, ls: 14 });
  arrow(s, p, 10.75, 3.02, 0.35, 0, { color: C.MUTED, width: 1.4 });
  s.addShape(p.ShapeType.ellipse, { x: 11.10, y: 2.77, w: 0.5, h: 0.5, fill: { color: C.AMBER }, line: { type: 'none' } });
  txt(s, '＋', { x: 11.10, y: 2.77, w: 0.5, h: 0.5, fontSize: 17, bold: true, color: C.INK, align: 'center', valign: 'middle' });
  arrow(s, p, 11.60, 3.02, 0.35, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 11.95, 2.60, 0.78, 0.85, 'y', { fill: C.WASH, line: C.RULE, size: 16, face: F.MATH });
  // 殘差 1（上方）
  arrow(s, p, 1.02, 2.08, 0, 0.52, { color: C.AMBER_D, width: 1.5, flipV: true, head: false });
  arrow(s, p, 1.02, 2.08, 5.23, 0, { color: C.AMBER_D, width: 1.5, head: false });
  arrow(s, p, 6.25, 2.08, 0, 0.69, { color: C.AMBER_D, width: 1.5 });
  txt(s, '殘差：直達路徑', { x: 2.6, y: 1.74, w: 2.6, h: 0.3, fontSize: 11.5, bold: true, color: C.AMBER_D });
  // 殘差 2（下方）
  arrow(s, p, 6.25, 3.45, 0, 0.55, { color: C.AMBER_D, width: 1.5, head: false });
  arrow(s, p, 6.25, 4.00, 5.10, 0, { color: C.AMBER_D, width: 1.5, head: false });
  arrow(s, p, 11.35, 3.45, 0, 0.55, { color: C.AMBER_D, width: 1.5, flipV: true });
  txt(s, '殘差：直達路徑', { x: 7.9, y: 4.04, w: 2.6, h: 0.3, fontSize: 11.5, bold: true, color: C.AMBER_D });

  formula(s, p, M, 4.56, CW, 0.98, 'h = x + Attention( RMSNorm(x) )          y = h + FFN( RMSNorm(h) )', { size: 18 });

  const b3 = [
    ['Attention 橫向', '唯一讓不同位置互通的地方', C.TEAL, C.TEAL_L],
    ['FFN 縱向', '對每個位置各做各的，位置之間完全不互動', C.AMBER_D, C.AMBER_L],
    ['形狀不變', '進去 (B,T,d)、出來 (B,T,d) —— 所以可以一直疊', C.MUTED, C.WASH],
  ];
  b3.forEach((b, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 5.72, 3.85, 1.0, b[3], { radius: 0.08 });
    txt(s, b[0], { x: x + 0.24, y: 5.84, w: 3.4, h: 0.3, fontSize: 13.5, bold: true, color: b[2], valign: 'middle' });
    txt(s, b[1], { x: x + 0.24, y: 6.14, w: 3.4, h: 0.5, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 15 });
  });
  s.addNotes('三件事：(1) 順序是 Norm 先，不是 FFN 先 (2) 那兩個 + 是殘差，下一張細講 (3) Attention 橫向、FFN 縱向——這個分工是理解 Transformer 的關鍵句。');

  // ══ 殘差 ═════════════════════════════════════════════════════
  s = contentSlide(p, 4, '那兩個「＋」為什麼是命脈', '沒有殘差的話，20 層以上基本訓不動');
  formula(s, p, M, 1.62, 5.9, 0.78, '前向：  y = x + f(x)', { size: 19, bg: C.WASH });
  formula(s, p, 6.82, 1.62, 5.91, 0.78, '反向：  ∂L/∂x = ∂L/∂y + f′( ∂L/∂y )', { size: 17, bg: C.AMBER_L });

  card(s, p, M, 2.68, CW, 1.9, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, '關鍵在那個孤零零的 ∂L/∂y', { x: M + 0.34, y: 2.84, w: 5, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  // 兩條路
  box(s, p, M + 0.34, 3.3, 2.5, 0.5, '上游梯度 ∂L/∂y', { fill: C.WASH, line: C.RULE, size: 11.5 });
  arrow(s, p, M + 2.94, 3.42, 1.3, 0, { color: C.AMBER_D, width: 2 });
  txt(s, '直達，完全不衰減', { x: M + 2.8, y: 3.12, w: 1.7, h: 0.26, fontSize: 9.5, color: C.AMBER_D, align: 'center' });
  arrow(s, p, M + 2.94, 3.68, 1.3, 0.42, { color: C.TEAL, width: 1.3 });
  txt(s, '穿過 f 的那條，會衰減', { x: M + 2.7, y: 4.14, w: 2.0, h: 0.26, fontSize: 9.5, color: C.TEAL, align: 'center' });
  box(s, p, M + 4.4, 3.3, 2.3, 0.5, '傳給下一層', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 11.5 });
  txt(s, '就算 f 那條路把梯度乘到剩 0.5 倍，直達路徑仍然原封不動地把 1 倍送下去。\n疊 N 層之後，沒有殘差是 0.5ᴺ（指數衰減），有殘差是 1 + 一點修正。',
    { x: M + 7.0, y: 3.24, w: 5.0, h: 1.1, fontSize: 12.5, color: C.TXT, valign: 'top', lineSpacing: 19 });

  txt(s, '一個通則，整份程式碼裡反覆出現', { x: M, y: 4.78, w: 8, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 5.2, CW, 1.5, C.INK, { radius: 0.1 });
  rich(s, [
    { text: '前向分岔  ⟹  反向相加', options: { bold: true, color: C.AMBER, fontSize: 24, breakLine: true } },
    { text: '一個張量被用了 k 次，它的梯度就是那 k 條路的總和。殘差是 k = 2；SwiGLU 裡 x 同時餵給 gate 和 up，也是 k = 2；Attention 裡 x 餵給 Q、K、V 三路，k = 3。', options: { color: 'C3CCD5', fontSize: 13.5 } },
  ], { x: M + 0.4, y: 5.2, w: CW - 0.8, h: 1.5, valign: 'middle', lineSpacing: 26 });
  s.addNotes('「前向分岔，反向相加」是這場最好用的一句口訣。之後每次遇到分岔（殘差、QKV、SwiGLU 雙路）都可以回頭指這句話，聽眾就不用背個別公式了。');

  // ══ 形狀與參數 ═══════════════════════════════════════════════
  s = contentSlide(p, 4, '資料的形狀怎麼變、參數住在哪裡', 'B = 64、T = 128、d = 256、|V| = 6,400');
  table(s, [
    [hdr('階段'), hdr('形狀'), hdr('說明')],
    [cell('token ids'), cell('(64, 128)', { fontFace: F.CODE }), cell('整數')],
    [cell('查表後'), cell('(64, 128, 256)', { fontFace: F.CODE }), cell('每個位置一個 256 維向量')],
    [cell('每個 Block 之後'), cell('(64, 128, 256)', { fontFace: F.CODE, bold: true }), cell('形狀不變，只是內容被加工', { bold: true, color: C.TEAL })],
    [cell('logits', { fill: { color: C.RED_L } }), cell('(64, 128, 6400)', { fontFace: F.CODE, bold: true, fill: { color: C.RED_L } }), cell('每個位置對整個詞表的分數', { fill: { color: C.RED_L } })],
    [cell('loss'), cell('( )', { fontFace: F.CODE }), cell('一個純量')],
  ], { x: M, y: 1.62, w: 6.6, colW: [2.2, 2.0, 2.4], rowH: 0.42, fontSize: 12.5 });

  card(s, p, 7.42, 1.62, 5.31, 2.52, C.RED_L, { radius: 0.1 });
  txt(s, 'logits 是訓練時最大的中間張量', { x: 7.72, y: 1.8, w: 4.7, h: 0.34, fontSize: 15, bold: true, color: C.RED, valign: 'middle' });
  stat(s, '5.24 × 10⁷', '個數字　（64 × 128 × 6,400）', { x: 7.72, y: 2.3, w: 4.7, size: 34, bh: 0.66, color: C.RED, lsize: 12 });
  txt(s, '記憶體大戶。而它的大小由詞表決定 —— 這就是為什麼「詞表大小是架構參數」。',
    { x: 7.72, y: 3.34, w: 4.7, h: 0.66, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  txt(s, '參數住在哪裡（本專案實際輸出）', { x: M, y: 4.4, w: 8, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 4.82, 6.2, 1.9, C.INK, { radius: 0.1 });
  txt(s, '  embedding                           1,638,400\n  block0 … block3         各    778,752\n  final_norm                                256\n  lm_head（與 embedding 共用）              0\n  ─────────────────────────────\n  總計                                  4,753,664',
    { x: M + 0.3, y: 4.94, w: 5.7, h: 1.7, fontSize: 11, color: 'C3CCD5', fontFace: F.CODE, valign: 'top', lineSpacing: 17 });

  card(s, p, 7.12, 4.82, 5.61, 1.9, C.WASH, { radius: 0.1 });
  txt(s, 'embedding 佔了 34.5%？', { x: 7.42, y: 4.96, w: 5.0, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  formula(s, p, 7.42, 5.34, 5.0, 0.56, '|V| · d      vs      O(d²)', { size: 16, bg: C.PAPER, color: C.TEAL });
  txt(s, '小模型的 |V| ≫ d，所以 embedding 佔比很誇張。真實大模型的 d 大得多，佔比會降到 5% 以下。',
    { x: 7.42, y: 6.0, w: 5.0, h: 0.6, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 16 });
  s.addNotes('「形狀不變」那一列要強調：Block 是可堆疊的，因為它保持形狀。這跟 CNN 每層改變形狀很不一樣。另外 lm_head 是 0，因為 weight tying——它跟 embedding 共用同一個張量。');

  // ══ 章節 5 ═══════════════════════════════════════════════════
  sectionSlide(p, 5, '三個零件拆開看', '每一個都拆到「可以自己寫出來」的程度',
    ['RMSNorm —— 參數最少，影響最大', 'Attention —— 唯一橫向交換資訊的地方', 'SwiGLU FFN —— 參數量佔三分之二', '每一層都有實測數值可以驗算']);

  // ══ RMSNorm 前向 ═════════════════════════════════════════════
  s = contentSlide(p, 5, 'RMSNorm：三個步驟，三個角色', '每一層的入口都會先過它。參數量最少，但它決定了進入 Attention 和 FFN 的數值尺度。');
  box(s, p, M, 2.0, 1.5, 0.7, 'x\n每個位置一個向量', { fill: C.WASH, line: C.RULE, size: 10.5, ls: 13 });
  arrow(s, p, 2.12, 2.35, 0.3, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 2.42, 2.0, 1.7, 0.7, '平方後取平均\nmean(x²)', { fill: C.WASH, line: C.RULE, size: 10.5, ls: 13 });
  arrow(s, p, 4.12, 2.35, 0.3, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 4.42, 2.0, 1.7, 0.7, '開根號\nr = √( · + ε )', { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 10.5, ls: 13 });
  arrow(s, p, 6.12, 2.35, 0.3, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 6.42, 2.0, 1.9, 0.7, 'x̂ = x / r\n長度被固定住', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 10.5, ls: 13 });
  arrow(s, p, 8.32, 2.35, 0.3, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 8.62, 2.0, 1.5, 0.7, '⊙　g\n（可學）', { fill: C.GREEN_L, line: C.GREEN, color: C.GREEN, size: 10.5, ls: 13 });
  arrow(s, p, 10.12, 2.35, 0.3, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 10.42, 2.0, 1.0, 0.7, 'y', { fill: C.WASH, line: C.RULE, size: 15, face: F.MATH });

  formula(s, p, M, 3.0, CW, 0.8, 'r = √( (1/n) Σ_{k} x_k² + ε )          x̂ = x / r          y = g ⊙ x̂', { size: 18 });

  table(s, [
    [hdr('符號'), hdr('是什麼'), hdr('可學嗎')],
    [cell('r', { fontFace: F.MATH, bold: true }), cell('這一列數字的均方根（root-mean-square）'), cell('否，算出來的')],
    [cell('x̂', { fontFace: F.MATH, bold: true }), cell('正規化後的向量，長度被拉到約 √n'), cell('否')],
    [cell('g', { fontFace: F.MATH, bold: true, color: C.GREEN }), cell('每個維度一個的縮放係數', { bold: true }), cell('是，初始值全為 1', { bold: true, color: C.GREEN })],
    [cell('ε', { fontFace: F.MATH, bold: true }), cell('防止除以零'), cell('否，固定 10⁻⁵')],
  ], { x: M, y: 4.0, w: 6.6, colW: [1.0, 4.0, 1.6], rowH: 0.4, fontSize: 12 });

  card(s, p, 7.42, 4.0, 5.31, 2.0, C.WASH, { radius: 0.1 });
  txt(s, '跟 LayerNorm 差在哪', { x: 7.72, y: 4.14, w: 4.7, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  rich(s, math('LayerNorm:  y = g ⊙ (x − μ)/√(σ²+ε) + b\nRMSNorm:    y = g ⊙ x/√(mean(x²)+ε)', { fontSize: 11.5, fontFace: F.MATH, color: C.TXT, bold: true }),
    { x: 7.72, y: 4.5, w: 4.7, h: 0.72, valign: 'top', lineSpacing: 19 });
  txt(s, 'RMSNorm 少了兩樣：① 不減平均數 μ　② 沒有偏置 b。少一次減法、少一組參數，實測效果幾乎一樣 —— 所以 LLaMA / Mistral / Qwen 現在幾乎都用它。',
    { x: 7.72, y: 5.24, w: 4.7, h: 0.68, fontSize: 11, color: C.TXT, valign: 'top', lineSpacing: 14 });
  txt(s, '沿最後一維（特徵維度）做，所以每個位置各自正規化，位置與位置之間互不影響。',
    { x: M, y: 6.14, w: 6.6, h: 0.34, fontSize: 11.5, color: C.MUTED, valign: 'middle' });
  s.addNotes('RMSNorm 是最容易被跳過的一層，但它是後面「梯度為什麼會被放大」的原因。下一張用真實數字看它做了什麼。');

  // ══ RMSNorm 實測 ═════════════════════════════════════════════
  s = contentSlide(p, 5, 'RMSNorm 實際上做了什麼：放大 50 倍', '從 traces/sample/train_step_0001.txt 抓的第一層 norm1');
  table(s, [
    [hdr(''), hdr('前 4 個維度')],
    [cell('x（輸入）'), cell('+0.01647    −0.04764    +0.02504    +0.00472', { fontFace: F.CODE })],
    [cell('r', { fontFace: F.MATH, bold: true, fill: { color: C.TEAL_L } }), cell('0.019711', { fontFace: F.CODE, bold: true, fill: { color: C.TEAL_L } })],
    [cell('x̂（正規化後）', { fill: { color: C.AMBER_L } }), cell('+0.83548    −2.41710    +1.27034    +0.23929', { fontFace: F.CODE, fill: { color: C.AMBER_L } })],
    [cell('g（初始值）'), cell('+1.00000    +1.00000    +1.00000    +1.00000', { fontFace: F.CODE })],
    [cell('y（輸出）'), cell('+0.83548    −2.41710    +1.27034    +0.23929', { fontFace: F.CODE })],
  ], { x: M, y: 1.68, w: 7.4, colW: [2.0, 5.4], rowH: 0.4, fontSize: 12 });

  card(s, p, 8.32, 1.68, 4.41, 2.4, C.INK, { radius: 0.1 });
  txt(s, '放大倍率', { x: 8.62, y: 1.84, w: 3.8, h: 0.3, fontSize: 12, color: '8D99A6' });
  stat(s, '50.7 ×', '1 / r  =  1 / 0.019711', { x: 8.62, y: 2.18, w: 3.8, size: 40, bh: 0.76, color: C.AMBER, lsize: 12 });
  txt(s, 'embedding 出來的向量很小（每個元素約 0.02），RMSNorm 把它們放大了 50 倍。',
    { x: 8.62, y: 3.2, w: 3.8, h: 0.72, fontSize: 11.5, color: 'C3CCD5', valign: 'top', lineSpacing: 16 });

  card(s, p, M, 4.28, 6.2, 1.1, C.WASH, { radius: 0.09 });
  txt(s, '驗算第一個維度', { x: M + 0.3, y: 4.38, w: 3, h: 0.3, fontSize: 12, bold: true, color: C.MUTED, valign: 'middle' });
  rich(s, math('x̂_{1} = 0.01647 / 0.019711 = 0.83557  ≈  0.83548　← 對得上', { fontSize: 14, fontFace: F.MATH, bold: true, color: C.TXT }),
    { x: M + 0.3, y: 4.7, w: 5.6, h: 0.4, valign: 'middle' });

  card(s, p, 7.12, 4.28, 5.61, 1.1, C.GREEN_L, { radius: 0.09 });
  txt(s, '這不是 bug，正是它的用途', { x: 7.42, y: 4.38, w: 5.0, h: 0.3, fontSize: 12.5, bold: true, color: C.GREEN, valign: 'middle' });
  txt(s, '把不同來源、不同尺度的向量，統一到同一個工作區間。', { x: 7.42, y: 4.7, w: 5.0, h: 0.4, fontSize: 12, color: C.TXT, valign: 'middle' });

  card(s, p, M, 5.56, CW, 1.14, C.RED_L, { radius: 0.1 });
  rich(s, [
    { text: '副作用：反向的時候梯度也跟著放大。', options: { bold: true, color: C.RED, fontSize: 16, breakLine: true } },
    { text: '前向除以 r，反向經過同一個除法時也會乘回來 —— 這就是為什麼梯度從第 1 層傳到 embedding 時會突然大一個數量級。稍後在反向傳播那一段會看到實測數字。', options: { color: C.TXT, fontSize: 13 } },
  ], { x: M + 0.36, y: 5.56, w: CW - 0.72, h: 1.14, valign: 'middle', lineSpacing: 24 });
  s.addNotes('讓聽眾自己拿計算機驗算 0.01647 / 0.019711。整場最重要的態度就是「每個數字都可以自己驗算」，這一張是最容易做到的示範。');
};
