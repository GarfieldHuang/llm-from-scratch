const L = require('./lib.js');
const { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell } = L;

module.exports = function partD(p) {
  let s;

  // ══ Attention 全貌（重繪）═════════════════════════════════════
  s = contentSlide(p, 5, 'Attention 的結構：三條線，逐步匯流', '整個模型裡唯一讓不同位置互看的地方 —— FFN 是每個位置各做各的');
  const CX = (i) => 0.62 + i * 1.76, CWD = 1.52;
  const LY = [1.82, 2.48, 3.14], LH = 0.52;      // Q / K / V 三條車道
  const cy = (i) => LY[i] + LH / 2;

  // c0  x
  box(s, p, CX(0), LY[0], CWD, 1.84, 'x\n(B, T, d)', { fill: C.WASH, line: C.RULE, size: 12, ls: 15 });
  // c1  三條投影
  [['q_proj　d → d', C.TEAL], ['k_proj　d → d', C.TEAL], ['v_proj　d → d', C.AMBER_D]].forEach((r, i) => {
    arrow(s, p, CX(0) + CWD, cy(i), 0.24, 0, { color: C.MUTED, width: 1.3 });
    box(s, p, CX(1), LY[i], CWD, LH, r[0], { fill: C.PAPER, line: r[1], color: r[1], size: 10.5 });
  });
  // c2  拆 head（三條一起）
  [0, 1, 2].forEach((i) => arrow(s, p, CX(1) + CWD, cy(i), 0.24, 0, { color: C.MUTED, width: 1.3 }));
  box(s, p, CX(2), LY[0], CWD, 1.84, '拆成 h 個 head\n(B, h, T, d_head)\n\n不是複製四份，\n是把 256 維切成四段', { fill: C.WASH, line: C.RULE, size: 9.5, ls: 12 });
  // c3  RoPE（V 不轉）
  [['RoPE 旋轉 Q', C.TEAL, C.TEAL_L], ['RoPE 旋轉 K', C.TEAL, C.TEAL_L], ['V 不旋轉', C.AMBER_D, C.AMBER_L]].forEach((r, i) => {
    arrow(s, p, CX(2) + CWD, cy(i), 0.24, 0, { color: C.MUTED, width: 1.3 });
    box(s, p, CX(3), LY[i], CWD, LH, r[0], { fill: r[2], line: r[1], color: r[1], size: 10.5 });
  });
  // c4  Q,K → 注意力權重 A
  arrow(s, p, CX(3) + CWD, cy(0), 0.24, 0, { color: C.TEAL, width: 1.3 });
  arrow(s, p, CX(3) + CWD, cy(1), 0.24, 0, { color: C.TEAL, width: 1.3 });
  box(s, p, CX(4), LY[0], CWD, 1.18, 'QKᵀ / √d_head\n↓\n因果遮罩 j > i → −∞\n↓\nsoftmax（每列和 = 1）\n= A', { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 9, ls: 11 });
  // c5  A · V —— V 從下方那條車道直接過來，不與上方交叉
  arrow(s, p, CX(3) + CWD, cy(2), CX(5) - CX(3) - CWD, 0, { color: C.AMBER_D, width: 1.4 });
  arrow(s, p, CX(4) + CWD, 2.41, 0.24, 0.24, { color: C.TEAL, width: 1.3 });
  box(s, p, CX(5), 2.26, CWD, 1.26, 'out = A · V\n\n每個位置的輸出\n＝它看得到的\n位置的加權平均', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 9.5, ls: 12 });
  // c6  合併 + o_proj → y
  arrow(s, p, CX(5) + CWD, 2.89, 0.24, 0, { color: C.MUTED, width: 1.3 });
  box(s, p, CX(6), 2.26, CWD, 1.26, '合併 h 個 head\n↓\no_proj　d → d\n↓\ny  (B, T, d)', { fill: C.WASH, line: C.RULE, size: 10, ls: 12.5 });

  txt(s, 'Q 車道', { x: CX(0), y: LY[0] - 0.3, w: 1.3, h: 0.26, fontSize: 9.5, bold: true, color: C.TEAL });
  txt(s, 'V 車道　全程不碰位置資訊，最後才加入', { x: CX(1), y: LY[2] + LH + 0.04, w: 5.2, h: 0.26, fontSize: 9.5, bold: true, color: C.AMBER_D });

  txt(s, '整件事其實只有兩個階段', { x: M, y: 4.12, w: 8, h: 0.34, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  const stage2 = [
    ['階段一：算出「誰該注意誰」', 'Q 和 K 決定一張 T×T 的權重表 A。位置 i 那一列，寫的是「我該從前面哪些位置拿東西、各拿多少」。因果遮罩保證只能往左看。', C.TEAL, C.TEAL_L],
    ['階段二：照這張表去拿內容', 'V 提供「內容」，A 提供「比例」。out = A · V 就是加權平均。這也是為什麼 V 不需要位置資訊 —— 位置只影響「誰該注意誰」，不影響「拿到什麼」。', C.AMBER_D, C.AMBER_L],
  ];
  stage2.forEach((c, i) => {
    const x = M + i * 6.25;
    card(s, p, x, 4.54, 5.84, 1.5, c[3], { radius: 0.1 });
    txt(s, c[0], { x: x + 0.28, y: 4.68, w: 5.28, h: 0.32, fontSize: 14, bold: true, color: c[2], valign: 'middle' });
    txt(s, c[1], { x: x + 0.28, y: 5.02, w: 5.28, h: 0.92, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 16 });
  });
  card(s, p, M, 6.18, CW, 0.56, C.INK, { radius: 0.08 });
  rich(s, [
    { text: '一句話：', options: { bold: true, color: C.AMBER, fontSize: 13 } },
    { text: 'Attention 是「先算一張誰看誰的表，再照這張表把內容加權平均起來」。前面那一大串投影、拆頭、旋轉，都只是為了把這張表算得準。', options: { color: 'C3CCD5', fontSize: 12.5 } },
  ], { x: M + 0.34, y: 6.18, w: CW - 0.68, h: 0.56, valign: 'middle' });
  s.addNotes('重繪版：三條車道由左到右，沒有迴轉也沒有交叉線。V 那條全程走最下面，到最後一步才匯入——這個視覺本身就解釋了「V 不做 RoPE」。講的時候先講下面那兩張卡（兩個階段），再回頭指圖，聽眾會比較容易接住。');

  // ══ QKV 三角色 + 多頭 ════════════════════════════════════════
  s = contentSlide(p, 5, '同一個 x 走三條線，變成三個角色', '因為 x 分岔成三路，反向時三份梯度要相加');
  const qkv = [
    ['Q', 'Query　查詢', '「我在找什麼」', C.TEAL],
    ['K', 'Key　鍵', '「我是什麼」', C.TEAL],
    ['V', 'Value　值', '「我能提供什麼內容」', C.AMBER_D],
  ];
  qkv.forEach((q, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 1.66, 3.85, 1.34, C.PAPER, { line: C.RULE, radius: 0.1 });
    s.addShape(p.ShapeType.roundRect, { x: x + 0.26, y: 1.9, w: 0.56, h: 0.56, rectRadius: 0.1, fill: { color: q[3] }, line: { type: 'none' } });
    txt(s, q[0], { x: x + 0.26, y: 1.9, w: 0.56, h: 0.56, fontSize: 22, bold: true, color: C.PAPER, align: 'center', valign: 'middle', fontFace: F.MATH });
    txt(s, q[1], { x: x + 0.96, y: 1.9, w: 2.7, h: 0.3, fontSize: 13.5, bold: true, color: C.TXT, valign: 'middle' });
    txt(s, q[2], { x: x + 0.96, y: 2.2, w: 2.7, h: 0.28, fontSize: 12, color: q[3], valign: 'middle' });
    txt(s, i === 2 ? '不做 RoPE —— 內容不需要位置資訊' : '會被 RoPE 旋轉，帶上位置資訊', { x: x + 0.26, y: 2.56, w: 3.4, h: 0.3, fontSize: 11, color: C.MUTED, valign: 'middle' });
  });
  formula(s, p, M, 3.14, CW, 0.66, '∂L/∂x  =  ∂L/∂x |_{Q}  +  ∂L/∂x |_{K}  +  ∂L/∂x |_{V}          前向分岔 ⟹ 反向相加', { size: 16 });

  txt(s, '多頭：不是複製四份，是把 256 維切成四段', { x: M, y: 4.06, w: 8, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  box(s, p, M, 4.56, 2.1, 0.7, '一個 256 維\n的向量', { fill: C.WASH, line: C.RULE, size: 11, ls: 13 });
  ['head 0\n維度 0…63', 'head 1\n維度 64…127', 'head 2\n維度 128…191', 'head 3\n維度 192…255'].forEach((h, i) => {
    const y = 4.52 + i * 0.0;
    const x = 3.35 + i * 1.85;
    box(s, p, x, 4.56, 1.7, 0.7, h, { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 10, ls: 12 });
    arrow(s, p, x - 0.15, 4.91, 0.15, 0, { color: C.TEAL, width: 1.1 });
  });
  arrow(s, p, 2.72, 4.91, 0.48, 0, { color: C.TEAL, width: 1.4 });
  box(s, p, 10.90, 4.56, 1.83, 0.7, '各自算各自的\n注意力', { fill: C.WASH, line: C.RULE, size: 10.5, ls: 13 });

  card(s, p, M, 5.52, CW, 1.2, C.AMBER_L, { radius: 0.1 });
  rich(s, [
    { text: '參數量完全沒有增加。', options: { bold: true, color: C.AMBER_D, fontSize: 16, breakLine: true } },
    { text: '但模型可以同時關注不同類型的關係 —— 例如一個 head 管語法、一個管指代。算完再接回 256 維，交給 o_proj 混合。沒有 o_proj 的話，各 head 算完就直接拼接，彼此之間永遠不會互動。', options: { color: C.TXT, fontSize: 13 } },
  ], { x: M + 0.36, y: 5.52, w: CW - 0.72, h: 1.2, valign: 'middle', lineSpacing: 24 });
  s.addNotes('「不是複製，是切開」是最常見的誤解之一。很多人以為多頭 = 多算幾次 = 參數變多。其實是同一個 256 維被切成 4 段各 64 維，參數量一模一樣。');

  // ══ 為什麼除以 √d_head ═══════════════════════════════════════
  s = contentSlide(p, 5, '為什麼要除以 √d_head', '不除的話，softmax 會飽和，梯度就消失了');
  formula(s, p, M, 1.66, CW, 0.78, 'S = QKᵀ / √d_head          S_{ij} = Σ_{k} q_{ik} k_{jk}', { size: 19 });

  const chain = [
    ['Q、K 各維度\n變異數約為 1', C.WASH, C.TXT],
    ['點積 d_head 個維度\n變異數變成 d_head', C.WASH, C.TXT],
    ['標準差 √d_head\nd=64 → 分數散到 ±8', C.RED_L, C.RED],
    ['softmax 變得極度尖銳\n幾乎變成 one-hot', C.RED_L, C.RED],
    ['飽和區梯度趨近 0\n梯度消失', C.RED, C.PAPER],
  ];
  chain.forEach((c, i) => {
    const x = M + i * 2.45;
    box(s, p, x, 2.66, 2.1, 0.9, c[0], { fill: c[1], line: c[2] === C.PAPER ? C.RED : (c[1] === C.WASH ? C.RULE : C.RED), color: c[2], size: 10.5, ls: 13 });
    if (i < 4) arrow(s, p, x + 2.1, 3.11, 0.35, 0, { color: C.MUTED, width: 1.4 });
  });

  card(s, p, M, 3.86, 6.2, 1.5, C.GREEN_L, { radius: 0.1 });
  txt(s, '除以 √d_head 之後', { x: M + 0.3, y: 4.0, w: 5.6, h: 0.32, fontSize: 14.5, bold: true, color: C.GREEN, valign: 'middle' });
  txt(s, '把變異數拉回 1，softmax 才處在有梯度的區間。',
    { x: M + 0.3, y: 4.36, w: 5.6, h: 0.3, fontSize: 13, color: C.TXT, valign: 'middle' });
  txt(s, '本專案：d_head = 64，scale = 1/√64 = 0.125', { x: M + 0.3, y: 4.7, w: 5.6, h: 0.32, fontSize: 12.5, bold: true, color: C.TXT, fontFace: F.CODE, valign: 'middle' });

  card(s, p, 7.12, 3.86, 5.61, 1.5, C.WASH, { radius: 0.1 });
  txt(s, '直覺版', { x: 7.42, y: 4.0, w: 5.0, h: 0.32, fontSize: 14.5, bold: true, color: C.MUTED, valign: 'middle' });
  txt(s, '維度越多，點積的數字自然越大 —— 就像加總 64 個隨機數會比加總 4 個大。softmax 吃到很大的數字時會把幾乎全部機率押在一格上，等於「還沒學就已經很有自信」，而且再也學不動。',
    { x: 7.42, y: 4.32, w: 5.0, h: 0.9, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  txt(s, '反向：這裡沒有 cross-entropy 接在後面，不能用 p − y 的簡化', { x: M, y: 5.62, w: 9, h: 0.34, fontSize: 14.5, bold: true, color: C.TXT, valign: 'middle' });
  formula(s, p, M, 6.0, CW, 0.72, '∂L/∂S_{ij}  =  A_{ij} ( ∂L/∂A_{ij}  −  Σ_{k} ∂L/∂A_{ik} A_{ik} )', { size: 17, bg: C.TEAL_L, color: C.TEAL });
  s.addNotes('括號裡那個 Σ 是「這一列的加權平均」，每列算一次就好，不需要真的建出 T×T 的 Jacobian。另外遮罩不用特別處理——被遮住的位置 Aij = 0，公式最前面的 Aij 自動讓那些梯度也是 0。');

  // ══ 因果遮罩 + 實測 ══════════════════════════════════════════
  s = contentSlide(p, 5, '因果遮罩：只能看已經看過的字', '也是為什麼一次前向就能同時訓練 T 個位置的預測');
  txt(s, '遮罩矩陣（● 可以看，空白被擋住）', { x: M, y: 1.6, w: 5.5, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  const mrows = [[hdr(''), hdr('j=0'), hdr('j=1'), hdr('j=2'), hdr('j=3'), hdr('j=4')]];
  for (let i = 0; i < 5; i++) {
    const r = [cell('i=' + i, { bold: true, fill: { color: C.WASH } })];
    for (let j = 0; j < 5; j++) {
      r.push(j <= i ? cell('●', { align: 'center', bold: true, color: C.GREEN, fill: { color: C.GREEN_L } })
        : cell('', { fill: { color: C.RED_L } }));
    }
    mrows.push(r);
  }
  table(s, mrows, { x: M, y: 1.98, w: 5.5, colW: [0.9, 0.92, 0.92, 0.92, 0.92, 0.92], rowH: 0.36, fontSize: 12 });
  formula(s, p, M, 4.24, 5.5, 0.6, 'S_{ij} ← −∞   if  j > i          exp(−∞) = 0', { size: 14, bg: C.RED_L, color: C.RED });
  txt(s, '下三角。每個位置看到的都是「只到自己為止」的資訊，互不干擾 —— 所以 T 個位置可以一次算完。',
    { x: M, y: 4.96, w: 5.5, h: 0.7, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  txt(s, '實際跑出來的注意力矩陣（訓練第 1 步、第 0 個 head）', { x: 6.62, y: 1.6, w: 6.1, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  const A = [
    ['1.00000', '', '', '', ''],
    ['0.43477', '0.56523', '', '', ''],
    ['0.35364', '0.28259', '0.36377', '', ''],
    ['0.21434', '0.32812', '0.22678', '0.23077', ''],
    ['0.23005', '0.15862', '0.16002', '0.20928', '0.24203'],
  ];
  const arows = [[hdr(''), hdr('j=0'), hdr('j=1'), hdr('j=2'), hdr('j=3'), hdr('j=4'), hdr('列和')]];
  A.forEach((row, i) => {
    const r = [cell('i=' + i, { bold: true, fill: { color: C.WASH } })];
    row.forEach((v) => r.push(cell(v, { align: 'right', fontFace: F.CODE, fontSize: 10.5, ...(v ? {} : { fill: { color: C.RED_L } }) })));
    r.push(cell('1.0', { align: 'right', bold: true, color: C.GREEN, fill: { color: C.GREEN_L } }));
    arows.push(r);
  });
  table(s, arows, { x: 6.62, y: 1.98, w: 6.11, colW: [0.66, 0.93, 0.93, 0.93, 0.93, 0.93, 0.8], rowH: 0.36, fontSize: 11 });

  const obs = [
    ['上三角全是空的', '因果遮罩生效了', C.RED],
    ['每列和都是 1.0', 'softmax 正確', C.GREEN],
    ['權重相當平均', '因為這是訓練第 1 步，權重還是亂數，模型還不知道該注意誰。訓練久了會出現明顯的尖峰。', C.AMBER_D],
  ];
  obs.forEach((o, i) => {
    const y = 4.24 + i * 0.84;
    card(s, p, 6.62, y, 6.11, 0.74, C.WASH, { radius: 0.08 });
    txt(s, String(i + 1), { x: 6.82, y, w: 0.34, h: 0.74, fontSize: 15, bold: true, color: o[2], fontFace: F.MATH, valign: 'middle' });
    txt(s, o[0], { x: 7.2, y, w: 1.85, h: 0.74, fontSize: 12.5, bold: true, color: C.TXT, valign: 'middle' });
    txt(s, o[1], { x: 9.1, y, w: 3.45, h: 0.74, fontSize: 10.5, color: C.MUTED, valign: 'middle', lineSpacing: 13 });
  });
  s.addNotes('第 3 點很值得停下來講：注意力矩陣在訓練初期是「平均分配」的，這是模型「什麼都還不知道」的樣子。可以對照 train_step_3000.txt 看它變尖銳。');

  // ══ Attention 記憶體 ═════════════════════════════════════════
  s = contentSlide(p, 5, '為什麼長 context 這麼貴', 'Attention 的成本是 O(T²)，而且訓練時 A 要存下來給反向用');
  formula(s, p, M, 1.66, CW, 0.78, 'S ∈ ℝ^{B × h × T × T}          64 × 4 × 128 × 128  =  4.19 × 10⁶', { size: 19 });

  txt(s, '序列長度加倍，這個張量會變成四倍', { x: M, y: 2.68, w: 7, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  const tsq = [['T = 128', '4.19 M', 1.0], ['T = 256', '16.8 M', 1.55], ['T = 512', '67.1 M', 2.35], ['T = 1024', '268 M', 3.4]];
  tsq.forEach((t, i) => {
    const x = M + i * 3.09;
    const h = 0.42 + t[2] * 0.52;
    card(s, p, x, 4.62 - h, 2.86, h, i === 3 ? C.RED : C.TEAL_L, { radius: 0.07 });
    txt(s, t[1], { x, y: 4.62 - h, w: 2.86, h, fontSize: 20, bold: true, color: i === 3 ? C.PAPER : C.TEAL, align: 'center', valign: 'middle', fontFace: F.MATH });
    txt(s, t[0], { x, y: 4.68, w: 2.86, h: 0.3, fontSize: 12.5, bold: true, color: C.TXT, align: 'center', valign: 'middle' });
  });

  card(s, p, M, 5.24, 6.2, 1.48, C.WASH, { radius: 0.1 });
  txt(s, '這就是 FlashAttention 要解決的問題', { x: M + 0.3, y: 5.38, w: 5.6, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  txt(s, '它的作法是分塊計算、不把完整的 S 物化出來 —— 數學上完全等價，省的是記憶體搬運。',
    { x: M + 0.3, y: 5.72, w: 5.6, h: 0.8, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  card(s, p, 7.12, 5.24, 5.61, 1.48, C.AMBER_L, { radius: 0.1 });
  txt(s, '呼應前面的 tokenizer', { x: 7.42, y: 5.38, w: 5.0, h: 0.32, fontSize: 14, bold: true, color: C.AMBER_D, valign: 'middle' });
  txt(s, 'BPE 讓序列短 29.4%。因為 attention 是 O(n²)，實際省下的計算量遠不只 29.4% —— 這就是「詞表大小是架構參數」的另一個面向。',
    { x: 7.42, y: 5.72, w: 5.0, h: 0.8, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });
  s.addNotes('把 tokenizer 那一段和 attention 這一段接起來。序列長度是 tokenizer 決定的，而 attention 對序列長度是平方級的成本——兩個看似無關的段落在這裡合流。');

  // ══ FFN 全貌（重繪）══════════════════════════════════════════
  s = contentSlide(p, 5, 'FFN（SwiGLU）的結構：一個閘門', 'Attention 橫向交換資訊，FFN 縱向加工 —— 它對第 7 個位置做的事，跟第 13 個位置完全獨立');
  const GY = 2.05, UY = 3.55, MY = 2.80, BH = 0.62;

  box(s, p, 0.62, MY, 1.05, BH, 'x\n256 維', { fill: C.WASH, line: C.RULE, size: 11, ls: 13 });
  // 分岔：x 同時餵給 gate 和 up 兩路
  arrow(s, p, 1.67, MY + BH / 2, 0.28, 0, { color: C.MUTED, width: 1.3, head: false });
  arrow(s, p, 1.95, GY + BH / 2, 0, MY + BH / 2 - GY - BH / 2, { color: C.MUTED, width: 1.3, head: false });
  arrow(s, p, 1.95, MY + BH / 2, 0, UY + BH / 2 - MY - BH / 2, { color: C.MUTED, width: 1.3, head: false });
  arrow(s, p, 1.95, GY + BH / 2, 0.25, 0, { color: C.MUTED, width: 1.3 });
  arrow(s, p, 1.95, UY + BH / 2, 0.25, 0, { color: C.MUTED, width: 1.3 });

  // 閘門那一路
  box(s, p, 2.20, GY, 1.60, BH, 'gate_proj\n256 → 672', { fill: C.PAPER, line: C.AMBER_D, color: C.AMBER_D, size: 10.5, ls: 13 });
  arrow(s, p, 3.80, GY + BH / 2, 0.25, 0, { color: C.AMBER_D, width: 1.3 });
  box(s, p, 4.05, GY, 1.60, BH, 'SiLU\ns = g · σ(g)', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 10.5, ls: 13 });
  arrow(s, p, 5.65, GY + BH / 2, 0.25, 0, { color: C.AMBER_D, width: 1.3 });
  box(s, p, 5.90, GY, 1.15, BH, 's\n閘門', { fill: C.AMBER, line: C.AMBER_D, color: C.INK, size: 11, ls: 13 });

  // 內容那一路
  box(s, p, 2.20, UY, 1.60, BH, 'up_proj\n256 → 672', { fill: C.PAPER, line: C.TEAL, color: C.TEAL, size: 10.5, ls: 13 });
  arrow(s, p, 3.80, UY + BH / 2, 2.10, 0, { color: C.TEAL, width: 1.3 });
  box(s, p, 5.90, UY, 1.15, BH, 'u\n內容', { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 11, ls: 13 });

  // 兩路匯入 ⊙
  s.addShape(p.ShapeType.ellipse, { x: 7.35, y: MY, w: 0.60, h: BH, fill: { color: C.INK }, line: { type: 'none' } });
  txt(s, '⊙', { x: 7.35, y: MY, w: 0.60, h: BH, fontSize: 19, bold: true, color: C.PAPER, align: 'center', valign: 'middle' });
  arrow(s, p, 7.05, GY + BH / 2, 0.60, 0, { color: C.AMBER_D, width: 1.3, head: false });
  arrow(s, p, 7.65, GY + BH / 2, 0, MY - GY - BH / 2, { color: C.AMBER_D, width: 1.3 });
  arrow(s, p, 7.05, UY + BH / 2, 0.60, 0, { color: C.TEAL, width: 1.3, head: false });
  arrow(s, p, 7.65, MY + BH, 0, UY + BH / 2 - MY - BH, { color: C.TEAL, width: 1.3, flipV: true });

  arrow(s, p, 7.95, MY + BH / 2, 0.30, 0, { color: C.MUTED, width: 1.3 });
  box(s, p, 8.25, MY, 1.50, BH, 'm = s ⊙ u\n672 維', { fill: C.WASH, line: C.RULE, size: 10.5, ls: 13 });
  arrow(s, p, 9.75, MY + BH / 2, 0.25, 0, { color: C.MUTED, width: 1.3 });
  box(s, p, 10.00, MY, 1.65, BH, 'down_proj\n672 → 256', { fill: C.PAPER, line: C.RULE, size: 10.5, ls: 13 });
  arrow(s, p, 11.65, MY + BH / 2, 0.30, 0, { color: C.MUTED, width: 1.3 });
  box(s, p, 11.95, MY, 0.78, BH, 'y\n256 維', { fill: C.WASH, line: C.RULE, size: 11, ls: 13 });

  // 升維區間的標示
  s.addShape(p.ShapeType.line, { x: 2.20, y: 4.32, w: 7.55, h: 0, line: { color: C.AMBER_D, width: 1, dashType: 'dash' } });
  s.addShape(p.ShapeType.line, { x: 2.20, y: 4.22, w: 0, h: 0.10, line: { color: C.AMBER_D, width: 1 } });
  s.addShape(p.ShapeType.line, { x: 9.75, y: 4.22, w: 0, h: 0.10, line: { color: C.AMBER_D, width: 1 } });
  txt(s, '這一整段都在 672 維工作（升維 2.6 倍），最後才由 down_proj 壓回 256', { x: 2.20, y: 4.38, w: 7.55, h: 0.28, fontSize: 10.5, bold: true, color: C.AMBER_D, align: 'center' });

  formula(s, p, M, 4.78, CW, 0.72, 'y  =  ( SiLU( x W_{gate}ᵀ )  ⊙  x W_{up}ᵀ )  W_{down}ᵀ', { size: 18 });

  const ffn2 = [
    ['閘門在做什麼', 's 和 u 是逐元素相乘 —— s 的第 3 維只乘 u 的第 3 維。所以 s 等於 672 個獨立的閥門，各自決定 u 那一維要放行多少。追蹤檔裡第 3 維的 u 乘上 s 之後只剩 −0.0069，幾乎被關掉。', C.AMBER_D, C.AMBER_L],
    ['為什麼要兩路', '原始 Transformer（2017）只有一路：ReLU(xW₁)W₂。SwiGLU 多出來的 up 那一路提供「內容」，gate 那一路提供「要不要放行」。多一個矩陣，所以 d_ff 取 8/3·d 而不是傳統的 4d，維持總參數量相當。', C.TEAL, C.TEAL_L],
  ];
  ffn2.forEach((c, i) => {
    const x = M + i * 6.25;
    card(s, p, x, 5.66, 5.84, 1.08, c[3], { radius: 0.1 });
    txt(s, c[0], { x: x + 0.28, y: 5.76, w: 5.28, h: 0.28, fontSize: 13, bold: true, color: c[2], valign: 'middle' });
    txt(s, c[1], { x: x + 0.28, y: 6.04, w: 5.28, h: 0.62, fontSize: 10.5, color: C.TXT, valign: 'top', lineSpacing: 14 });
  });
  s.addNotes('重繪版：兩條車道（閘門 / 內容）平行往右，在 ⊙ 匯流。橘色是閘門路、藍色是內容路，顏色本身就在講故事。下面那條虛線標出「672 維工作區」，把升維再降維畫出來了——原本這件事只有文字寫著 d → d_ff，圖上看不出來。');

  // ══ FFN 升維 + 參數量 ════════════════════════════════════════
  s = contentSlide(p, 5, '為什麼先升維再降回來 —— 以及它為什麼這麼肥', '一層裡三分之二的參數都在 FFN');
  box(s, p, M, 1.78, 2.4, 0.85, 'd = 256', { fill: C.WASH, line: C.RULE, size: 14, face: F.MATH });
  arrow(s, p, 3.05, 2.20, 0.5, 0, { color: C.AMBER_D, width: 1.6 });
  box(s, p, 3.55, 1.78, 2.9, 0.85, 'd_ff = 672\n放大 2.6 倍', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 13, ls: 17 });
  arrow(s, p, 6.45, 2.20, 0.5, 0, { color: C.AMBER_D, width: 1.6 });
  box(s, p, 6.95, 1.78, 2.4, 0.85, '非線性加工', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 13 });
  arrow(s, p, 9.35, 2.20, 0.5, 0, { color: C.AMBER_D, width: 1.6 });
  box(s, p, 9.85, 1.78, 2.88, 0.85, 'd = 256\n降回原本維度', { fill: C.WASH, line: C.RULE, size: 13, ls: 17 });
  txt(s, '在高維空間裡，資料更容易被非線性「切開」。先升維、加工、再壓回來 —— 這是 FFN 的標準套路。',
    { x: M, y: 2.78, w: CW, h: 0.34, fontSize: 13, color: C.MUTED, valign: 'middle' });

  card(s, p, M, 3.3, 6.2, 1.66, C.AMBER_L, { radius: 0.1 });
  txt(s, 'FFN 的參數量', { x: M + 0.3, y: 3.44, w: 5.6, h: 0.3, fontSize: 13.5, bold: true, color: C.AMBER_D, valign: 'middle' });
  rich(s, math('3 × d × d_ff  =  3 × 256 × 672  =  516,096', { fontSize: 16, fontFace: F.MATH, bold: true, color: C.TXT }),
    { x: M + 0.3, y: 3.8, w: 5.6, h: 0.44, valign: 'middle' });
  txt(s, '三個矩陣：gate、up、down', { x: M + 0.3, y: 4.3, w: 5.6, h: 0.3, fontSize: 11.5, color: C.MUTED, valign: 'middle' });

  card(s, p, 7.12, 3.3, 5.61, 1.66, C.TEAL_L, { radius: 0.1 });
  txt(s, 'Attention 的參數量', { x: 7.42, y: 3.44, w: 5.0, h: 0.3, fontSize: 13.5, bold: true, color: C.TEAL, valign: 'middle' });
  rich(s, math('4 × d²  =  4 × 256²  =  262,144', { fontSize: 16, fontFace: F.MATH, bold: true, color: C.TXT }),
    { x: 7.42, y: 3.8, w: 5.0, h: 0.44, valign: 'middle' });
  txt(s, '四個矩陣：q、k、v、o', { x: 7.42, y: 4.3, w: 5.0, h: 0.3, fontSize: 11.5, color: C.MUTED, valign: 'middle' });

  card(s, p, M, 5.2, CW, 0.9, C.INK, { radius: 0.09 });
  rich(s, [
    { text: 'FFN 大約是 Attention 的兩倍。', options: { bold: true, color: C.AMBER, fontSize: 17 } },
    { text: '　大家都在談 attention，但參數其實大多住在 FFN 裡。', options: { color: 'C3CCD5', fontSize: 14 } },
  ], { x: M + 0.36, y: 5.2, w: CW - 0.72, h: 0.9, valign: 'middle' });

  card(s, p, M, 6.24, CW, 0.66, C.RED_L, { radius: 0.08 });
  rich(s, [
    { text: '記憶體：', options: { bold: true, color: C.RED, fontSize: 12.5 } },
    { text: '反向需要 g、s、u，所以前向時全都要存下來，形狀都是 (B,T,d_ff)。bf16 下一層約 33 MB，四層 132 MB —— 這就是「訓練比推論吃記憶體」的主因之一。推論不用存這些，算完就丟。', options: { color: C.TXT, fontSize: 12 } },
  ], { x: M + 0.34, y: 6.24, w: CW - 0.68, h: 0.66, valign: 'middle' });
  s.addNotes('兩個工程上很有用的數字：(1) 參數三分之二在 FFN (2) 訓練吃記憶體是因為要存活化值給反向用。第二點解釋了為什麼 inference 可以用小很多的卡。');

  // ══ 閘門實測 ═════════════════════════════════════════════════
  s = contentSlide(p, 5, '閘門在做什麼：用真實數值看', '從 traces/sample/train_step_0001.txt 抓的第一層 FFN，前 4 個維度');
  table(s, [
    [hdr('維度'), hdr('g（gate）'), hdr('s = SiLU(g)'), hdr('u（up）'), hdr('m = s ⊙ u')],
    [cell('1'), cell('−1.00882', { align: 'right', fontFace: F.CODE }), cell('−0.26957', { align: 'right', fontFace: F.CODE }), cell('−0.59599', { align: 'right', fontFace: F.CODE }), cell('+0.16066', { align: 'right', fontFace: F.CODE, bold: true })],
    [cell('2'), cell('−0.32442', { align: 'right', fontFace: F.CODE }), cell('−0.13613', { align: 'right', fontFace: F.CODE }), cell('+0.28378', { align: 'right', fontFace: F.CODE }), cell('−0.03863', { align: 'right', fontFace: F.CODE, bold: true })],
    [cell('3', { fill: { color: C.RED_L } }), cell('+0.33926', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } }), cell('+0.19813', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } }), cell('−0.03482', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } }), cell('−0.00690', { align: 'right', fontFace: F.CODE, bold: true, color: C.RED, fill: { color: C.RED_L } })],
    [cell('4'), cell('−1.21415', { align: 'right', fontFace: F.CODE }), cell('−0.27800', { align: 'right', fontFace: F.CODE }), cell('−0.69361', { align: 'right', fontFace: F.CODE }), cell('+0.19282', { align: 'right', fontFace: F.CODE, bold: true })],
  ], { x: M, y: 1.68, w: 7.9, colW: [1.1, 1.7, 1.7, 1.7, 1.7], rowH: 0.4, fontSize: 12 });

  card(s, p, 8.72, 1.68, 4.01, 2.4, C.RED_L, { radius: 0.1 });
  txt(s, '看第 3 個維度', { x: 9.0, y: 1.84, w: 3.4, h: 0.32, fontSize: 14, bold: true, color: C.RED, valign: 'middle' });
  txt(s, 'u₃ = −0.03482 本身就很小，乘上 s₃ = 0.198 之後變成 −0.0069 —— 幾乎被關掉了。',
    { x: 9.0, y: 2.2, w: 3.4, h: 0.86, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });
  txt(s, '這就是 gating：即使 u 那一路算出了東西，s 可以決定讓它過多少。',
    { x: 9.0, y: 3.1, w: 3.4, h: 0.8, fontSize: 12, bold: true, color: C.RED, valign: 'top', lineSpacing: 16 });

  txt(s, '驗算', { x: M, y: 4.3, w: 3, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  formula(s, p, M, 4.66, 6.2, 1.0, 'SiLU(−1.00882) = −1.00882 × σ(−1.00882)\n= −1.00882 × 0.2672 = −0.2696　← 對得上', { size: 13, bg: C.WASH, lineSpacing: 20 });
  formula(s, p, 7.12, 4.66, 5.61, 1.0, 'm_{1} = s_{1} × u_{1}\n= (−0.26957) × (−0.59599) = +0.16066　← 對得上', { size: 13, bg: C.WASH, lineSpacing: 20 });

  card(s, p, M, 5.86, CW, 0.86, C.GREEN_L, { radius: 0.08 });
  rich(s, [
    { text: 'SiLU 對比 ReLU：', options: { bold: true, color: C.GREEN, fontSize: 13 } },
    { text: '負數區不是直接砍成 0，而是留下一點負值（最小約 −0.278），所以沒有 ReLU 的「死神經元」問題；而且處處可微，沒有 z=0 的折點。', options: { color: C.TXT, fontSize: 12.5 } },
  ], { x: M + 0.34, y: 5.86, w: CW - 0.68, h: 0.86, valign: 'middle' });
  s.addNotes('這張是「閘門」這個抽象詞的具體化。第 3 個維度就是一個被關掉的例子。再次強調：所有數字都可以自己驗算，SiLU(-1.00882) 拿計算機按一次就出來。');
};
