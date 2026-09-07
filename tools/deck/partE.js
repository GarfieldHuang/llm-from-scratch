const L = require('./lib.js');
const { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell } = L;

module.exports = function partE(p) {
  let s;

  // ══ 章節 6 ═══════════════════════════════════════════════════
  sectionSlide(p, 6, '反向傳播全鏈', '從 p − y 一路走回 E。中間每一層都不知道任務是什麼。',
    ['整條鏈只有一格知道「對錯」', 'dz = p − y 為什麼這麼簡潔', '每一層都在做同一件翻譯', '反向傳播不會修改任何權重']);

  // ══ 只有一格知道對錯 ═════════════════════════════════════════
  s = contentSlide(p, 6, '整條鏈只有一個地方知道「對錯」', '往下每一層都只是在做微分 —— 不知道任務是什麼，也不知道「貓是動物」');
  const chainB = [
    ['cross-entropy\n知道正確答案', C.RED, C.RED_L, 'dz = p − y'],
    ['輸出層', C.MUTED, C.PAPER, 'dh'],
    ['final RMSNorm', C.MUTED, C.PAPER, 'd'],
    ['Block N … Block 1', C.MUTED, C.PAPER, 'dx = 修正單'],
    ['Embedding\nscatter-add', C.TEAL, C.TEAL_L, ''],
  ];
  chainB.forEach((b, i) => {
    const x = M + i * 2.45;
    box(s, p, x, 1.72, 2.1, 0.92, b[0], { fill: b[2], line: b[0].indexOf('cross') === 0 ? C.RED : (i === 4 ? C.TEAL : C.RULE), color: b[1] === C.MUTED ? C.TXT : b[1], size: 11.5, ls: 14 });
    if (i < 4) {
      arrow(s, p, x + 2.1, 2.18, 0.35, 0, { color: C.MUTED, width: 1.4 });
      txt(s, b[3], { x: x + 1.75, y: 2.62, w: 1.9, h: 0.28, fontSize: 10, color: C.MUTED, align: 'center', fontFace: F.MATH });
    }
  });

  card(s, p, M, 3.12, CW, 1.24, C.INK, { radius: 0.1 });
  rich(s, [
    { text: '只有最上面那一格含有「答案」的資訊。', options: { bold: true, color: C.AMBER, fontSize: 19, breakLine: true } },
    { text: '從那裡往下，每一層做的事情一模一樣：把「我的輸出該怎麼動」翻譯成「那我的輸入該怎麼動」。翻譯的方法就是用自己的導數。它不需要知道任務是什麼。', options: { color: 'C3CCD5', fontSize: 13.5 } },
  ], { x: M + 0.4, y: 3.12, w: CW - 0.8, h: 1.24, valign: 'middle', lineSpacing: 24 });

  txt(s, '一直往上追，源頭只有一個', { x: M, y: 4.6, w: 6, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 5.02, 6.2, 1.68, C.WASH, { radius: 0.1 });
  txt(s, 'dz = p − y            ← 唯一的源頭，只有這裡有「對錯」的資訊\n  ↓  lm_head.backward\n  ↓  final_norm.backward\n  ↓  block3 … block0        每一步都只是「翻譯」\ndx = 送到 embedding 的修正單',
    { x: M + 0.3, y: 5.14, w: 5.6, h: 1.46, fontSize: 10.5, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 17 });

  card(s, p, 7.12, 5.02, 5.61, 1.68, C.RED_L, { radius: 0.1 });
  txt(s, '追到最頂端就沒有「上一層」了', { x: 7.42, y: 5.16, w: 5.0, h: 0.32, fontSize: 14, bold: true, color: C.RED, valign: 'middle' });
  formula(s, p, 7.42, 5.54, 5.0, 0.56, '∂L/∂z  =  p − y', { size: 18, bg: C.PAPER, color: C.RED });
  txt(s, '這裡不是從別人那裡拿來的，是算出來的 —— p 是模型的預測，y 是正確答案。',
    { x: 7.42, y: 6.16, w: 5.0, h: 0.42, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 15 });
  s.addNotes('「每一層都不知道任務是什麼」是個很反直覺但很重要的觀念。模型沒有「理解」，只有一條把 p−y 機械式傳遞下去的鏈。');

  // ══ 第一棒 dz = p − y ════════════════════════════════════════
  s = contentSlide(p, 6, '第一棒：∂L/∂z = p − y', 'softmax 和 cross-entropy 分開推都很醜，但合起來推會神奇地簡化');
  card(s, p, M, 1.62, 6.2, 2.1, C.WASH, { radius: 0.1 });
  txt(s, 'softmax 的 Jacobian（會出現 |V| × |V| 的矩陣）', { x: M + 0.3, y: 1.76, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: C.MUTED, valign: 'middle' });
  formula(s, p, M + 0.3, 2.08, 5.6, 0.5, '∂p_{i}/∂z_{j} = p_{i} ( δ_{ij} − p_{j} )', { size: 15, bg: C.PAPER });
  txt(s, '接上 cross-entropy（L = −log pᵧ）之後：', { x: M + 0.3, y: 2.66, w: 5.6, h: 0.3, fontSize: 12, color: C.TXT, valign: 'middle' });
  formula(s, p, M + 0.3, 2.98, 5.6, 0.56, '= −(1/p_{y}) · p_{y} ( δ_{yj} − p_{j} ) = p_{j} − δ_{yj}', { size: 13, bg: C.PAPER });

  card(s, p, 7.12, 1.62, 5.61, 2.1, C.RED_L, { radius: 0.1 });
  txt(s, '整個 Jacobian 消失，只剩：', { x: 7.42, y: 1.8, w: 5.0, h: 0.32, fontSize: 13.5, bold: true, color: C.RED, valign: 'middle' });
  formula(s, p, 7.42, 2.2, 5.0, 0.72, '∂L/∂z  =  p − y', { size: 26, bg: C.PAPER, color: C.RED });
  txt(s, '翻成白話：正解那一格的分數要拉高，其他全部壓低。', { x: 7.42, y: 3.04, w: 5.0, h: 0.5, fontSize: 13, bold: true, color: C.TXT, valign: 'middle' });

  txt(s, '追蹤檔裡看到的（train_step_0001.txt 第 493 行）', { x: M, y: 3.94, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('token id'), hdr('p'), hdr('y'), hdr('dz = (p−y)/N'), hdr('作用')],
    [cell('4656'), cell('+0.00048', { align: 'right', fontFace: F.CODE }), cell('0', { align: 'right', fontFace: F.CODE }), cell('+5.920e−08', { align: 'right', fontFace: F.CODE }), cell('壓下去')],
    [cell('220'), cell('+0.00046', { align: 'right', fontFace: F.CODE }), cell('0', { align: 'right', fontFace: F.CODE }), cell('+5.575e−08', { align: 'right', fontFace: F.CODE }), cell('壓下去')],
    [cell('5868', { bold: true, fill: { color: C.GREEN_L } }), cell('+0.00026', { align: 'right', fontFace: F.CODE, fill: { color: C.GREEN_L } }), cell('1', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.GREEN_L } }), cell('−0.00012', { align: 'right', fontFace: F.CODE, bold: true, color: C.GREEN, fill: { color: C.GREEN_L } }), cell('拉上來（正解）', { bold: true, color: C.GREEN, fill: { color: C.GREEN_L } })],
  ], { x: M, y: 4.34, w: CW, colW: [2.0, 2.2, 1.6, 2.8, 3.49], rowH: 0.42, fontSize: 12.5 });

  card(s, p, M, 6.02, CW, 0.74, C.INK, { radius: 0.08 });
  rich(s, [
    { text: '正解那格 y = 1，所以 dz 是負的。', options: { bold: true, color: C.AMBER, fontSize: 13.5 } },
    { text: '　更新時 E ← E − η·dz，負負得正，分數被推高。其他格全是正的，分數被壓低。這一行就是整個訓練的全部監督訊號。', options: { color: 'C3CCD5', fontSize: 13 } },
  ], { x: M + 0.34, y: 6.02, w: CW - 0.68, h: 0.74, valign: 'middle' });
  s.addNotes('「除以 N」是因為 loss 對所有有效位置取了平均。強調正解那格是負的、其他是正的——這個符號差別就是全部的學習訊號。');

  // ══ 中間每一棒 ═══════════════════════════════════════════════
  s = contentSlide(p, 6, '中間每一棒都在做同一件事', '「我的輸出該怎麼動」→「那我的輸入該怎麼動」');
  const bwRules = [
    ['Linear', 'y = x Wᵀ', '∂L/∂x = ∂L/∂y W\n∂L/∂W = (∂L/∂y)ᵀ x', '點積對其中一邊微分，得到的就是另一邊', C.TEAL],
    ['逐元素非線性', 'z = f(a)', '∂L/∂a = ∂L/∂z ⊙ f′(a)', '不用加總，直接乘上導數', C.AMBER_D],
    ['殘差', 'y = x + f(x)', '∂L/∂x = ∂L/∂y + f′(∂L/∂y)', '那個孤零零的 ∂L/∂y 就是直達路徑', C.GREEN],
  ];
  bwRules.forEach((r, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 1.66, 3.85, 2.5, C.PAPER, { line: C.RULE, radius: 0.1 });
    txt(s, r[0], { x: x + 0.26, y: 1.82, w: 3.35, h: 0.32, fontSize: 15, bold: true, color: r[4], valign: 'middle' });
    formula(s, p, x + 0.26, 2.2, 3.35, 0.44, r[1], { size: 13, bg: C.WASH });
    rich(s, math(r[2], { fontSize: 12, fontFace: F.MATH, bold: true, color: C.TXT }), { x: x + 0.26, y: 2.76, w: 3.35, h: 0.7, valign: 'top', lineSpacing: 18 });
    txt(s, r[3], { x: x + 0.26, y: 3.5, w: 3.35, h: 0.5, fontSize: 11, color: C.MUTED, valign: 'top', lineSpacing: 14 });
  });

  card(s, p, M, 4.42, 6.2, 1.28, C.WASH, { radius: 0.1 });
  txt(s, 'SiLU 的導數（用 σ′ = σ(1−σ) 推）', { x: M + 0.3, y: 4.54, w: 5.6, h: 0.3, fontSize: 12.5, bold: true, color: C.MUTED, valign: 'middle' });
  formula(s, p, M + 0.3, 4.86, 5.6, 0.72, 'SiLU′(z) = σ(z) ( 1 + z ( 1 − σ(z) ) )', { size: 15, bg: C.PAPER });

  card(s, p, 7.12, 4.42, 5.61, 1.28, C.AMBER_L, { radius: 0.1 });
  txt(s, '忘記公式的時候：用形狀反推', { x: 7.42, y: 4.54, w: 5.0, h: 0.3, fontSize: 12.5, bold: true, color: C.AMBER_D, valign: 'middle' });
  txt(s, '∂L/∂x 要是 (1,3)  →  dy(1,2) @ W(2,3)\n∂L/∂W 要是 (2,3)  →  dyᵀ(2,1) @ x(1,3)',
    { x: 7.42, y: 4.9, w: 5.0, h: 0.64, fontSize: 11, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 17 });

  card(s, p, M, 5.94, CW, 0.8, C.INK, { radius: 0.09 });
  rich(s, [
    { text: '梯度的形狀永遠跟被微分的東西一樣。', options: { bold: true, color: C.AMBER, fontSize: 15 } },
    { text: '　把形狀寫出來，答案通常只有一種排法 —— 這是實務上最好用的記憶法。', options: { color: 'C3CCD5', fontSize: 13.5 } },
  ], { x: M + 0.36, y: 5.94, w: CW - 0.72, h: 0.8, valign: 'middle' });
  s.addNotes('「用形狀反推」是給工程師最實用的一招。不用記公式，把 shape 寫出來，能接得起來的排法通常只有一種。');

  // ══ 梯度接力實測 ═════════════════════════════════════════════
  s = contentSlide(p, 6, '「接力」的實際樣子', '每一層的 dx 就是下一層的 dy —— 數字完全相同');
  table(s, [
    [hdr('層'), hdr('‖dy‖（上游進來）'), hdr('‖dx‖（往下傳出去）')],
    [cell('block3'), cell('0.006687', { align: 'right', fontFace: F.CODE }), cell('0.007068', { align: 'right', fontFace: F.CODE, bold: true, color: C.TEAL })],
    [cell('block2'), cell('0.007068', { align: 'right', fontFace: F.CODE, bold: true, color: C.TEAL }), cell('0.007881', { align: 'right', fontFace: F.CODE, bold: true, color: C.TEAL })],
    [cell('block1'), cell('0.007881', { align: 'right', fontFace: F.CODE, bold: true, color: C.TEAL }), cell('0.012044', { align: 'right', fontFace: F.CODE, bold: true, color: C.TEAL })],
    [cell('block0'), cell('0.012044', { align: 'right', fontFace: F.CODE, bold: true, color: C.TEAL }), cell('0.297985', { align: 'right', fontFace: F.CODE, bold: true, color: C.RED, fill: { color: C.RED_L } })],
  ], { x: M, y: 1.68, w: 6.6, colW: [1.6, 2.5, 2.5], rowH: 0.42, fontSize: 12.5 });
  txt(s, '粗體的數字上下對得起來 —— 這就是「接力」，不是比喻，是同一個張量。',
    { x: M, y: 3.86, w: 6.6, h: 0.34, fontSize: 12, color: C.MUTED, valign: 'middle' });

  card(s, p, 7.42, 1.68, 5.31, 2.5, C.RED_L, { radius: 0.1 });
  txt(s, '最後那個大跳躍', { x: 7.72, y: 1.84, w: 4.7, h: 0.32, fontSize: 14.5, bold: true, color: C.RED, valign: 'middle' });
  stat(s, '0.012 → 0.298', '約 25 倍', { x: 7.72, y: 2.2, w: 4.7, size: 26, bh: 0.6, color: C.RED, lsize: 12 });
  txt(s, '原因是 RMSNorm。embedding 向量的 rms 約 0.02，前向除以它等於放大約 50 倍；反向經過同一個除法時，梯度也跟著放大。',
    { x: 7.72, y: 3.1, w: 4.7, h: 0.9, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  card(s, p, M, 4.44, CW, 0.86, C.WASH, { radius: 0.09 });
  rich(s, [
    { text: '為什麼實測倍率（25×）比 1/r（50×）小？', options: { bold: true, color: C.TXT, fontSize: 13 } },
    { text: '　因為 RMSNorm 的反向除了除以 r，還會把梯度中「沿著 x̂ 方向」的分量扣掉 —— x̂ 的長度已經被固定住，往它自己的方向推是無效的。那一項抵消掉了大約一半。', options: { color: C.TXT, fontSize: 12.5 } },
  ], { x: M + 0.34, y: 4.44, w: CW - 0.68, h: 0.86, valign: 'middle' });

  txt(s, '最後一棒：embedding 只做路由', { x: M, y: 5.46, w: 8, h: 0.34, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 5.86, CW, 0.88, C.TEAL_L, { radius: 0.09 });
  rich(s, [
    { text: '前向 gather  ⟺  反向 scatter-add。', options: { bold: true, color: C.TEAL, fontSize: 15 } },
    { text: '　embedding 的前向是「原封不動抄出來」，導數是 1，所以反向也是「原封不動接回去」—— 它不對梯度做任何變換，只負責送回正確的列。這個對稱不是設計出來的，是導數為 1 的必然結果。', options: { color: C.TXT, fontSize: 12.5 } },
  ], { x: M + 0.34, y: 5.86, w: CW - 0.68, h: 0.88, valign: 'middle' });
  s.addNotes('這張把 RMSNorm 那一段和反向傳播接起來。可以問聽眾：為什麼 block0 的輸出突然大了 25 倍？答案在前面 RMSNorm 那張。');

  // ══ 誤解② ════════════════════════════════════════════════════
  s = contentSlide(p, 6, '誤解 ②「反向傳播會修改權重」', '錯。backward() 跑完，權重一個數字都沒變。');
  card(s, p, M, 1.66, CW, 1.3, C.INK, { radius: 0.1 });
  txt(s, 'model.backward(dlogits)      # 只算出「該往哪修」，存進另一個張量\nopt.step()                   # 這一行才真的改',
    { x: M + 0.4, y: 1.66, w: CW - 0.8, h: 1.3, fontSize: 14, color: 'C3CCD5', fontFace: F.CODE, valign: 'middle', lineSpacing: 26 });
  s.addShape(p.ShapeType.roundRect, { x: 9.6, y: 2.34, w: 2.9, h: 0.4, rectRadius: 0.07, fill: { color: C.AMBER }, line: { type: 'none' } });
  txt(s, '↑ 兩件事之間隔著一整張梯度表', { x: 9.6, y: 2.34, w: 2.9, h: 0.4, fontSize: 11, bold: true, color: C.INK, align: 'center', valign: 'middle' });

  txt(s, '而且方向是相反的，步伐小得多', { x: M, y: 3.2, w: 8, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr(''), hdr('數值'), hdr('')],
    [cell('更新前 E[...]'), cell('−0.01533508', { align: 'right', fontFace: F.CODE }), cell('')],
    [cell('梯度 g'), cell('+0.24444881', { align: 'right', fontFace: F.CODE }), cell('')],
    [cell('若直接「加上梯度」', { fill: { color: C.RED_L } }), cell('+0.22911373', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.RED_L } }), cell('← 錯誤的直覺', { color: C.RED, bold: true, fill: { color: C.RED_L } })],
    [cell('AdamW 實際結果', { fill: { color: C.GREEN_L } }), cell('−0.01583501', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.GREEN_L } }), cell('← 差了 489 倍', { color: C.GREEN, bold: true, fill: { color: C.GREEN_L } })],
  ], { x: M, y: 3.6, w: 7.6, colW: [2.6, 2.6, 2.4], rowH: 0.42, fontSize: 12.5 });

  card(s, p, 8.42, 3.6, 4.31, 2.52, C.WASH, { radius: 0.1 });
  txt(s, '兩層都要注意', { x: 8.72, y: 3.76, w: 3.7, h: 0.32, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
  txt(s, '① 梯度是先彼此相加存進梯度表，不是加到 E 上\n\n② E 真的要改的時候是「減」，而且要乘上很小的 lr',
    { x: 8.72, y: 4.14, w: 3.7, h: 1.7, fontSize: 12.5, color: C.TXT, valign: 'top', lineSpacing: 19 });

  card(s, p, M, 6.28, CW, 0.5, C.AMBER_L, { radius: 0.07 });
  txt(s, '怎麼當場證明：追蹤檔第 04 節的最後，E 還是原值；到第 06 節 AdamW 那裡才變。',
    { x: M + 0.34, y: 6.28, w: CW - 0.68, h: 0.5, fontSize: 12.5, bold: true, color: C.AMBER_D, valign: 'middle' });
  s.addNotes('這個誤解很多人有，因為「反向傳播」這個詞聽起來就像在做更新。實際上它只是計算。真正動手的是 optimizer，這也是為什麼 optimizer 是可以替換的元件。');

  // ══ 章節 7 ═══════════════════════════════════════════════════
  sectionSlide(p, 7, '權重是怎麼被改的', '反向算完之後，才輪到 optimizer',
    ['SGD：為什麼是「減」', 'AdamW 的五行', '一個反直覺的性質：位移量與梯度大小無關', '梯度裁剪與 learning rate schedule']);

  // ══ SGD ══════════════════════════════════════════════════════
  s = contentSlide(p, 7, '最直覺的版本：SGD', '「減掉梯度乘上學習率」');
  formula(s, p, M, 1.66, CW, 1.0, 'θ  ←  θ  −  η · g', { size: 32 });
  txt(s, '為什麼是「減」？', { x: M, y: 2.94, w: 8, h: 0.36, fontSize: 17, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 3.36, 6.2, 1.5, C.WASH, { radius: 0.1 });
  txt(s, '梯度的定義是「往哪個方向動，L 會變大」：', { x: M + 0.3, y: 3.5, w: 5.6, h: 0.3, fontSize: 13, color: C.TXT, valign: 'middle' });
  formula(s, p, M + 0.3, 3.82, 5.6, 0.52, 'g  =  ∂L/∂θ', { size: 18, bg: C.PAPER });
  txt(s, '我們要 L 變小，所以走反方向。這就是梯度「下降」的下降。',
    { x: M + 0.3, y: 4.42, w: 5.6, h: 0.32, fontSize: 13, bold: true, color: C.AMBER_D, valign: 'middle' });

  card(s, p, 7.12, 3.36, 5.61, 1.5, C.RED_L, { radius: 0.1 });
  txt(s, '注意：← 不是 =', { x: 7.42, y: 3.5, w: 5.0, h: 0.32, fontSize: 14, bold: true, color: C.RED, valign: 'middle' });
  txt(s, '這是「更新」：用右邊算出來的值取代左邊。\n\n寫成 θ = θ − ηg 在數學上是個無解的方程式（除非 ηg = 0）。這是讀論文時很容易忽略的一個符號。',
    { x: 7.42, y: 3.84, w: 5.0, h: 0.9, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  txt(s, '而 AdamW 只是把那個 g 換掉', { x: M, y: 5.1, w: 8, h: 0.36, fontSize: 17, bold: true, color: C.TXT, valign: 'middle' });
  formula(s, p, M, 5.54, CW, 1.16, 'SGD:      θ ← θ − η · g\nAdamW:  θ ← θ − η · m̂ / ( √v̂ + ε )', { size: 20, lineSpacing: 30 });
  s.addNotes('先講 SGD 讓大家有個錨。AdamW 看起來複雜，但骨架跟 SGD 一模一樣，只是把 g 換成一個正規化過的方向。');

  // ══ AdamW 五行 ═══════════════════════════════════════════════
  s = contentSlide(p, 7, 'AdamW 的五行', '跟 SGD 的差別只在最後一行');
  const ad = [
    ['①', 'θ ← θ − η λ θ', 'decoupled weight decay（與梯度無關）', C.TEAL],
    ['②', 'm ← β₁ m + (1−β₁) g', '一階動量：梯度的移動平均', C.MUTED],
    ['③', 'v ← β₂ v + (1−β₂) g²', '二階動量：梯度平方的移動平均', C.MUTED],
    ['④', 'm̂ ← m/(1−β₁ᵗ)　v̂ ← v/(1−β₂ᵗ)', 'bias correction', C.MUTED],
    ['⑤', 'θ ← θ − η · m̂ / ( √v̂ + ε )', '真正的更新', C.AMBER_D],
  ];
  ad.forEach((a, i) => {
    const y = 1.66 + i * 0.72;
    card(s, p, M, y, 8.5, 0.62, i === 4 ? C.AMBER_L : C.WASH, { radius: 0.07 });
    txt(s, a[0], { x: M + 0.2, y, w: 0.44, h: 0.62, fontSize: 15, bold: true, color: a[3], align: 'center', valign: 'middle' });
    rich(s, math(a[1], { fontSize: 14, fontFace: F.MATH, bold: true, color: C.TXT }), { x: M + 0.75, y, w: 3.6, h: 0.62, valign: 'middle' });
    txt(s, a[2], { x: M + 4.5, y, w: 3.85, h: 0.62, fontSize: 11.5, color: a[3] === C.MUTED ? C.MUTED : a[3], bold: a[3] !== C.MUTED, valign: 'middle' });
  });

  card(s, p, 9.42, 1.66, 3.31, 1.34, C.INK, { radius: 0.1 });
  txt(s, 'W 的由來', { x: 9.7, y: 1.8, w: 2.8, h: 0.3, fontSize: 13, bold: true, color: C.AMBER, valign: 'middle' });
  txt(s, '舊的 Adam 把 λθ 加進梯度裡，於是 weight decay 也被 √v̂ 正規化。AdamW 把它拆出來獨立做，強度固定是 ηλ。',
    { x: 9.7, y: 2.12, w: 2.8, h: 0.8, fontSize: 10.5, color: 'C3CCD5', valign: 'top', lineSpacing: 14 });

  card(s, p, 9.42, 3.1, 3.31, 1.34, C.WASH, { radius: 0.1 });
  txt(s, 'bias correction 在做什麼', { x: 9.7, y: 3.24, w: 2.8, h: 0.3, fontSize: 13, bold: true, color: C.TXT, valign: 'middle' });
  txt(s, 'm、v 都從 0 出發，前幾步會系統性偏小。除以 1−β₁ᵗ 剛好把偏差補回來（t 大了之後趨近 1，就沒作用了）。',
    { x: 9.7, y: 3.56, w: 2.8, h: 0.8, fontSize: 10.5, color: C.TXT, valign: 'top', lineSpacing: 14 });

  card(s, p, 9.42, 4.54, 3.31, 1.34, C.RED_L, { radius: 0.1 });
  txt(s, '一個實務上的坑', { x: 9.7, y: 4.68, w: 2.8, h: 0.3, fontSize: 13, bold: true, color: C.RED, valign: 'middle' });
  txt(s, 'torch.optim.AdamW 預設 λ=0.01，而且會套用到所有參數，包括 embedding 和 norm 的 gain。',
    { x: 9.7, y: 5.0, w: 2.8, h: 0.8, fontSize: 10.5, color: C.TXT, valign: 'top', lineSpacing: 14 });

  card(s, p, M, 5.42, 8.5, 1.32, C.INK, { radius: 0.1 });
  txt(s, "AdamW(model.params(), lr=..., no_decay=('.g', '.E'))", { x: M + 0.32, y: 5.56, w: 8.0, h: 0.36, fontSize: 12.5, color: C.AMBER, fontFace: F.CODE, valign: 'middle' });
  txt(s, '生產級訓練通常把 embedding 和 norm 排除 —— 因為很少拿到梯度的列會被 weight decay 慢慢磨向 0。這份程式預設就排除了。',
    { x: M + 0.32, y: 5.94, w: 8.0, h: 0.68, fontSize: 11.5, color: 'C3CCD5', valign: 'top', lineSpacing: 16 });
  s.addNotes('不要逐行推導。重點是 ⑤：g 被換成了 m̂/(√v̂+ε)。下一張講那個替換造成的關鍵性質。右邊三張小卡是給有經驗的人看的細節，可以快速帶過。');

  // ══ 位移量與梯度無關 ═════════════════════════════════════════
  s = contentSlide(p, 7, '關鍵性質：位移量幾乎與梯度大小無關', '分子是梯度的平均，分母是梯度大小的平均 —— 量級相同，相除之後大約落在 ±1');
  formula(s, p, M, 1.62, CW, 0.86, '| Δθ |  ≈  η          第一步時這件事是精確成立的', { size: 22 });

  card(s, p, M, 2.66, 6.2, 1.44, C.WASH, { radius: 0.1 });
  txt(s, '代入 m₀ = v₀ = 0：', { x: M + 0.3, y: 2.78, w: 5.6, h: 0.28, fontSize: 12, bold: true, color: C.MUTED, valign: 'middle' });
  formula(s, p, M + 0.3, 3.08, 5.6, 0.9, 'm̂_{1} = g ,   v̂_{1} = g²\nm̂_{1}/√v̂_{1} = g/|g| = sign(g)   ⟹   Δθ = ± η', { size: 13, bg: C.PAPER, lineSpacing: 22 });

  card(s, p, 7.12, 2.66, 5.61, 1.44, C.AMBER_L, { radius: 0.1 });
  txt(s, '白話', { x: 7.42, y: 2.78, w: 5.0, h: 0.28, fontSize: 12, bold: true, color: C.AMBER_D, valign: 'middle' });
  txt(s, 'Adam 幾乎不看梯度「有多大」，只看它「往哪邊」。所有參數走一樣遠，差別只在方向。',
    { x: 7.42, y: 3.1, w: 5.0, h: 0.86, fontSize: 13, bold: true, color: C.TXT, valign: 'top', lineSpacing: 18 });

  txt(s, '追蹤檔裡可以直接驗證（η = 5×10⁻⁴）', { x: M, y: 4.28, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr(''), hdr('維度 1'), hdr('維度 2'), hdr('維度 3'), hdr('維度 4')],
    [cell('梯度 g', { fill: { color: C.RED_L } }), cell('+0.244449', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } }), cell('−0.286912', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } }), cell('−0.239161', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } }), cell('+0.236112', { align: 'right', fontFace: F.CODE, fill: { color: C.RED_L } })],
    [cell('m̂ / (√v̂ + ε)'), cell('+1.000000', { align: 'right', fontFace: F.CODE }), cell('−1.000000', { align: 'right', fontFace: F.CODE }), cell('−1.000000', { align: 'right', fontFace: F.CODE }), cell('+1.000000', { align: 'right', fontFace: F.CODE })],
    [cell('位移 Δθ', { fill: { color: C.GREEN_L } }), cell('−0.000500', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.GREEN_L } }), cell('+0.000500', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.GREEN_L } }), cell('+0.000500', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.GREEN_L } }), cell('−0.000500', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.GREEN_L } })],
  ], { x: M, y: 4.68, w: CW, colW: [2.5, 2.4, 2.4, 2.4, 2.39], rowH: 0.42, fontSize: 12 });

  card(s, p, M, 6.1, CW, 0.66, C.INK, { radius: 0.08 });
  rich(s, [
    { text: '四個維度的梯度大小不同（0.236 ~ 0.287），位移卻一模一樣。', options: { bold: true, color: C.AMBER, fontSize: 14 } },
    { text: '　推論：跑 T 步之後，任何參數的總位移上限約是 T · η̄。實測 3000 × 2.75×10⁻⁴ ≈ 0.825，而平均位移落在 0.63 ~ 0.86 —— 正好卡在上限。', options: { color: 'C3CCD5', fontSize: 12 } },
  ], { x: M + 0.34, y: 6.1, w: CW - 0.68, h: 0.66, valign: 'middle' });
  s.addNotes('這是全場最反直覺的一張。大家以為「梯度大 = 走得多」，Adam 完全不是這樣。延伸：高頻 token 走得遠但方向沒怎麼變（各種上下文互相抵消），中頻 token 轉得最多。位移量被封頂，真正有差別的是「動得有沒有方向」。');

  // ══ 護欄 ═════════════════════════════════════════════════════
  s = contentSlide(p, 7, '兩道護欄：梯度裁剪與 learning rate schedule', '防止偶爾出現的異常大梯度一步把模型炸掉');
  card(s, p, M, 1.66, 6.2, 2.5, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, '梯度裁剪　opt.clip_grad_norm(1.0)', { x: M + 0.3, y: 1.82, w: 5.6, h: 0.32, fontSize: 14.5, bold: true, color: C.TXT, valign: 'middle' });
  formula(s, p, M + 0.3, 2.2, 5.6, 0.94, 'G = √( Σ_{所有參數} ‖g‖² )\n若 G > G_max :   g ← g · G_max / (G + 10⁻⁶)', { size: 13, bg: C.WASH, lineSpacing: 22 });
  txt(s, '注意是「整體」按比例縮小 —— 各參數之間的相對比例不變。這跟逐元素裁剪（clip by value）不同，後者會扭曲梯度的方向。',
    { x: M + 0.3, y: 3.24, w: 5.6, h: 0.8, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  card(s, p, 7.12, 1.66, 5.61, 2.5, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, 'lr schedule　cosine_lr(step, total, lr, warmup=100)', { x: 7.42, y: 1.82, w: 5.0, h: 0.32, fontSize: 13, bold: true, color: C.TXT, valign: 'middle' });
  // 迷你 lr 曲線
  const lx = 7.42, ly = 2.3, lw = 5.0, lh = 0.86;
  card(s, p, lx, ly, lw, lh, C.WASH, { radius: 0.06 });
  const pts = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const v = t < 0.033 ? t / 0.033 : 0.1 + 0.9 * (1 + Math.cos(Math.PI * (t - 0.033) / 0.967)) / 2;
    pts.push([lx + 0.12 + t * (lw - 0.24), ly + lh - 0.12 - v * (lh - 0.28)]);
  }
  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    arrow(s, p, Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1), { color: C.AMBER_D, width: 1.8, head: false, flipV: y2 < y1 });
  }
  txt(s, 'warmup', { x: lx + 0.05, y: ly + lh + 0.03, w: 0.9, h: 0.22, fontSize: 9, color: C.MUTED });
  txt(s, 'cosine 衰減', { x: lx + 2.0, y: ly + lh + 0.03, w: 1.6, h: 0.22, fontSize: 9, color: C.MUTED });
  txt(s, 'warmup：一開始權重是亂數，梯度方向不可靠，用大 η 容易走歪。\ncosine 衰減：後期要小步微調，不然會在最低點附近震盪。',
    { x: 7.42, y: 3.48, w: 5.0, h: 0.62, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 16 });

  txt(s, '兩者一起看：訓練前期與後期的 lr 差了十倍', { x: M, y: 4.44, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('step'), hdr('1'), hdr('100'), hdr('1000'), hdr('2000'), hdr('3000')],
    [cell('lr'), cell('3.000e−05', { align: 'right', fontFace: F.CODE }), cell('3.000e−03', { align: 'right', fontFace: F.CODE, bold: true, color: C.AMBER_D }), cell('2.408e−03', { align: 'right', fontFace: F.CODE }), cell('1.018e−03', { align: 'right', fontFace: F.CODE }), cell('3.000e−04', { align: 'right', fontFace: F.CODE })],
    [cell('loss'), cell('8.8149', { align: 'right', fontFace: F.CODE }), cell('6.3556', { align: 'right', fontFace: F.CODE }), cell('3.6166', { align: 'right', fontFace: F.CODE }), cell('3.0630', { align: 'right', fontFace: F.CODE }), cell('2.7806', { align: 'right', fontFace: F.CODE, bold: true, color: C.GREEN })],
  ], { x: M, y: 4.84, w: CW, colW: [1.5, 2.15, 2.15, 2.15, 2.15, 1.99], rowH: 0.42, fontSize: 12 });
  txt(s, 'traces/train_log.txt 記了每一步的 η，可以直接畫成曲線。', { x: M, y: 6.3, w: CW, h: 0.34, fontSize: 11.5, color: C.MUTED, valign: 'middle' });
  s.addNotes('梯度裁剪的重點是「整體縮放，不改方向」。lr schedule 的重點是 warmup 存在的理由：一開始權重是亂數，梯度方向根本不可靠。');
};
