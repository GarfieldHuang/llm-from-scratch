const L = require('./lib.js');
const { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell } = L;

module.exports = function partA(p) {
  let s;

  // ══ 1. 封面 ══════════════════════════════════════════════════
  s = p.addSlide();
  s.background = { color: C.INK };
  txt(s, '從零手刻一個 LLM', {
    x: M + 0.1, y: 1.85, w: 11.5, h: 1.15, fontSize: 52, bold: true, color: C.PAPER, valign: 'middle',
  });
  s.addShape(p.ShapeType.rect, { x: M + 0.14, y: 3.12, w: 1.5, h: 0.055, fill: { color: C.AMBER }, line: { type: 'none' } });
  txt(s, '不用 autograd、不用 nn.Module —— 每一條 backward 都自己推、自己驗', {
    x: M + 0.1, y: 3.36, w: 11.2, h: 0.44, fontSize: 19, color: 'A9B4C0', valign: 'middle',
  });
  txt(s, '給有程式經驗、但沒碰過機器學習的人', {
    x: M + 0.1, y: 3.86, w: 11.2, h: 0.38, fontSize: 15, color: C.AMBER, valign: 'middle',
  });
  // 底部：管線縮影（視覺母題預告）
  const pipe = ['文字', 'id', '向量', 'Block ×4', 'logits', 'loss'];
  pipe.forEach((t, i) => {
    const x = M + 0.14 + i * 1.42;
    box(s, p, x, 5.5, 1.2, 0.46, t, { fill: i === 5 ? C.RED : C.INK2, color: i === 5 ? C.PAPER : '8D99A6', size: 10.5, radius: 0.05 });
    if (i < 5) arrow(s, p, x + 1.2, 5.73, 0.22, 0, { color: '4A5764', width: 1 });
  });
  arrow(s, p, M + 0.14, 6.28, 7.06, 0, { color: C.TEAL, width: 1.25, dash: true, flipH: true });
  txt(s, '反向：把 loss 翻譯成每一層的修正單', { x: M + 0.2, y: 6.36, w: 7.2, h: 0.3, fontSize: 10.5, color: C.TEAL });
  txt(s, '對應專案　llm-from-scratch\n475 萬參數 · 詞表 6,400 · 3,000 步 / 79 秒', {
    x: 9.0, y: 5.5, w: 3.7, h: 1.0, fontSize: 12, color: '76828F', align: 'right', valign: 'top', lineSpacing: 20,
  });
  s.addNotes('開場。核心賣點：市面上的教學都在講「Transformer 的架構圖」，這份不一樣——它把每一層的 backward 手推出來，再用有限差分驗證。所以你會知道「梯度」不是抽象名詞，是一個可以印出來、可以驗算的數字。');

  // ══ 2. 這 60 分鐘要回答的問題 ═════════════════════════════════
  s = contentSlide(p, '·', '這 60 分鐘，要回答四個問題', '每一題都不需要數學系背景。但幾乎每個人第一次都會答錯。');
  const qs = [
    ['詞表 6,400 個 id，\n這個數字是誰決定的？', '不是算出來的，是人工挑的停止條件。', C.AMBER, C.AMBER_L],
    ['embedding 到底是\n「資料」還是「模型」？', '是權重，跟 q_proj 平起平坐。', C.TEAL, C.TEAL_L],
    ['反向傳播會不會\n改動權重？', 'backward() 只算斜率，AdamW 才改。', C.RED, C.RED_L],
    ['token id 是整數，\n那要怎麼微分？', '不對 id 微分，對的是 E[id] 那一列。', C.GREEN, C.GREEN_L],
  ];
  qs.forEach((q, i) => {
    const x = M + i * 3.09, w = 2.86;
    card(s, p, x, 1.72, w, 3.15, q[3], { shadow: true, radius: 0.1 });
    s.addShape(p.ShapeType.roundRect, { x: x + 0.26, y: 1.98, w: 0.4, h: 0.4, rectRadius: 0.08, fill: { color: q[2] }, line: { type: 'none' } });
    txt(s, String(i + 1), { x: x + 0.26, y: 1.98, w: 0.4, h: 0.4, fontSize: 14, bold: true, color: C.PAPER, align: 'center', valign: 'middle', fontFace: F.MATH });
    txt(s, q[0], { x: x + 0.26, y: 2.56, w: w - 0.52, h: 1.4, fontSize: 16.5, bold: true, color: C.TXT, valign: 'top', lineSpacing: 25 });
    txt(s, q[1], { x: x + 0.26, y: 4.16, w: w - 0.52, h: 0.55, fontSize: 12.5, color: q[2], bold: true, valign: 'top', lineSpacing: 18 });
  });
  card(s, p, M, 5.28, CW, 0.86, C.WASH, { radius: 0.08 });
  rich(s, [
    { text: '共通點：', options: { bold: true, color: C.TXT, fontSize: 15 } },
    { text: '這四個誤解都不是「懂不懂數學」的問題，而是', options: { color: C.TXT, fontSize: 15 } },
    { text: '「哪個東西是變數、哪個是常數」', options: { bold: true, color: C.AMBER_D, fontSize: 15 } },
    { text: ' 沒分清楚。分清楚之後，公式反而是最簡單的部分。', options: { color: C.TXT, fontSize: 15 } },
  ], { x: M + 0.32, y: 5.28, w: CW - 0.64, h: 0.86, valign: 'middle' });
  s.addNotes('先把四個問題丟出來，讓聽眾帶著問題聽。結尾會回來一題一題收。強調：這不是數學能力的問題，是「變數 vs 常數」的分類問題。');

  // ══ 2.5 模型在學什麼（新增）═══════════════════════════════════
  s = contentSlide(p, '·', '模型在學什麼：看 0…t，猜 t+1', '整個訓練的監督訊號只有這一句 —— 沒有任何人標註「貓是動物」');
  const toks = ['貓', '在', '墊', '子', '上', '睡', '覺'];
  const PITCH = 0.97, BW2 = 0.85;
  txt(s, 'x（餵進去的）', { x: M, y: 1.78, w: 1.7, h: 0.5, fontSize: 12.5, bold: true, color: C.MUTED, align: 'right', valign: 'middle' });
  toks.slice(0, 6).forEach((t, i) => {
    box(s, p, 2.5 + i * PITCH, 1.78, BW2, 0.5, t, { fill: C.WASH, line: C.RULE, size: 15 });
  });
  txt(s, 'y（要它猜的）', { x: M, y: 2.6, w: 1.7, h: 0.5, fontSize: 12.5, bold: true, color: C.AMBER_D, align: 'right', valign: 'middle' });
  toks.slice(1, 7).forEach((t, i) => {
    box(s, p, 2.5 + (i + 1) * PITCH, 2.6, BW2, 0.5, t, { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 15 });
  });
  arrow(s, p, 2.93, 2.28, 0.97, 0.32, { color: C.AMBER_D, width: 1.3 });
  txt(s, '往右位移一格', { x: 3.95, y: 2.24, w: 1.9, h: 0.3, fontSize: 11, bold: true, color: C.AMBER_D, valign: 'middle' });
  txt(s, '看到「貓」要猜「在」，看到「貓在」要猜「墊」…… 一句話裡有幾個 token，就有幾道題目。',
    { x: 2.5, y: 3.2, w: 8.5, h: 0.32, fontSize: 12, color: C.MUTED, valign: 'middle' });

  card(s, p, 9.4, 1.78, 3.33, 1.32, C.INK, { radius: 0.09 });
  txt(s, '# train.py:139–141', { x: 9.64, y: 1.88, w: 2.9, h: 0.26, fontSize: 10, color: '76828F', fontFace: F.CODE });
  txt(s, 'x = data[i     : i+T]\ny = data[i + 1 : i+T+1]', { x: 9.64, y: 2.16, w: 2.9, h: 0.7, fontSize: 10.5, color: 'C3CCD5', fontFace: F.CODE, valign: 'top', lineSpacing: 18 });

  const learn3 = [
    ['一次訓練 T 個位置', '不是「一句話一道題」。因果遮罩讓每個位置只看得到自己左邊，所以 128 個位置可以同時算、互不干擾。', C.TEAL, C.TEAL_L],
    ['N = B × T = 8,192', '一個 batch 有 8,192 道題。loss 是這 8,192 個 −log p[正解] 取平均 —— 這就是 dz 要除以 N 的原因。', C.AMBER_D, C.AMBER_L],
    ['teacher forcing', '不管模型上一步猜對還猜錯，下一步餵的都是正確答案。所以 id 是資料集給定的常數 —— 訓練全程可微。', C.GREEN, C.GREEN_L],
  ];
  learn3.forEach((c, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 3.72, 3.85, 1.66, c[3], { radius: 0.09 });
    txt(s, c[0], { x: x + 0.26, y: 3.86, w: 3.35, h: 0.34, fontSize: 14, bold: true, color: c[2], valign: 'middle' });
    txt(s, c[1], { x: x + 0.26, y: 4.24, w: 3.35, h: 1.0, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 16 });
  });

  card(s, p, M, 5.62, CW, 1.1, C.INK, { radius: 0.1 });
  rich(s, [
    { text: '「看位置 0…t，猜位置 t+1。」', options: { bold: true, color: C.AMBER, fontSize: 20, breakLine: true } },
    { text: '整個訓練的監督訊號只有這個。沒有人標註詞性、沒有人標註「貓是動物」—— 語法、語意、常識全部是從「猜下一個字」這一件事裡長出來的副產品。', options: { color: 'C3CCD5', fontSize: 13 } },
  ], { x: M + 0.4, y: 5.62, w: CW - 0.8, h: 1.1, valign: 'middle', lineSpacing: 24 });
  s.addNotes('這張是全場的地基，一定要講。非 ML 背景的人最常卡在「模型到底在學什麼」——答案樸素到會讓人失望：把文字往右位移一格當答案，猜下一個字。所有能力都是這件事的副產品。teacher forcing 那格也順便鋪好了後面「訓練可微、生成不可微」的伏筆。');

  // ══ 3. 全景圖 ════════════════════════════════════════════════
  s = contentSlide(p, '·', '全景：訓練就是這個迴圈跑幾千次', '一句話進去，到權重被改動，中間發生了什麼');
  const FW = ['一句話\n貓在墊子上睡覺', 'tokenizer\n切成 id', 'Embedding\nx = E[ids]', 'N 個 transformer Block\n注意力＋FFN', '最後一道\nRMSNorm', '輸出層\nz (logits) = h @ Eᵀ', 'softmax → loss\np = softmax(z)\nL = −log p[正解]'];
  FW.forEach((t, i) => {
    const x = M + i * 1.77;
    const last = i === 6;
    box(s, p, x, 1.72, 1.46, 0.78, t, {
      fill: last ? C.RED_L : (i === 2 ? C.TEAL_L : C.WASH),
      line: last ? C.RED : (i === 2 ? C.TEAL : C.RULE),
      color: last ? C.RED : C.TXT, size: 10, ls: 13,
    });
    if (i < 6) arrow(s, p, x + 1.46, 2.11, 0.31, 0, { color: C.MUTED, width: 1.4 });
  });
  txt(s, '前　向', { x: M, y: 1.36, w: 2.0, h: 0.28, fontSize: 11, bold: true, color: C.MUTED });

  arrow(s, p, 11.97, 2.50, 0, 1.08, { color: C.RED, width: 1.6 });
  txt(s, 'dz = p − y', { x: 10.1, y: 2.72, w: 1.75, h: 0.32, fontSize: 12, bold: true, color: C.RED, align: 'right', fontFace: F.MATH });

  const BW = [['唯一知道「對錯」的地方\ndz = p − y', C.RED], ['backward() 反向傳播\n逐層往回算出每層該怎麼改', C.TEAL], ['scatter-add\n各 id 的修正量累加進 dE\n（此時 E 還沒動）', C.TEAL], ['AdamW\nE ← E − lr ·（梯度表方向）', C.GREEN]];
  const bx = [10.15, 6.95, 3.75, 0.62];
  BW.forEach((b, i) => {
    box(s, p, bx[i], 3.58, 2.55, 0.76, b[0], { fill: C.PAPER, line: b[1], lineW: 1.25, color: b[1], size: 10.5, ls: 13 });
    if (i < 3) arrow(s, p, bx[i] - 0.65, 3.96, 0.65, 0, { color: C.TEAL, width: 1.4, dash: true, flipH: true });
  });
  txt(s, '反　向', { x: M, y: 3.22, w: 2.0, h: 0.28, fontSize: 11, bold: true, color: C.TEAL });

  // 迴圈：AdamW → 回到 Embedding
  arrow(s, p, 1.90, 3.02, 0, 0.56, { color: C.GREEN, width: 1.4, dash: true, flipV: true, head: false });
  arrow(s, p, 1.90, 3.02, 2.99, 0, { color: C.GREEN, width: 1.4, dash: true, head: false });
  arrow(s, p, 4.89, 2.50, 0, 0.52, { color: C.GREEN, width: 1.4, dash: true, flipV: true });
  txt(s, '重複幾千次', { x: 2.1, y: 2.68, w: 1.6, h: 0.3, fontSize: 10.5, bold: true, color: C.GREEN });

  const tk = [
    ['整條鏈只有一格知道答案', '最後那個 cross-entropy。往下每一層都只是機械式微分，不知道任務是什麼。'],
    ['反向不改權重', 'backward() 只算出「該往哪修」，存進另一張梯度表 dE。'],
    ['optimizer AdamW 才動手', '而且方向相反、步伐只有 lr 那麼小。'],
  ];
  tk.forEach((t, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 4.86, 3.85, 1.32, C.WASH, { radius: 0.08 });
    txt(s, t[0], { x: x + 0.24, y: 5.0, w: 3.4, h: 0.36, fontSize: 14, bold: true, color: C.TXT, valign: 'middle' });
    txt(s, t[1], { x: x + 0.24, y: 5.38, w: 3.4, h: 0.7, fontSize: 11.5, color: C.MUTED, valign: 'top', lineSpacing: 15 });
  });
  txt(s, '實線＝前向　·　虛線＝反向　　（z、p、y 這幾個符號稍後會有專頁定義，現在只要看流程）', { x: M, y: 6.36, w: 9, h: 0.3, fontSize: 11, color: C.MUTED });
  s.addNotes('這張是全場的地圖，後面每一段都會回頭指這張。三個重點：(1) 只有最右邊那格有「對錯」的資訊 (2) 反向不改權重 (3) 那條綠色虛線回到 Embedding，就是訓練迴圈。留意 Embedding 被標成藍色——它同時是前向的第一站，也是反向的最後一站。');

  // ══ 4. 路線圖 ════════════════════════════════════════════════
  s = contentSlide(p, '·', '路線圖', '前半場建立直覺，後半場拆零件，最後跑一遍真的');
  const road = [
    ['1', '讀懂公式的最小前置', '⊙ 和 @ 都叫「乘」，但完全是兩回事', C.AMBER],
    ['2', '文字怎麼變成數字', 'BPE、詞表 6,400 的由來', C.AMBER],
    ['3', 'Embedding 與輸出層', '查表、h @ Eᵀ、z / p / y / L', C.TEAL],
    ['4', '模型的骨架', 'Block、殘差、形狀怎麼變', C.TEAL],
    ['5', '三個零件拆開看', 'RMSNorm / Attention / FFN', C.TEAL],
    ['6', '反向傳播全鏈', '從 p − y 一路走回 E', C.RED],
    ['7', '權重是怎麼被改的', 'AdamW 的關鍵性質', C.GREEN],
    ['8', '怎麼確定沒推錯 · 跑一遍', '有限差分驗證、loss 曲線、生成結果', C.GREEN],
    ['9', '最後一個坑', '權重與 tokenizer 是綁死的', C.RED],
  ];
  road.forEach((r, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * 4.09, y = 1.62 + row * 1.68;
    card(s, p, x, y, 3.85, 1.44, C.PAPER, { line: C.RULE, radius: 0.08 });
    s.addShape(p.ShapeType.roundRect, { x: x + 0.24, y: y + 0.24, w: 0.4, h: 0.4, rectRadius: 0.08, fill: { color: r[3] }, line: { type: 'none' } });
    txt(s, r[0], { x: x + 0.24, y: y + 0.24, w: 0.4, h: 0.4, fontSize: 14, bold: true, color: C.PAPER, align: 'center', valign: 'middle', fontFace: F.MATH });
    txt(s, r[1], { x: x + 0.76, y: y + 0.24, w: 2.85, h: 0.4, fontSize: 14.5, bold: true, color: C.TXT, valign: 'middle' });
    txt(s, r[2], { x: x + 0.24, y: y + 0.76, w: 3.4, h: 0.5, fontSize: 11.5, color: C.MUTED, valign: 'top', lineSpacing: 15 });
  });
  txt(s, '橘＝進模型之前　·　藍＝模型內部　·　紅＝梯度與陷阱　·　綠＝更新與驗證', { x: M, y: 6.72, w: CW, h: 0.32, fontSize: 11.5, color: C.MUTED });
  s.addNotes('九站。時間分配：1-2 各 5 分鐘，3-5 是主體約 25 分鐘，6-7 約 15 分鐘，8-9 收尾 10 分鐘。');

  // ══ 5. 章節 1 ════════════════════════════════════════════════
  sectionSlide(p, 1, '讀懂公式的最小前置', '只有一個觀念要先建立：兩種「乘法」的差別',
    ['⊙ 逐元素相乘', '@ 矩陣乘法', '為什麼這決定了模型的學習能力', '符號速查表']);

  // ══ 6. ⊙ vs @ ════════════════════════════════════════════════
  s = contentSlide(p, 1, '⊙ 與 @：兩個都念「乘」，但完全是兩回事', '混淆這兩個，後面每一條公式都會讀錯');
  formula(s, p, M, 1.62, 5.9, 0.62, 'A = [ 1  2 ; 3  4 ]        B = [ 5  6 ; 7  8 ]', { size: 16, bg: C.WASH });
  formula(s, p, 6.82, 1.62, 5.91, 0.62, '同樣的輸入，兩種算法', { size: 16, bg: C.WASH, color: C.MUTED });

  // 左：⊙
  card(s, p, M, 2.44, 5.9, 3.05, C.AMBER_L, { radius: 0.1 });
  txt(s, '⊙　逐元素相乘（Hadamard 哈達馬乘積）', { x: M + 0.3, y: 2.62, w: 5.3, h: 0.4, fontSize: 17, bold: true, color: C.AMBER_D, valign: 'middle' });
  rich(s, math('(A ⊙ B)_{ij} = A_{ij} × B_{ij}', { fontSize: 16, fontFace: F.MATH, color: C.TXT, bold: true }), { x: M + 0.3, y: 3.06, w: 5.3, h: 0.36, valign: 'middle' });
  formula(s, p, M + 0.3, 3.5, 5.3, 0.92, '[ 1×5   2×6 ;   3×7   4×8 ]  =  [ 5   12 ;   21   32 ]', { size: 15, bg: C.PAPER, color: C.TXT });
  txt(s, '每一格只跟「同一格」的對手相乘，跟其他格完全無關。形狀不變。', { x: M + 0.3, y: 4.52, w: 5.3, h: 0.72, fontSize: 12.5, color: C.TXT, valign: 'top', lineSpacing: 17 });

  // 右：@
  card(s, p, 6.82, 2.44, 5.91, 3.05, C.TEAL_L, { radius: 0.1 });
  txt(s, '@　矩陣乘法', { x: 7.12, y: 2.62, w: 5.3, h: 0.4, fontSize: 17, bold: true, color: C.TEAL, valign: 'middle' });
  rich(s, math('(A @ B)_{ij} = Σ_{k} A_{ik} × B_{kj}', { fontSize: 16, fontFace: F.MATH, color: C.TXT, bold: true }), { x: 7.12, y: 3.06, w: 5.3, h: 0.36, valign: 'middle' });
  formula(s, p, 7.12, 3.5, 5.31, 0.92, '[ 1×5+2×7   1×6+2×8 ;   3×5+4×7   3×6+4×8 ]  =  [ 19   22 ;   43   50 ]', { size: 13, bg: C.PAPER, color: C.TXT });
  txt(s, '左邊的「列」跟右邊的「行」做點積 —— 用到了一整列和一整行，也就是所有維度。', { x: 7.12, y: 4.52, w: 5.31, h: 0.72, fontSize: 12.5, color: C.TXT, valign: 'top', lineSpacing: 17 });

  // 對照條
  card(s, p, M, 5.66, CW, 1.06, C.INK, { radius: 0.09 });
  txt(s, '同一個左上角', { x: M + 0.34, y: 5.66, w: 1.9, h: 1.06, fontSize: 13, bold: true, color: '8D99A6', valign: 'middle' });
  stat(s, '5', 'A ⊙ B　＝ 1 × 5', { x: M + 2.4, y: 5.76, w: 2.5, size: 34, bh: 0.62, color: C.AMBER, lsize: 11.5 });
  stat(s, '19', 'A @ B　＝ 1 × 5 + 2 × 7', { x: M + 5.2, y: 5.76, w: 3.2, size: 34, bh: 0.62, color: '5FB3D4', lsize: 11.5 });
  txt(s, '可執行對照：python tools/demo_operators.py', { x: 9.2, y: 5.66, w: 3.3, h: 1.06, fontSize: 11, color: '76828F', fontFace: F.CODE, align: 'right', valign: 'middle' });
  s.addNotes('這是全場唯一「純數學」的一張，但只要 2 分鐘。重點不是算式，是左上角 5 vs 19 這個對比：同樣兩個矩陣，兩種乘法給出完全不同的答案。下一張講為什麼這個差別很重要。');

  // ══ 7. 會不會混合維度 ═════════════════════════════════════════
  s = contentSlide(p, 1, '真正的差別：會不會把不同維度混在一起', '這一件事決定了它們在模型裡的角色');
  // 左圖：⊙ 平行
  card(s, p, M, 1.66, 5.9, 3.4, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, '⊙　每條線都是平行的', { x: M + 0.3, y: 1.84, w: 5.3, h: 0.36, fontSize: 15, bold: true, color: C.AMBER_D, valign: 'middle' });
  ['維度 1', '維度 2', '維度 3'].forEach((d, i) => {
    const y = 2.42 + i * 0.68;
    box(s, p, M + 0.55, y, 1.7, 0.5, d, { fill: C.AMBER_L, size: 11.5, color: C.AMBER_D });
    arrow(s, p, M + 2.35, y + 0.25, 1.2, 0, { color: C.AMBER, width: 1.4 });
    box(s, p, M + 3.65, y, 1.7, 0.5, d, { fill: C.AMBER_L, size: 11.5, color: C.AMBER_D });
  });
  txt(s, '資訊不會橫向流動', { x: M + 0.3, y: 4.56, w: 5.3, h: 0.32, fontSize: 12, color: C.MUTED, align: 'center' });
  // 右圖：@ 全連接
  card(s, p, 6.82, 1.66, 5.91, 3.4, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, '@　每個輸出都吃進所有輸入', { x: 7.12, y: 1.84, w: 5.3, h: 0.36, fontSize: 15, bold: true, color: C.TEAL, valign: 'middle' });
  const srcY = [2.42, 3.10, 3.78], dstY = [2.76, 3.44];
  srcY.forEach((y, i) => box(s, p, 7.38, y, 1.7, 0.5, '維度 ' + (i + 1), { fill: C.TEAL_L, size: 11.5, color: C.TEAL }));
  dstY.forEach((y, j) => box(s, p, 10.55, y, 1.75, 0.5, '新維度 ' + (j + 1), { fill: C.TEAL_L, size: 11.5, color: C.TEAL }));
  srcY.forEach((y) => dstY.forEach((yy) => {
    const y1 = y + 0.25, y2 = yy + 0.25;
    arrow(s, p, 9.18, Math.min(y1, y2), 1.27, Math.abs(y2 - y1), { color: C.TEAL, width: 0.9, flipV: y2 < y1 });
  }));
  txt(s, '所有維度都被混在一起', { x: 7.12, y: 4.56, w: 5.31, h: 0.32, fontSize: 12, color: C.MUTED, align: 'center' });

  card(s, p, M, 5.24, CW, 1.5, C.INK, { radius: 0.1 });
  rich(s, [
    { text: '⊙ 永遠不會把不同維度的資訊混在一起。', options: { bold: true, color: C.AMBER, fontSize: 19, breakLine: true } },
    { text: '所以模型的「學習能力」幾乎全部來自 @，⊙ 只負責調整強度 —— 這也解釋了為什麼「疊很多個 ⊙」沒有用：不管乘幾次，維度 1 永遠只影響維度 1。', options: { color: 'C3CCD5', fontSize: 13.5 } },
  ], { x: M + 0.36, y: 5.24, w: CW - 0.72, h: 1.5, valign: 'middle', lineSpacing: 22 });
  s.addNotes('這張是本段的重點。程式設計師的類比：⊙ 像對陣列做 map，@ 像做 reduce 之後再重組。map 再多次也不會讓元素之間交換資訊。');

  // ══ 8. 各自出現在哪 ═══════════════════════════════════════════
  s = contentSlide(p, 1, '所以它們在模型裡的分工是固定的', '看到符號就知道那一步在做什麼');
  table(s, [
    [hdr('符號'), hdr('出現在哪一層'), hdr('公式'), hdr('在做什麼')],
    [cell('@', { bold: true, fontSize: 15, color: C.TEAL, align: 'center' }), cell('Linear（全連接）'), cell('y = x @ Wᵀ', { fontFace: F.CODE }), cell('把 d 維換成另外 d_out 維')],
    [cell('@', { bold: true, fontSize: 15, color: C.TEAL, align: 'center' }), cell('Attention'), cell('Q @ Kᵀ', { fontFace: F.CODE }), cell('每個位置跟每個位置做點積')],
    [cell('@', { bold: true, fontSize: 15, color: C.TEAL, align: 'center' }), cell('輸出層'), cell('h @ Eᵀ', { fontFace: F.CODE }), cell('隱藏狀態跟每個詞向量做點積')],
    [cell('⊙', { bold: true, fontSize: 15, color: C.AMBER_D, align: 'center' }), cell('RMSNorm'), cell('y = g ⊙ x̂', { fontFace: F.CODE }), cell('每個維度乘上自己的縮放係數')],
    [cell('⊙', { bold: true, fontSize: 15, color: C.AMBER_D, align: 'center' }), cell('SwiGLU（FFN）'), cell('m = s ⊙ u', { fontFace: F.CODE }), cell('閘門：每個維度各自決定放行多少')],
    [cell('⊙', { bold: true, fontSize: 15, color: C.AMBER_D, align: 'center' }), cell('RoPE（位置編碼）'), cell('x ⊙ cos', { fontFace: F.CODE }), cell('每個維度乘上自己的旋轉係數')],
  ], { x: M, y: 1.62, w: CW, colW: [1.0, 2.7, 2.4, 5.99], rowH: 0.42, fontSize: 12.5 });

  txt(s, '反向傳播的規則也跟著不一樣', { x: M, y: 4.68, w: 6, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 5.12, 5.9, 1.62, C.AMBER_L, { radius: 0.1 });
  rich(s, math('y = a ⊙ b\n∂L/∂a = ∂L/∂y ⊙ b', { fontSize: 15, fontFace: F.MATH, color: C.TXT, bold: true }), { x: M + 0.32, y: 5.26, w: 5.3, h: 0.76, valign: 'top', lineSpacing: 20 });
  txt(s, '一個輸入只影響一個輸出 → 不用加總，直接乘', { x: M + 0.32, y: 6.08, w: 5.3, h: 0.5, fontSize: 12, color: C.AMBER_D, bold: true, valign: 'top', lineSpacing: 16 });
  card(s, p, 6.82, 5.12, 5.91, 1.62, C.TEAL_L, { radius: 0.1 });
  rich(s, math('y = x @ Wᵀ\n∂L/∂x = ∂L/∂y @ W　　∂L/∂W = (∂L/∂y)ᵀ @ x', { fontSize: 15, fontFace: F.MATH, color: C.TXT, bold: true }), { x: 7.14, y: 5.26, w: 5.3, h: 0.76, valign: 'top', lineSpacing: 20 });
  txt(s, '一個輸入影響多個輸出 → 要把所有路徑加起來', { x: 7.14, y: 6.08, w: 5.3, h: 0.5, fontSize: 12, color: C.TEAL, bold: true, valign: 'top', lineSpacing: 16 });
  s.addNotes('不用背公式。記憶法在下一張：用形狀反推，梯度的形狀永遠跟被微分的東西一樣，通常只有一種排法接得起來。');

  // ══ 9. 符號速查 ══════════════════════════════════════════════
  s = contentSlide(p, 1, '符號速查：只要記這些就夠了', '後面所有公式都只用到這張表');
  txt(s, '形狀與維度', { x: M, y: 1.6, w: 6, h: 0.34, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('符號'), hdr('意思'), hdr('本專案的值')],
    [cell('B', { fontFace: F.MATH, bold: true }), cell('batch size，一次處理幾段文字'), cell('64', { align: 'right', fontFace: F.CODE })],
    [cell('T', { fontFace: F.MATH, bold: true }), cell('序列長度，一段文字幾個 token'), cell('128', { align: 'right', fontFace: F.CODE })],
    [cell('d', { fontFace: F.MATH, bold: true }), cell('模型維度（hidden size）'), cell('256', { align: 'right', fontFace: F.CODE })],
    [cell('h', { fontFace: F.MATH, bold: true }), cell('attention head 數'), cell('4', { align: 'right', fontFace: F.CODE })],
    [cell('d_head', { fontFace: F.MATH, bold: true }), cell('每個 head 的維度 ＝ d / h'), cell('64', { align: 'right', fontFace: F.CODE })],
    [cell('d_ff', { fontFace: F.MATH, bold: true }), cell('FFN 的中間維度'), cell('672', { align: 'right', fontFace: F.CODE })],
    [cell('|V|', { fontFace: F.MATH, bold: true }), cell('詞表大小'), cell('6,400', { align: 'right', fontFace: F.CODE })],
  ], { x: M, y: 2.0, w: 5.9, colW: [1.15, 3.55, 1.2], rowH: 0.365, fontSize: 12 });
  txt(s, '張量形狀寫成 (B, T, d)，最後一維永遠是「特徵維度」。', { x: M, y: 4.98, w: 5.9, h: 0.34, fontSize: 11.5, color: C.MUTED });

  txt(s, '微分符號（白話版）', { x: 6.82, y: 1.6, w: 6, h: 0.34, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  const dsym = [
    ['∂L/∂x', '把 x 動一丁點，L 會變多少', C.TEAL],
    ['∂L/∂x（粗體 x）', '對整個向量微分，結果跟 x 同形狀', C.TEAL],
    ['‖x‖', '向量長度 √( Σ x_{k}² )', C.MUTED],
    ['δ_{ij}', 'i = j 時為 1，否則為 0', C.MUTED],
    ['θ ← θ − η g', '「更新」，不是等式 —— 用右邊的值取代左邊', C.RED],
  ];
  dsym.forEach((d, i) => {
    const y = 2.0 + i * 0.72;
    card(s, p, 6.82, y, 5.91, 0.62, i === 4 ? C.RED_L : C.WASH, { radius: 0.07 });
    txt(s, d[0], { x: 7.06, y, w: 2.1, h: 0.62, fontSize: 14, bold: true, color: d[2], fontFace: F.MATH, valign: 'middle' });
    txt(s, d[1], { x: 9.2, y, w: 3.35, h: 0.62, fontSize: 11.5, color: C.TXT, valign: 'middle', lineSpacing: 15 });
  });
  txt(s, '程式碼裡為了簡潔，∂L/∂x 一律寫成 dx。', { x: 6.82, y: 5.68, w: 5.91, h: 0.34, fontSize: 11.5, color: C.MUTED });
  s.addNotes('不要唸完整張表。只強調兩件事：(1) ∂L/∂x 的白話就是「動一丁點會變多少」(2) ← 是賦值不是等式，θ = θ − ηg 在數學上是無解方程式。其他讓聽眾之後回頭查。');
};
