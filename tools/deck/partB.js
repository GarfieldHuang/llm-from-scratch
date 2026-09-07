const L = require('./lib.js');
const { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell } = L;

module.exports = function partB(p) {
  let s;

  // ══ 章節 2 ═══════════════════════════════════════════════════
  sectionSlide(p, 2, '文字怎麼變成數字', 'tokenizer —— 整條流程裡唯一沒有神經網路的階段',
    ['BPE 到底在做什麼', '「6,400」這個數字的由來', '為什麼它同時是模型的架構參數', '換 tokenizer 之後 loss 不能直接比']);

  // ══ tokenizer 是模型外的獨立階段 ═══════════════════════════════
  s = contentSlide(p, 2, 'tokenizer 是模型「外面」的一個階段', '它在訓練開始前就跑完，之後整個訓練過程都不會再變動');
  const flow = [
    ['原始語料', C.WASH, C.TXT], ['預切分\n切成詞塊', C.WASH, C.TXT],
    ['起始詞表\n＝所有相異字元', C.AMBER_L, C.AMBER_D], ['統計相鄰配對次數', C.AMBER_L, C.AMBER_D],
    ['合併最高頻那組\n詞表 +1', C.AMBER, C.INK], ['詞表定案\nvocab_size = 6,400', C.GREEN_L, C.GREEN],
  ];
  flow.forEach((f, i) => {
    const x = M + i * 2.04;
    box(s, p, x, 1.72, 1.72, 0.92, f[0], { fill: f[1], color: f[2], size: 11, ls: 14 });
    if (i < 5) arrow(s, p, x + 1.72, 2.18, 0.32, 0, { color: C.MUTED, width: 1.4 });
  });
  // 迴圈箭頭：合併 → 回到統計
  arrow(s, p, 9.22, 2.64, 0, 0.44, { color: C.AMBER_D, width: 1.4, dash: true, head: false });
  arrow(s, p, 7.18, 3.08, 2.04, 0, { color: C.AMBER_D, width: 1.4, dash: true, flipH: true, head: false });
  arrow(s, p, 7.18, 2.64, 0, 0.44, { color: C.AMBER_D, width: 1.4, dash: true, flipV: true });
  txt(s, '還沒到 6,400 就再合併一次', { x: 7.3, y: 3.12, w: 3.0, h: 0.3, fontSize: 10.5, color: C.AMBER_D, bold: true });

  card(s, p, M, 3.66, CW, 1.02, C.INK, { radius: 0.09 });
  rich(s, [
    { text: '這裡沒有任何學習。', options: { bold: true, color: C.AMBER, fontSize: 19 } },
    { text: '　沒有梯度、沒有神經網路、沒有 loss —— 純粹是貪婪的頻率統計。', options: { color: 'C3CCD5', fontSize: 15 } },
  ], { x: M + 0.36, y: 3.66, w: CW - 0.72, h: 1.02, valign: 'middle' });

  txt(s, '但它決定了模型的四件事', { x: M, y: 4.94, w: 6, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  const dec = [
    ['詞表大小 |V|', '直接等於 embedding 的列數'],
    ['embedding 參數量', '|V| · d ＝ 6,400 × 256 ＝ 1.6M'],
    ['logits 張量大小', 'B × T × |V| ，訓練時最大的中間張量'],
    ['實際序列長度', 'BPE 比字元級短約 30%，而 attention 是 O(n²)'],
  ];
  dec.forEach((d, i) => {
    const x = M + i * 3.09;
    card(s, p, x, 5.36, 2.86, 1.28, C.WASH, { radius: 0.08 });
    txt(s, d[0], { x: x + 0.22, y: 5.5, w: 2.45, h: 0.36, fontSize: 13.5, bold: true, color: C.AMBER_D, valign: 'middle' });
    txt(s, d[1], { x: x + 0.22, y: 5.88, w: 2.45, h: 0.66, fontSize: 11, color: C.TXT, valign: 'top', lineSpacing: 14 });
  });
  s.addNotes('程式設計師的類比：tokenizer 像是編譯器的 lexer，是一個獨立的前處理階段，跑完就固定。它沒有「訓練」的意義上的學習，只是統計。但它產生的常數會硬編進模型的架構裡。');

  // ══ 6400 哪來的 ══════════════════════════════════════════════
  s = contentSlide(p, 2, '「詞表 6,400 個 token，這個數字是哪來的？」', '不是算出來的 —— 是你自己挑的停止條件');
  formula(s, p, M, 1.66, CW, 0.92, '|V|  =  |V_{0}|  +  M          起始符號數  ＋  合併次數', { size: 22 });
  txt(s, 'BPE 每合併一次，詞表就多一個。想要 |V| = 6,400，就合併到總數變成 6,400 為止。',
    { x: M, y: 2.72, w: CW, h: 0.36, fontSize: 14, color: C.TXT, valign: 'middle' });

  card(s, p, M, 3.2, 5.9, 1.66, C.INK, { radius: 0.1 });
  txt(s, '實測（tests/test_bpe.py）', { x: M + 0.32, y: 3.34, w: 5.3, h: 0.3, fontSize: 11.5, color: '8D99A6' });
  stat(s, '5,227', '起始詞表\n4 特殊 ＋ 5,223 字元', { x: M + 0.32, y: 3.66, w: 1.75, size: 30, bh: 0.56, color: C.PAPER, lsize: 10.5 });
  txt(s, '＋', { x: M + 2.05, y: 3.66, w: 0.44, h: 0.56, fontSize: 22, bold: true, color: C.AMBER, align: 'center', valign: 'middle', fontFace: F.MATH });
  stat(s, '1,173', '需要合併\n（次數）', { x: M + 2.5, y: 3.66, w: 1.55, size: 30, bh: 0.56, color: C.PAPER, lsize: 10.5 });
  txt(s, '＝', { x: M + 4.0, y: 3.66, w: 0.44, h: 0.56, fontSize: 22, bold: true, color: C.AMBER, align: 'center', valign: 'middle', fontFace: F.MATH });
  stat(s, '6,400', '最終詞表', { x: M + 4.44, y: 3.66, w: 1.4, size: 30, bh: 0.56, color: C.AMBER, lsize: 10.5 });

  card(s, p, 6.82, 3.2, 5.91, 1.66, C.WASH, { radius: 0.1 });
  txt(s, '合併紀錄長什麼樣（traces/bpe_training.txt）', { x: 7.1, y: 3.34, w: 5.4, h: 0.3, fontSize: 11.5, bold: true, color: C.MUTED });
  txt(s, "合併 #1      '可'+'以' → '可以'    23,763 次\n合併 #1000   '生'+'產' → '生產'         393 次",
    { x: 7.1, y: 3.72, w: 5.4, h: 0.9, fontSize: 12, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 21 });

  card(s, p, M, 5.06, CW, 1.62, C.AMBER_L, { radius: 0.1 });
  txt(s, '一個很實用的判斷', { x: M + 0.34, y: 5.22, w: 4.4, h: 0.34, fontSize: 15, bold: true, color: C.AMBER_D, valign: 'middle' });
  rich(s, [
    { text: '注意合併次數一路遞減：23,763 → 393。', options: { bold: true, color: C.TXT, fontSize: 13.5, breakLine: true } },
    { text: '如果最後幾次合併已經降到個位數，代表詞表開太大了 —— 那些 token 學不到什麼東西。反過來，如果停下來時次數還有好幾百，代表還可以再擴。', options: { color: C.TXT, fontSize: 13 } },
  ], { x: M + 0.34, y: 5.6, w: CW - 0.68, h: 0.94, valign: 'top', lineSpacing: 19 });
  s.addNotes('這是最常被問的問題，答案卻極其樸素：它就是個 while 迴圈的停止條件。5227 + 1173 = 6400，投影出來讓大家自己加一次。順帶給出「合併次數遞減」這個實用的調參直覺。');

  // ══ 詞表大小＝架構參數 ═══════════════════════════════════════
  s = contentSlide(p, 2, '詞表大小不只是 tokenizer 的參數', '它同時是模型的架構參數 —— 改它會改變模型的形狀');
  txt(s, '切分效果（同一份語料、同一個模型）', { x: M, y: 1.6, w: 6.5, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  card(s, p, M, 2.0, 7.4, 1.5, C.WASH, { radius: 0.09 });
  txt(s, '原文', { x: M + 0.26, y: 2.14, w: 0.8, h: 0.3, fontSize: 11.5, bold: true, color: C.MUTED, valign: 'middle' });
  txt(s, '機器學習是人工智慧的一個分支', { x: M + 1.1, y: 2.14, w: 6.0, h: 0.3, fontSize: 13, color: C.TXT, valign: 'middle' });
  txt(s, '字元級', { x: M + 0.26, y: 2.5, w: 0.8, h: 0.3, fontSize: 11.5, bold: true, color: C.MUTED, valign: 'middle' });
  txt(s, "['機','器','學','習','是','人','工','智','慧','的','一','個','分','支']    14 tokens",
    { x: M + 1.1, y: 2.5, w: 6.0, h: 0.3, fontSize: 10.5, color: C.TXT, fontFace: F.CODE, valign: 'middle' });
  txt(s, 'BPE', { x: M + 0.26, y: 2.9, w: 0.8, h: 0.3, fontSize: 11.5, bold: true, color: C.AMBER_D, valign: 'middle' });
  txt(s, "['機器學習', '是', '人', '工智慧', '的一個', '分', '支']    7 tokens",
    { x: M + 1.1, y: 2.9, w: 6.0, h: 0.3, fontSize: 10.5, color: C.AMBER_D, bold: true, fontFace: F.CODE, valign: 'middle' });

  card(s, p, 8.32, 2.0, 4.41, 1.5, C.AMBER_L, { radius: 0.09 });
  stat(s, '−29.4%', '序列長度\n（500 段實測）', { x: 8.6, y: 2.16, w: 1.9, size: 27, bh: 0.52, lsize: 10.5 });
  txt(s, 'Attention 是 O(n²)\n序列短一半，注意力\n計算量剩四分之一', { x: 10.6, y: 2.2, w: 1.95, h: 1.1, fontSize: 11, color: C.TXT, valign: 'top', lineSpacing: 15 });

  txt(s, "「機器學習」變成一個 token —— 模型不必再從四個字元學會它們常常一起出現，那件事在 tokenizer 階段就處理掉了。",
    { x: M, y: 3.62, w: CW, h: 0.34, fontSize: 12.5, color: C.MUTED, valign: 'middle' });

  card(s, p, M, 4.16, 5.9, 2.5, C.PAPER, { line: C.TEAL, lineW: 1.25, radius: 0.1 });
  txt(s, '詞表 → embedding 參數量', { x: M + 0.3, y: 4.32, w: 5.3, h: 0.34, fontSize: 15, bold: true, color: C.TEAL, valign: 'middle' });
  formula(s, p, M + 0.3, 4.72, 5.3, 0.6, '|θ_{embed}|  =  |V| × d', { size: 17, bg: C.TEAL_L, color: C.TEAL });
  txt(s, '詞表開兩倍，embedding 就大兩倍。而且因為 weight tying，同一個矩陣還兼任輸出層 —— 所以 logits 張量也跟著變大。',
    { x: M + 0.3, y: 5.44, w: 5.3, h: 0.9, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });

  card(s, p, 6.82, 4.16, 5.91, 2.5, C.RED_L, { radius: 0.1 });
  txt(s, '注意：換 tokenizer 之後，loss 不能直接比', { x: 7.12, y: 4.32, w: 5.3, h: 0.34, fontSize: 15, bold: true, color: C.RED, valign: 'middle' });
  txt(s, '字元級 3,195 詞表 → loss 2.24\nBPE 6,400 詞表 → loss 2.76', { x: 7.12, y: 4.72, w: 5.3, h: 0.62, fontSize: 12.5, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 18 });
  txt(s, '後者看起來比較差，但每個 token 攜帶的資訊量更大、猜對更難。要公平比較得換算成 bits-per-character：',
    { x: 7.12, y: 5.4, w: 5.3, h: 0.62, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 15 });
  formula(s, p, 7.12, 6.02, 5.31, 0.5, 'bpc = L · N_{token} / ( N_{char} · ln2 )', { size: 14, bg: C.PAPER, color: C.RED });
  s.addNotes('重點：詞表大小是「架構參數」，不是「前處理參數」。以及那個 loss 比較的陷阱——很多人換了 tokenizer 看到 loss 變高就以為模型退步了，其實是單位變了。');

  // ══ 章節 3 ═══════════════════════════════════════════════════
  sectionSlide(p, 3, 'Embedding 與輸出層', '同一張表 E，進來時用來「讀」，出去時用來「寫」',
    ['查表為什麼是矩陣乘法', '輸出層：從 256 維回到 6,400 個字', 'z / p / y / L 四個符號', '反向為什麼是相加（scatter-add）', '「離散的 id 怎麼微分？」']);

  // ══ 誤解① ════════════════════════════════════════════════════
  s = contentSlide(p, 3, '誤解 ①「embedding 是資料，不是模型的一部分」', '錯。它是權重，跟 q_proj 完全平起平坐。');
  card(s, p, M, 1.66, 6.2, 2.72, C.INK, { radius: 0.1 });
  txt(s, '$ python train.py --steps 1 --trace-steps 1', { x: M + 0.3, y: 1.82, w: 5.6, h: 0.3, fontSize: 11.5, color: C.AMBER, fontFace: F.CODE });
  txt(s, 'model.embed.E          1,638,400\nblock0.attn.q_proj.W      65,536\nblock0.ffn.gate_proj.W   172,032\n...\noptimizer 拿到 38 個參數張量',
    { x: M + 0.3, y: 2.24, w: 5.6, h: 1.5, fontSize: 12, color: 'C3CCD5', fontFace: F.CODE, valign: 'top', lineSpacing: 20 });
  s.addShape(p.ShapeType.roundRect, { x: M + 4.3, y: 2.22, w: 1.85, h: 0.32, rectRadius: 0.06, fill: { color: C.AMBER }, line: { type: 'none' } });
  txt(s, '← 就是它', { x: M + 4.3, y: 2.22, w: 1.85, h: 0.32, fontSize: 11, bold: true, color: C.INK, align: 'center', valign: 'middle' });

  card(s, p, 7.12, 1.66, 5.61, 2.72, C.WASH, { radius: 0.1 });
  txt(s, '追蹤檔裡兩者的更新', { x: 7.42, y: 1.82, w: 5.0, h: 0.3, fontSize: 12.5, bold: true, color: C.MUTED, valign: 'middle' });
  txt(s, '位移 E[...]      −0.000500  +0.000500  +0.000500\n位移 q_proj      +0.000500  +0.000500  −0.000500',
    { x: 7.42, y: 2.28, w: 5.0, h: 0.8, fontSize: 11, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 20 });
  card(s, p, 7.42, 3.16, 5.0, 0.98, C.AMBER_L, { radius: 0.08 });
  txt(s, '同一個 optimizer、同一套規則、同樣的 ±lr。\n「embedding 有一套自己的訓練方法」這件事不存在。',
    { x: 7.62, y: 3.16, w: 4.6, h: 0.98, fontSize: 12.5, bold: true, color: C.AMBER_D, valign: 'middle', lineSpacing: 18 });

  txt(s, '換個框架就通了 —— E 就是「第 0 層」', { x: M, y: 4.62, w: 8, h: 0.36, fontSize: 16, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr(''), hdr('輸入'), hdr('權重'), hdr('輸出')],
    [cell('第 0 層（embedding）', { bold: true, fill: { color: C.AMBER_L } }), cell('one-hot（固定資料）', { fill: { color: C.AMBER_L } }), cell('E', { bold: true, fontFace: F.MATH, color: C.AMBER_D, fill: { color: C.AMBER_L } }), cell('詞向量', { fill: { color: C.AMBER_L } })],
    [cell('第 1 層', { bold: true }), cell('詞向量'), cell('w_q, w_k, w_v, FFN', { fontFace: F.CODE }), cell('新的向量')],
  ], { x: M, y: 5.04, w: CW, colW: [3.1, 3.0, 3.0, 3.0], rowH: 0.44, fontSize: 12.5 });
  txt(s, '第 1 層的「輸入」就是第 0 層的「輸出」。梯度傳到第 1 層的輸入時不會停在那裡 —— 它繼續往下，變成 E 的梯度。',
    { x: M, y: 6.44, w: CW, h: 0.34, fontSize: 13, color: C.MUTED, valign: 'middle' });
  s.addNotes('這是全場最重要的一張之一。用「第 0 層」的框架去重述，聽眾通常當場就通了。可以問現場：有多少人以為 embedding 是先用 word2vec 之類的東西「預先算好」的？其實在 LLM 裡它就是隨機初始化然後一起訓練。');

  // ══ 查表＝矩陣乘法 ═══════════════════════════════════════════
  s = contentSlide(p, 3, '查表，其實是一次 one-hot 矩陣乘法', '把它看成矩陣乘法很重要 —— 因為反向就是從這裡推出來的');
  formula(s, p, M, 1.62, CW, 0.72, 'x = onehot(v) @ E ,          E ∈ ℝ^{|V| × d}', { size: 19 });
  // 圖示
  card(s, p, M, 2.56, CW, 2.16, C.WASH, { radius: 0.1 });
  txt(s, 'onehot(4)', { x: M + 0.35, y: 2.74, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: C.MUTED });
  [0, 0, 0, 0, 1, 0, 0].forEach((v, i) => {
    box(s, p, M + 0.35 + i * 0.44, 3.1, 0.4, 0.4, String(v), {
      fill: v ? C.AMBER : C.PAPER, color: v ? C.INK : C.MUTED, line: v ? C.AMBER_D : C.RULE, size: 12, face: F.CODE,
    });
  });
  txt(s, '第 4 個是 1，其餘全 0', { x: M + 0.35, y: 3.58, w: 3.2, h: 0.3, fontSize: 10.5, color: C.MUTED });
  txt(s, '@', { x: M + 3.65, y: 3.1, w: 0.5, h: 0.4, fontSize: 20, bold: true, color: C.TEAL, align: 'center', valign: 'middle', fontFace: F.MATH });
  // E 矩陣
  txt(s, 'E　（|V| × d）', { x: M + 4.35, y: 2.74, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: C.MUTED });
  for (let r = 0; r < 6; r++) {
    const hit = r === 4;
    box(s, p, M + 4.35, 2.98 + r * 0.28, 2.5, 0.26, hit ? 'E[4]  =  −0.0153   +0.0284  …' : '', {
      fill: hit ? C.AMBER_L : C.PAPER, line: hit ? C.AMBER_D : C.RULE, color: C.AMBER_D, size: 9, face: F.CODE, radius: 0.03,
    });
  }
  arrow(s, p, M + 7.05, 3.28, 0.7, 0, { color: C.AMBER_D, width: 1.6 });
  txt(s, '結果 x', { x: M + 7.9, y: 2.74, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: C.MUTED });
  box(s, p, M + 7.9, 3.1, 3.5, 0.4, '−0.0153   +0.0284   +0.0111   …   （256 維）', { fill: C.AMBER_L, color: C.AMBER_D, size: 10.5, face: F.CODE, line: C.AMBER_D });
  txt(s, '就是把 E 的第 4 列原封不動抄出來', { x: M + 7.9, y: 3.58, w: 3.6, h: 0.3, fontSize: 10.5, color: C.MUTED });

  card(s, p, M, 4.96, 6.2, 1.72, C.INK, { radius: 0.1 });
  txt(s, '那實作為什麼寫成 x = E[ids]？', { x: M + 0.32, y: 5.12, w: 5.6, h: 0.32, fontSize: 14, bold: true, color: C.AMBER, valign: 'middle' });
  txt(s, '因為 (|V|−1)/|V| = 99.98% 的乘法都是「乘以 0」。兩者結果逐位元相同，實測查表快 20 倍以上、省 8 倍記憶體。',
    { x: M + 0.32, y: 5.5, w: 5.6, h: 0.96, fontSize: 12.5, color: 'C3CCD5', valign: 'top', lineSpacing: 18 });
  card(s, p, 7.12, 4.96, 5.61, 1.72, C.TEAL_L, { radius: 0.1 });
  txt(s, '所以前向沒有任何「學習」動作', { x: 7.42, y: 5.12, w: 5.0, h: 0.32, fontSize: 14, bold: true, color: C.TEAL, valign: 'middle' });
  txt(s, '就是陣列索引。導數是 1 —— 這也預告了反向：它不會對梯度做任何變換，只負責把梯度送回正確的那一列。',
    { x: 7.42, y: 5.5, w: 5.0, h: 0.96, fontSize: 12.5, color: C.TXT, valign: 'top', lineSpacing: 18 });
  s.addNotes('對工程師：「查表就是 O(1) 索引，為什麼要繞一圈講成矩陣乘法？」答案是——反向傳播的推導需要它。你不寫成 X = O·E，就推不出 scatter-add，只能把它當成一條要背的特殊規則。');

  // ══ 輸出層 h @ Eᵀ（新增）══════════════════════════════════════
  s = contentSlide(p, 3, '輸出層：從 256 維回到 6,400 個字', 'E 的第二個用途 —— 進來時「id → 向量」，出去時「向量 → 每個 id 的分數」');
  card(s, p, M, 1.62, 5.9, 1.0, C.WASH, { radius: 0.1 });
  txt(s, 'h', { x: M + 0.3, y: 1.62, w: 0.5, h: 1.0, fontSize: 26, bold: true, color: C.TEAL, fontFace: F.MATH, valign: 'middle' });
  txt(s, '256 個數字', { x: M + 0.9, y: 1.74, w: 4.7, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  txt(s, '模型想說的話，但還沒翻譯成詞表的語言', { x: M + 0.9, y: 2.08, w: 4.7, h: 0.34, fontSize: 12, color: C.MUTED, valign: 'middle' });
  card(s, p, 6.82, 1.62, 5.91, 1.0, C.RED_L, { radius: 0.1 });
  txt(s, '|V|', { x: 7.12, y: 1.62, w: 0.7, h: 1.0, fontSize: 22, bold: true, color: C.RED, fontFace: F.MATH, valign: 'middle' });
  txt(s, '6,400 個 token', { x: 7.92, y: 1.74, w: 4.5, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  txt(s, '數量對不上 —— 所以 h 不可能是「對每個詞的分數」', { x: 7.92, y: 2.08, w: 4.5, h: 0.34, fontSize: 12, color: C.RED, valign: 'middle' });

  formula(s, p, M, 2.78, CW, 0.72, 'logits[t][v]  =  h[t] · E[v]          隱藏狀態 · 詞向量', { size: 19 });

  box(s, p, M, 3.86, 1.4, 0.86, 'h\n(256 維)', { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 11.5, ls: 14 });
  arrow(s, p, 2.02, 4.29, 0.4, 0, { color: C.MUTED, width: 1.4 });
  card(s, p, 2.42, 3.76, 6.4, 1.06, C.WASH, { radius: 0.08 });
  txt(s, '跟詞表裡每一個詞向量各做一次點積', { x: 2.64, y: 3.84, w: 6.0, h: 0.26, fontSize: 11, bold: true, color: C.MUTED });
  txt(s, 'h · E[0] = −0.05929      h · E[1] = +0.48077      h · E[2] = −0.21864\nh · E[3] = −0.18226      h · E[4] = +0.63529      ⋯  共 6,400 次',
    { x: 2.64, y: 4.1, w: 6.0, h: 0.62, fontSize: 10, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 15 });
  arrow(s, p, 8.82, 4.29, 0.4, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, 9.22, 3.86, 3.51, 0.86, 'z\n6,400 個分數', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 11.5, ls: 14 });

  table(s, [
    [hdr(''), hdr('方向'), hdr('E 被拿來做什麼')],
    [cell('輸入端'), cell('id → 向量'), cell('查表，抄出 E[5868] 那一列')],
    [cell('輸出端'), cell('向量 → 分數'), cell('跟每個 E[v] 做點積，得到 6,400 個分數')],
  ], { x: M, y: 5.3, w: 7.4, colW: [1.5, 2.2, 3.7], rowH: 0.38, fontSize: 12 });
  txt(s, '同一張表，進來時用來「讀」，出去時用來「寫」。這是一個選擇（weight tying）—— 不共用的話這裡要多 1,638,400 個參數。',
    { x: M, y: 6.5, w: 7.4, h: 0.34, fontSize: 10.5, color: C.MUTED, valign: 'middle' });

  card(s, p, 8.22, 5.3, 4.51, 1.42, C.AMBER_L, { radius: 0.1 });
  txt(s, '注意：點積不是「距離」', { x: 8.46, y: 5.42, w: 4.05, h: 0.3, fontSize: 13.5, bold: true, color: C.AMBER_D, valign: 'middle' });
  rich(s, math('h · E[v] = |h| × |E[v]| × cos θ', { fontSize: 12.5, fontFace: F.MATH, bold: true, color: C.TXT }),
    { x: 8.46, y: 5.74, w: 4.05, h: 0.3, valign: 'middle' });
  txt(s, '長度也算分 —— 詞向量夠長，角度沒那麼準也可能贏。高頻詞可以靠養長向量拿到基礎加分。',
    { x: 8.46, y: 6.06, w: 4.05, h: 0.6, fontSize: 10.5, color: C.TXT, valign: 'top', lineSpacing: 14 });
  s.addNotes('這張回答「h 不就已經是分數了嗎」。關鍵在形狀：256 ≠ 6400，一定要有東西把它換過去。順便鋪好 weight tying——下一張的兩條梯度路徑就是因為 E 在這裡被用了第二次。「點積不是距離」是給有數學背景的人的防呆，講一句帶過就好。');

  // ══ 輸出端的四個符號（新增）══════════════════════════════════
  s = contentSlide(p, 3, '輸出端的四個符號：z → p → y → L', '全景圖上那個 dz = p − y，符號都在這裡');
  const sym4 = [
    ['z', '6,400 個原始分數', C.AMBER_D, C.AMBER_L],
    ['p', '6,400 個機率，和 = 1', C.TEAL, C.TEAL_L],
    ['p[正解]', '取出一格 → 一個數字', C.GREEN, C.GREEN_L],
    ['L', '一個純量', C.RED, C.RED_L],
  ];
  sym4.forEach((c, i) => {
    const x = M + i * 3.2;
    box(s, p, x, 1.72, 2.5, 0.9, c[0] + '\n' + c[1], { fill: c[3], line: c[2], color: c[2], size: 12, ls: 16 });
    if (i < 3) arrow(s, p, x + 2.5, 2.17, 0.7, 0, { color: C.MUTED, width: 1.4 });
  });
  ['softmax', '取正解那格', '−log'].forEach((t, i) => {
    txt(s, t, { x: M + 2.4 + i * 3.2, y: 2.62, w: 1.9, h: 0.28, fontSize: 10, color: C.MUTED, align: 'center' });
  });

  table(s, [
    [hdr('符號'), hdr('是什麼'), hdr('形狀'), hdr('第 1 步的實測值（正解 = id 5868）')],
    [cell('z', { fontFace: F.MATH, bold: true, color: C.AMBER_D }), cell('輸出層算出的原始分數，可正可負'), cell('6,400', { align: 'right', fontFace: F.CODE }), cell('−0.05929  +0.48077  −0.21864  ⋯', { fontFace: F.CODE })],
    [cell('p', { fontFace: F.MATH, bold: true, color: C.TEAL }), cell('softmax(z)，機率，全部加起來 = 1'), cell('6,400', { align: 'right', fontFace: F.CODE }), cell('最高的是 id 4656 → 0.00048', { fontFace: F.CODE })],
    [cell('y', { fontFace: F.MATH, bold: true, color: C.MUTED }), cell('正解的 one-hot，只有一格是 1'), cell('6,400', { align: 'right', fontFace: F.CODE }), cell('第 5868 格 = 1，其餘 6,399 格 = 0', { fontFace: F.CODE })],
    [cell('L', { fontFace: F.MATH, bold: true, color: C.RED }), cell('−log p[正解]', { bold: true }), cell('1', { align: 'right', fontFace: F.CODE }), cell('−log(0.000265) = 8.2359', { fontFace: F.CODE, bold: true })],
  ], { x: M, y: 3.12, w: CW, colW: [1.1, 4.3, 1.4, 5.29], rowH: 0.4, fontSize: 12 });

  card(s, p, M, 5.24, 5.9, 1.5, C.WASH, { radius: 0.1 });
  txt(s, '為什麼只看正解那一格？', { x: M + 0.3, y: 5.36, w: 5.3, h: 0.3, fontSize: 13.5, bold: true, color: C.TXT, valign: 'middle' });
  rich(s, math('L = −Σ_{v} y_{v} log p_{v}   ⟹   其他 6,399 項都乘以 0', { fontSize: 12.5, fontFace: F.MATH, bold: true, color: C.TXT }),
    { x: M + 0.3, y: 5.68, w: 5.3, h: 0.32, valign: 'middle' });
  txt(s, '不是偷懶只看一格，是化簡後的結果。而且 softmax 已經把 6,400 格綁死（和 = 1），推高正解就等於壓低其他 —— 一個數字足以約束整張表。',
    { x: M + 0.3, y: 6.02, w: 5.3, h: 0.6, fontSize: 11, color: C.TXT, valign: 'top', lineSpacing: 15 });

  txt(s, '−log 把「機率」翻譯成「驚訝程度」', { x: 6.82, y: 5.24, w: 5.91, h: 0.32, fontSize: 13.5, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('p[正解]'), hdr('L'), hdr('')],
    [cell('1.0', { align: 'right', fontFace: F.CODE }), cell('0', { align: 'right', fontFace: F.CODE }), cell('完全正確，不罰')],
    [cell('1 / 6400', { align: 'right', fontFace: F.CODE }), cell('8.76', { align: 'right', fontFace: F.CODE }), cell('等於瞎猜')],
    [cell('0.000265', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.AMBER_L } }), cell('8.24', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.AMBER_L } }), cell('第 1 步的實測', { bold: true, fill: { color: C.AMBER_L } })],
    [cell('→ 0', { align: 'right', fontFace: F.CODE }), cell('→ ∞', { align: 'right', fontFace: F.CODE, color: C.RED, bold: true }), cell('自信地猜錯，無上限重罰', { color: C.RED })],
  ], { x: 6.82, y: 5.62, w: 5.91, h: 1.1, colW: [1.5, 1.1, 3.31], rowH: 0.26, fontSize: 11 });
  s.addNotes('這張把全景圖上沒解釋的符號一次補齊。最容易卡的是 y——聽眾會以為 y 是「正確答案的 id」（一個整數），但它是 6,400 維的 one-hot，不然 p − y 這個減法不成立。最後那格「自信地猜錯無上限重罰」會在第 9 段的 tokenizer 綁定再出現一次。');

  // ══ scatter-add ══════════════════════════════════════════════
  s = contentSlide(p, 3, '反向：scatter-add', '前向 gather，反向 scatter-add —— 這個對稱不是設計出來的，是導數為 1 的必然結果');
  formula(s, p, M, 1.62, CW, 0.68, '∂L/∂E  =  Oᵀ G          展開後：   ∂L/∂E_{v}  =  Σ_{t : id_t = v}  G_{t}', { size: 18 });
  txt(s, '白話：把每個位置的梯度，加回它當初查表的那一列；同一列有多筆就相加。',
    { x: M, y: 2.42, w: CW, h: 0.34, fontSize: 14, bold: true, color: C.AMBER_D, valign: 'middle' });

  card(s, p, M, 2.9, 6.2, 2.3, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, '前向　gather', { x: M + 0.3, y: 3.04, w: 3, h: 0.3, fontSize: 13, bold: true, color: C.TXT, valign: 'middle' });
  box(s, p, M + 0.3, 3.44, 1.9, 1.4, 'E\n詞表', { fill: C.WASH, line: C.RULE, size: 12 });
  arrow(s, p, M + 2.35, 4.14, 1.4, 0, { color: C.AMBER_D, width: 1.6 });
  txt(s, '照 id 抄', { x: M + 2.3, y: 3.82, w: 1.5, h: 0.28, fontSize: 10.5, color: C.AMBER_D, align: 'center' });
  box(s, p, M + 3.9, 3.44, 2.0, 1.4, '位置 0…T\n的向量', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 12 });

  card(s, p, 7.12, 2.9, 5.61, 2.3, C.PAPER, { line: C.RULE, radius: 0.1 });
  txt(s, '反向　scatter-add', { x: 7.42, y: 3.04, w: 3, h: 0.3, fontSize: 13, bold: true, color: C.TEAL, valign: 'middle' });
  box(s, p, 7.42, 3.44, 2.0, 1.4, '每個位置\n的梯度', { fill: C.TEAL_L, line: C.TEAL, color: C.TEAL, size: 12 });
  arrow(s, p, 9.57, 4.14, 1.4, 0, { color: C.TEAL, width: 1.6 });
  txt(s, '加回原本那列\n重複的相加', { x: 9.42, y: 3.66, w: 1.7, h: 0.44, fontSize: 10, color: C.TEAL, align: 'center', lineSpacing: 12 });
  box(s, p, 11.12, 3.44, 1.31, 1.4, 'dE', { fill: C.WASH, line: C.RULE, size: 13, face: F.MATH });

  txt(s, '兩個常被誤會成「特殊規則」的性質，其實都只是算出來的結果', { x: M, y: 5.42, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('現象'), hdr('為什麼')],
    [cell('同一個 token 出現 k 次 → 那一列收到 k 筆梯度相加'), cell('Oᵀ 那一列有 k 個 1，矩陣乘法自然就加起來了')],
    [cell('沒出現過的 token → 梯度是 0'), cell('Oᵀ 那一列全是 0')],
  ], { x: M, y: 5.82, w: CW, colW: [5.5, 6.59], rowH: 0.36, fontSize: 12.5 });
  s.addNotes('強調「不是特殊規則」。很多教材會說「embedding 的反向是特殊的 scatter-add」，好像是額外加的機制。其實把它寫成 X = O·E，套最普通的線性層公式 ∂L/∂W = (∂L/∂y)ᵀ x，展開就是它。');

  // ══ 兩條梯度路徑 ═════════════════════════════════════════════
  s = contentSlide(p, 3, '但 E 的梯度其實有兩條路', '因為 weight tying —— E 同時是輸出層的權重');
  // 圖
  card(s, p, M, 1.62, CW, 1.9, C.WASH, { radius: 0.1 });
  box(s, p, M + 0.4, 2.06, 1.9, 0.9, 'E\n（同一個張量）', { fill: C.INK, color: C.PAPER, size: 12, ls: 15 });
  arrow(s, p, M + 2.45, 2.32, 1.35, 0, { color: C.AMBER_D, width: 1.6 });
  txt(s, '輸入側：查表', { x: M + 2.4, y: 2.02, w: 1.55, h: 0.26, fontSize: 10, color: C.AMBER_D, align: 'center' });
  arrow(s, p, M + 2.45, 2.76, 1.35, 0, { color: C.TEAL, width: 1.6 });
  txt(s, '輸出側：logits = h @ Eᵀ', { x: M + 2.2, y: 2.86, w: 2.0, h: 0.26, fontSize: 10, color: C.TEAL, align: 'center' });
  box(s, p, M + 3.95, 2.06, 2.3, 0.5, 'scatter-add 回來', { fill: C.AMBER_L, color: C.AMBER_D, size: 11 });
  box(s, p, M + 3.95, 2.6, 2.3, 0.5, 'softmax 那條路回來', { fill: C.TEAL_L, color: C.TEAL, size: 11 });
  arrow(s, p, M + 6.4, 2.32, 0.7, 0.26, { color: C.MUTED, width: 1.4 });
  arrow(s, p, M + 6.4, 2.58, 0.7, 0.26, { color: C.MUTED, width: 1.4, flipV: true });
  box(s, p, M + 7.25, 2.32, 1.5, 0.5, '相加', { fill: C.INK, color: C.PAPER, size: 12 });
  arrow(s, p, M + 8.9, 2.57, 0.7, 0, { color: C.MUTED, width: 1.4 });
  box(s, p, M + 9.75, 2.32, 1.9, 0.5, '總梯度 dE', { fill: C.WASH, line: C.RULE, size: 12, face: F.MATH });

  table(s, [
    [hdr('來源'), hdr('∂L/∂E₄[0]'), hdr('有幾列非零'), hdr('稀疏還是稠密')],
    [cell('輸入側（scatter-add）'), cell('+0.01673', { align: 'right', fontFace: F.CODE }), cell('272 / 3,195', { align: 'right', fontFace: F.CODE }), cell('稀疏 —— 只有出現過的 token', { color: C.AMBER_D, bold: true })],
    [cell('輸出側（weight tying）'), cell('+0.00500', { align: 'right', fontFace: F.CODE }), cell('3,195 / 3,195', { align: 'right', fontFace: F.CODE }), cell('稠密 —— 每一步每一列都被碰到', { color: C.TEAL, bold: true })],
    [cell('合計', { bold: true, fill: { color: C.WASH } }), cell('+0.02174', { align: 'right', bold: true, fontFace: F.CODE, fill: { color: C.WASH } }), cell('3,195 / 3,195', { align: 'right', bold: true, fontFace: F.CODE, fill: { color: C.WASH } }), cell('稠密', { bold: true, fill: { color: C.WASH } })],
  ], { x: M, y: 3.76, w: CW, colW: [3.1, 2.0, 2.2, 4.79], rowH: 0.42, fontSize: 12.5 });

  card(s, p, M, 5.6, CW, 1.12, C.RED_L, { radius: 0.1 });
  rich(s, [
    { text: '所以「embedding 的梯度是稀疏的」這句話，在有 weight tying 的模型上並不成立。', options: { bold: true, color: C.RED, fontSize: 16, breakLine: true } },
    { text: '想親眼驗證：把 TinyLM(..., tie_weights=False) 跑一次，非零列就會剩下 272。', options: { color: C.TXT, fontSize: 13 } },
  ], { x: M + 0.36, y: 5.6, w: CW - 0.72, h: 1.12, valign: 'middle', lineSpacing: 24 });
  s.addNotes('這是連老手都會踩的一個點。「embedding gradient is sparse」是很常見的說法，來自沒有 weight tying 的年代。現代 LLM 幾乎都 tie，所以那個最佳化假設不成立。');

  // ══ 誤解③ 離散 id ════════════════════════════════════════════
  s = contentSlide(p, 3, '誤解 ③「token id 是離散的，所以沒辦法微分」', '混淆了兩個東西 —— 不對 id 微分，它是資料不是參數');
  table(s, [
    [hdr(''), hdr('是什麼'), hdr('離散 / 連續'), hdr('微分嗎')],
    [cell('id = 4', { fontFace: F.CODE, bold: true }), cell('資料'), cell('離散'), cell('不用，也不能', { bold: true, color: C.RED })],
    [cell('E[4][0] = −0.0153', { fontFace: F.CODE, bold: true }), cell('參數'), cell('連續實數'), cell('對它微分', { bold: true, color: C.GREEN })],
  ], { x: M, y: 1.62, w: CW, colW: [3.4, 2.6, 2.6, 3.49], rowH: 0.44, fontSize: 13 });

  txt(s, '梯度問的是這件事，跟 id 是不是離散完全無關：', { x: M, y: 3.12, w: 8, h: 0.32, fontSize: 13.5, color: C.TXT, valign: 'middle' });
  formula(s, p, M, 3.5, 6.2, 0.78, '∂L/∂E_{4,0}  =  lim  [ L(E_{4,0}+ε) − L(E_{4,0}−ε) ] / 2ε', { size: 15 });
  card(s, p, 7.12, 3.5, 5.61, 0.78, C.WASH, { radius: 0.08 });
  rich(s, math('f(a,b,c) = a + b     ⟹     ∂f/∂c = 0', { fontSize: 15, fontFace: F.MATH, bold: true, color: C.TXT }), { x: 7.3, y: 3.5, w: 5.25, h: 0.78, align: 'center', valign: 'middle' });

  txt(s, '有限差分實測（ε = 10⁻³，句子含重複的「在」）', { x: M, y: 4.5, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('格子'), hdr('(L₊ − L₋) / 2ε'), hdr('公式算出來'), hdr('說明')],
    [cell('E₂,₀（在）', { fontFace: F.MATH }), cell('0.246048', { align: 'right', fontFace: F.CODE }), cell('0.246000', { align: 'right', fontFace: F.CODE }), cell('查兩次，斜率加倍')],
    [cell('E₀,₀（貓）', { fontFace: F.MATH }), cell('0.123024', { align: 'right', fontFace: F.CODE }), cell('0.123000', { align: 'right', fontFace: F.CODE }), cell('查一次')],
    [cell('E₁,₀（狗）', { fontFace: F.MATH, fill: { color: C.RED_L } }), cell('0.000000', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.RED_L } }), cell('0.000000', { align: 'right', fontFace: F.CODE, bold: true, fill: { color: C.RED_L } }), cell('沒被查到', { bold: true, color: C.RED, fill: { color: C.RED_L } })],
  ], { x: M, y: 4.9, w: CW, colW: [2.3, 2.6, 2.4, 4.79], rowH: 0.4, fontSize: 12.5 });
  rich(s, [
    { text: '「狗」那一列 L(w+ε) 和 L(w−ε) 完全相等 —— ', options: { color: C.TXT, fontSize: 13 } },
    { text: '不是「不能微分」，是「算出來就是 0」。', options: { bold: true, color: C.RED, fontSize: 13 } },
  ], { x: M, y: 6.56, w: CW, h: 0.34, valign: 'middle' });
  s.addNotes('這一題最能展現「變數 vs 常數」的分類。有限差分是最有說服力的示範：真的去把那個數字加減 ε，跑兩次 forward，看 loss 差多少。「狗」那列跑出來一模一樣，因為它根本沒參與計算。');
};
