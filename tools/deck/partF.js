const L = require('./lib.js');
const { C, F, SW, SH, M, CW, math, contentSlide, sectionSlide, card, formula, txt, rich, box, arrow, stat, table, hdr, cell } = L;

module.exports = function partF(p) {
  let s;

  // ══ 章節 8 ═══════════════════════════════════════════════════
  sectionSlide(p, 8, '怎麼確定沒推錯 · 跑一遍', '不用 autograd 對答案 —— 直接回到梯度的定義',
    ['有限差分驗證九層', 'loss 從 8.81 掉到 2.78', '生成出來的東西長什麼樣', '每一個數字都可以自己驗算']);

  // ══ 有限差分驗證 ═════════════════════════════════════════════
  s = contentSlide(p, 8, '不用 PyTorch 對答案，直接量斜率', '如果拿 autograd 當標準答案，你只是驗證「我抄對了」，不是「我推對了」');
  formula(s, p, M, 1.62, CW, 0.78, '∂L/∂w  ≈  [ L(w+ε) − L(w−ε) ] / 2ε', { size: 22 });

  card(s, p, M, 2.6, 6.2, 2.16, C.INK, { radius: 0.1 });
  txt(s, '# scratch/gradcheck.py（節錄）', { x: M + 0.3, y: 2.72, w: 5.6, h: 0.28, fontSize: 11, color: '76828F', fontFace: F.CODE });
  txt(s, 'y0 = layer.forward(x)\ndy = torch.randn(y0.shape)      # 隨機的上游梯度\ndx = layer.backward(dy)         # 手推公式算出來的\n\nflat[i] = orig + eps;  lp = (layer.forward(x) * dy).sum()\nflat[i] = orig − eps;  lm = (layer.forward(x) * dy).sum()\nnum = (lp − lm) / (2 * eps)     # 有限差分算出來的',
    { x: M + 0.3, y: 3.02, w: 5.6, h: 1.6, fontSize: 10, color: 'C3CCD5', fontFace: F.CODE, valign: 'top', lineSpacing: 15 });

  card(s, p, 7.12, 2.6, 5.61, 2.16, C.WASH, { radius: 0.1 });
  txt(s, '三個設計上的細節', { x: 7.42, y: 2.74, w: 5.0, h: 0.3, fontSize: 13.5, bold: true, color: C.TXT, valign: 'middle' });
  const gc = [
    ['隨機的 dy', 'backward 的正確性要求它對任何 dy 都成立 —— 這正是各層能自由組合的前提'],
    ['中央差分', '兩邊各推一次，誤差是 O(ε²) 而不是 O(ε)'],
    ['跑在 float64', 'float32 只有約 7 位有效數字，兩個相近的數相減會被捨入誤差吃掉'],
  ];
  gc.forEach((g, i) => {
    const y = 3.08 + i * 0.56;
    txt(s, '·', { x: 7.42, y, w: 0.2, h: 0.5, fontSize: 16, bold: true, color: C.AMBER_D, valign: 'top' });
    txt(s, g[0], { x: 7.64, y, w: 1.35, h: 0.5, fontSize: 11.5, bold: true, color: C.AMBER_D, valign: 'top' });
    txt(s, g[1], { x: 9.02, y, w: 3.5, h: 0.5, fontSize: 10, color: C.TXT, valign: 'top', lineSpacing: 13 });
  });

  const gr = [
    ['Linear（無 bias）', '2.29e−12'], ['Linear（有 bias）', '2.03e−12'], ['RMSNorm', '2.13e−07'],
    ['Embedding (scatter-add)', '6.80e−14'], ['TiedOutputHead', '5.61e−12'], ['SwiGLU (FFN)', '1.69e−06'],
    ['RoPE', '2.19e−12'], ['CausalSelfAttention', '3.32e−07'], ['Softmax+CrossEntropy', '7.50e−08'],
  ];
  gr.forEach((g, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * 4.09, y = 5.06 + row * 0.5;
    card(s, p, x, y, 3.85, 0.42, C.GREEN_L, { radius: 0.06 });
    txt(s, g[0], { x: x + 0.18, y, w: 2.0, h: 0.42, fontSize: 10.5, color: C.TXT, valign: 'middle' });
    txt(s, g[1], { x: x + 2.2, y, w: 0.95, h: 0.42, fontSize: 10.5, color: C.TXT, fontFace: F.CODE, align: 'right', valign: 'middle' });
    txt(s, 'PASS', { x: x + 3.2, y, w: 0.5, h: 0.42, fontSize: 10, bold: true, color: C.GREEN, valign: 'middle' });
  });
  txt(s, '誤差大小的差異是有意義的：純矩陣運算的層在 10⁻¹² 等級；含開根號、除法、σ(1−σ) 的層在 10⁻⁶ ~ 10⁻⁷，因為浮點誤差會累積。',
    { x: M, y: 6.62, w: CW, h: 0.34, fontSize: 11.5, color: C.MUTED, valign: 'middle' });
  s.addNotes('這一段是「工程師的誠實」。有限差分很慢，但它是唯一不依賴任何框架的驗證方法。九層全過，而且誤差量級跟每層的運算複雜度對得起來——這比單純的 PASS 更有說服力。');

  // ══ loss 曲線 ════════════════════════════════════════════════
  s = contentSlide(p, 8, '實際跑一遍：loss 從 8.81 掉到 2.78', 'RTX PRO 6000 · 4 層 · 475 萬參數 · 3,000 步 · 79 秒');
  const steps = [1, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 1920, 2040, 2160, 2280, 2400, 2520, 2640, 2760, 2880, 3000];
  const losses = [8.8149, 6.0609, 5.1124, 4.5695, 4.2454, 3.9605, 3.8504, 3.7543, 3.6738, 3.5001, 3.3466, 3.4276, 3.1896, 3.2272, 3.2002, 3.2319, 3.1012, 3.1449, 3.0731, 3.0104, 2.8574, 2.9581, 2.8899, 2.8810, 2.8959, 2.7806];
  s.addChart(p.ChartType.line, [{ name: 'loss', labels: steps.map(String), values: losses }], {
    x: M, y: 1.6, w: 8.4, h: 3.5,
    showTitle: false, showLegend: false,
    chartColors: [C.AMBER_D], lineSize: 2.5, lineDataSymbol: 'none', lineSmooth: false,
    catAxisLabelColor: C.MUTED, valAxisLabelColor: C.MUTED,
    catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
    catAxisLabelFrequency: 5,
    valAxisMinVal: 2, valAxisMaxVal: 9, valAxisMajorUnit: 1,
    valGridLine: { color: C.RULE, size: 0.75 }, catGridLine: { style: 'none' },
    catAxisTitle: 'step', showCatAxisTitle: true, catAxisTitleColor: C.MUTED, catAxisTitleFontSize: 10,
  });

  const ms = [['8.8149', 'step 1'], ['3.6166', 'step 1000'], ['2.7806', 'step 3000'], ['79.4 秒', '總耗時']];
  ms.forEach((m, i) => {
    const y = 1.6 + i * 0.9;
    card(s, p, 9.42, y, 3.31, 0.78, i === 3 ? C.INK : C.WASH, { radius: 0.08 });
    txt(s, m[0], { x: 9.66, y, w: 1.85, h: 0.78, fontSize: 21, bold: true, color: i === 3 ? C.AMBER : C.AMBER_D, fontFace: F.MATH, valign: 'middle' });
    txt(s, m[1], { x: 11.5, y, w: 1.0, h: 0.78, fontSize: 11, color: i === 3 ? '8D99A6' : C.MUTED, align: 'right', valign: 'middle' });
  });

  const lz = [
    ['前 100 步是 warmup', 'lr 從 3e−05 爬到 3e−03，所以一開始掉得慢'],
    ['1500 步之後開始震盪', 'batch 之間的差異開始蓋過進步幅度 —— 這是正常的，不是壞掉'],
    ['最後仍在下降', '代表還沒訓練飽和。這個規模繼續跑還有空間'],
  ];
  lz.forEach((z, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 5.3, 3.85, 1.42, C.WASH, { radius: 0.08 });
    txt(s, z[0], { x: x + 0.24, y: 5.44, w: 3.4, h: 0.34, fontSize: 13, bold: true, color: C.TXT, valign: 'middle' });
    txt(s, z[1], { x: x + 0.24, y: 5.8, w: 3.4, h: 0.76, fontSize: 11.5, color: C.MUTED, valign: 'top', lineSpacing: 15 });
  });
  s.addNotes('曲線後半的抖動很重要——很多人第一次看到會以為訓練壞了。那是 batch 之間的自然差異。整體趨勢還在下降，代表沒有訓練飽和。');

  // ══ 生成結果 ═════════════════════════════════════════════════
  s = contentSlide(p, 8, '它生出來的東西長什麼樣', 'python generate.py --prompt "機器學習是"');
  const gens = [
    ['機器學習是', '一項重要的領域，即將資料能夠從大量資料中學習和推理。機器學習可以用於學習、自然語言處理和自然語言生成等各種任務。', '語意通順、術語正確', C.GREEN, C.GREEN_L],
    ['台灣的天氣', '如今，在這裡有著令人無法抗衰老的環境和繁華性災難。這些災難事件可以用於對當地居民群落和探討。', '語法沒問題，但語意會飄', C.AMBER_D, C.AMBER_L],
    ['貓是一種', '常見的寵物，並且常常會受到許多人關注和感悟。狗通常很快地奔跑、游泳、跳傘⋯⋯', '看得出在模仿句式', C.AMBER_D, C.AMBER_L],
  ];
  gens.forEach((g, i) => {
    const y = 1.66 + i * 1.32;
    card(s, p, M, y, 9.2, 1.18, C.PAPER, { line: C.RULE, radius: 0.09 });
    s.addShape(p.ShapeType.roundRect, { x: M + 0.22, y: y + 0.18, w: 1.5, h: 0.34, rectRadius: 0.06, fill: { color: C.INK }, line: { type: 'none' } });
    txt(s, g[0], { x: M + 0.22, y: y + 0.18, w: 1.5, h: 0.34, fontSize: 11, bold: true, color: C.PAPER, align: 'center', valign: 'middle' });
    txt(s, g[1], { x: M + 1.88, y: y + 0.14, w: 7.1, h: 0.9, fontSize: 12.5, color: C.TXT, valign: 'top', lineSpacing: 19 });
    card(s, p, 10.0, y, 2.73, 1.18, g[4], { radius: 0.09 });
    txt(s, g[2], { x: 10.2, y, w: 2.35, h: 1.18, fontSize: 12, bold: true, color: g[3], valign: 'middle', lineSpacing: 17 });
  });

  card(s, p, M, 5.64, CW, 1.1, C.INK, { radius: 0.1 });
  rich(s, [
    { text: '這就是 475 萬參數、吃 424 萬 token 該有的水準。', options: { bold: true, color: C.AMBER, fontSize: 17, breakLine: true } },
    { text: '重點不是它寫得好不好，而是「它確實學到了東西」—— 第 1 步時正解的機率是 0.00026（幾乎等於 1/6400 的隨機猜測），第 3000 步明顯高出來。順帶一提：「機器學習」被 BPE 合併成單一 token，所以模型對它的處理特別穩。', options: { color: 'C3CCD5', fontSize: 12.5 } },
  ], { x: M + 0.4, y: 5.64, w: CW - 0.8, h: 1.1, valign: 'middle', lineSpacing: 22 });
  s.addNotes('誠實面對輸出品質。第一段很好是因為「機器學習」是單一 token；後兩段語法對、語意飄，這正是這個規模該有的表現。不要為了好看只放第一段。');

  // ══ 章節 9 ═══════════════════════════════════════════════════
  sectionSlide(p, 9, '最後一個坑', '權重跟 tokenizer 是綁死的 —— 而且比多數人以為的更嚴格',
    ['不只詞表大小要一樣', '每個 token 對應到哪個 id 都要相同', '用錯比沒訓練過還糟，而且不會報錯', '可以擴充，不能重建']);

  // ══ 綁定實測 ═════════════════════════════════════════════════
  s = contentSlide(p, 9, '拿別人的權重，tokenizer 也要一樣嗎？', '對，而且比多數人以為的更嚴格 —— 不只詞表大小，每個 token 對應到哪個 id 都必須相同');
  card(s, p, M, 1.66, CW, 1.1, C.INK, { radius: 0.1 });
  txt(s, "訓練時        id 4 = '的'   →  E[4] 學成了「的」的向量\n換 tokenizer 後   id 4 = '，'   →  模型拿「的」的向量去理解「，」",
    { x: M + 0.4, y: 1.66, w: CW - 0.8, h: 1.1, fontSize: 13, color: 'C3CCD5', fontFace: F.CODE, valign: 'middle', lineSpacing: 24 });

  txt(s, '而且因為 weight tying，E 同時是輸出層 —— 輸入查表跟輸出分類會一起錯亂。',
    { x: M, y: 2.9, w: CW, h: 0.34, fontSize: 13.5, bold: true, color: C.RED, valign: 'middle' });

  txt(s, '實測：同一個模型、同一句話，只是把 id 對應打亂', { x: M, y: 3.4, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  const pp = [
    ['正確的 tokenizer', '3.61', '37', C.GREEN, C.GREEN_L],
    ['完全沒訓練過的模型', '—', '6,400', C.MUTED, C.WASH],
    ['id 對應被打亂', '12.87', '389,971', C.RED, C.RED_L],
  ];
  pp.forEach((q, i) => {
    const x = M + i * 4.09;
    card(s, p, x, 3.82, 3.85, 1.86, q[4], { radius: 0.1 });
    txt(s, q[0], { x: x + 0.26, y: 3.96, w: 3.35, h: 0.32, fontSize: 13, bold: true, color: q[3], valign: 'middle' });
    txt(s, 'loss ' + q[1], { x: x + 0.26, y: 4.3, w: 3.35, h: 0.28, fontSize: 11.5, color: C.MUTED, valign: 'middle', fontFace: F.CODE });
    txt(s, q[2], { x: x + 0.26, y: 4.62, w: 3.35, h: 0.66, fontSize: 34, bold: true, color: q[3], fontFace: F.MATH, valign: 'middle' });
    txt(s, 'perplexity（等效於在幾個選項中亂猜）', { x: x + 0.26, y: 5.3, w: 3.35, h: 0.3, fontSize: 9.5, color: C.MUTED, valign: 'middle' });
  });

  card(s, p, M, 5.9, CW, 0.84, C.RED_L, { radius: 0.1 });
  rich(s, [
    { text: '用錯 tokenizer 比完全沒訓練過的模型還糟 61 倍。', options: { bold: true, color: C.RED, fontSize: 16 } },
    { text: '　為什麼會比隨機還差？因為隨機模型是均勻分布，至少「不確定」；而載錯 tokenizer 的模型會很有自信地猜錯 —— 把高機率押在錯誤的 token 上。', options: { color: C.TXT, fontSize: 12.5 } },
  ], { x: M + 0.36, y: 5.9, w: CW - 0.72, h: 0.84, valign: 'middle' });
  s.addNotes('「比隨機還差」這件事值得停下來講。均勻分布至少誠實地說「我不知道」；載錯 tokenizer 的模型是自信地猜錯，cross-entropy 會重罰這種行為。');

  // ══ 三種情況 ═════════════════════════════════════════════════
  s = contentSlide(p, 9, '三種情況，危險程度完全不同', '真正的坑是第二種 —— 它不會報錯');
  box(s, p, M, 2.6, 2.2, 0.7, '拿到別人的權重', { fill: C.WASH, line: C.RULE, size: 12 });
  arrow(s, p, 2.82, 2.95, 0.5, 0, { color: C.MUTED, width: 1.4 });
  s.addShape(p.ShapeType.diamond, { x: 3.32, y: 2.35, w: 2.0, h: 1.2, fill: { color: C.INK }, line: { type: 'none' } });
  txt(s, '詞表大小\n一樣嗎？', { x: 3.32, y: 2.35, w: 2.0, h: 1.2, fontSize: 11, bold: true, color: C.PAPER, align: 'center', valign: 'middle', lineSpacing: 14 });
  arrow(s, p, 4.32, 3.55, 0, 0.7, { color: C.MUTED, width: 1.4 });
  txt(s, '否', { x: 4.38, y: 3.68, w: 0.4, h: 0.3, fontSize: 11, bold: true, color: C.MUTED });
  box(s, p, 3.02, 4.25, 2.6, 0.8, 'RuntimeError\nsize mismatch', { fill: C.AMBER_L, line: C.AMBER_D, color: C.AMBER_D, size: 11, ls: 14 });
  arrow(s, p, 5.32, 2.95, 0.5, 0, { color: C.MUTED, width: 1.4 });
  txt(s, '是', { x: 5.42, y: 2.62, w: 0.4, h: 0.3, fontSize: 11, bold: true, color: C.MUTED });
  s.addShape(p.ShapeType.diamond, { x: 5.82, y: 2.35, w: 2.0, h: 1.2, fill: { color: C.INK }, line: { type: 'none' } });
  txt(s, 'id 對應\n一樣嗎？', { x: 5.82, y: 2.35, w: 2.0, h: 1.2, fontSize: 11, bold: true, color: C.PAPER, align: 'center', valign: 'middle', lineSpacing: 14 });
  arrow(s, p, 6.82, 3.55, 0, 0.7, { color: C.RED, width: 1.6 });
  txt(s, '否', { x: 6.88, y: 3.68, w: 0.4, h: 0.3, fontSize: 11, bold: true, color: C.RED });
  box(s, p, 5.42, 4.25, 2.8, 1.0, '載得進去 · 跑得動\n輸出是垃圾\n沒有任何警告', { fill: C.RED, color: C.PAPER, size: 11, ls: 14 });
  arrow(s, p, 7.82, 2.95, 0.5, 0, { color: C.GREEN, width: 1.4 });
  txt(s, '是', { x: 7.92, y: 2.62, w: 0.4, h: 0.3, fontSize: 11, bold: true, color: C.GREEN });
  box(s, p, 8.32, 2.6, 1.7, 0.7, '正常', { fill: C.GREEN_L, line: C.GREEN, color: C.GREEN, size: 13 });

  card(s, p, 10.4, 1.66, 2.33, 3.6, C.RED_L, { radius: 0.1 });
  txt(s, '為什麼\n第二種\n最危險', { x: 10.6, y: 1.84, w: 1.95, h: 1.0, fontSize: 16, bold: true, color: C.RED, valign: 'top', lineSpacing: 22 });
  txt(s, '你會以為模型壞了、資料有問題、超參數不對，花好幾天調參 —— 實際上只是 tokenizer 拿錯了。',
    { x: 10.6, y: 2.94, w: 1.95, h: 2.1, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 17 });

  table(s, [
    [hdr('情況'), hdr('會怎樣'), hdr('危險嗎')],
    [cell('詞表大小不同'), cell('RuntimeError: size mismatch', { fontFace: F.CODE }), cell('不危險 —— 馬上就知道出事了', { bold: true, color: C.AMBER_D })],
    [cell('大小相同、id 對應不同', { fill: { color: C.RED_L } }), cell('安然載入、正常執行、輸出垃圾', { fill: { color: C.RED_L } }), cell('非常危險 —— 沒有任何提示', { bold: true, color: C.RED, fill: { color: C.RED_L } })],
  ], { x: M, y: 5.56, w: CW, colW: [3.3, 4.3, 4.49], rowH: 0.42, fontSize: 12.5 });
  s.addNotes('「不會報錯的錯誤」是最貴的錯誤。這一段的價值在於它是可以直接帶走用的工程教訓，不限於 LLM。');

  // ══ 防呆 ═════════════════════════════════════════════════════
  s = contentSlide(p, 9, '防呆：指紋能「驗證」，但不能「尋找」', '這個差別看起來很小，實際上這份專案自己就踩過');
  card(s, p, M, 1.62, 6.2, 2.16, C.INK, { radius: 0.1 });
  txt(s, '# tokenizer/bpe.py', { x: M + 0.3, y: 1.74, w: 5.6, h: 0.26, fontSize: 10.5, color: '76828F', fontFace: F.CODE });
  txt(s, "def fingerprint(self):\n    s = '|'.join(self.itos) + \\\n        '|'.join('%s>%s' % m for m in self.merges)\n    return hashlib.sha256(\n        s.encode('utf-8')).hexdigest()[:16]",
    { x: M + 0.3, y: 2.02, w: 5.6, h: 1.2, fontSize: 10.5, color: 'C3CCD5', fontFace: F.CODE, valign: 'top', lineSpacing: 16 });
  txt(s, '指紋涵蓋 id 的順序（itos 是有序的），所以順序一變就會被抓到。', { x: M + 0.3, y: 3.3, w: 5.6, h: 0.34, fontSize: 11, color: C.AMBER, valign: 'middle' });

  card(s, p, 7.12, 1.62, 5.61, 2.16, C.RED_L, { radius: 0.1 });
  txt(s, '拿錯就直接擋下來', { x: 7.42, y: 1.76, w: 5.0, h: 0.3, fontSize: 13, bold: true, color: C.RED, valign: 'middle' });
  txt(s, 'ValueError: tokenizer 不符！\n  權重存檔時用的 : dcb6ef062b94a20e\n  現在載入的     : deadbeefdeadbeef\n  詞表大小可能一樣，但 id 對應不同——\n  模型會照跑，輸出卻是垃圾。',
    { x: 7.42, y: 2.1, w: 5.0, h: 1.5, fontSize: 10, color: C.TXT, fontFace: F.CODE, valign: 'top', lineSpacing: 15 });

  card(s, p, M, 3.96, CW, 1.0, C.WASH, { radius: 0.1 });
  rich(s, [
    { text: '但這份專案自己踩過的坑：', options: { bold: true, color: C.TXT, fontSize: 14 } },
    { text: '　generate.py 當初用「猜檔名」的方式找 tokenizer。平常沒事，因為 out/ 底下只有一份詞表 —— 但只要跑過一次 tests/test_bpe.py，就會多出一份 6,400 的 BPE 詞表，於是這行開始把 6,400 的詞表交給 3,195 的權重。指紋確實擋下來了，可是使用者收到的是「不符」，不是「對的那份在這裡」。', options: { color: C.TXT, fontSize: 12 } },
  ], { x: M + 0.36, y: 3.96, w: CW - 0.72, h: 1.0, valign: 'middle', lineSpacing: 17 });

  txt(s, '所以 checkpoint 要存的不只是指紋，還有 tokenizer 的「身分」', { x: M, y: 5.1, w: 9, h: 0.34, fontSize: 15, bold: true, color: C.TXT, valign: 'middle' });
  table(s, [
    [hdr('順序'), hdr('依據'), hdr('用在什麼情況')],
    [cell('①'), cell('checkpoint 記的 tokenizer_path', { fontFace: F.CODE }), cell('正常情況，直接讀，不用猜')],
    [cell('②'), cell('指紋比對候選詞表'), cell('舊 checkpoint 沒存路徑，或詞表被搬走了')],
    [cell('③'), cell('明確報錯，列出試過哪些、指紋各是多少'), cell('真的找不到 —— 講清楚，而不是丟 IndexError')],
  ], { x: M, y: 5.5, w: CW, colW: [1.0, 5.2, 5.89], rowH: 0.34, fontSize: 12 });
  txt(s, '教訓：「靠檔案系統的狀態去推斷模型的身分」是個很容易寫出來、又很難發現的錯 —— 它在乾淨環境下永遠正常，只在「跑過別的東西之後」才出錯。模型的身分應該由模型自己攜帶。',
    { x: M, y: 6.9, w: CW, h: 0.34, fontSize: 11, color: C.MUTED, valign: 'middle' });
  s.addNotes('這是全場最有「帶得走的工程教訓」味道的一張。「驗證」和「尋找」是兩件事——這個區分在任何有 artifact 依賴的系統裡都適用，不限於 LLM。');

  // ══ 擴充 vs 重建 ═════════════════════════════════════════════
  s = contentSlide(p, 9, '如果真的需要改 tokenizer', '只能往後擴充，不能重建');
  const opts = [
    ['可以：擴充（append）', '保留原有的 id 0…V−1 完全不動，只在後面接新的列。',
      "p[:old_vocab].copy_(old_E)      # 原有 id 原封不動\np[old_vocab:] = old_E.mean(0)   # 新列用平均值起步",
      '實測 6,400 → 6,450：同一句話 loss = 3.6130（原本 3.6129）\n幾乎沒有影響，因為原有的 id 對應沒被動到。', C.GREEN, C.GREEN_L],
    ['不行：重建（rebuild）', '用新語料重訓一個 tokenizer，即使詞表大小相同、甚至詞彙集合相同。',
      'BPE 的 id 順序取決於合併的順序，\n而合併順序取決於語料的統計。',
      '換語料、換種子、甚至換一個實作的 tie-breaking 規則，順序就會不一樣 —— 只要 id 分配順序不同，權重就報廢了。', C.RED, C.RED_L],
  ];
  opts.forEach((o, i) => {
    const x = M + i * 6.25;
    card(s, p, x, 1.66, 5.84, 3.5, o[5], { radius: 0.1 });
    txt(s, o[0], { x: x + 0.28, y: 1.82, w: 5.28, h: 0.34, fontSize: 15, bold: true, color: o[4], valign: 'middle' });
    txt(s, o[1], { x: x + 0.28, y: 2.2, w: 5.28, h: 0.5, fontSize: 12, color: C.TXT, valign: 'top', lineSpacing: 16 });
    card(s, p, x + 0.28, 2.76, 5.28, 0.86, C.PAPER, { radius: 0.07 });
    txt(s, o[2], { x: x + 0.44, y: 2.76, w: 4.96, h: 0.86, fontSize: 9.5, color: C.TXT, fontFace: F.CODE, valign: 'middle', lineSpacing: 15 });
    txt(s, o[3], { x: x + 0.28, y: 3.74, w: 5.28, h: 1.2, fontSize: 11.5, color: C.TXT, valign: 'top', lineSpacing: 16 });
  });

  card(s, p, M, 5.32, 6.2, 1.42, C.WASH, { radius: 0.1 });
  txt(s, '折衷：詞表移植（tokenizer transplant）', { x: M + 0.3, y: 5.44, w: 5.6, h: 0.3, fontSize: 13, bold: true, color: C.TXT, valign: 'middle' });
  txt(s, '對新詞表的每個 token：舊詞表裡有同樣字串就把舊向量搬過來，否則用子詞平均初始化。能救回一部分，但一定會掉品質。業界做跨語言適配時常用這招，但那是不得已，不是首選。',
    { x: M + 0.3, y: 5.76, w: 5.6, h: 0.86, fontSize: 11, color: C.TXT, valign: 'top', lineSpacing: 15 });

  card(s, p, 7.12, 5.32, 5.61, 1.42, C.INK, { radius: 0.1 });
  rich(s, [
    { text: '權重跟 tokenizer 是一組的。', options: { bold: true, color: C.AMBER, fontSize: 16, breakLine: true } },
    { text: '發布時要一起發布，換模型時要一起換。這也是為什麼 HuggingFace 上的模型 repo 一定會把 tokenizer.json 跟 model.safetensors 放在同一個目錄。', options: { color: 'C3CCD5', fontSize: 11.5 } },
  ], { x: 7.44, y: 5.32, w: 4.97, h: 1.42, valign: 'middle', lineSpacing: 22 });
  s.addNotes('新列的初始化有三種常見選擇：既有向量平均（最簡單）、小亂數（跟從頭訓練一致）、子詞平均（新 token 是「機器學習」就用「機」「器」「學」「習」四個舊向量的平均）。第三種通常最好。');

  // ══ 回顧 ═════════════════════════════════════════════════════
  s = contentSlide(p, '·', '回到開場的四個問題', '');
  const ans = [
    ['1', '詞表 6,400 是誰決定的？', '你自己。|V| = |V₀| + M，它就是個 while 迴圈的停止條件。5,227 + 1,173 = 6,400。而且它同時是模型的架構參數。', C.AMBER],
    ['2', 'embedding 是資料還是模型？', '是權重。它在 optimizer 的 38 個張量裡，跟 q_proj 用同一套規則、同樣的 ±lr。把它想成「第 0 層」就通了。', C.TEAL],
    ['3', '反向傳播會改權重嗎？', '不會。backward() 只算出「該往哪修」存進梯度表，opt.step() 才動手 —— 而且方向相反、步伐只有 lr 那麼小。', C.RED],
    ['4', 'token id 是整數，怎麼微分？', '不對 id 微分。id 是資料（常數），E[4][0] 才是參數。「狗」那列的有限差分算出來就是 0，不是「不能算」。', C.GREEN],
  ];
  ans.forEach((a, i) => {
    const y = 1.44 + i * 1.32;
    card(s, p, M, y, CW, 1.18, C.PAPER, { line: C.RULE, radius: 0.09 });
    s.addShape(p.ShapeType.roundRect, { x: M + 0.26, y: y + 0.3, w: 0.56, h: 0.56, rectRadius: 0.1, fill: { color: a[3] }, line: { type: 'none' } });
    txt(s, a[0], { x: M + 0.26, y: y + 0.3, w: 0.56, h: 0.56, fontSize: 20, bold: true, color: C.PAPER, align: 'center', valign: 'middle', fontFace: F.MATH });
    txt(s, a[1], { x: M + 1.02, y, w: 3.5, h: 1.18, fontSize: 14.5, bold: true, color: C.TXT, valign: 'middle', lineSpacing: 20 });
    txt(s, a[2], { x: M + 4.7, y, w: 7.3, h: 1.18, fontSize: 12.5, color: C.TXT, valign: 'middle', lineSpacing: 18 });
  });
  s.addNotes('一題一題收。可以先問聽眾，再翻答案。這四題如果都答對，這場的目的就達到了。');

  // ══ 自己跑一遍 ═══════════════════════════════════════════════
  s = contentSlide(p, '·', '自己跑一遍', '固定 seed（預設 42），所以數字可以完全重現');
  card(s, p, M, 1.62, 7.5, 3.5, C.INK, { radius: 0.1 });
  txt(s, 'rm -rf out traces\n\npython tests/test_layers.py\n\npython train.py --tokenizer bpe --vocab 6400 --steps 3000 \\\n                --dim 256 --layers 4 --heads 4 --seq 128 \\\n                --batch 64 --trace-steps 1,3000\n\npython generate.py --prompt "機器學習是"\n\npython tests/test_tokenizer_binding.py',
    { x: M + 0.34, y: 1.62, w: 6.9, h: 3.5, fontSize: 11.5, color: 'C3CCD5', fontFace: F.CODE, valign: 'middle', lineSpacing: 18 });

  const demos = [
    ['tools/demo_operators.py', '⊙ 與 @ 的差別，用最小的矩陣算給你看'],
    ['tools/demo_backward.py', '兩條反向規則的逐元素推導'],
    ['tests/test_layers.py', '逐層梯度驗證表 → traces/gradcheck.txt'],
    ['tests/test_bpe.py', 'BPE 合併過程 → traces/bpe_training.txt'],
  ];
  demos.forEach((d, i) => {
    const y = 1.62 + i * 0.9;
    card(s, p, 8.42, y, 4.31, 0.78, C.WASH, { radius: 0.07 });
    txt(s, d[0], { x: 8.66, y: y + 0.08, w: 3.85, h: 0.28, fontSize: 10, bold: true, color: C.AMBER_D, fontFace: F.CODE, valign: 'middle' });
    txt(s, d[1], { x: 8.66, y: y + 0.36, w: 3.85, h: 0.34, fontSize: 10, color: C.MUTED, valign: 'middle', lineSpacing: 13 });
  });

  card(s, p, M, 5.34, CW, 1.4, C.AMBER_L, { radius: 0.1 });
  txt(s, '如果只做一件事：打開 traces/train_step_0001.txt，只看這四行', { x: M + 0.36, y: 5.44, w: 8, h: 0.3, fontSize: 13.5, bold: true, color: C.AMBER_D, valign: 'middle' });
  const jump = [
    ['第 25 行', '查表就是索引', '這一批用到 2,036 個 token，詞表 6,400'],
    ['第 493 行', 'dz = (p − y)/N', '整條鏈唯一知道答案的地方'],
    ['第 851 行', 'scatter-add 兩條路', '輸入側稀疏 ＋ 輸出側稠密，符號還相反'],
    ['第 889 行', '位移恰好等於 lr', '梯度差 67 倍，位移全是 ±3e−05'],
  ];
  jump.forEach((j, i) => {
    const x = M + i * 3.05;
    txt(s, j[0], { x: x + 0.1, y: 5.8, w: 1.0, h: 0.26, fontSize: 10.5, bold: true, color: C.AMBER_D, fontFace: F.CODE });
    txt(s, j[1], { x: x + 1.15, y: 5.8, w: 1.8, h: 0.26, fontSize: 10.5, bold: true, color: C.TXT });
    txt(s, j[2], { x: x + 0.1, y: 6.08, w: 2.85, h: 0.5, fontSize: 9.5, color: C.MUTED, valign: 'top', lineSpacing: 12 });
  });
  s.addNotes('給聽眾一個明確的、五分鐘內可以做完的下一步。不要說「去 GitHub 看看」，要說「跑這一行、看這兩個行號」。');

  // ══ 結尾 ═════════════════════════════════════════════════════
  s = p.addSlide();
  s.background = { color: C.INK };
  txt(s, '整場只有一句話', { x: M + 0.1, y: 1.9, w: 11.5, h: 0.44, fontSize: 17, color: C.AMBER, valign: 'middle' });
  txt(s, '沒有魔法，只有\n一個迴圈跑幾千次', {
    x: M + 0.1, y: 2.5, w: 11.5, h: 1.9, fontSize: 46, bold: true, color: C.PAPER, valign: 'middle', lineSpacing: 62,
  });
  txt(s, '前向算出分數 → cross-entropy 說錯了多少 → 每一層機械式地把它翻譯成自己的修正單 → optimizer 往反方向走一小步。\n這條鏈上沒有任何一個地方「理解」語言。', {
    x: M + 0.1, y: 4.66, w: 10.6, h: 1.0, fontSize: 15, color: 'A9B4C0', valign: 'top', lineSpacing: 24,
  });
  const finalPipe = ['p − y', '逐層翻譯', 'scatter-add', 'AdamW', '重複 3,000 次'];
  finalPipe.forEach((t, i) => {
    const x = M + 0.14 + i * 2.3;
    box(s, p, x, 5.92, 2.0, 0.5, t, { fill: i === 0 ? C.RED : C.INK2, color: i === 0 ? C.PAPER : '8D99A6', size: 10.5, radius: 0.05 });
    if (i < 4) arrow(s, p, x + 2.0, 6.17, 0.3, 0, { color: '4A5764', width: 1 });
  });
  txt(s, 'llm-from-scratch　·　docs/ 00–11', { x: 9.0, y: 6.6, w: 3.7, h: 0.32, fontSize: 11, color: '76828F', align: 'right' });
  s.addNotes('收尾。把「模型不理解語言」這句話留在最後——對非 ML 背景的聽眾來說，這是最有價值的除魅。');
};
