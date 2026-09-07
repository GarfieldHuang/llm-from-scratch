const pptxgen = require('pptxgenjs');
const p = new pptxgen();

p.defineLayout({ name: 'W16x9', width: 13.333, height: 7.5 });
p.layout = 'W16x9';
p.author = 'llm-from-scratch';
p.title = '從零手刻一個 LLM';
p.subject = '給有程式經驗、但不熟機器學習的開發者';

require('./partA.js')(p);
require('./partB.js')(p);
require('./partC.js')(p);
require('./partD.js')(p);
require('./partE.js')(p);
require('./partF.js')(p);

p.writeFile({ fileName: 'llm-from-scratch-教學簡報.pptx' }).then((f) => console.log('written:', f));
