"""
把所有層組裝成一個完整的語言模型

------------------------------------------------------------------
一個 Block 的結構（Pre-LN，跟 LLaMA / minimind 一致）
------------------------------------------------------------------
    h = x + Attention( RMSNorm(x) )
    y = h + FFN( RMSNorm(h) )

注意順序：先 Norm，再 Attention，然後 FFN 排第三。
以及那兩個「+」——殘差連接（residual）。

殘差為什麼重要：
    前向 y = x + f(x)   →   反向 dx = dy + f'(dy)
    梯度有一條「直達」的路可以走（那個 dy），不會因為層數深就消失。
    沒有殘差的話，20 層以上基本訓不動。

「前向相加 → 反向也相加」是通則：一個張量被用了幾次，
它的梯度就是那幾條路的梯度總和。

------------------------------------------------------------------
整個模型
------------------------------------------------------------------
    ids  →  Embedding 查表
         →  Block × N
         →  最後一道 RMSNorm
         →  輸出層（跟 Embedding 共用權重）
         →  logits
"""
import torch

from .attention import CausalSelfAttention
from .embedding import Embedding, TiedOutputHead
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .tracer import NULL


class Block:
    def __init__(self, dim, n_heads, hidden, max_len, layer_id=0,
                 rope_base=10000.0, device='cpu', dtype=None):
        self.name = 'block%d' % layer_id
        self.norm1 = RMSNorm(dim, device=device, dtype=dtype, name=self.name + '.norm1')
        self.attn = CausalSelfAttention(dim, n_heads, max_len, rope_base,
                                        device, dtype, name=self.name + '.attn')
        self.norm2 = RMSNorm(dim, device=device, dtype=dtype, name=self.name + '.norm2')
        self.ffn = SwiGLU(dim, hidden, device=device, dtype=dtype, name=self.name + '.ffn')

    def forward(self, x, tr=NULL):
        if tr.enabled:
            tr.section('===== %s =====' % self.name)
            tr.formula('h = x + Attention( RMSNorm(x) )')
            tr.formula('y = h + FFN( RMSNorm(h) )')

        a = self.attn.forward(self.norm1.forward(x, tr), tr)
        h = x + a                                   # 殘差 ①

        f = self.ffn.forward(self.norm2.forward(h, tr), tr)
        y = h + f                                   # 殘差 ②

        if tr.enabled:
            tr.scalar('‖x‖   (進來)', float(x.norm()))
            tr.scalar('‖a‖   (attention 的貢獻)', float(a.norm()))
            tr.scalar('‖h‖   (第一個殘差之後)', float(h.norm()))
            tr.scalar('‖f‖   (FFN 的貢獻)', float(f.norm()))
            tr.scalar('‖y‖   (出去)', float(y.norm()))
            tr.note('每個子層只在主幹上「加一點東西」，不是整個取代掉。\n'
                    '這就是殘差網路的精神——主幹保持暢通，各層做增量修正。')
        return y

    def backward(self, dy, tr=NULL):
        if tr.enabled:
            tr.section('===== %s · backward =====' % self.name)
            tr.formula('前向 y = h + f(h)  →  反向 dh = dy + f\'(dy)', '殘差讓梯度多一條直達的路')

        df = self.ffn.backward(dy, tr)
        dh = dy + self.norm2.backward(df, tr)       # 殘差 ② 的反向：相加

        da = self.attn.backward(dh, tr)
        dx = dh + self.norm1.backward(da, tr)       # 殘差 ① 的反向：相加

        if tr.enabled:
            tr.scalar('‖dy‖  (上游進來)', float(dy.norm()))
            tr.scalar('‖dx‖  (往下傳出去)', float(dx.norm()))
        return dx

    def zero_grad(self):
        for m in (self.norm1, self.attn, self.norm2, self.ffn):
            m.zero_grad()

    def params(self):
        out = []
        for m in (self.norm1, self.attn, self.norm2, self.ffn):
            out += m.params()
        return out

    def numel(self):
        return sum(m.numel() for m in (self.norm1, self.attn, self.norm2, self.ffn))


class TinyLM:
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4,
                 hidden=None, max_len=256, rope_base=10000.0,
                 tie_weights=True, device='cpu', dtype=None):
        hidden = hidden or int(dim * 8 / 3 / 32) * 32 or dim * 2
        self.cfg = dict(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                        n_heads=n_heads, hidden=hidden, max_len=max_len,
                        rope_base=rope_base, tie_weights=tie_weights)

        self.embed = Embedding(vocab_size, dim, device=device, dtype=dtype, name='embed')
        self.blocks = [Block(dim, n_heads, hidden, max_len, i, rope_base, device, dtype)
                       for i in range(n_layers)]
        self.norm_f = RMSNorm(dim, device=device, dtype=dtype, name='final_norm')

        if tie_weights:
            self.head = TiedOutputHead(self.embed, name='lm_head(共用權重)')
        else:
            from .linear import Linear
            self.head = Linear(dim, vocab_size, device=device, dtype=dtype, name='lm_head')

        self._modules = [self.embed] + self.blocks + [self.norm_f, self.head]

    # ---------------------------------------------------------------
    def forward(self, ids, tr=NULL):
        if tr.enabled:
            tr.title('前向傳播', 'ids shape %s，%d 層，dim=%d'
                     % (tuple(ids.shape), len(self.blocks), self.cfg['dim']))
        x = self.embed.forward(ids, tr)
        for b in self.blocks:
            x = b.forward(x, tr)
        x = self.norm_f.forward(x, tr)
        logits = self.head.forward(x, tr)
        self._h = x
        return logits

    def backward(self, dlogits, tr=NULL):
        if tr.enabled:
            tr.title('反向傳播', '從 loss 一路回到 embedding')
        d = self.head.backward(dlogits, tr)
        d = self.norm_f.backward(d, tr)
        for b in reversed(self.blocks):
            d = b.backward(d, tr)
        self.embed.backward(d, tr)
        return None

    # ---------------------------------------------------------------
    def zero_grad(self):
        for m in self._modules:
            m.zero_grad()

    def params(self):
        out = []
        for m in self._modules:
            out += m.params()
        return out

    def numel(self):
        return sum(m.numel() for m in self._modules)

    def summary(self):
        lines = ['模型結構', '=' * 62]
        lines.append('  %-30s %14s' % ('embedding', '{:,}'.format(self.embed.numel())))
        for b in self.blocks:
            lines.append('  %-30s %14s' % (b.name, '{:,}'.format(b.numel())))
        lines.append('  %-30s %14s' % ('final_norm', '{:,}'.format(self.norm_f.numel())))
        if self.head.numel():
            lines.append('  %-30s %14s' % ('lm_head', '{:,}'.format(self.head.numel())))
        else:
            lines.append('  %-30s %14s' % ('lm_head (與 embedding 共用)', '0'))
        lines.append('  ' + '-' * 46)
        lines.append('  %-30s %14s' % ('總計', '{:,}'.format(self.numel())))
        lines.append('  %-30s %13.1f%%' % ('embedding 佔比',
                                           100 * self.embed.numel() / self.numel()))
        return '\n'.join(lines)

    # ---------------------------------------------------------------
    def state(self):
        return {name: p for p, _, name in self.params()}

    def save(self, path, tokenizer_fingerprint=None,
             tokenizer_path=None, tokenizer_kind=None):
        """
        權重跟 tokenizer 是綁死的，所以把 tokenizer 的身分一起存進去。
        見 tests/test_tokenizer_binding.py 的實驗：用錯 tokenizer 不會報錯，
        但 perplexity 會比完全沒訓練的模型還糟。

        存兩種資訊，各有各的用途：
          fingerprint  用來「驗證」——載入時比對，不合就擋下來。
          path / kind  用來「尋找」——下次要載入時，直接知道去哪拿，不用猜檔名。
        只存指紋是不夠的：指紋能告訴你手上這份不對，卻不能告訴你對的在哪。
        """
        torch.save({'cfg': self.cfg,
                    'tokenizer_fingerprint': tokenizer_fingerprint,
                    'tokenizer_path': tokenizer_path,
                    'tokenizer_kind': tokenizer_kind,
                    'params': {n: p.cpu() for n, p in self.state().items()}}, path)

    def load(self, path, device='cpu', tokenizer_fingerprint=None, strict_tokenizer=True):
        d = torch.load(path, map_location=device)

        saved = d.get('tokenizer_fingerprint')
        if saved and tokenizer_fingerprint and saved != tokenizer_fingerprint:
            nl = chr(10)
            msg = ('tokenizer 不符！' + nl +
                   '  權重存檔時用的 : ' + str(saved) + nl +
                   '  現在載入的     : ' + str(tokenizer_fingerprint) + nl +
                   '  詞表大小可能一樣，但 id 對應不同——模型會照跑，輸出卻是垃圾。' + nl +
                   '  要強制載入請傳 strict_tokenizer=False。')
            if strict_tokenizer:
                raise ValueError(msg)
            print('警告：' + msg)

        cur = self.state()
        for n, v in d['params'].items():
            if n in cur:
                cur[n].copy_(v.to(device))
        return self
