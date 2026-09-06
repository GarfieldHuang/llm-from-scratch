# llm-from-scratch

從零手刻一個語言模型。**每一條公式、每一個反向傳播，都自己寫**，不用任何深度學習框架的自動微分。

torch 在這裡只被當成「會用 GPU 的 NumPy」——只用它的張量與基本運算（加減乘除、矩陣乘法、索引），
不用 `nn.Module`、不用 `autograd`、不用 `loss.backward()`。

---

## 這份專案的規則

| 禁止 | 改成 |
|---|---|
| `nn.Linear` | 自己寫 `y = x @ W.T + b`，並手推 `dW`、`dx`、`db` |
| `F.softmax` | 自己寫 `exp(z − max z) / Σexp(...)`，並手推 Jacobian |
| `F.scaled_dot_product_attention` | 自己寫 Q/K/V、遮罩、softmax、加權和，全部手推反向 |
| `loss.backward()` | 每一層自己的 `backward()`，逐層串接 |
| `torch.optim.AdamW` | 自己寫動量、bias correction、decoupled weight decay |

**每一個手推的 backward，都用有限差分驗證過**：

```
∂L/∂w  ≈  ( L(w + ε) − L(w − ε) ) / 2ε
```

跑 `python tests/test_layers.py` 會逐層驗證，結果寫進 `traces/gradcheck.txt`。

---

## 快速開始

```bash
# 需要 python 3.9+ 與 torch（只用張量，不用 autograd）
pip install torch

# 1. 驗證所有手推的公式都正確
python tests/test_layers.py

# 2. 訓練（預設會完整追蹤第 1 步）
python train.py                          # 字元級 tokenizer
python train.py --tokenizer bpe --vocab 6400   # 手刻 BPE

# 3. 生成
python generate.py --prompt "貓是一種"

# 4. 看 BPE 詞表怎麼從 3653 長到 6400
python tests/test_bpe.py

# 5. 證明權重與 tokenizer 綁死
python tests/test_tokenizer_binding.py
```

詳細的教學文件在 [`docs/`](docs/)——markdown 與排版好的 PDF 放在同一個目錄。

```bash
python tools/build_pdf.py            # 合併成一份 PDF
python tools/build_pdf.py --split    # 每份文件各一個 PDF
```

---

## 每一步的數值都會寫成檔案

這是這份專案的重點。跑完之後 `traces/` 底下會有：

| 檔案 | 內容 |
|---|---|
| `gradcheck.txt` | 每一層手推公式 vs 有限差分的比對 |
| `model_summary.txt` | 模型結構、參數量、超參數 |
| `train_step_0001.txt` | **第 1 步的完整軌跡**：每一層的前向數值、每一層的反向數值、AdamW 的每個中間量 |
| `train_log.txt` | 每一步的 loss / lr / 梯度範數 |
| `inference.txt` | 一次推論的完整前向 |

追蹤檔長這樣——公式跟數字並排：

```
  embed · backward（scatter-add，輸入側）
----------------------------------------------------------------
    公式:  dE[v] += Σ_{t : id_t = v}  dx[t]
           等價於 dE += O^T @ dx，O 是 one-hot 矩陣
    「id 4」在這一批出現 27 次
    看第 0 維，這 27 筆修正單相加：
        位置 1      dx[0] = +1.09540625e-02
        位置 12     dx[0] = +5.40754059e-03
        ...
        這 27 筆相加 =               +1.67316496e-02

    這一列的梯度其實是兩條路加起來的：
                              來源      dE[4][0]     有幾列非零
      輸入側（本步 scatter-add）        +0.01673     272 / 3195
        輸出側（weight tying）        +0.00500    3195 / 3195
              合計 dE[4][0]        +0.02174    3195 / 3195
```

用 `--trace-steps 1,500,3000` 指定要完整追蹤哪幾步。預設只追蹤第 1 步，
因為反向傳播每一步都長得一樣，看一次就夠，全記會產生幾 GB 的檔案。

---

## 建議的閱讀順序

程式碼本身就是教材，每個檔案開頭都有完整的公式推導。

| 順序 | 檔案 | 學到什麼 |
|---|---|---|
| 1 | `scratch/linear.py` | `Ax + B`，以及反向的三條公式。所有其他層都是它的組合 |
| 2 | `scratch/embedding.py` | 查表其實是 one-hot 矩陣乘法；scatter-add 是它的反向 |
| 3 | `scratch/loss.py` | softmax + cross-entropy 合併後神奇地簡化成 `p − y` |
| 4 | `scratch/rmsnorm.py` | 一個需要用到商法則的反向推導 |
| 5 | `scratch/swiglu.py` | FFN，以及「前向分岔 → 反向相加」 |
| 6 | `scratch/rope.py` | 位置資訊為什麼可以用旋轉表示 |
| 7 | `scratch/attention.py` | 最複雜的一層，含 softmax 的完整 Jacobian |
| 8 | `scratch/model.py` | 殘差連接，以及它為什麼讓深層網路訓得動 |
| 9 | `scratch/adamw.py` | 梯度算完之後，權重到底怎麼被改的 |
| 10 | `tokenizer/bpe.py` | 詞表大小是怎麼決定的（「6400 哪來的」） |
| 11 | `tests/test_tokenizer_binding.py` | 為什麼權重與詞表必須一起發布 |

---

## 三個最容易誤會的地方

這份專案特別把這三點寫清楚（都在程式碼註解與追蹤檔裡）。

**① embedding 不是資料，是權重**

`E` 是一個 `vocab × dim` 的權重矩陣，跟 `q_proj` 完全平起平坐，
在同一份 `params()` 清單裡、用同一個 optimizer、同樣的步伐更新。
「embedding 有一套自己的訓練方法」這件事不存在。

**② 反向傳播從不修改權重**

`backward()` 只算出「該往哪修」，存在另一個張量裡。
真正動手的是 `optimizer.step()`，而且方向相反、步伐只有 `lr` 那麼小。

**③ 離散的 token id 不影響可微性**

不對 id 微分——它是資料不是參數。微分是對 `E` 裡的每一個實數做的，
那些數字是連續的。沒被查到的列梯度算出來就是 0，不是特別規定。

三點的詳細說明與當場驗證方法：[docs/06-常見誤解.md](docs/06-常見誤解.md)

---

## 跟真實模型的差異

這份程式為了好懂做了幾個簡化，**都不影響機制的正確性**：

| | 這份專案 | 真實 LLM |
|---|---|---|
| Tokenizer | 字元級 **或** 手刻 BPE（起始符號是字元） | byte-level BPE |
| Attention | 標準多頭 | 常用 GQA（K/V head 數較少） |
| 精度 | float32 | bf16 混合精度 |
| KV cache | 無（生成時重算） | 有 |
| 規模 | 數十萬到數百萬參數 | 數十億以上 |

架構順序、公式、反向傳播的推導全部跟 LLaMA / minimind 一致。

---

## 授權

MIT
