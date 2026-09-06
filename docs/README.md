# 教學文件

程式碼註解裡有完整的公式推導，這裡放的是**程式碼講不了的東西**：流程圖、全景視角、
以及做簡報時的敘事順序。

## 建議順序

| | 文件 | 對應程式碼 | 一句話 |
|---|---|---|---|
| 1 | [全景圖](01-全景圖.md) | — | 一句話進去，到權重被改動，中間發生了什麼 |
| 2 | [詞表大小是怎麼決定的](02-tokenizer.md) | `tokenizer/bpe.py` | 「6400 個詞哪來的」 |
| 3 | [Embedding 與 scatter-add](03-embedding.md) | `scratch/embedding.py` | 查表為什麼是矩陣乘法，反向為什麼是相加 |
| 4 | [反向傳播全鏈](04-反向傳播.md) | 全部 | 從 `p − y` 一路走回 `E` |
| 5 | [權重是怎麼被改的](05-adamw.md) | `scratch/adamw.py` | 反向算完之後才輪到 optimizer |
| 6 | [三個最常見的誤解](06-常見誤解.md) | — | 做簡報時最值得講的三件事 |

## 附：怎麼產生講解用的素材

```bash
# 逐層梯度驗證表（可直接貼進簡報）
python tests/test_layers.py          # -> traces/gradcheck.txt

# BPE 合併過程，看詞表怎麼從 3653 長到 6400
python tests/test_bpe.py             # -> traces/bpe_training.txt

# 第 1 步的完整數值軌跡（前向 + 反向 + AdamW 每個中間量）
python train.py --trace-steps 1      # -> traces/train_step_0001.txt

# 一次推論的完整前向
python generate.py --prompt "貓是一種" --trace   # -> traces/inference.txt
```

`traces/sample/` 裡有一份預先跑好的，不用執行就能看。
