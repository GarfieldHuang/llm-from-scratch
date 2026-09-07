# 教學簡報產生器

60 張的教學簡報，對應 `docs/` 底下 00–11 那幾份文件。

```bash
npm install                # 只需一次
pip install lxml fonttools # 後處理與驗證用
./make.sh                  # 產生 → 後處理 → 驗證 → 輸出到 docs/
```

## 為什麼需要 postprocess.py

**pptxgenjs 的輸出直接給 PowerPoint 會跳「發現內容有問題，是否修復」。**
`postprocess.py` 修掉這些：

| 問題 | 說明 |
|---|---|
| **屬性值寫成 `Infinity` / `NaN`** | **真正會觸發修復的原因。** 零尺寸的 `roundRect` 會讓 pptxgenjs 在 `<a:gd fmla>` 寫出 `val Infinity` |
| 字型缺字 | Calibri 和 Cambria **Bold** 都沒有 `⊙`。含缺字的 run 會被拆開，只有那幾個字元改用 Cambria Math |
| `[Content_Types].xml` 幽靈宣告 | 每張投影片寫一筆 `slideMasterN.xml`，但只有 slideMaster1 存在 |
| `<p:bgPr>` 缺 `<a:effectLst/>` | schema 上是必要元素 |
| 重複的 shape id | 表格和圖形用兩套 id 計數器 |
| `notesMasterIdLst` 順序 | `CT_Presentation` 是嚴格序列，必須排在 `sldIdLst` 之前 |
| slideMaster / notesMaster 共用 theme | 一個 theme part 只能被一個 master 參照 |
| 懸空的圖表 `<c:axId>` | 折線圖參照 3 個軸卻只宣告 2 個 |
| ZIP 封裝 | pptxgenjs 不壓縮且含資料夾項目；重新打包後 2.2 MB → 280 KB |

**只有第一項會真的讓 PowerPoint 要求修復**，其餘 PowerPoint 都容忍
（用最陽春的 pptxgenjs 輸出實測過）。但既然知道了就一併修掉。

`verify.py` 在交付前擋下這八類問題。`subset.py` 可抽取投影片子集，
用二分法定位是哪一張出問題——這是唯一可靠的除錯方法，
比對「PowerPoint 修復後的檔案」會找到一堆無關的差異。

## 注意

- 字型檢查會讀本機字型（含 Office 藏在 app bundle 裡的），換機器結果可能不同
- 產生後**必須**由人從 Finder 雙擊開一次確認。用 AppleScript `open` 驗證會給假陰性
