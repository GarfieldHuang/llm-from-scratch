#!/bin/sh
# 產生 → 後處理 → 驗證。三步一定要一起跑，少了後處理 PowerPoint 會要求修復。
set -e
cd "$(dirname "$0")"
OUT="llm-from-scratch-教學簡報.pptx"
node build.js
python3 postprocess.py "$OUT"
python3 verify.py "$OUT"
mv "$OUT" ../../docs/
echo "已輸出到 docs/$OUT"
