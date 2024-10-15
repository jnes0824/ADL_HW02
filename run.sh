#!/bin/bash

# 檢查是否提供了足夠的參數
if [ "$#" -ne 2 ]; then
    echo "用法: ./run.sh /path/to/input.jsonl /path/to/output.jsonl"
    exit 1
fi

# 取得參數
INPUT_FILE=$1
OUTPUT_FILE=$2

# 運行 Python 程序進行推理
echo "正在運行推理模型..."
python ./inference.py $INPUT_FILE $OUTPUT_FILE 

# 檢查推理是否成功
if [ $? -eq 0 ]; then
    echo "推理完成，預測結果保存在 $OUTPUT_FILE"
else
    echo "推理失敗"
    exit 1
fi
