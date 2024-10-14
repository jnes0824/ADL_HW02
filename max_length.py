import json

# 打開和讀取JSON檔案
with open("./data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 初始化變數來儲存總長度和數量
total_title_length = 0
total_maintext_length = 0
num_entries = len(data)

# 初始化最大長度變數
max_title_length = 0
max_maintext_length = 0
min_title_length = 1000

# 遍歷每一項，累積長度並計算最大長度
for entry in data:
    title_length = len(entry["title"])
    maintext_length = len(entry["maintext"])
    
    total_title_length += title_length
    total_maintext_length += maintext_length
    
    if title_length > max_title_length:
        max_title_length = title_length
    if maintext_length > max_maintext_length:
        max_maintext_length = maintext_length
    if title_length < min_title_length:
        min_title_length = title_length
# 計算平均長度
average_title_length = total_title_length / num_entries
average_maintext_length = total_maintext_length / num_entries

# 顯示結果
print(f"最大 title 長度: {max_title_length}")
print(f"最大 maintext 長度: {max_maintext_length}")
print(f"平均 title 長度: {average_title_length}")
print(f"平均 maintext 長度: {average_maintext_length}")
print(f"最小 title 長度: {min_title_length}")

