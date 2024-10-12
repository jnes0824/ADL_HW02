import json
import random

# 定義分割比例，例如 80% 的資料作為訓練集，20% 作為驗證集
SPLIT_RATIO = 0.9

def split_train_validation(input_file, train_output_file, validation_output_file, split_ratio=SPLIT_RATIO):
    # 讀取原始的 train.json 資料
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # 打亂資料順序以確保隨機性
    random.shuffle(data)

    # 計算分割的索引
    split_index = int(len(data) * split_ratio)

    # 分割資料集
    train_data = data[:split_index]
    validation_data = data[split_index:]

    print(f"Splitting {len(data)} samples into {len(train_data)} samples for training and {len(validation_data)} samples for validation...")

    # 將訓練集寫入 train.json
    with open(train_output_file, "w", encoding="utf-8") as train_outfile:
        json.dump(train_data, train_outfile, ensure_ascii=False, indent=4)

    # 將驗證集寫入 validation.json
    with open(validation_output_file, "w", encoding="utf-8") as validation_outfile:
        json.dump(validation_data, validation_outfile, ensure_ascii=False, indent=4)

    print(f"Split complete! {len(train_data)} samples in train set, {len(validation_data)} samples in validation set.")

# 使用範例
split_train_validation("./data/train.json", "./data/train_split.json", "./data/validation.json")
