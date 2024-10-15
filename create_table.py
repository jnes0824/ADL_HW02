import pandas as pd
import json

# 定義讀取 .jsonl 文件並生成表格的函數
def jsonl_to_dataframe(jsonl_file):
    data = []
    
    # 讀取 jsonl 文件
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每行 json
            json_data = json.loads(line)

            params = json_data['params']

            # 提取參數中的值，例如 num_beams 或 temperature，如果存在
            if 'num_beams' in params:
                param_value = params['num_beams']
                params_type = 'num_beams'
            elif 'temperature' in params:
                param_value = params['temperature']
                params_type = 'temperature'
            elif 'top_k' in params:
                param_value = params['top_k']
                params_type = 'top_k'
            elif 'top_p' in params:
                param_value = params['top_p'] 
                params_type = 'top_p'
            else:
                param_value = None
                params_type = None
            # 提取數據並加入列表
            data.append({
                "Strategy": json_data['strategy'],
                "Param Type": params_type,
                # "Params": json_data['params'],
                "Param Value": param_value,
                "ROUGE-1 (f)": json_data['rouge_scores']['rouge-1']['f'],
                "ROUGE-2 (f)": json_data['rouge_scores']['rouge-2']['f'],
                "ROUGE-L (f)": json_data['rouge_scores']['rouge-l']['f']
            })
    
    # 生成 DataFrame
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by=["Strategy", "Param Value"])
    return df_sorted

# 調用函數並生成表格
jsonl_file_path = 'rouge_results.jsonl'  # 替換為你的 jsonl 文件路徑
df = jsonl_to_dataframe(jsonl_file_path)

# 顯示生成的表格
print(df)
