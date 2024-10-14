import sys
import argparse
from transformers import pipeline
from datasets import Dataset
import json
from tqdm import tqdm

# 定義參數
def parse_args():
    parser = argparse.ArgumentParser(description="Text Summarization with Different Generation Methods")
    parser.add_argument("--model_path", type=str, default="./output_mt5_all_train2", help="Path to the fine-tuned model")
    return parser.parse_args()

# 根據策略和參數選擇不同的 summarizer，並使用默認的長度參數
def get_summarizer(strategy, model_path, top_k=None, top_p=None, temperature=None, num_beams=None):
    if strategy == "beam_search":
        return pipeline("summarization", model=model_path, device=0, num_beams=num_beams, max_length=128)
    elif strategy == "top_k":
        return pipeline("summarization", model=model_path, device=0, num_beams=1, top_k=top_k, max_length=128)
    elif strategy == "top_p":
        return pipeline("summarization", model=model_path, device=0, num_beams=1, top_p=top_p, max_length=1280)
    elif strategy == "temperature":
        return pipeline("summarization", model=model_path, device=0, num_beams=1, temperature=temperature, max_length=128)
    elif strategy == "combined":
        # 使用 num_beams + top_p + temperature 同時生成
        return pipeline("summarization", model=model_path, device=0, num_beams=num_beams, top_p=top_p, temperature=temperature, max_length=128)
    else:  # greedy search
        return pipeline("summarization", model=model_path, device=0, num_beams=1, max_length=128)
    

# 主函數
def main():
    args = parse_args()

    # 加載數據
    data = []
    with open("./data/public.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))


    # 將 JSON 轉為 datasets 格式
    dataset = Dataset.from_list(data)

    all_generated_summaries = []
    all_reference_summaries = []
    
    # 定義測試策略及參數組合
    strategies = [
        {"strategy": "beam_search", "params": {"num_beams": 4}},
        {"strategy": "beam_search", "params": {"num_beams": 6}},
        {"strategy": "top_k", "params": {"top_k": 50}},
        {"strategy": "top_k", "params": {"top_k": 100}},
        {"strategy": "top_p", "params": {"top_p": 0.9}},
        {"strategy": "top_p", "params": {"top_p": 0.95}},
        {"strategy": "temperature", "params": {"temperature": 0.7}},
        {"strategy": "temperature", "params": {"temperature": 1.2}},
        {"strategy": "greedy", "params": {}},
    ]
    
    # 打開檔案進行結果紀錄
    with open("./rouge_results.txt", "w", encoding="utf-8") as result_file:
        # 開始測試每個策略
        for strat in strategies:
            strategy = strat['strategy']
            params = strat['params']
            
            print(f"Testing strategy: {strategy}, params: {params}")
            summarizer = get_summarizer(strategy, args.model_path, **params)
            
            all_generated_summaries.clear()
            all_reference_summaries.clear()

            # 定義摘要函數
            def summarize_batch(batch):
                text = ["summarize: " + item for item in batch['maintext']]
                summaries = summarizer(text, batch_size=2)
                summary_texts = [summary['summary_text'].replace("<extra_id_0>", "").replace("<extra_id_1>", "").replace("<extra_id_2>", "") for summary in summaries]
                batch['summary'] = summary_texts
                all_generated_summaries.extend(summary_texts)
                all_reference_summaries.extend(batch['title'])
                return batch

            # 對數據集進行摘要
            result_dataset = dataset.map(summarize_batch, batched=True, batch_size=2)

            # 計算 ROUGE 分數
            from tw_rouge import get_rouge
            rouge_scores = get_rouge(all_generated_summaries, all_reference_summaries)
            print(f"Strategy: {strategy}, Params: {params}, ROUGE Scores: {rouge_scores}")
            sys.modules.pop('tw_rouge', None)
            # 將結果寫入檔案
            result_file.write(f"Strategy: {strategy}, Params: {params}, ROUGE Scores: {rouge_scores}\n")

if __name__ == "__main__":
    main()
