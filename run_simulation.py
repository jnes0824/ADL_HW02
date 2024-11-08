import sys
import argparse
from transformers import pipeline
from datasets import Dataset
import json
from tqdm import tqdm
from convert_to_full import convert_to_fullwidth
import os
import torch




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
        return pipeline("summarization", model=model_path, device=0, num_beams=5, top_k=top_k, max_length=128, do_sample=True)
    elif strategy == "top_p":
        return pipeline("summarization", model=model_path, device=0, num_beams=5, top_p=top_p, max_length=128, do_sample=True)
    elif strategy == "temperature":
        return pipeline("summarization", model=model_path, device=0, num_beams=5, temperature=temperature, max_length=128, do_sample=True)
    elif strategy == "combined":
        # 使用 num_beams + top_p + temperature 同時生成
        return pipeline("summarization", model=model_path, device=0, num_beams=num_beams, top_p=top_p, temperature=temperature, max_length=128)
    else:  # greedy search
        return pipeline("summarization", model=model_path, device=0, num_beams=1, max_length=128)
    

# 主函數
def main():
    args = parse_args()
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    original_cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", None)

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
        # {"strategy": "beam_search", "params": {"num_beams": 4}},
        {"strategy": "beam_search", "params": {"num_beams": 5}},
        # {"strategy": "beam_search", "params": {"num_beams": 6}},
        # {"strategy": "top_k", "params": {"top_k": 50}},
        # {"strategy": "top_k", "params": {"top_k": 100}},
        # {"strategy": "top_p", "params": {"top_p": 0.9}},
        # {"strategy": "top_p", "params": {"top_p": 0.5}},
        # {"strategy": "top_p", "params": {"top_p": 0.95}},
        # {"strategy": "temperature", "params": {"temperature": 0.7}},
        # {"strategy": "temperature", "params": {"temperature": 1.2}},
        # {"strategy": "greedy", "params": {}},
        # {"strategy": "temperature", "params": {"temperature": 2.0}},
        # {"strategy": "temperature", "params": {"temperature": 0.1}},
        # {"strategy": "top_k", "params": {"top_k": 10}},
    ]
    
    # 開始測試每個策略
    for strat in strategies:
        torch.cuda.empty_cache()
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
            summary_texts = [convert_to_fullwidth(summary['summary_text'].replace("<extra_id_0>", "").replace("<extra_id_1>", "").replace("<extra_id_2>", "")) for summary in summaries]
            batch['summary'] = summary_texts
            all_generated_summaries.extend(summary_texts)
            all_reference_summaries.extend(batch['title'])
            return batch

        # 對數據集進行摘要
        result_dataset = dataset.map(summarize_batch, batched=True, batch_size=2)

        # 計算 ROUGE 分數
        torch.cuda.empty_cache()
        from tw_rouge import get_rouge
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        rouge_scores = get_rouge(all_generated_summaries, all_reference_summaries)
        print(f"Strategy: {strategy}, Params: {params}, ROUGE Scores: {rouge_scores}")
        sys.modules.pop('tw_rouge', None)
        if original_cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_device
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]  # 如果原本沒有設置，則刪除變量
        # 將結果寫入檔案
        with open("./rouge_results.jsonl", "a", encoding="utf-8") as result_file:
            result = {
                "strategy": strategy,
                "params": params,
                "rouge_scores": rouge_scores
            }
            result_file.write(json.dumps(result) + "\n")
    print("All done!")

if __name__ == "__main__":
    main()

