import json
import argparse
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from datasets import Dataset
from convert_to_full import convert_to_fullwidth
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Summarization with full-width conversion.")
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_path", type=str, help="Path to save the output JSONL file")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading the model...")
    # Load your fine-tuned model for summarization
    model_path = "./final_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=1024)
    summarizer = pipeline("summarization", model=model_path, device=0, num_beams=5, tokenizer=tokenizer,  max_length=1024, truncation=True) 
    # torch.cuda.set_per_process_memory_fraction(1/3, device=0)

    # Load the JSON file
    data = []
    with open(args.input_path, "r", encoding="utf-8") as file:
        for line in file:
                data.append(json.loads(line))

    # Convert JSON to datasets.Dataset
    dataset = Dataset.from_list(data)

    output_data = []
    for entry in tqdm(data, desc="Processing entries"):
        # 在 maintext 欄位中加入 'summarize:' 前綴
        text_to_summarize = "summarize: " + entry["maintext"]
    
        # 生成摘要
        summary = summarizer(text_to_summarize, max_length=128, do_sample=False)[0]["summary_text"]
        summary_text = convert_to_fullwidth(summary.replace("<extra_id_0>", "").replace("<extra_id_1>", "").replace("<extra_id_2>", ""))
        # 建立新格式的字典並加入結果列表
        output_data.append({
            "title": summary_text,
            "id": entry["id"]
        })

    # 將結果寫入 JSONL 文件
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    main()
