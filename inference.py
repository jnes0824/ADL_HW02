import json
import argparse
from transformers import pipeline
from tqdm import tqdm
from datasets import Dataset
from convert_to_full import convert_to_fullwidth

def parse_args():
    parser = argparse.ArgumentParser(description="Summarization with full-width conversion.")
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_path", type=str, help="Path to save the output JSONL file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load your fine-tuned model for summarization
    summarizer = pipeline("summarization", model="./output_mt5_all_train2", device=0, num_beams=4)

    # Load the JSON file
    data = []
    with open(args.input_path, "r", encoding="utf-8") as file:
        for line in file:
                data.append(json.loads(line))

    # Convert JSON to datasets.Dataset
    dataset = Dataset.from_list(data)

    # Define a function for summarization that can be applied to each row
    def summarize_batch(batch):
        text = ["summarize: " + item for item in batch['maintext']]
        summaries = summarizer(text, batch_size=2, max_length=128) 
        summary_texts = [convert_to_fullwidth(summary['summary_text'].replace("<extra_id_0>", "").replace("<extra_id_1>", "").replace("<extra_id_2>", "")) for summary in summaries]
        batch['summary'] = summary_texts
        return batch

    # Apply the summarization function to the dataset in batches
    result_dataset = dataset.map(summarize_batch, batched=True, batch_size=2)

    # Convert the result dataset to list of dictionaries for saving in jsonl format
    result_list = result_dataset.to_dict()

    # Save the summaries back to a JSONL file (one JSON object per line)
    with open(args.output_path, "w", encoding="utf-8") as out_file:
        for i in range(len(result_list['summary'])):
            json_record = {
                'title': result_list['summary'][i], 
                'id': result_list['id'][i]
            }
            out_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")  # Write each record as a line
