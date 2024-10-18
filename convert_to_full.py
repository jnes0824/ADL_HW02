import json

def convert_to_fullwidth(text):
    halfwidth_symbols = u",.?!:;()[]{}<>\"\'"
    fullwidth_symbols = u"，。？！：；（）［］｛｝＜＞＂＇"
    trans_table = str.maketrans(halfwidth_symbols, fullwidth_symbols)
    return text.translate(trans_table)

# 讀取 jsonl 檔案，並對 title 欄位進行全半形轉換
def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record = json.loads(line)
            if 'title' in record:
                record['title'] = convert_to_fullwidth(record['title'])
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    # 測試範例
    input_file = './data/summary_output3.jsonl'  # 替換成你的 jsonl 檔案路徑
    output_file = './data/full.jsonl'
    process_jsonl_file(input_file, output_file)

if __name__ == '__main__':
    main()
