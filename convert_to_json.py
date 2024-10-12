import json

# Read the .jsonl file, convert it to .json, and process the data
def convert_and_process(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        json_list = []
        
        # Iterate through each line in the input file
        for line in infile:
            # Parse the line as JSON and automatically decode any Unicode characters
            data = json.loads(line)
            
            # Optional: Apply any additional processing here, e.g., modify the structure or content
            # In this example, we will just print the title and ID
            # print(f"Processing: {data['title']} (ID: {data['id']})")
            
            # Add the processed data to the list
            json_list.append(data)
        
        # Write the processed JSON list to the output file with pretty printing
        json.dump(json_list, outfile, ensure_ascii=False, indent=4)  # ensure_ascii=False keeps the Unicode characters

# Call the function to process the file
convert_and_process("./data/train.jsonl", "./data/train.json")
convert_and_process("./data/sample_test.jsonl", "./data/sample_test.json")
convert_and_process("./data/sample_submission.jsonl", "./data/sample_submission.json")
convert_and_process("./data/public.jsonl", "./data/public.json")
print("Conversion and processing complete!")

