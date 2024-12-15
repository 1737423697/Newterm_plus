import json
import re
import argparse

parser = argparse.ArgumentParser(description='Process input and output file paths.')

parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input JSONL file.')
parser.add_argument('--output_file_path', type=str, required=True, help='Path to the output JSON file.')

args = parser.parse_args()

input_file_path = args.input_file_path
output_file_path = args.output_file_path

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line.strip())
        
        response_value = data.get('response', '')
        
        match = re.search(r'1\..*', response_value, re.DOTALL)
        term = data['term']
        if match:  
            content = match.group(0)
            
            text_items = content.split('\n')
            
            for item in text_items:
                if re.match(r'^\d+\.\s', item.strip()):
                    new_entry = {
                        'text': item.strip()[3:].lstrip(),
                        'prefix': f'Please create a sentence using the term "{term}":',
                    }
                    outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')