import json
import argparse
import re
import torch
from copy import deepcopy
import os
from utils import build_prompt, get_response_open_source
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_finetuned_model(model_name, lora_weights=None):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    if lora_weights is not None and os.path.exists(lora_weights):
        print(f'Loading LoRA weights from path: {lora_weights}')
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,trust_remote_code=True)    
    return model, tokenizer

def test_open_source(task, model_name, year, path,lora_weights=None):
    model, tokenizer = load_finetuned_model(model_name, lora_weights)
    file_name = os.path.join(path, f'benchmark_{year}/{task}_clean.jsonl')
    if lora_weights is not None and os.path.exists(lora_weights):
        if not re.search(r'checkpoint-(\d+)', lora_weights) == None:
            epoch_num = re.search(r'checkpoint-(\d+)', lora_weights).group(1)
        else:
            epoch_num="None"
    else:
        epoch_num = re.search(r'(\d+)','0').group(1)
    with open(file_name, 'r', encoding='utf-8') as f, \
         open(os.path.join(path, f'results_{year}', f'{task}_{model_name.split("/")[-1]}-LoRA{epoch_num}.jsonl'), 'w', encoding='utf-8') as w:
        
        for line in f:
            line = json.loads(line.strip())
            
            # Loop through prompt variants
            for p in range(3):
                lsb = deepcopy(line)
                lsb['prompt_id'] = p
                system, prompt, _ = build_prompt(line, task, p)
                prompt = [system, prompt]
                lsb['response'] = get_response_open_source(prompt, model_name, model, tokenizer)
                w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
    
    del model
    del tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['COMA', 'COST', 'CSJ', 'ALL'], default='ALL', help='Which task to test.')
    parser.add_argument('--model', type=str, default='/data/dhx/LLMs/Llama-2-7b-chat-hf', help='Which model to test.')
    parser.add_argument('--year', type=str, default='2023', help='Which year to test.')
    parser.add_argument('--path', type=str, default='data/', help='Base path for data files.')
    parser.add_argument('--lora-weights', type=str, required=True, help='Path to LoRA weights for the model.')

    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.path, f"results_{args.year}")):
        os.makedirs(os.path.join(args.path, f"results_{args.year}"))
    
    task_func = test_open_source
    if args.task == 'ALL':
        for task in ['COMA', 'COST', 'CSJ']:
            task_func(task, args.model, args.year, args.path, args.lora_weights)
    else:
        task_func(args.task, args.model, args.year, args.path, args.lora_weights)
