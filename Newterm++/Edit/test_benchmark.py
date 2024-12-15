import os
import json
import argparse
from my_utils import eval_qa
from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Edit examples and evaluate accuracy.')
parser.add_argument('--batch_size', type=int, default=1, help='Number to edit at once (default: 1)')
parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
parser.add_argument('--model_type', type=str, choices=['llama', 'qwen'], required=True, help='Type of the model to use (llama or qwen)')
parser.add_argument('--path', type=str, required=True, help='Path to the COMA, COST, CSJ questions')
args = parser.parse_args()

batch_sizes = [args.batch_size]
for batch_size in batch_sizes:
    data_path = args.data_path
    with open(data_path, 'r') as f:
        data = json.load(f)

if args.model_type == 'qwen':
    hparams = MEMITHyperParams.from_hparams('Edit/EasyEdit/hparams/MEMIT/qwen-7b.yaml')
elif args.model_type == 'llama':
    hparams = MEMITHyperParams.from_hparams('Edit/EasyEdit/hparams/MEMIT/llama-7b.yaml')

    total_examples = len(data["prompts"])
    num_iterations = total_examples // batch_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    coma_tot_cnt, coma_cor_cnt, cost_tot_cnt, cost_cor_cnt, csj_tot_cnt, csj_cor_cnt = 0, 0, 0, 0, 0, 0

    for i in range(num_iterations):
        editor = BaseEditor.from_hparams(hparams)
        prompts = data["prompts"][i * batch_size:(i + 1) * batch_size]
        ground_truth = [None] * batch_size
        target_new = data["target_new"][i * batch_size:(i + 1) * batch_size]
        subject = data["subject"][i * batch_size:(i + 1) * batch_size]

        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            sequential_edit=True
        )

        # Pass the path argument to eval_qa
        coma_tot, coma_cor = eval_qa(meanings=target_new, model=edited_model, tokenizer=tokenizer, task='COMA', model_name=args.model_type, path=args.path)
        coma_tot_cnt += coma_tot
        coma_cor_cnt += coma_cor
        
        cost_tot, cost_cor = eval_qa(meanings=target_new, model=edited_model, tokenizer=tokenizer, task='COST', model_name=args.model_type, path=args.path)
        cost_tot_cnt += cost_tot
        cost_cor_cnt += cost_cor
        
        csj_tot, csj_cor = eval_qa(meanings=target_new, model=edited_model, tokenizer=tokenizer, task='CSJ', model_name=args.model_type, path=args.path)
        csj_tot_cnt += csj_tot
        csj_cor_cnt += csj_cor

    COMA_AVG = coma_cor_cnt / coma_tot_cnt if coma_tot_cnt else 0
    COST_AVG = cost_cor_cnt / cost_tot_cnt if cost_tot_cnt else 0
    CSJ_AVG = csj_cor_cnt / csj_tot_cnt if csj_tot_cnt else 0
    AVG = (COMA_AVG + COST_AVG + CSJ_AVG) / 3

    print("*********************************************************")
    print("i = " + str(i))
    print("COMA = " + str(COMA_AVG) + "    Total=" + str(coma_tot_cnt))
    print("COST = " + str(COST_AVG) + "    Total=" + str(cost_tot_cnt))
    print("CSJ = " + str(CSJ_AVG) + "    Total=" + str(csj_tot_cnt))
    print("*********************************************************")

    output_dir = 'Edit/results'
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "COMA_AVG": COMA_AVG,
        "COST_AVG": COST_AVG,
        "CSJ_AVG": CSJ_AVG,
        "AVG": AVG
    }

    output_file_path = f'{output_dir}/{args.model_type}_{batch_size}_benchmark.json'
    with open(output_file_path, 'w') as outfile:
        json.dump(results, outfile)
    print(f"Results saved to {output_file_path}")
