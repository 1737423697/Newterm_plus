## Step1:Generate sentences

### Usage

To run the script, use the following command:

```bash
python ft/generate_sentence.py --input_file your/path/to/new_terms.jsonl --output_file path/to/in_context_output.jsonl --num_sentences number of sentences to generate for each term
```

if  you want  to generate with llama, please modify ft/genetate_sentence.py line 107 and 108

```python
  # generate_sentences_llama(args.input_file, args.output_file, args.num_sentences,"LLMs/Llama-3-70B-Instruct")
```



## Step2:Generate lora data

### Usage

To run the script, use the following command:

```bash
python your_script.py --input_file_path "path/to/in_context_output.jsonl.jsonl" --output_file_path "path/to/lora_data.json"
```

The `input_file_path` should be identical to the `output_file` from the first step.



## Step3:Git clone Parrot

### Usage

```bash
git clone https://github.com/wxjiao/ParroT.git ft/Parrot/
```



## Step4:Adjust Directory Structure

```BASH
mv ft/run_clm_lora_eos_change.py ft/ParroT/transformers/examples/pytorch/language-modeling
```



## Step5:Finetune Model

### Usage example

```bash
export MASTER_ADDR=localhost

export MASTER_PORT=29411

export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1

cd ft/ParroT/

train_path=transformers/examples/pytorch/language-modeling/run_clm_lora_eos_change.py



model_path=LLMs/Llama-2-7b-chat-hf

model_save=LLMs/Llama-2-7b-Lora-eos


torchrun --nnodes 1 --nproc_per_node 2 \

  --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \

  ${train_path} \

  --deepspeed train/deepspeed_config_zero2.json \

  --model_name_or_path ${model_path} \

  --train_file ft/lora_data.json \

  --use_lora True \

  --lora_config train/lora_config.json \

  --preprocessing_num_workers 16 \

  --dataloader_num_workers 8 \

  --dataloader_pin_memory True \

  --per_device_train_batch_size 16 \

  --per_device_eval_batch_size 2 \

  --gradient_accumulation_steps 4 \

  --max_steps 1000 \

  --save_strategy "steps" \

  --save_steps 50 \

  --save_total_limit 20 \

  --learning_rate 3e-4 \

  --weight_decay 0. \

  --warmup_ratio 0.03 \

  --lr_scheduler_type "cosine" \

  --logging_steps 10 \

  --block_size 512 \

  --do_train \

  --evaluation_strategy "no" \

  --validation_split_percentage 0 \

  --bf16=True \

  --bf16_full_eval True \

  --ddp_timeout 3600 \

  --seed 1 \

  --gradient_checkpointing True \

  --output_dir ${model_save}
```



## Step6:Evaluate LoRA Model

### Usage example(100 steps)

```bash
cd ~/NewTerm++

python ft/evaluate_lora.py --path ft/ --lora-weights LLMs/Llama-2-7b-Lora-eos/checkpoint-100/adapter_model --model LLMs/Llama-2-7b-chat-hf --year 2023 --task ALL
```

And then you can use RAG/get_results.py to check the performance.