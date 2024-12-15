import json
import argparse
from utils import MultiChat, load_open_source, get_response_open_source

def generate_sentences_gpt(input_file, output_file, num_sentences):
    with open(input_file, 'r', encoding='utf-8') as f:
        terms = [json.loads(line) for line in f]

    
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    chat = MultiChat(config,
        save_path=output_file,
        model="gpt-4o-mini",
        temperature=0
    )
    chat.start()

    for term_info in terms:
        word = term_info["term"]
        meaning = term_info["meaning"]

        prompt = (f'I created a new term "{word}", which means "{meaning}". '
                  f'Please generate {num_sentences} different sentences using this term in the format: '
                  f'"1. 2." Each sentence should clearly demonstrate the meaning of the term.')

        lsb = {
            "term": word,
            "meaning": meaning,
            "type":term_info["type"],
            "prompt": [
                {
                    "role": "system",
                    "content": 'Please generate sentences following the specified format and ensure clarity.',
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        
        chat.post(lsb)  

    chat.wait_finish()


def llama_3_response(prompt, model, tokenizer, num_sentences):
    messages = [{"role": "system", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        num_beams=1,
        do_sample=False,
        max_new_tokens=64 * num_sentences,
        eos_token_id=terminators
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_sentences_llama(input_file, output_file, num_sentences, model_name):
    model, tokenizer = load_open_source(model_name)

    with open(input_file, 'r', encoding='utf-8') as f:
        terms = [json.loads(line) for line in f]

    with open(output_file, 'w', encoding='utf-8') as w:
        for term_info in terms:
            word = term_info["term"]
            meaning = term_info["meaning"]

            prompt = (
                f'I created a new term "{word}", which means "{meaning}". '
                f'Please generate {num_sentences} different sentences using this term in the format: '
                f'"1. 2." Each sentence should clearly demonstrate the meaning of the term.'
            )

            response = llama_3_response(prompt, model, tokenizer, num_sentences)
            result = {
                "term": word,
                "meaning": meaning,
                "type": term_info["type"],
                "response": response
            }

            w.write(json.dumps(result, ensure_ascii=False) + "\n")
            w.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentences')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input term.jsonl file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONL file')
    parser.add_argument('--num_sentences', type=int, default=10, help='Number of sentences to generate for each term')

    args = parser.parse_args()

    generate_sentences_gpt(args.input_file, args.output_file, args.num_sentences)
    # generate_sentences_llama(args.input_file, args.output_file, args.num_sentences,"LLMs/Llama-3-70B-Instruct")