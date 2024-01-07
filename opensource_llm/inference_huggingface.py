import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

import torch


context = (
    'Your are an idea filter is designed to weed out ideas that are '
    'sloppy, off-topic (i.e., not sustainability related), unsuitable, or '
    'vague (such as the over-generic content that prioritizes form over substance, offering generalities instead of specific details). This filtration system helps concentrate human evaluators\' time and resources on concepts that are meticulously crafted, well-articulated, and hold tangible relevance.'
    'The intended audience is venture capitalists.\n'
    'When evaluating ideas you will first succinctly evaluate it on each criterion, then output a word of recommendation: "pass" or "fail" for each criterion and overall\n'
    'The output format will be like:\n'
    'EVALUATION: ... /PASS/ or /FAIL/\n'
    'OVERALL: /PASS/ or /FAIL/'
)

# problem = "PLASTIC RECYCLING"
# solution = "PLASTIC PRODUCTS RECYCLING PROCESS"

# problem = 'Shrimp shells waste'
# solution = 'produce 4 products fro shrimp shells'

# problem = 'This solution helps organizers of various events, conferences and concerts to be ego prepared so that no catastrophe may arise '
# solution = 'Event Changing'

# problem = 'Today, millions of tons of textile waste end up in landfill, leading to severe environmental damage. The fashion industry is one of the most polluting industries in the world and is primarily linear, meaning it produces a significant amount of waste that is not effectively reused or recycled.'
# solution = "A 'Clothes-as-a-Service' model could transform the fashion industry. Under this model, brands maintain ownership of the clothes and lease them to customers for a period. This service allows customers to get the style and variety they want, discouraging the tendency for impulse purchases that end up in the landfill. Once clothes are returned, they are cleaned, repaired, and leased again. When they are no longer rentable, these clothes can be recycled in a way that keeps the raw materials in the cycle. This model also offers an opportunity for companies to invest in more durable, high-quality garments, which are more environmental-friendly."

# problem = problem.strip()
# solution = solution.strip()
# prompt = (
#     "[INST]<> "
#     f'{context}<>\n'
#     f'The texs is: ```problem: {problem} and solution: {solution}```\n'
#     "[/INST]\n"
# )

def tokenize_mistral_instruct(context, problem, solution, tokenizer):
    chat = [
        {"role": "user", "content": f'{context}\nproblem: ```{problem}```, solution: ```{solution}```\n'},
    ]
    model_inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')

    print('prompt:\n', tokenizer.apply_chat_template(chat, tokenize=False))

    return model_inputs


def tokenize_zephyr(context, problem, solution, tokenizer):
    chat = [
        {"role": "system", "content": f'{context}'},
        {"role": "user", "content": f'Problem: {problem} \nSolution: {solution}'},
    ]

    model_inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')

    print('prompt:\n', tokenizer.apply_chat_template(chat, tokenize=False))

    return model_inputs


if __name__ == '__main__':
    dataset = 'AI EarthHack Dataset.csv'
    dataset = pd.read_csv(dataset, encoding='latin-1')

    # model_name = 'mistralai/Mistral-7B-v0.1'
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    model_name = 'HuggingFaceH4/zephyr-7b-beta'

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto", load_in_4bit=True,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
    # generated_ids = model.generate(**model_inputs)
    # tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

    for idx in range(len(dataset)):
        problem_id = dataset.loc[idx, 'id']
        problem = dataset.loc[idx, 'problem'].strip().replace('\n', '')
        solution = dataset.loc[idx, 'solution']

        if not solution or not isinstance(solution, str):
            continue

        solution = solution.strip().replace('\n', '')

        # model_inputs = tokenize_mistral_instruct(context, problem, solution, tokenizer)
        model_inputs = tokenize_zephyr(context, problem, solution, tokenizer)

        # model_inputs = tokenizer(
        #     [prompt], return_tensors="pt", padding=True
        # ).to("cuda")

        generated_ids = model.generate(model_inputs, max_new_tokens=500)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print('output:\n\n', output[0].strip())

        with open(f'{problem_id}.txt', 'w') as f:
            f.write(output[0].strip())

        # import pdb; pdb.set_trace()
