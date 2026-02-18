import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import torch.profiler
from random import randint
def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    cpu_kv_cache = True
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, cpu_kv_cache=cpu_kv_cache)

    max_tokens = 200
    sampling_params = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    seqs = llm.config.num_sequences
    long_prompt = ""
    with open("long_prompt.txt", "r") as f:
        long_prompt = f.read()
    prompts = ["Calculate the result of 2923+2132 by step by step:" for _ in range(seqs[1])]
    # for i in range(seqs[1]):
    #     prompts.append(long_prompt)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  
        )
        for prompt in prompts
    ]

    # for i in range(seqs[1]):
    #     prompts.append([randint(0, 10000) for _ in range(4096)])
    # print(f"short prompt length: {len(prompts[0])}")
    # print(f"long prompt length: {len(prompts[-1])}")
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for prompt, output in zip(prompts, outputs):
        # pass
        #print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
