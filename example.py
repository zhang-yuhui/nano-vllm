import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    cpu_kv_cache = True
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, cpu_kv_cache=cpu_kv_cache)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=400)
    prompts = [
        "introduce yourself",
        "1231+3141=?",
        "hello"
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
