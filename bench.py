import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 32* 2
    max_input_len = 1024 * 4
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(1000, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(500, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    _, stats = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True, stats=True)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    ttft = [stats[i]["prefill"][0] for i in stats.keys()]
    tpot = [stats[i]["decode"][0]/stats[i]["decode"][1] for i in stats.keys()]
    print(f"TTFT: {sum(ttft)/len(ttft) * 1000:.2f}ms, TPOT = {sum(tpot)/len(tpot)*1000:.2f}ms")
    #print(stats)
if __name__ == "__main__":
    main()
