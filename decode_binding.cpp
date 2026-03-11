#include <torch/all.h>
#include <torch/library.h>

void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap);

TORCH_LIBRARY(sgl_kernel, m) {
  m.def(
      "decode_attention_cpu(Tensor query, Tensor k_cache, Tensor v_cache, Tensor(a!) output, Tensor key, Tensor value, "
      "Tensor loc, Tensor attn_logits, Tensor req_to_token, Tensor req_pool_indices, Tensor seq_lens, float sm_scale, "
      "float logit_cap) -> ()");
  m.impl("decode_attention_cpu", torch::kCPU, &decode_attention_cpu);
}