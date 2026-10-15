#ifndef DCUT_CAUSAL_CONV1D_TORCH_ADPT_H
#define DCUT_CAUSAL_CONV1D_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_dcut_causal_conv1d(const at::Tensor& output, const at::Tensor& x, const at::Tensor& weight,
                                  const at::Tensor& conv_state, const c10::optional<at::Tensor>& bias,
                                  const c10::optional<at::Tensor>& query_start_loc,
                                  const c10::optional<at::Tensor>& cache_indices,
                                  const c10::optional<at::Tensor>& num_accepted_tokens, int64_t activation_mode,
                                  int64_t pad_slot_id) {
  TORCH_CHECK(query_start_loc.has_value(), "query_start_loc cannot be empty.");
  TORCH_CHECK(cache_indices.has_value(), "cache_indices cannot be empty.");
  TORCH_CHECK(num_accepted_tokens.has_value(), "num_accepted_tokens cannot be empty.");

  TORCH_CHECK(conv_state.dim() == 3 && conv_state[0].is_contiguous(),
              "Dcut convolution state must have packed inner dimensions.");
  int64_t store_mode = 1;
  int64_t state_row_stride = conv_state.stride(0);
  EXEC_NPU_CMD(aclnnDcutCausalConv1d, x, weight, bias, conv_state, query_start_loc, cache_indices, c10::nullopt,
               num_accepted_tokens, activation_mode, pad_slot_id, store_mode, state_row_stride, output);
  return output;
}

}  // namespace vllm_ascend
#endif
