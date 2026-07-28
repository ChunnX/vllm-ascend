/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ADN_FUSED_INFER_ATTENTION_TORCH_ADPT_H
#define ADN_FUSED_INFER_ATTENTION_TORCH_ADPT_H

#include <string>

namespace vllm_ascend {

namespace detail {

inline void run_adn_fused_infer_attention(
    const at::Tensor& query,
    at::TensorList key,
    at::TensorList value,
    const c10::optional<at::Tensor>& attn_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_q,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& block_table,
    int64_t num_heads,
    double scale_value,
    c10::string_view input_layout,
    int64_t num_key_value_heads,
    int64_t block_size,
    int64_t inner_precise,
    at::Tensor& output)
{
    TORCH_CHECK(key.size() == 1, "ADN expects exactly one paged key-cache tensor, got ", key.size());
    TORCH_CHECK(value.size() == 1, "ADN expects exactly one paged value-cache tensor, got ", value.size());
    TORCH_CHECK(
        actual_seq_lengths_q.has_value() && !actual_seq_lengths_q->empty(),
        "ADN requires non-empty actual_seq_lengths_q");
    TORCH_CHECK(
        actual_seq_lengths_kv.has_value() && !actual_seq_lengths_kv->empty(),
        "ADN requires non-empty actual_seq_lengths_kv");
    TORCH_CHECK(
        actual_seq_lengths_q->size() == actual_seq_lengths_kv->size(),
        "ADN q/kv sequence-length batches differ: ",
        actual_seq_lengths_q->size(),
        " vs ",
        actual_seq_lengths_kv->size());
    TORCH_CHECK(
        block_table.has_value() && block_table->defined(),
        "ADN paged attention requires block_table");
    TORCH_CHECK(
        output.sizes().equals(query.sizes()),
        "ADN output shape must match query shape, got ",
        output.sizes(),
        " vs ",
        query.sizes());
    TORCH_CHECK(
        output.scalar_type() == query.scalar_type(),
        "ADN output dtype must match query dtype, got ",
        output.scalar_type(),
        " vs ",
        query.scalar_type());
    TORCH_CHECK(
        output.device() == query.device(),
        "ADN output device must match query device, got ",
        output.device(),
        " vs ",
        query.device());

    const auto q_lengths_sym =
        actual_seq_lengths_q.value_or(at::ArrayRef<c10::SymInt>{});
    const auto kv_lengths_sym =
        actual_seq_lengths_kv.value_or(at::ArrayRef<c10::SymInt>{});
    const auto q_lengths = c10::asIntArrayRefUnchecked(q_lengths_sym);
    const auto kv_lengths = c10::asIntArrayRefUnchecked(kv_lengths_sym);
    const std::string input_layout_string(input_layout);
    const char* input_layout_char = input_layout_string.c_str();

    EXEC_NPU_CMD(
        aclnnAdnFusedInferAttention,
        query,
        key,
        value,
        attn_mask,
        q_lengths,
        kv_lengths,
        block_table,
        num_heads,
        scale_value,
        input_layout_char,
        num_key_value_heads,
        block_size,
        inner_precise,
        output);
}

}  // namespace detail

at::Tensor npu_adn_fused_infer_attention(
    const at::Tensor& query,
    at::TensorList key,
    at::TensorList value,
    const c10::optional<at::Tensor>& attn_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_q,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& block_table,
    int64_t num_heads,
    double scale_value,
    c10::string_view input_layout,
    int64_t num_key_value_heads,
    int64_t block_size,
    int64_t inner_precise)
{
    at::Tensor output = at::empty_symint(query.sym_sizes(), query.options());
    detail::run_adn_fused_infer_attention(
        query,
        key,
        value,
        attn_mask,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
        num_heads,
        scale_value,
        input_layout,
        num_key_value_heads,
        block_size,
        inner_precise,
        output);
    return output;
}

at::Tensor npu_adn_fused_infer_attention_out(
    const at::Tensor& query,
    at::TensorList key,
    at::TensorList value,
    at::Tensor& output,
    const c10::optional<at::Tensor>& attn_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_q,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& block_table,
    int64_t num_heads,
    double scale_value,
    c10::string_view input_layout,
    int64_t num_key_value_heads,
    int64_t block_size,
    int64_t inner_precise)
{
    detail::run_adn_fused_infer_attention(
        query,
        key,
        value,
        attn_mask,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
        num_heads,
        scale_value,
        input_layout,
        num_key_value_heads,
        block_size,
        inner_precise,
        output);
    return output;
}

}  // namespace vllm_ascend

#endif
