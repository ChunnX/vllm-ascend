source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/bin/set_env.bash
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODEL="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_KV_GROUP_MIN_SIZE=16
export MODEL="/opt/foundation_model/Qwen3.6-27B"
export VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU=v4
export VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=0

python -m vllm.entrypoints.openai.api_server --model $MODEL \
--port 46008 \
--data-parallel-size 1 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.9 \
--trust-remote-code \
--served-model-name qwen \
--max-model-len 32768 \
--max-num-batched-tokens 4096 \
--max-num-seqs 4 \
--host="0.0.0.0" \
--block-size=128 \
--tokenizer-mode auto \
--enable-auto-tool-choice \
--enable-prefix-caching \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3 \
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 7,"method":"dspark","model":"/opt/w00958190/dspark_gabbages/tige_training_output/0818_vocab64000_qwen3.6_27b_1000k/step90792"}' 
2>&1 | tee log.txt
