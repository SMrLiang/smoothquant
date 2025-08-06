GPU=3

export CUDA_VISIBLE_DEVICES=$GPU

# python examples/export_int8_model.py
# meta-llama/Llama-2-7b-hf
# meta-llama/Llama-3.2-3B

python examples/generate_act_scales.py \
    --model-name "meta-llama/Llama-2-7b-hf" \
    --output-path "act_scales/llama-2-7b.pt" \
    --num-samples 512 \
    --seq-len 512 \
    --dataset-path "data/val.jsonl.zst"

# python examples/export_int8_model.py
python smoothquant/ppl_eval.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --act_scales_path act_scales/llama-2-7b.pt \
    --smooth \
    --alpha 0.85 \
    --quantize \
    --w_bit 4 \
    --a_bit 8

# python smoothquant/acc_eval.py \
#     --model_path meta-llama/Llama-2-7b-hf \
#     --act_scales_path act_scales/llama-2-7b.pt \
#     --smooth \
#     --alpha 0.85 \
#     --w_bit 8 \
#     --a_bit 8 \

# python smoothquant/acc_eval.py \
#     --model_path meta-llama/Llama-2-7b-hf \
#     --act_scales_path act_scales/llama-2-7b.pt \
#     --smooth \
#     --alpha 0.85 \
#     --w_bit 8 \
#     --a_bit 8 \
#     --quantize 

