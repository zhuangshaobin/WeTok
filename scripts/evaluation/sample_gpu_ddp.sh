CHUNKS=$1

# #---------------------------------------------------------------------------
# ###WeTok
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
#         --ckpt "/path/to/your/llamagen-b/ckpt" \
#         --o "/path/to/save/samples" \
#         --config "configs/WeToK/GenStage/imagenet_conditional_llama_L_GFQ.yaml" \
#         -k 0,0,0,0 \
#         -p 1.0,1.0,1.0,1.0 \
#         -t 1.0,1.0,1.0,1.0 \
#         -n 50 \
#         --token_factorization \
#         --batch_size 50 \
#         --cfg_scale 1.95,2.05,1.95,1.95 \
#         --global_seed 42 \
#         --model WeTok \
#         --num_chunks $CHUNKS \
#         --chunk_idx $IDX &
# done

# wait

# echo "combining"

# ## logdir format
# python combine_npz.py --logdir "/path/to/save/npzs"

# #---------------------------------------------------------------------------
# ###WeTok
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
#         --ckpt "/path/to/your/llamagen-l/ckpt" \
#         --o "/path/to/save/samples" \
#         --config "configs/WeToK/GenStage/imagenet_conditional_llama_B_GFQ.yaml" \
#         -k 0,0,0,0 \
#         -p 1.0,1.0,1.0,1.0 \
#         -t 1.0,1.0,1.0,1.0 \
#         -n 50 \
#         --token_factorization \
#         --batch_size 50 \
#         --cfg_scale 1.95,1.8,2.0,1.9 \
#         --global_seed 42 \
#         --model WeTok \
#         --num_chunks $CHUNKS \
#         --chunk_idx $IDX &
# done

# wait

# echo "combining"

# ## logdir format
# python combine_npz.py --logdir "/path/to/save/npzs"

#---------------------------------------------------------------------------
###WeTok
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "/path/to/your/llamagen-xl/ckpt" \
        --o "/path/to/save/samples" \
        --config "configs/WeToK/GenStage/imagenet_conditional_llama_XL_GFQ.yaml" \
        -k 0,0,0,0 \
        -p 1.0,1.0,1.0,1.0 \
        -t 1.0,1.0,1.0,1.0 \
        -n 50 \
        --token_factorization \
        --batch_size 50 \
        --cfg_scale 1.75,1.8,2.0,2.0 \
        --global_seed 42 \
        --model WeTok \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

## logdir format
python combine_npz.py --logdir "/path/to/save/npzs"

