NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-12345}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


echo $MASTER_ADDR
echo $MASTER_PORT

# torchrun --standalone --nproc_per_node=8 evaluation_original_reso_dist.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio24_mscoco.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio24/WeTok.ckpt \
#  --original_reso \
#  --model WeTok \
#  --batch_size 1 \
#  --workers 8

#  torchrun --standalone --nproc_per_node=8 evaluation_original_reso_dist.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio48_mscoco.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio48/WeTok.ckpt \
#  --original_reso \
#  --model WeTok \
#  --batch_size 1 \
#  --workers 8

#  torchrun --standalone --nproc_per_node=8 evaluation_original_reso_dist.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio192_mscoco.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio192/WeTok.ckpt \
#  --original_reso \
#  --model WeTok \
#  --batch_size 1 \
#  --workers 8

 torchrun --standalone --nproc_per_node=8 evaluation_original_reso_dist.py \
 --config_file configs/WeToK/Inference/GeneralDomain_compratio768_mscoco.yaml \
 --ckpt_path /your/ckpt/path/GeneralDomain/compratio768/WeTok.ckpt \
 --original_reso \
 --model WeTok \
 --batch_size 1 \
 --workers 8