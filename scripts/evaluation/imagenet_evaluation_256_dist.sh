NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-12345}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo $MASTER_ADDR
echo $MASTER_PORT

# torchrun --standalone --nproc_per_node=8  evaluation_image_ddp.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio24_imagenet.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio24/WeTok.ckpt \
#  --image_size 256 \
#  --batch_size 100 \
#  --model WeTok \
#  --workers 8 \

#  torchrun --standalone --nproc_per_node=8  evaluation_image_ddp.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio48_imagenet.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio48/WeTok.ckpt \
#  --image_size 256 \
#  --batch_size 100 \
#  --model WeTok \
#  --workers 8 \

#   torchrun --standalone --nproc_per_node=8  evaluation_image_ddp.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio192_imagenet.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio192/WeTok.ckpt \
#  --image_size 256 \
#  --batch_size 100 \
#  --model WeTok \
#  --workers 8 \

#   torchrun --standalone --nproc_per_node=8  evaluation_image_ddp.py \
#  --config_file configs/WeToK/Inference/GeneralDomain_compratio768_imagenet.yaml \
#  --ckpt_path /your/ckpt/path/GeneralDomain/compratio768/WeTok.ckpt \
#  --image_size 256 \
#  --batch_size 100 \
#  --model WeTok \
#  --workers 8 \

#   torchrun --standalone --nproc_per_node=8  evaluation_image_ddp.py \
#  --config_file configs/WeToK/Inference/ImageNet_downsample8_imagenet.yaml \
#  --ckpt_path /your/ckpt/path/ImageNet/downsample8/WeTok.ckpt \
#  --image_size 256 \
#  --batch_size 100 \
#  --model WeTok \
#  --workers 8 \

  torchrun --standalone --nproc_per_node=8  evaluation_image_ddp.py \
 --config_file configs/WeToK/Inference/ImageNet_downsample16_imagenet.yaml \
 --ckpt_path /your/ckpt/path/ImageNet/downsample16/WeTok.ckpt \
 --image_size 256 \
 --batch_size 100 \
 --model WeTok \
 --workers 8 \



