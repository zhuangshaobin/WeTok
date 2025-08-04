python reconstruct_image.py \
--config_file "configs/WeToK/Inference/GeneralDomain_compratio24_imagenet.yaml" \
--ckpt_path  /your/ckpt/path/GeneralDomain/compratio24/WeTok.ckpt \
--save_dir "./visualize" \
--version  "GeneralDomain/compratio24" \
--image_num 8 \
--image_size 512 \
--model WeTok \

python reconstruct_image.py \
--config_file "configs/WeToK/Inference/GeneralDomain_compratio48_imagenet.yaml" \
--ckpt_path  /your/ckpt/path/GeneralDomain/compratio48/WeTok.ckpt \
--save_dir "./visualize" \
--version  "GeneralDomain/compratio48" \
--image_num 8 \
--image_size 512 \
--model WeTok \


python reconstruct_image.py \
--config_file "configs/WeToK/Inference/GeneralDomain_compratio192_imagenet.yaml" \
--ckpt_path  /your/ckpt/path/WeTok/GeneralDomain/compratio192/WeTok.ckpt \
--save_dir "./visualize" \
--version  "GeneralDomain/compratio192" \
--image_num 8 \
--image_size 512 \
--model WeTok \

python reconstruct_image.py \
--config_file "configs/WeToK/Inference/GeneralDomain_compratio768_imagenet.yaml" \
--ckpt_path  /your/ckpt/path/GeneralDomain/compratio768/WeTok.ckpt \
--save_dir "./visualize" \
--version  "GeneralDomain/compratio768" \
--image_num 8 \
--image_size 512 \
--model WeTok \

python reconstruct_image.py \
--config_file "configs/WeToK/Inference/ImageNet_downsample8_imagenet.yaml" \
--ckpt_path  /your/ckpt/path/ImageNet/downsample8/WeTok.ckpt \
--save_dir "./visualize" \
--version  "ImageNet/downsample8" \
--image_num 8 \
--image_size 512 \
--model WeTok \

python reconstruct_image.py \
--config_file "configs/WeToK/Inference/ImageNet_downsample16_imagenet.yaml" \
--ckpt_path  /your/ckpt/path/ImageNet/downsample16/WeTok.ckpt \
--save_dir "./visualize" \
--version  "ImageNet/downsample16" \
--image_num 8 \
--image_size 512 \
--model WeTok \