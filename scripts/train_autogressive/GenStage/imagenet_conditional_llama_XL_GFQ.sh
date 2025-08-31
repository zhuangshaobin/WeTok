#! /bin/bash

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_BLOCKING_WAIT=1

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-12345}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo $MASTER_ADDR
echo $MASTER_PORT

##GPU
NODE_RANK=$NODE_RANK python main.py fit --config configs/WeToK/GenStage/imagenet_conditional_llama_XL_GFQ.yaml
