#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    --rdzv_endpoint=$MASTER_ADDR:$PORT \
    --rdzv_backend=c10d \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}