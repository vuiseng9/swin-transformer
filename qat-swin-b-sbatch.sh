#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="swin-jpqd"

SWIN_CFG=configs/swin/qat-swin-b-p4-w7-224_22kto1k.yaml
# PRETRAINED_CKPT=/data2/vchua/run/dgx4-swin-opt/swin/pretrained/swin_base_patch4_window7_224_22k.pth
PRETRAINED_CKPT=/data2/vchua/run/dgx4-swin-opt/swin/pretrained/split_qkv_swin_base_patch4_window7_224_22k.pth

BS=64
MASTER_PORT=12345
RUNID=qat-pt-swin-b-22kto1k-dgx4-bs${BS}-splitqkv

DATADIR=/data/dataset/imagenet/ilsvrc2012/torchvision/

CONDAROOT=/data1/vchua/miniconda3
CONDAENV=dgx4-swin-opt

WORKDIR=/data2/vchua/dev/dgx4-swin-opt/swin-transformer
OUTROOT=/data2/vchua/run/dgx4-swin-opt/swin
# ---------------------------------------------------------------------------------------------
OUTDIR=$OUTROOT/$RUNID
mkdir -p $OUTDIR
cd $WORKDIR

python -m torch.distributed.launch \
   --nproc_per_node 4 \
   --master_port $MASTER_PORT \
   main.py \
    --cfg $SWIN_CFG \
    --pretrained $PRETRAINED_CKPT \
    --data-path $DATADIR \
    --output $OUTDIR \
   --batch-size $BS \
   --wandb_id $RUNID \
   --accumulation-steps 4

# original config used 8 cards
# ------------



