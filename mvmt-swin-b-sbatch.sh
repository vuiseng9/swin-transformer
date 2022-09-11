#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="swin-jpqd"

SWIN_CFG=configs/swin/mvmt-swin-b-p4-w7-224_22kto1k.yaml
PRETRAINED_CKPT=/data/vchua/run/msft-swin/swin/pretrained/swin_base_patch4_window7_224_22k.pth

BS=128
MASTER_PORT=13254

RUNID=mvmt-swin-b-22kto1k-tuned-bs${BS}-r0.01-e5-15-filled-init-0.1

DATADIR=/data/dataset/imagenet/ilsvrc2012/torchvision/

CONDAROOT=/data/vchua/miniconda3
CONDAENV=msft-swin

WORKDIR=/data/vchua/dev/msft-swin/swin-transformer
OUTROOT=/data/vchua/run/msft-swin/swin
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
   --accumulation-steps 2

# original config used 8 cards with bs64, 2 step gradient accumulation
# ------------


