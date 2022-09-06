#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="sixer-swin"

SWIN_CFG=configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml
PRETRAINED_CKPT=/data5/vchua/run/test-a100x6/swin/pretrained/swin_base_patch4_window7_224_22k.pth

MASTER_PORT=12345
BS=128

RUNID=swin-b-p4-w7-r224-22kto1k-ftuned-bs${BS}-regression

DATADIR=/data1/dataset/imagenet/ilsvrc2012/torchvision/

CONDAROOT=/data5/vchua/miniconda3
CONDAENV=test-a100x6

WORKDIR=/data5/vchua/dev/test-a100x6/swin-transformer
OUTROOT=/data5/vchua/run/test-a100x6/swin

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

# original config used 8 cards, 64 bs each card
# ------------

