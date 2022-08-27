#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="swin-jpqd"

RUNID=distill-swin-l-to-swin-b-pt22k-t5
SWIN_CFG=configs/swin/kd-swin-l-to-swin-b-p4-w7-224_22kto1k.yaml
# PRETRAINED_CKPT=/data/vchua/run/msft-swin/swin/swin_base_patch4_window7_224_22kto1k_finetune/default/ckpt_epoch_29.pth
PRETRAINED_CKPT=/data/vchua/run/msft-swin/swin/pretrained/swin_base_patch4_window7_224_22k.pth

BS=256
MASTER_PORT=12345

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
   --nproc_per_node 2 \
   --master_port $MASTER_PORT \
   main.py \
    --cfg $SWIN_CFG \
    --pretrained $PRETRAINED_CKPT \
    --data-path $DATADIR \
    --output $OUTDIR \
   --batch-size $BS \
   --wandb_id $RUNID \
   --accumulation-steps 2

   #  --nncf_cfg $NNCF_CFG \
# original config used 8 cards
# ------------



