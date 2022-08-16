#!/usr/bin/env bash

RUNID=swin-b-p4-w7-r224-22kto1k-ftuned
SWIN_CFG=configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml
PRETRAINED_CKPT=/data/vchua/run/msft-swin/swin/pretrained/swin_base_patch4_window7_224_22k.pth

BS=64
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
   --nproc_per_node 4 \
   --master_port $MASTER_PORT \
   main.py \
    --cfg $SWIN_CFG \
    --pretrained $PRETRAINED_CKPT \
    --data-path $DATADIR \
    --output $OUTDIR \
   --batch-size $BS \
   --accumulation-steps 2

# original config used 8 cards, 64 bs each card
# ------------

