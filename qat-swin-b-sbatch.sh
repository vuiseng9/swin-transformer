#!/usr/bin/env bash

SWIN_CFG=configs/swin/qat-swin-b-p4-w7-224_22kto1k.yaml
NNCF_CFG=nncfcfg/swin_base_int8.json
PRETRAINED_CKPT=/data/vchua/run/msft-swin/swin/swin_base_patch4_window7_224_22kto1k_finetune/default/ckpt_epoch_29.pth

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
    --nncf_cfg $NNCF_CFG \
    --pretrained $PRETRAINED_CKPT \
    --data-path $DATADIR \
    --output $OUTDIR \
   --batch-size $BS \
   --accumulation-steps 2

# original config used 8 cards
# ------------



