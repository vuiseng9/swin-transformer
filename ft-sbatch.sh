#!/usr/bin/env bash

cd /data/vchua/dev/msft-swin/swin-transformer

#python -m torch.distributed.launch \
#    --nproc_per_node 4 \
#    --master_port 12345 \
#    main.py \
#    --cfg configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml \
#    --pretrained pretrained/swin_base_patch4_window7_224_22k.pth \
#    --data-path /data/dataset/imagenet/ilsvrc2012/torchvision/ \
#    --batch-size 64 \
#    --accumulation-steps 2

# original config used 8 cards

# ------------

SWIN_CFG=configs/swin/swin_large_patch4_window7_224_22kto1k_finetune.yaml
PRETRAINED_CKPT=pretrained/swin_large_patch4_window7_224_22k.pth

DATADIR=/data/dataset/imagenet/ilsvrc2012/torchvision/ 
OUTDIR=$OUTROOT/$RUNID


python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 12345 \
    main.py \
    --cfg $SWIN_CFG \
    --pretrained $PRETRAINED_CKPT \
    --data-path $DATADIR \
    --output $OUTDIR \
    --batch-size 64 \
    --accumulation-steps 2

