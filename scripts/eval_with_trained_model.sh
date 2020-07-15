#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --config-file configs/kitti/pob/mask.yaml \
    --ckpt models/kitti/pob/smrcnn.pth \
    OUTPUT_DIR models/kitti/pob \
    DATASETS.TEST "('kitti_val_pob',)" \
    DATALOADER.NUM_WORKERS 0

python tools/split_predictions.py --prediction_path models/kitti/pob/inference/kitti_val_pob/predictions.pth --split_path data/kitti/object/split_set/val_set.txt
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/kitti/pob/idispnet.yaml MODEL.DISPNET.TRAINED_MODEL models/kitti/pob/idispnet.pth OUTPUT_DIR models/kitti/pob DATASETS.TEST "('kitti_val_pob',)" SOLVER.OFFLINE_2D_PREDICTIONS models/kitti/pob/inference/kitti_val_pob/predictions DATALOADER.NUM_WORKERS 0
python tools/split_predictions.py --prediction_path models/kitti/pob/inference/kitti_val_pob/predictions.pth --split_path data/kitti/object/split_set/val_set.txt
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/kitti/pob/rcnn.yaml --ckpt  models/kitti/pob/pointrcnn.pth OUTPUT_DIR models/kitti/pob DATASETS.TEST "('kitti_val_pob',)" SOLVER.OFFLINE_2D_PREDICTIONS models/kitti/pob/inference/kitti_val_pob/predictions DATALOADER.NUM_WORKERS 0