#!/bin/bash

export NGPUS=4
export WORLD_SIZE=${NGPUS}

python -m torch.distributed.launch --nproc_per_node ${NGPUS} tools/train_net.py --config-file configs/kitti/pob/mask.yaml

#python tools/train_net.py --config-file configs/kitti/pob/mask.yaml
