#!/bin/bash
export NGPUS=4
export WORLD_SIZE=${NGPUS}

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py --config-file configs/kitti/pob/rcnn.yaml
