#!/usr/bin/env python
#
# Creates training and validation splits based on images in training/image_2 directory.
# The training/val lists are random shuffled and 80/20 split
# Also create testing list based images in testing/image_2 directory. Test list is also random shuffled.
#
# Takes the path to KITTI dataset 'object' directory as input.
############################################################################
import os
import argparse
import glob
import random

def createLists(args):
    ###
    object_dir = os.path.abspath(args.object_dir)

    # Get list of training images
    trn_imgs_path = os.path.join(object_dir, 'training', 'image_2')
    os.chdir(trn_imgs_path)
    trn_imgs = glob.glob('*.png')

    # Get list of testing images
    tst_imgs_path = os.path.join(object_dir, 'testing', 'image_2')
    os.chdir(tst_imgs_path)
    tst_imgs = glob.glob('*.png')

    print(f'Found {len(trn_imgs)} training images and {len(tst_imgs)} test images.')

    random.shuffle(trn_imgs)
    nval = int(len(trn_imgs)*args.val_train_ratio) # Get 80/20 split of training and validation
    trn_list = [img[:-4] for img in trn_imgs[:-nval]]
    val_list = [img[:-4] for img in trn_imgs[len(trn_imgs)-nval:]]

    # Write lists
    split_dir = os.path.join(object_dir, 'split_set')
    os.makedirs(split_dir, exist_ok=True)

    outFile = open(os.path.join(split_dir, 'train_set.txt'), 'w')
    outFile.write('\n'.join(trn_list))
    outFile.write('\n')
    outFile.close()
    outFile = open(os.path.join(split_dir, 'val_set.txt'), 'w')
    outFile.write('\n'.join(val_list))
    outFile.write('\n')
    outFile.close()
    print(f'Created {len(trn_list)} training and {len(val_list)} validation image IDs.')

    random.shuffle(tst_imgs)
    tst_list = [img[:-4] for img in tst_imgs]
    outFile = open(os.path.join(split_dir, 'test_set.txt'), 'w')
    outFile.write('\n'.join(tst_list))
    outFile.write('\n')
    outFile.close()
    print(f'Created {len(tst_list)} testing image IDs.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("A utility to create train, val and test lists for DispRCNN")
    parser.add_argument(
        "--object-dir", "-d", help="Path to 'object' directory in the dataset", default=None, required=True
    )
    parser.add_argument(
        "--val-train-ratio", "-r", help="Ratio of validation to training images (e.g. 0.2 for 20% validation images)", default=0.2, type=float
    )

    args = parser.parse_args()
    if (args.val_train_ratio < 0.02 or args.val_train_ratio > 0.5):
        print(f"The validation to training ratio seems to be invalid ({args.val_train_ratio}). Try a value between 0.02 and 0.5")
        exit(1)
    createLists(args)
