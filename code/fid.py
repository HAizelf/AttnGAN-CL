#fid.py
from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from miscc.utils import collapse_dirs, mv_to_paths
from datasets import TextDataset, ImageFolderDataset
from trainer import condGANTrainer
import pytorch_fid.fid_score
from tqdm import tqdm

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as transforms

from nltk.tokenize import RegexpTokenizer
from miscc.config import cfg, cfg_from_file
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    split_dir = 'test'
    bshuffle = True

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    save_dir = '../models/netG_epoch_600/valid'
    final_dir_g = '../models/netG_epoch_600/metrics'
    root_dir_g = save_dir

    device = torch.device( 'cuda' if (torch.cuda.is_available()) else 'cpu' )
    final_dir_g = str( Path( root_dir_g ).parent/'metrics' )
    num_metrics = 0

    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
    print( 'Computing FID...' )
    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
    orig_paths_g, final_paths_g = collapse_dirs(root_dir_g, final_dir_g )
    # -- #
    data_dir_r = '%s/CUB_200_2011' % dataset.data_dir if dataset.bbox is not None else dataset.data_dir
    root_dir_r = os.path.join( data_dir_r, 'images' )
    final_dir_r = os.path.join( root_dir_r, f'{imsize}x{imsize}' )
    orig_paths_r, final_paths_r = collapse_dirs( root_dir_r, final_dir_r, copy = True, ext = '.' + cfg.EXT_IN )
    dataset_rsz = ImageFolderDataset( img_paths = final_paths_r,
                                        transform = image_transform,  # transforms.Compose([transforms.Resize((imsize, imsize,))]),
                                        save_transformed = True )
    dataloader_rsz = torch.utils.data.DataLoader( dataset_rsz, batch_size = cfg.TRAIN.BATCH_SIZE,
                                                    drop_last = False, shuffle = False, num_workers = int(cfg.WORKERS) )
    dl_itr = iter( dataloader_rsz )
    print( f'Resizing real images to that of generated images and then saving into {final_dir_r}' )
    for batch_itr in tqdm( range( len( dataloader_rsz ) ) ):
        next( dl_itr )
    # -- #
    
    # print( f'Number of generated images to be used in FID calculation: {len( final_paths_g )}' )
    print( f'Number of real images to be used in FID calculation: {len( final_paths_r )}' )
    fid_value = pytorch_fid.fid_score.calculate_fid_given_paths( paths = [ final_dir_g, final_dir_r ],
                                                                            batch_size = 50,
                                                                            device = device,
                                                                            dims = 2048 )
    mv_to_paths( final_paths_g, orig_paths_g )
    with open( os.path.join( final_dir_g, 'metrics.txt' ), 'w' if num_metrics == 0 else 'a' ) as f:
        f.write( 'Frechet Inception Distance (FID): {:f}\n'.format( fid_value ) )
        f.write( 'Root Directories for Datasets used in Calculation: {}, {}\n\n'.format( root_dir_g, root_dir_r ) )
        num_metrics += 1
    print( '---> Frechet Inception Distance (FID): {:f}\n'.format( fid_value ) )
    #calculate fid score
