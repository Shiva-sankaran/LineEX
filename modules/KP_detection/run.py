from cProfile import label
from html.entities import name2codepoint
from operator import mod
from cv2 import sqrt, threshold
import torch
import sys

from yaml import parse
sys.path.append("/home/vp.shivasan/ChartIE")
from models.my_model import Model
import os
import argparse
from datasets.coco_line import build as build_line
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
from tqdm import tqdm
import random
from utils import *


## PE-FORMER ARGS ##
parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

parser.add_argument('--batch_size', default=42, type=int)
parser.add_argument('--patch_size', default=16, type=int)


parser.add_argument('--position_embedding', default='enc_xcit',
                    type=str, choices=('enc_sine', 'enc_learned', 'enc_xcit',
                                        'learned_cls', 'learned_nocls', 'none'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', "glu"),
                    help="Activation function used for the transformer decoder")

parser.add_argument('--input_size', nargs="+", default=[288, 384], type=int,
                    help="Input image size. Default is %(default)s")
parser.add_argument('--hidden_dim', default=384, type=int,
                    help="Size of the embeddings for the DETR transformer")

parser.add_argument('--vit_dim', default=384, type=int,
                    help="Output token dimension of the VIT")
parser.add_argument('--vit_weights', type=str, default="https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth",
                    help="Path to the weights for vit (must match the vit_arch, input_size and patch_size).")

parser.add_argument('--vit_dropout', default=0., type=float,
                    help="Dropout applied in the vit backbone")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=1536, type=int,
                     help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--dropout', default=0., type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=64, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--data_path', default="/home/vp.shivasan/data/data/ChartOCR_lines/line/images", type=str)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
                                                        (default from simple baselines is %(default)s)")

parser.add_argument('--num_workers', default=24, type=int)


## CUSTOM ARGS ##

parser.add_argument('--input_path',default="sample_input")
parser.add_argument('--save_path',default="sample_output")
parser.add_argument('--show_LineEX_D',default = True)
parser.add_argument('--LineEX_D_Path',default="ckpts/ckpt_L.t7")
parser.add_argument('--show_LineEX_DA',default = True)
parser.add_argument('--LineEX_DA_Path',default="ckpts/ckpt_L+D.t7")
parser.add_argument('--use_gpu',default=True)
parser.add_argument('--cuda_id',default=3)



args = parser.parse_args()

assert (args.show_LineEX_D or args.show_LineEX_DA) == True, "Atleast one model should be True"

if(args.use_gpu):
    CUDA_ = 'cuda:'+str(args.cuda_id)

else:
    CUDA_ = 'cpu'

if(args.show_LineEX_D):
    LineEX_D  = Model(args)
    LineEX_D = LineEX_D.to(CUDA_)
    state = torch.load(args.LineEX_D_Path, map_location = 'cpu')
    LineEX_D.load_state_dict(state['state_dict'])
    LineEX_D = LineEX_D.to(CUDA_)
    LineEX_D.eval()


if(args.show_LineEX_DA):
    LineEX_DA  = Model(args)
    LineEX_DA = LineEX_DA.to(CUDA_)
    state2 = torch.load(args.LineEX_DA_Path, map_location = 'cpu')
    LineEX_DA.load_state_dict(state2['state_dict'])
    LineEX_DA = LineEX_DA.to(CUDA_)
    LineEX_DA.eval()



torch.manual_seed(317)
torch.backends.cudnn.benchmark = True  
num_gpus = torch.cuda.device_count()

def save_keypoints(image_path,model,model_type = "DA"):
    pred_kp = keypoints(model = model, image_path= image_path,input_size= (args.input_size[0], args.input_size[1]), CUDA_ = CUDA_)
    timage= cv2.imread(image_path)
    for x,y in pred_kp:
        timage = cv2.circle(timage,(x,y),radius=5,color=(0,0,255),thickness=-1)

    save_image_path = args.save_path + "/"+ model_type + "_" + image_path.split("/")[-1]
    print("Saving: " ,save_image_path)
    cv2.imwrite(save_image_path,timage)
    

input_files = os.listdir(args.input_path)
print("Running keypoint detection model on {} images".format(len(input_files)))

for file in input_files:
    file_path = args.input_path + "/" + file
    print("Processing: ",file_path)
    
    if(args.show_LineEX_D):
        save_keypoints(image_path = file_path,model = LineEX_D, model_type = "D")
    if(args.show_LineEX_DA):
        save_keypoints(image_path = file_path,model = LineEX_DA, model_type = "DA")
    
