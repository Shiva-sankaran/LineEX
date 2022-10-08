import torch
from models.my_model import Model
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm
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

# parser.add_argument('--vit_weights', type=str,
#                     help="Path to the weights for vit (must match the vit_arch, input_size and patch_size).")

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
parser.add_argument('--dataset',default="Ours", help="Ours,ExcelChart400k,Adobe")
parser.add_argument('--image_dir',default="/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/images/",help = "Path to image directory of the dataset")
parser.add_argument('--anno_dir',default="/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/anno/line_anno.json",help = "Path to annotations directory of the dataset")
parser.add_argument('--chartOCR_pred',default="/home/vp.shivasan/ChartIE/val_synth.json",help = "Path to chartOCR predictions on the dataset")


parser.add_argument('--show_LineEX_D',default = True)
parser.add_argument('--LineEX_D_Path',default="ckpts/ckpt_L.t7")
parser.add_argument('--show_LineEX_DA',default = True)
parser.add_argument('--LineEX_DA_Path',default="ckpts/ckpt_L+D.t7")
parser.add_argument('--use_gpu',default=True)
parser.add_argument('--cuda_id',default=3)


args = parser.parse_args()

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

                
def keypoints(image_name,ANNO_FILE,model,dataset = "ChartOCR"):
    image_path = IMAGE_DIR+image_name
    image= cv2.imread(image_path)
    timage = image
    
    h,w,_ = image.shape
    image = cv2.resize(image, (args.input_size[0], args.input_size[1]))
    image = np.asarray(image)
    image = image.astype(np.float32) / 255
    
    image = torch.from_numpy(image)
    image = image.permute((2, 0, 1))
    image = torch.unsqueeze(image,0)

    image = image.to(CUDA_, non_blocking=True)
    
    
    output = model(image,return_attn = True)
    out_bbox = output["pred_boxes"][0]
    out_bbox = out_bbox.cpu().detach().numpy()
    x_cords = (out_bbox[:,0]*w).astype(np.uint32)
    y_cords = (out_bbox[:,1]*h).astype(np.uint32)

    with open(ANNO_FILE) as f:
        ANNO = json.load(f)
        for file in ANNO["images"]:
            if file["file_name"] == image_name:
                id = file["id"]
        ground_kp = []
        ground_lines = []
        for anno in ANNO["annotations"]:
            if(anno["image_id"] == id):
                ground_kp.extend(anno["bbox"])  
                ground_lines.append(anno['bbox'])
    g_lines = []
    for line in ground_lines:
        g_x_cords = np.array(line[0::2]).astype(int)
        g_y_cords = np.array(line[1::2]).astype(int)
        temp = []
        for x,y in zip(g_x_cords,g_y_cords):
            temp.append((x,y))
        g_lines.append(temp)

    pred_kp = []
    g_kp = []
    g_x_cords = np.array(ground_kp[0::2]).astype(int)
    g_y_cords = np.array(ground_kp[1::2]).astype(int)
    for x,y in zip(x_cords,y_cords):
        if(checkifbackground((x,y),timage)):
            continue

        pred_kp.append((x,y))
    for x,y in zip(g_x_cords,g_y_cords):
        g_kp.append((x,y))

    chartocr_kps = get_chartocr_kp(CHART_OCR_KP,image_name,dataset=dataset)
    
    return pred_kp,g_kp,g_lines,chartocr_kps




# IMAGEPATH = "/home/vp.shivasan/data/data/ChartOCR_lines/line/images/test2019/f4a2a26ef3d6b6da14e661c013590327_d3d3LnNpZS5nb2IuZG8JMzUuMTg1LjgzLjIwNw==-0-0.png"
# image_name = "f4a2a26ef3d6b6da14e661c013590327_d3d3LnNpZS5nb2IuZG8JMzUuMTg1LjgzLjIwNw==-0-0.png"


#CHARTOCR DATASET
# IMAGE_DIR =  "/home/vp.shivasan/data/data/ChartOCR_lines/line/images/test2019/"
# ANNO_FILE = "/home/vp.shivasan/data/data/ChartOCR_lines/line/annotations_cleaned/clean_instancesLine(1023)_test2019.json"
# CHART_OCR_KP = "/home/vp.shivasan/ChartIE/test.json"
# print("RUnning with chartOCR dataset")

# dataset = "chartOCR"

# OUR SYNTHETIC DATASET
# IMAGE_DIR = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/images/'
# ANNO_FILE = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/anno/line_anno.json'
# CHART_OCR_KP = "/home/vp.shivasan/ChartIE/val_synth.json"
# print("running with our synth dataset")

# dataset = "chartOCR"

#ADOBE DATASET
# IMAGE_DIR =  "/home/md.hassan/charts/data/data/Adobe_Synth/line/train/images/"
# ANNO_FILE = "/home/md.hassan/charts/data/data/Adobe_Synth/line/train/anno/adobe_line_anno.json"
# CHART_OCR_KP = "/home/md.hassan/charts/s_CornerNet/predicted/line/anno/adobe_synth_train.json"
# print("RUnning with Adobe dataset")

# dataset = "adobe"




# pdj_accs2 = []
# oks_accs2 = []

# c_pdj_accs = []
# c_oks_accs = []

dataset = args.dataset
IMAGE_DIR = args.image_dir
ANNO_FILE = args.anno_dir
CHART_OCR_KP = args.chartOCR_pred

print("Evaluating metrics on {} dataset".format(dataset))


file_names = []
with open(ANNO_FILE) as f:
        ANNO = json.load(f)
        for file in ANNO["images"]:
            file_names.append(file["file_name"])


oks_recall1 = []
oks_recall2 = []
oks_recall_c = []

oks_prec1 = []
oks_prec2 = []
oks_prec_c = []

oks_f11 = []
oks_f12 = []
oks_f1_c= []


oks_recall1_p = []
oks_recall2_p = []
oks_recall_c_p = []


oks_prec1_p = []
oks_prec2_p = []
oks_prec_c_p = []


oks_f11_p = []
oks_f12_p = []
oks_f1_c_p= []

for file_name in tqdm(file_names[:1]):
    try:
        file_path = IMAGE_DIR+file_name
        

        pred_kp,ground_kp,g_lines1,chart_kp = keypoints(file_name,ANNO_FILE,LineEX_D,dataset = dataset)
        pred_kp2,ground_kp2,g_lines2,chart_kp2 = keypoints(file_name,ANNO_FILE,LineEX_DA,dataset = dataset)

        recall_oks1,precision_oks1,F1_oks1 = metric(pred_kp,ground_kp,g_lines1,file_path,relaxed = False)
        recall_oks2,precision_oks2,F1_oks2 = metric(pred_kp2,ground_kp2,g_lines2,file_path,relaxed= False)
        recall_oks_c,precision_oks_c,F1_oks_c = metric(chart_kp2,ground_kp2,g_lines1,file_path,relaxed=False)

        recall_oks1_p,precision_oks1_p,F1_oks1_p = metric(pred_kp,ground_kp,g_lines1,file_path,relaxed = True)
        recall_oks2_p,precision_oks2_p,F1_oks2_p = metric(pred_kp2,ground_kp2,g_lines2,file_path,relaxed = True)
        recall_oks_c_p,precision_oks_c_p,F1_oks_c_p = metric(chart_kp2,ground_kp2,g_lines1,file_path,relaxed = True)


        oks_recall1.append(recall_oks1)
        oks_recall2.append(recall_oks2)
        oks_recall_c.append(recall_oks_c)

        oks_prec1.append(precision_oks1)
        oks_prec2.append(precision_oks2)
        oks_prec_c.append(precision_oks_c)

        oks_f11.append(F1_oks1)
        oks_f12.append(F1_oks2)
        oks_f1_c.append(F1_oks_c)

        oks_recall1_p.append(recall_oks1_p)
        oks_recall2_p.append(recall_oks2_p)
        oks_recall_c_p.append(recall_oks_c_p)

        oks_prec1_p.append(precision_oks1_p)
        oks_prec2_p.append(precision_oks2_p)
        oks_prec_c_p.append(precision_oks_c_p)

        oks_f11_p.append(F1_oks1_p)
        oks_f12_p.append(F1_oks2_p)
        oks_f1_c_p.append(F1_oks_c_p)

    except: # LEAVE DUBIOUS IMAGES
        continue

print("Showing stricter version metric\n\n")
print("##########################################################\n")

print("Model:LineEX_D")
print(" recall : {}".format(np.mean(oks_recall1)))
print(" precision : {}".format(np.mean(oks_prec1)))
print(" F1 score : {}\n".format(np.mean(oks_f11)))

print("Model:LineEX_DA")
print(" recall : {}".format(np.mean(oks_recall2)))
print(" precision : {}".format(np.mean(oks_prec2)))
print(" F1 score : {}\n".format(np.mean(oks_f12)))

print("Model:ChartOCR")
print(" recall : {}".format(np.mean(oks_recall_c)))
print(" precision : {}".format(np.mean(oks_prec_c)))
print(" F1 score : {}\n".format(np.mean(oks_f1_c)))
print("##########################################################\n")

print("Showing relaxed version metric\n\n")
print("##########################################################\n")

print("Model:LineEX_D")
print(" recall : {}".format(np.mean(oks_recall1_p)))
print(" precision : {}".format(np.mean(oks_prec1_p)))
print(" F1 score : {}\n".format(np.mean(oks_f11_p)))

print("Model:LineEX_DA")
print(" recall : {}".format(np.mean(oks_recall2_p)))
print(" precision : {}".format(np.mean(oks_prec2_p)))
print(" F1 score : {}\n".format(np.mean(oks_f12_p)))

print("Model:ChartOCR")
print(" recall : {}".format(np.mean(oks_recall_c_p)))
print(" precision : {}".format(np.mean(oks_prec_c_p)))
print(" F1 score : {}\n".format(np.mean(oks_f1_c_p)))

print("##########################################################\n")
