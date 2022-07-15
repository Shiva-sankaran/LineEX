from cv2 import FileNode_NAMED
import torch, cv2, os, sys, json
import numpy as np
import tqdm as tq
from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
from modules.Grouping_legend_mapping.legend_models.net import *
from modules.KP_detection.models.my_model import Model
import matplotlib.pyplot as plt
from modules.KP_detection.utils import *
from modules.Grouping_legend_mapping.legend_models.MLP import legend_network
num_gpus = torch.cuda.device_count()
import argparse


parser = argparse.ArgumentParser('Set transformer detector', add_help=False)


## PE-former args ##
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
parser.add_argument('--vit_weights', type=str,
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
parser.add_argument('--show_keypoints', default=True, type=bool)

## CUSTOM ARGS ##
parser.add_argument('--input_path',default="LineEX/input")
parser.add_argument('--output_path',default="LineEX/output")
parser.add_argument('--data_path',default='/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/',help = "path to data (Ours, Adobe)")
parser.add_argument('--LineEX_path',default="LineEX/modules/KP_detection/pretrained_ckpts/ckpt_L.t7")
parser.add_argument('--use_gpu',default=True)
parser.add_argument('--cuda_id',default=3)

args, unknown = parser.parse_known_args()
if(args.use_gpu):
    CUDA_ = "cuda:"+str(args.cuda_id)
else:
    CUDA_ = "cpu"


MLP = legend_network(200)
MLP = MLP.to(CUDA_)
state = torch.load('/home/vp.shivasan/ChartIE/legend_mapping2/legend_network/training/cat_deepranking_mod_legend_line/line_latest_ckpt.t7', map_location = 'cpu')
MLP.load_state_dict(state['state_dict'])
MLP = MLP.to(CUDA_)
MLP.eval()

emb_model = TripletNet(resnet101())
state = torch.load('/home/vp.shivasan/image-similarity-using-deep-ranking/checkpoint3/ckpt_30.t7', map_location = 'cpu')
emb_model.load_state_dict(state['state_dict'])
emb_model.eval()
emb_model = emb_model.to(CUDA_)

LineEX  = Model(args)
LineEX = LineEX.to(CUDA_)
state2 = torch.load('/home/vp.shivasan/ChartIE/checkpoint/checkpoint_l1_ang_150_0.99.t7', map_location = 'cpu')
LineEX.load_state_dict(state2['state_dict'])
model2 = LineEX.to(CUDA_)
LineEX.eval()


pred_line = {}
transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

input_files = os.listdir(args.input_path)
print("Running whole pipeling for {} images".format(len(input_files)))
for image_name in tq.tqdm(input_files):
    print("Running: {}".format(image_name))
    file_path =args.input_path + "/" + image_name
    
    legend_bboxes = [] # ###################### HEEERREEEEE ###################
    if(legend_bboxes == []):
        continue
    pred_line[image_name] = []


    all_kps = keypoints(model=LineEX,image_path=file_path,input_size=args.input_size,CUDA_ = CUDA_)
    image_cls = Image.open(file_path)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    legends_list = []

    # ------ (Grouping and Legend mapping on GT keypoints) ------
    Scores = np.zeros((len(legend_bboxes), len(all_kps))) # Legends in Rows, Lines in Cols
    draw = ImageDraw.Draw(image_cls)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
    for bbox in legend_bboxes:
        try:
            draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
            crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]
            crop = Image.fromarray(crop).convert('RGB')
            tcrop = crop.copy()
            crop = transform_test(crop).reshape(1, 3, 224, 224)
        except:
            print("legend bbox out of bounds")
            continue
        
        legends_list.append(crop)

    for legend_idx, legend in enumerate(legends_list):
        legend = legend.to(CUDA_)
        legend_vec,_,_ = emb_model(legend,legend,legend)

        for kp_idx, kp in enumerate(all_kps):

            x, y = kp                
            bbox = [x - 20, y - 10, 40, 20]
            try:
                crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]
                crop = Image.fromarray(crop).convert('RGB')
                crop = transform_test(crop).reshape(1, 3, 224, 224)
            except:
                print("keypoint crop out of bound; skipping. image_name = ", + str(image_name))
                continue
            crop =crop.to(CUDA_)
            kp_vec,_,_ = emb_model(crop,crop,crop)
            with torch.no_grad():
                output = MLP(legend_vec, kp_vec)
                match_confidence = output.item()
            Scores[legend_idx][kp_idx] = match_confidence

    kp_mapping = Scores.argmax(axis=0)
    lines = {}
    for i in range(len(legend_bboxes)):
        kp_indices = np.where(kp_mapping == i)[0]
        line = np.array(all_kps)[kp_indices]
        sorted_line = sorted(line, key=lambda x: x[0])
        line = [tuple(l) for l in sorted_line]
        line_ = [(int(l[0]),int(l[1])) for l in sorted_line]
        
        draw.line(line, fill=(0, 255, 0), width=2)
        lines[i] = line
        pred_line[image_name].append(line_)
    

    for line_idx_, line in lines.items():
        if len(line) == 0 :
            continue
        legend_bbox = legend_bboxes[line_idx_]
        draw.text((line[-1][0], line[-1][1]), str(len(line)), font = fnt, fill = (255, 0, 0))
        xy_list = [(line[-1][0], line[-1][1]), (legend_bbox[0], legend_bbox[1])]
        draw.line(xy_list, fill=(255, 0, 0), width=1)
    save_path = args.output_path + "/" + image_name
    image_cls.save(save_path)


