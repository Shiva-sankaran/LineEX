import torch, cv2, os, sys
import numpy as np
import tqdm as tq
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from util import box_ops

from modules.CE_detection.models import build_det_model
import modules.CE_detection.util.misc as utils
from modules.CE_detection.util.utils import run_element_det

from modules.Grouping_legend_mapping.legend_models.net import *
from modules.Grouping_legend_mapping.legend_models.MLP import legend_network

from modules.KP_detection.models.my_model import Model
from modules.KP_detection.utils import *

num_gpus = torch.cuda.device_count()
import argparse
sys.path.append('/home/vp.shivasan/LineEX')
sys.path.append('/home/md.hassan/charts/LineEX')

parser = argparse.ArgumentParser()
parser_det = argparse.ArgumentParser()

## Detection args ##
parser_det.add_argument('--lr', default=1e-4, type=float)
parser_det.add_argument('--lr_backbone', default=1e-5, type=float)
parser_det.add_argument('--batch_size', default=1, type=int)
parser_det.add_argument('--weight_decay', default=1e-4, type=float)
parser_det.add_argument('--epochs', default=300, type=int)
parser_det.add_argument('--lr_drop', default=200, type=int)
parser_det.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Model parameters
parser_det.add_argument('--weights', type=str, default='/home/md.hassan/charts/detr/charts/ckpt/figqa_dataset/checkpoint110.pth')
# parser_det.add_argument('--weights', type=str, default='/home/md.hassan/charts/detr/charts/ckpt/final_dataset/checkpoint_latest.pth')

# * Backbone
parser_det.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser_det.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser_det.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser_det.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser_det.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser_det.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser_det.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser_det.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser_det.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser_det.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser_det.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser_det.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser_det.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser_det.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser_det.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser_det.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser_det.add_argument('--mask_loss_coef', default=1, type=float)
parser_det.add_argument('--dice_loss_coef', default=1, type=float)
parser_det.add_argument('--bbox_loss_coef', default=5, type=float)
parser_det.add_argument('--giou_loss_coef', default=2, type=float)
parser_det.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser_det.add_argument('--coco_path', type=str)
parser_det.add_argument('--dataset_file', default='charts')
parser_det.add_argument('--output_dir', default='/home/md.hassan/charts/detr/charts',
                    help='path where to save, empty for no saving')
parser_det.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser_det.add_argument('--num_workers', default=2, type=int)

parser_det.add_argument('--distributed', default=False, type=bool)

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
# parser.add_argument('--data_path', default="/home/vp.shivasan/data/data/ChartOCR_lines/line/images", type=str)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
                                                        (default from simple baselines is %(default)s)")

parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--show_keypoints', default=True, type=bool)

## CUSTOM ARGS ##
parser.add_argument('--input_path',default="sample_input")
parser.add_argument('--output_path',default="sample_output/")
# parser.add_argument('--data_path',default='/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/',help = "path to data (Ours, Adobe)")
# parser.add_argument('--LineEX_path',default="LineEX/modules/KP_detection/pretrained_ckpts/ckpt_L.t7")
parser.add_argument('--use_gpu',default=True)
parser.add_argument('--cuda_id',default=3)

args, unknown = parser.parse_known_args()
if(args.use_gpu):
    CUDA_ = "cuda:"+str(args.cuda_id)
else:
    CUDA_ = "cpu"
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

args_det = parser_det.parse_args()
args_det.device = CUDA_
det_model, _, _ = build_det_model(args_det)
checkpoint = torch.load(args_det.weights, map_location='cpu')
det_model.load_state_dict(checkpoint['model'])
det_model.to(CUDA_)
det_model.eval()
print("Loaded element detection model at Epoch {}".format(checkpoint['epoch']))

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

args.input_path = '/home/md.hassan/charts/LineEX/sample_input'
args.output_path = '/home/md.hassan/charts/LineEX/sample_output/'
input_files = os.listdir(args.input_path)
for f in os.listdir(args.output_path):
    os.remove(args.output_path + '/' + f) 

print("Running whole pipeling for {} images".format(len(input_files)))
for image_name in tq.tqdm(input_files):
    print("Running: {}".format(image_name))
    file_path = args.input_path + "/" + image_name

    legend_bboxes, legend_text, legend_text_boxes, legend_ele_boxes,  xticks_info, yticks_info, unique_boxes = run_element_det(det_model, file_path, image_name, args.output_path)
    x_text, x_coords, x_ratio, x_med_ids = xticks_info
    y_text, y_coords, y_ratio, y_med_ids = yticks_info
    if(legend_bboxes == []):
        continue
    pred_line[image_name] = []

    all_kps = keypoints(model=LineEX,image_path=file_path,input_size=args.input_size,CUDA_ = CUDA_)
    image_cls = Image.open(file_path)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # data1 = ratio * (pixel1 - pixel0) + data0
    scaled_kps = np.array(all_kps).copy()
    try:
        scaled_kps[:, 0] = (np.array(all_kps)[:, 0] - x_coords[x_med_ids[0]][0]) * x_ratio + x_text[x_med_ids[0]]
        scaled_kps[:, 1] = -1 * (np.array(all_kps)[:, 1] - y_coords[y_med_ids[0]][1]) * y_ratio + y_text[y_med_ids[0]]		
        for i, kp in enumerate(all_kps):
            image = cv2.circle(image, (int(kp[0]), int(kp[1])), radius=3, color=(0,255,0), thickness=-1)
            cv2.putText(image, str(round(float(str(scaled_kps[i,0])),1))+', '+str(round(float(str(scaled_kps[i,1])),1)), (int(kp[0]), int(kp[1])), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    except:
        temp = 0
    cv2.imwrite(args.output_path + 'kp_' + image_name, image)

    legends_list = []
    # ------ (Grouping and Legend mapping on GT keypoints) ------
    Scores = np.zeros((len(legend_bboxes), len(all_kps))) # Legends in Rows, Lines in Cols
    draw = ImageDraw.Draw(image_cls)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
    legend_bboxes = np.array(legend_bboxes)
    legend_bboxes[:, 0] = legend_bboxes[:, 0] - legend_bboxes[:, 2]/2
    legend_bboxes[:, 1] = legend_bboxes[:, 1] - legend_bboxes[:, 3]/2
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
    save_path = args.output_path + 'mapped_' + image_name
    image_cls.save(save_path)


