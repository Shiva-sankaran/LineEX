import torch, cv2, os, sys
import argparse, random
import numpy as np
import tqdm as tq
from models import build_det_model
import util.misc as utils
from util.utils import run_element_det
sys.path.append('/home/vp.shivasan/LineEX')
sys.path.append('/home/md.hassan/charts/LineEX')

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Model parameters
parser.add_argument('--weights', type=str, default='ckpts/checkpoint110.pth')
# parser.add_argument('--weights', type=str, default='/home/md.hassan/charts/detr/charts/ckpt/final_dataset/checkpoint_latest.pth')

# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser.add_argument('--coco_path', type=str)
parser.add_argument('--dataset_file', default='charts')
parser.add_argument('--output_dir', default='/home/md.hassan/charts/detr/charts',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--num_workers', default=2, type=int)

parser.add_argument('--distributed', default=False, type=bool)

num_gpus = torch.cuda.device_count()

# custom args
parser.add_argument('--input_path',default="sample_input/")
parser.add_argument('--output_path',default="sample_output/")

CUDA_ = 'cuda:3'
SEED = 42

seed = SEED + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# image_path = '/home/md.hassan/charts/metrics/det/chartie/Adobe_synth/test/images'
# image_path = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/images'
# image_path = '/home/md.hassan/charts/data/data/FigureSeerDataset/Annotated images'
# image_path = 'modules/CE_detection/sample_input'

# image_save_path = '/home/md.hassan/charts/ChartIE/results_synth/'
# image_save_path = 'modules/CE_detection/sample_output/'

# txt_save_path = '/home/md.hassan/charts/metrics/det/chartie/Adobe_synth/test/detections_postprocess/'
# txt_save_path = '/home/md.hassan/charts/metrics/det/chartie/synth/old_val_data/detections_postprocess/'
# txt_save_path = '/home/md.hassan/charts/metrics/det/chartie/FigureSeer/detections_postprocess/'



# Loading element detection model
args_det.device = CUDA_
det_model, _, _ = build_det_model(args_det)
checkpoint = torch.load(args_det.weights, map_location='cpu')
det_model.load_state_dict(checkpoint['model'])
det_model.to(CUDA_)
det_model.eval()
print("Loaded element detection model at Epoch {}".format(checkpoint['epoch']))

for f in os.listdir(args_det.output_path):
    os.remove(args_det.output_path + f) 

for image_name in tq.tqdm(os.listdir(args_det.input_path)):
    image_path = args_det.input_path + "/" + image_name
    legend_bboxes, legend_text, legend_text_boxes, legend_ele_boxes,  xticks_info, yticks_info, unique_boxes = run_element_det(det_model, image_path, image_name, args_det.output_path)

	# # FOR METRICS
	# txt_str = ""
	# for c, box in unique_boxes.items():
	# 	box = box_ops.box_cxcywh_to_xyxy(box)
	# 	txt_str += str(c)+" 1.0 "+str(int(box[0]))+" "+str(int(box[1]))+" "+str(int(box[2]))+" "+str(int(box[3]))+"\n"
	# txt_str = prepare_txt(txt_str, np.array(xticks_info[1]), 6)
	# txt_str = prepare_txt(txt_str, np.array(yticks_info[1]), 6)
	# txt_str = prepare_txt(txt_str, np.array(legend_bboxes), 7)
	# txt_str = prepare_txt(txt_str, np.array(legend_text_boxes), 8)
	# txt_str = prepare_txt(txt_str, np.array(legend_ele_boxes), 9)
	# txt_str = txt_str[:-1]
	# with open(txt_save_path +image_name[:-4]+'.txt', 'w') as f:
	# 	f.write(txt_str)
    x_text, x_coords, x_ratio, x_med_ids = xticks_info
    y_text, y_coords, y_ratio, y_med_ids = yticks_info
