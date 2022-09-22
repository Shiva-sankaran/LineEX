import torch, cv2, os, sys, json
import numpy as np
import tqdm as tq
sys.path.append("/home/vp.shivasan/LineEX/")
print(os.getcwd())
from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
from scipy.interpolate import interp1d
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
parser.add_argument('--pred_save_path',default="LineEX/modules/grouping_legend/grouping_results")
parser.add_argument('--test_path',default='/home/md.hassan/charts/data/data/synth_lines/val/',help = "path to data (Ours, Adobe)")
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

# data_path = args.test_path
data_path = '/home/md.hassan/charts/data/data/synth_lines/val/'
image_path = data_path + 'images'
anno_path = data_path + 'anno/'


with open(anno_path + 'cls_anno.json') as f:
  cls_annos = json.load(f)
with open(anno_path + 'line_anno.json') as f:
  line_annos = json.load(f)

pred_line = {}
transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

def match_embeds(lines_annos, lines_embeds):
    annotations = {}
    for key, line_anno in lines_annos.items():
        x_temp, y_temp = (np.array(line_anno[0::2]).astype(int), np.array(line_anno[1::2]).astype(int))


        x_min_offset = x_temp[1] - x_temp[0]
        x_max_offset = x_temp[-1] - x_temp[-2]
        x_min = x_temp[0] - x_min_offset
        x_max = x_temp[-1] + x_max_offset

        f = interp1d(x_temp, y_temp, fill_value="extrapolate")

        x_new = np.linspace(x_min, x_max, num = (x_max - x_min + 1))
        y_new = f(x_new)
        annotations[key] = [x_new, y_new]        

    final_matching = {}
    matchings = {}
    for line_embed_id, line_embed in enumerate(lines_embeds):
        point_matching = [0]*len(line_embed)
        for point_idx, point in enumerate(line_embed):
            x = int(point[0])
            y = int(point[1])
            y_dist = 10000000
            outside_flag = 0
            for anno_key, line_anno in annotations.items():
                if x < line_anno[0][0] or x > line_anno[0][-1]: # if point outside of line, increase flag
                    outside_flag += 1
                else:
                    idx = np.where(line_anno[0]==x)
                    y_gt = line_anno[1][idx]
                    if abs(y_gt - y) < y_dist:
                        y_dist = abs(y_gt - y)
                        point_matching[point_idx] = anno_key
            if outside_flag == len(annotations): # if point outside of ALL lines, mark it as -1
                point_matching[point_idx] = -1

        matchings[line_embed_id] = list(np.unique(point_matching, return_counts=True)) # returns (values, frequencies)

    # for line_embed_id, anno_frequencies in matchings.items():
    #     if len(anno_frequencies[0]) > 0:
    #         max_freq = max(anno_frequencies[1])
    #         max_freq_idx = np.argmax(anno_frequencies[1])
    #         max_freq_id = anno_frequencies[0][max_freq_idx] if len(anno_frequencies[0]) > 1 else anno_frequencies[0][0]
    #         final_matching[line_embed_id] = (max_freq_id, max_freq)

    final_matching = {}
    matchings_ = matchings.copy()
    min_id = min(annotations.keys())
    max_id = max(annotations.keys())
    while min_id <= max_id:
        max_f = 0
        for line_embed_id, anno_frequencies in matchings_.items():
            if min_id in anno_frequencies[0]:
                if max_f < anno_frequencies[1][np.argwhere(anno_frequencies[0] == min_id)]:
                    max_f =  anno_frequencies[1][np.argwhere(anno_frequencies[0] == min_id)]
                    # print(max_f, line_embed_id)
                    if line_embed_id not in  final_matching.keys():
                        final_matching[line_embed_id] = (min_id, max_f[0][0])
                    elif final_matching[line_embed_id][1] < max_f:
                        final_matching[line_embed_id] = (min_id, max_f[0][0])
                        
        for line_embed_id, anno_frequencies in matchings_.items():
            if min_id in anno_frequencies[0]:
                idx__ = np.argwhere(anno_frequencies[0] == min_id)
                anno_frequencies[0] = np.delete(anno_frequencies[0], idx__)
                anno_frequencies[1] = np.delete(anno_frequencies[1], idx__)
        min_id += 1
    return final_matching


TP, FP, FN, Prec, Rec, F1, F1_w = 0, 0, 0, 0, 0, 0, 0
total_unmatched, total_lines = 0, 0
accuracy_dict = {}
for image_name in tq.tqdm(os.listdir(image_path)[:3000]):
    
    # if image_name == '4224.png' or image_name =="4212.png":
    #     continue
    for file in cls_annos["images"]:
      if file["file_name"] == image_name:
        id = file["id"]

    legend_bboxes = []
    legend_bboxes_ids = []
    for anno in cls_annos["annotations"]:
        if anno["image_id"] == id and anno["category_id"] == 7:
            legend_bboxes.append(anno["bbox"])
            legend_bboxes_ids.append(anno['line_id'])
    if(legend_bboxes == []):
        continue
    pred_line[image_name] = []

    line_kps = []
    lines_annos = {}
    for anno in line_annos["annotations"]:
        if anno["image_id"] == id:
            lines_annos[anno["id"]] = anno["bbox"]#anno["id"] is id of a line in the chart-> will map to marker image
            temp = []
            for i in range(0, len(anno["bbox"]), 2):
                if anno["bbox"][i] != 0.0 and anno["bbox"][i+1] != 0.0:
                    temp.append(anno["bbox"][i])
                    temp.append(anno["bbox"][i+1])
            if len(temp) > 1:
                x_temp, y_temp = (np.array(temp[0::2]).astype(int), np.array(temp[1::2]).astype(int))
                line_kps.append((x_temp, y_temp))

    all_kps = keypoints(model=LineEX,image_path=image_path +'/'+ image_name,input_size=args.input_size,CUDA_ = CUDA_)
    image = cv2.imread(image_path +'/'+ image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    legends_list = []

    Scores = np.zeros((len(legend_bboxes), len(all_kps))) # Legends in Rows, Lines in Cols
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
    for bbox in legend_bboxes:

        try:
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
    lines = []
    for i in range(len(legend_bboxes)):
        kp_indices = np.where(kp_mapping == i)[0]
        line = np.array(all_kps)[kp_indices]
        sorted_line = sorted(line, key=lambda x: x[0])
        line = [tuple(l) for l in sorted_line]
        line_ = [(int(l[0]),int(l[1])) for l in sorted_line]
        pred_line[image_name].append(line_)
        lines.append(line)
        
    matching = match_embeds(lines_annos, lines)
    # Result in form : {0: 13034, 1: 13035}. Grouped line no. matched to GT line.
    # Also, ith grouped line is predicted to match with the ith legend.
    # So, matching is a prediction of {..., legend no: GT line no, ...}
    # Also, number of predicted lines is always less than or equal to number of legend boxes
    # If pred lines less, then matching algo will also return less matchings than legend boxes. 
    # Handling this in metric calculation

    # DEBUG
    im = image.copy()
    if image_name != "7930.png":
        continue
    try:
        for line, box in zip(lines, legend_bboxes):
            cv2.line(im, line[-1], (int(box[0]), int(box[1])), (0, 0, 255), 1)
            cv2.polylines(im, [np.array(line, np.int32)], False, (0, 255, 0), 1)
            for i in line:
                cv2.circle(im, i, 2, (0, 0, 0), -1)        
        # for legend_idx, line_item in matching.items():
        #     cv2.putText(im, str(line_item[1]), lines[legend_idx][-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite('/home/md.hassan/charts/LineEX/modules/Grouping_legend_mapping/sample_output/' + image_name, im)
    except:
        print(image_name)
        continue

    line_ids = [i[0] for i in matching.values()]
    legend_ids = [i for i in matching.keys()]
    CM = np.zeros((len(matching), len(matching))) # along column: actual, along row: pred
    for legend_idx, line_item in matching.items():
        if legend_bboxes_ids[legend_idx] == line_item[0]:
            CM[legend_ids.index(legend_idx)][line_ids.index(line_item[0])] = 1
        else:
            CM[legend_ids.index(legend_idx)][line_ids.index(line_item[0])] = 1
            CM[line_ids.index(line_item[0])][legend_ids.index(legend_idx)] = 1

    F1 = 0.0
    total_lines += len(legends_list)
    # total_lines += len(matching)
    for i in range(len(matching)):
        TP = CM[i, i]
        FP = sum(CM[:, i]) - CM[i, i]
        FN = sum(CM[i, :]) - CM[i, i]
        Prec = TP / (TP + FP)
        Rec = TP / (TP + FN)
        if TP == 0:
            F1 += 0.0
        else:
            F1 += 2 * Prec * Rec / (Prec + Rec)
    if len(matching) < len(legends_list):
        F1 /= (len(matching) + 1 + (len(legends_list) - len(matching))) 
    # 1 for null class, rest for missing classes, F1 score for these classes will be 0, so just div
    else:
        F1 /= len(matching)
    print("F1 for chart: {}".format(F1))
    F1_w += F1 * len(legends_list)
    # F1_w += F1 * len(matching)

F1_w /= total_lines
print("Total lines = {}".format(total_lines))
print("Total F1 score weighted = {}".format(F1_w))

# # with open(args.pred_save_path+ "/pred.json", "w") as outfile:
# #     json.dump(pred_line, outfile)

    
#     # for line_idx_, line in lines.items():
#     #     if len(line) == 0 :
#     #         continue
#     #     legend_bbox = legend_bboxes[line_idx_]
#     #     draw.text((line[-1][0], line[-1][1]), str(len(line)), font = fnt, fill = (255, 0, 0))
#     #     xy_list = [(line[-1][0], line[-1][1]), (legend_bbox[0], legend_bbox[1])]
#     #     draw.line(xy_list, fill=(255, 0, 0), width=1)
#     # image_cls.save('/home/vp.shivasan/ChartIE/legend_mapping2/results/temp.png')


