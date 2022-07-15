from scipy.interpolate import interp1d
import json, numpy as np
import tqdm as tq
import math
import os


import argparse


parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

parser.add_argument('--ground_truth', default="/home/md.hassan/charts/data/data/Adobe_Synth/line/train/anno/adobe_line_anno.json", type=str)
parser.add_argument('--prediction', default="/home/vp.shivasan/ChartIE/legend_mapping2/adobe_lines_150.json", type=str)
parser.add_argument('--test_path',default='/home/md.hassan/charts/data/data/Adobe_Synth/line/train/',help = "path to data (Ours, Adobe)")

args, unknown = parser.parse_known_args()


def Err(P, G, i):
    N = len(P)
    P_x = [P[i][0] for i in range(N)]
    P_y = [P[i][1] for i in range(N)]

    f = interp1d(P_x, P_y, fill_value='extrapolate')
    I = f(G[i][0])
    return min(1, abs((G[i][1] - I)/G[i][1]))

def Intv(G, i):
    M = len(G)
    if i == 0:
        return (G[i+1][0] - G[i][0])/2
    if i == M-1:
        return (G[i][0] - G[i-1][0])/2
    else:
        return (G[i+1][0] - G[i-1][0])/2

def Rec(P, G):
    M = len(G)
    rec = 0
    for i in range(M):
        rec += (1 - Err(P, G, i))*Intv(G, i)/(G[M-1][0] - G[0][0])
    return rec


data_path = args.test_path
image_path = data_path + 'images'
anno_path = data_path + 'anno/'


with open(anno_path + 'adobe_cls_anno.json') as f:
  cls_annos = json.load(f)
with open(anno_path + 'adobe_line_anno.json') as f:
    line_annos = json.load(f)


def line_metric():
    with open(args.ground_truth) as f:
        GT_data = json.load(f)
    with open(args.prediction) as f:
        Pred_data = json.load(f)

    avg_score = 0
    total = 0
    num_files = 0
    npres = 0
    nempt = 0
    for file in tq.tqdm(GT_data["images"]):
        file_name = file["file_name"]
        if file_name not in os.listdir(image_path):
            continue
        legend_bboxes = []

        for file in cls_annos["images"]:
            if file["file_name"] == file_name:
                 id = file["id"]
        for anno in cls_annos["annotations"]:
            if anno["image_id"] == id and anno["category_id"] == 6:
                legend_bboxes.append(anno["bbox"])
        if(legend_bboxes == []):
            continue
        id = file["id"]
        try:
            Pred_full = Pred_data[file_name] # multiple lines
        except:
            npres+=1
            continue

        if(Pred_full == []):
            nempt+=1
            continue
        num_files+=1
        GT_raw = []
        for f in GT_data["annotations"]:
            if f["image_id"] == id:
                 GT_raw.append(f["bbox"])
        GT_full = []
        for line in GT_raw:
            temp = []
            for i in range(0, len(line), 2):
                temp.append([line[i], line[i+1]])
            if len(temp) > 1:
                GT_full.append(temp)
        
        N = len(Pred_full)
        M = len(GT_full)
        Scores = np.zeros((N, M))
        flag = 0
        for i in range(N):
            for j in range(M):
                if(Pred_full[i] == [] or len(Pred_full[i]) == 1):
                    flag = 1
                    break

                Scores[i][j] = Rec(Pred_full[i], GT_full[j])
            
            if(flag ==1 ):
                break
        if(flag == 1):
            continue
    
        if GT_full == []:
            continue
        elif Pred_full == []:
            continue
        elif N < M:
            s = np.amax(Scores, axis = 1)  # max in each row
            avg_score += np.mean(s)
        elif N >= M:
            s = np.amax(Scores, axis = 0)  # max in each column. equality?
            avg_score += np.mean(s)
        total += 1
        if math.isnan(avg_score):
            print("nan")

    print("\n\nAvg = " + str(avg_score/total))
    

line_metric()  
print("\n\n")
