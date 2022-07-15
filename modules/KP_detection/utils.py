import numpy as np
import cv2
import json

import torch

def checkifbackground(kp,image):

    alpha = max(image.shape[0],image.shape[1])*0.05
    t1 = int(alpha/2)
    t2 = int(alpha/6)
    x,y = kp
    threshold = 0.999   #0.98 for adobe,0.999 for chartOCR and synt
        
    sections = [
        

        image[y-t1:y-t2,x-t1:x-t2],
        image[y-t1:y-t2,x-t2:x+t2],
        image[y-t1:y-t2,x+t2:x+t1],
        image[y-t2:y+t2,x-t1:x-t2],
        image[y-t2:y+t2,x-t2:x+t2],
        image[y-t2:y+t2,x+t2:x+t1],
        image[y+t2:y+t1,x-t1:x-t2],
        image[y+t2:y+t1,x-t2:x+t2],
        image[y+t2:y+t1,x+t2:x+t1],
        ]
    

    sec_hists = []
    
    for i,section in enumerate(sections):
        hist = cv2.calcHist([section], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        sec_hists.append(hist)
        

    for idx1,h1 in enumerate(sec_hists):
        for idx2,h2 in enumerate(sec_hists):
            if(idx1 == idx2):
                continue
            val = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        
            if(val < threshold):
                return False
    return True



def min_distance_gkp(kp,kp_list,image):
    kp_list = np.array(kp_list)
    leftbottom = np.array(kp)
    distances = np.linalg.norm(kp_list-leftbottom, axis=1)
    min_idx = np.argmin(distances)
    return distances[min_idx],tuple(kp_list[min_idx])

def oks(d,s,k):
    return np.exp(-(d**2)/(2*(s**2)*(k**2)))

def find_line(lines,pt):
    for line in lines:
        if(pt in line):
            return line
def enc_pt(pt,g_pt,line):
    idx = -1
    for i,p in enumerate(line):
        if(p == g_pt):
            idx = i
            break
    if(g_pt[0]>pt[0]):
        if(idx ==0):
            return -1,-1
        return g_pt,line[idx-1]
    else:
        if(idx == (len(line)-1)):
            return -1,-1
        return g_pt,line[idx+1] 
    

def metric(pred_kp,ground_kp,ground_lines,image_path,relaxed = True):
    image= cv2.imread(image_path)
    image2 = image.copy()
    h,w,_ = image.shape
    s = np.sqrt(h*w)
    d = np.sqrt(h**2+w**2)
    k = 0.025
    threshold_pdj = 0.05*d
    threshold_oks = 0.5
    threshold2= 0.007*d
    tp_oks = 0
    fp_oks = 0
    fn_oks = 0
    found_ground_kps_oks = []
    for kp in pred_kp:
        min_dist,gkp = min_distance_gkp(kp,ground_kp,image)
        oks_val = oks(min_dist,s,k)
        if(oks_val>threshold_oks):
            if(gkp not in found_ground_kps_oks):
                found_ground_kps_oks.append(gkp)
                tp_oks+=1
        else:
            if(relaxed):
                gline = find_line(ground_lines,gkp)
                gpt1,gpt2 = enc_pt(kp,gkp,gline)
                if(gpt1 == -1):
                    fp_oks+=1
                else:
                    gpt1 = np.asarray(gpt1)
                    gpt2 = np.asarray(gpt2)
                    d = np.linalg.norm(np.cross(gpt2-gpt1, gpt1-kp))/np.linalg.norm(gpt2-gpt1)
                    if(d<threshold2):
                        if(gkp not in found_ground_kps_oks):
                            found_ground_kps_oks.append(gkp)
                            tp_oks+=1
                    else:
                        fp_oks+=1
            else:
                fp_oks+=1


    
    fn_oks = len(ground_kp) - len(found_ground_kps_oks)

    recall_oks = tp_oks/(tp_oks+fn_oks)
    precision_oks = tp_oks/(tp_oks+fp_oks)
    F1_oks = 2*tp_oks/(2*tp_oks + fp_oks + fn_oks)

    return recall_oks,precision_oks,F1_oks


def get_chartocr_kp(json_path,image_name,dataset):
    kps = []
    if(dataset == "ExcelChart400k" or dataset == "Ours"):
        with open(json_path) as file:
            json_file = json.load(file)
            for ele in json_file[image_name]:
                if(ele["score"]>0.5):
                    kps.append((int(ele["bbox"][0]),int(ele["bbox"][1])))
    elif(dataset == "Adobe"):
        with open(json_path) as file:
            json_file = json.load(file)
            for ele in json_file[image_name]:
                for pt in ele:
                    kps.append((int(pt[0]),int(pt[1])))
    else:
        assert 1==0

    return kps


def keypoints(model,image_path,input_size,CUDA_ = "cpu"):

    image= cv2.imread(image_path)
    timage = image.copy()
    h,w,_ = image.shape
    image = cv2.resize(image, (input_size[0],input_size[1]))
    image = np.asarray(image)
    image = image.astype(np.float32) / 255
    
    image = torch.from_numpy(image)
    image = image.permute((2, 0, 1))
    image = torch.unsqueeze(image,0)

    torch.tensor(image, dtype=torch.float32)
    image = image.to(CUDA_, non_blocking=True)
    
    
    output = model(image,return_attn =False)
    out_bbox = output["pred_boxes"][0]
    out_bbox = out_bbox.cpu().detach().numpy()
    x_cords = (out_bbox[:,0]*w).astype(np.uint32)
    y_cords = (out_bbox[:,1]*h).astype(np.uint32)
    pred_kp  = []
    for x,y in zip(x_cords,y_cords):
        if(checkifbackground((x,y),timage)):
            continue

        pred_kp.append((x,y))
    return pred_kp

