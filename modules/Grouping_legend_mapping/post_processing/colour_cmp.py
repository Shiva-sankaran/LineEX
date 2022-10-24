import torch, cv2, os, sys
import numpy as np
def hist_cmp(img1,img2,METHOD=None):
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    # img1 = img1[0].permute((1,2,0))
    # img2 = img2[0].permute((1,2,0))
    # img1 = img1.cpu().detach().numpy()
    # img2 = img2.cpu().detach().numpy()

    # cv2.imwrite("/home/vp.shivasan/LineEX/temp/kp_patch.png",img1)
    # cv2.imwrite("/home/vp.shivasan/LineEX/temp/legend_patch.png",img2)
    

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist_img1 = cv2.calcHist([img1], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_img2 = cv2.calcHist([img2], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_KL_DIV)

    if(metric_val == 0):
        return 10000000

    return 1/metric_val