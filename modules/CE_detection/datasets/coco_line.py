"""
COCO Person dataset.
Persons (Cropped) with keypoints.

Code adapted from the simplebaselines repo:
https://github.com/microsoft/human-pose-estimation.pytorch/tree/master/lib/dataset

"""

from curses import KEY_UNDO
from tkinter import Y
import torch
import torchvision
from pathlib import Path
import copy
import cv2
import random
import matplotlib.pyplot as plt

from utils.sb_transforms import fliplr_joints, affine_transform, get_affine_transform

import datasets.transforms as T
from torchvision import transforms

from PIL import Image
from typing import Any, Tuple, List
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img, target):
        do_it = random.random() <= self.prob
        if not do_it:
            return img, target

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))), target


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return ImageOps.solarize(img), target
        else:
            return img, target


class ColorJitter(object):

    def __init__(self, jitter_p=0.8, gray_p=0.2):
        color_jitter = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=0.4,
                                                                                          contrast=0.4,
                                                                                          saturation=0.2,
                                                                                          hue=0.1)],
                                                                  p=jitter_p),
                                          transforms.RandomGrayscale(p=gray_p)])
        self.tr = color_jitter

    def __call__(self, img, target):
        return self.tr(img), target


def make_coco_person_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.NormalizePerson([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO move resize/augment operations here
    # instead of the dataset
    if image_set == 'train':
        # tr = T.Compose([ColorJitter(0.8, 0.2),
                        # GaussianBlur(0.1),
                        # Solarization(0.2),
                        # normalize])
        return normalize  # tr

    if image_set == 'val':
        return normalize

    raise ValueError(f'unknown {image_set}')

# from ChartOCR
def _get_border(border, size):
  i = 1
  while size - border // i <= border // i:
    i *= 2
  return border // i

# from ChartOCR
def random_crop_line(image, detections, random_scales, view_size, border=64):
    view_height, view_width = (view_size['h'], view_size['w'])
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:-1:2] -= x0
    cropped_detections[:, 1:-1:2] -= y0
    cropped_detections[:, 0:-1:2] += cropped_ctx - left_w
    cropped_detections[:, 1:-1:2] += cropped_cty - top_h

    return cropped_image, cropped_detections, scale

def crop_image(image, center, new_size):
  cty, ctx = center
  height, width = new_size
  im_height, im_width = image.shape[0:2]
  cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

  x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
  y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

  left, right = ctx - x0, x1 - ctx
  top, bottom = cty - y0, y1 - cty

  cropped_cty, cropped_ctx = height // 2, width // 2
  y_slice = slice(cropped_cty - top, cropped_cty + bottom)
  x_slice = slice(cropped_ctx - left, cropped_ctx + right)
  cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

  border = np.array([
    cropped_cty - top,
    cropped_cty + bottom,
    cropped_ctx - left,
    cropped_ctx + right
  ], dtype=np.float32)

  offset = np.array([
    cty - height // 2,
    ctx - width // 2
  ])

  return cropped_image, border, offset

def color_jittering_(data_rng, image):
  functions = [brightness_, contrast_, saturation_]
  random.shuffle(functions)

  gs = grayscale(image)
  gs_mean = gs.mean()
  for f in functions:
    f(data_rng, image, gs, gs_mean, 0.4)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
  alpha = data_rng.normal(scale=alphastd, size=(3,))
  image += np.dot(eigvec, eigval * alpha)

def saturation_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs_mean)

def blend_(alpha, image1, image2):
  image1 *= alpha
  image2 *= (1 - alpha)
  image1 += image2

def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class CocoLine(torchvision.datasets.VisionDataset):
    '''
    "keypoints": {
        0: "line"
    },
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, root, ann_file, image_set, transforms=None, is_train=False,
                 input_size=(224, 224), scale_factor=0.3):
        super().__init__(root)
        self.calculate_anchors = True
        self.image_set = image_set
        self.is_train = is_train
        self.image_size = input_size
        self.split = image_set
        self.coco = COCO(ann_file)
        self.image_set_index = self.coco.getImgIds()

        self.db = self._get_db()
        self.anchor_db = {}
        

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        if(self.image_set == "train"):
            image_file = str(self.root) + '/'  + str(db_rec['file_name'])
            
        else:
            image_file = str(self.root) + '/' + self.image_set + '2019/' + str(db_rec['file_name'])
            
        

        image  = cv2.imread(image_file)
        height, width, _ = image.shape
        image = cv2.resize(image, (self.image_size[0], self.image_size[1]))
        image = np.asarray(image)

        bboxes = db_rec['objs']
        bboxes = [bbox.tolist() for bbox in bboxes]

        m = 0
        for i in bboxes:
            m = max(m, len(i))                
        max_len = m
        for ind_bbox in range(len(bboxes)):   #if in the same chart, one line has more points... 
            if len(bboxes[ind_bbox]) < max_len: #then pad others to match length.CAUSES PREDICTIONS @ORIGIN?
                bboxes[ind_bbox] = np.pad(bboxes[ind_bbox], (0, max_len - len(bboxes[ind_bbox])), 'constant',
                                        constant_values=(0, 0))

        bboxes = np.array(bboxes)
        X_cords = bboxes[:, 0::2] *( 1 / width)
        Y_cords = bboxes[:, 1::2] *( 1 / height) # xy coords become 0 to 1
        X_cords = X_cords.flatten()
        Y_cords = Y_cords.flatten()
        bboxes = np.array((X_cords,Y_cords)).T
        temp1 = bboxes>0
        temp2 = bboxes<1
        z_ind = np.where(np.sum(temp1*temp2,axis=1)!=2)
        bboxes = np.delete(list(bboxes),list(z_ind[0]),axis=0)
        if(bboxes.shape[0]<64):
            mask = np.vstack((np.ones((bboxes.shape[0],2)),np.zeros((64-bboxes.shape[0],2))))
            mask_len = bboxes.shape[0]
            bboxes = np.vstack((bboxes,np.zeros((64-bboxes.shape[0],2))))
        elif(bboxes.shape[0]>=64):
            mask = np.ones((64,2))
            mask_len = bboxes.shape[0]
            bboxes = bboxes[:64]
            
        ten = np.array([0.0,0.0])
        temp1=0
        temp2 = 0
        for i in mask:
            if((i==ten).all()):
                temp1+=1
        for i in bboxes:
            if((i==ten).all()):
                temp2+=1
        if(temp1!=temp2):
            print(temp1,temp2)


        
        image = image.astype(np.float32) / 255.

        # randomly change color and lighting
        if self.split == 'train':
            color_jittering_(np.random.RandomState(), image)


        image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        




        l=[]
        for char in image_file:
            l.append(ord(char))
        image_file_arr = np.array(l)

        
        if(image_file not in self.anchor_db.keys()):
            img1 = cv2.imread(image_file)
            img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            X_cords = bboxes[:, 0::2] *(width)
            Y_cords = bboxes[:, 1::2] *(height) 
            X_cords = X_cords.flatten()
            Y_cords = Y_cords.flatten()
            keypoints = np.array((X_cords,Y_cords),dtype=np.int32).T
            anchors_x,anchors_y = self.get_anchors(image_file,keypoints)
            anchors_x = anchors_x * ( 1 / width)
            anchors_y = anchors_y * ( 1 / height)
            anchors = np.array((anchors_x,anchors_y)).T
            self.anchor_db[image_file] = anchors


            
        # print(self.image_set)
        if(self.image_set == "train"):
            target = {
                'image':torch.tensor(image, dtype=torch.float32),
                'size': torch.tensor(list(self.image_size)),
                'orig_size': torch.tensor([width, height]),
                'image_id': torch.tensor([db_rec['image_id']], dtype=torch.int64),
                'bboxes': torch.as_tensor(bboxes, dtype=torch.float32),
                'labels': np.arange(1),
                'masks':mask.astype(float),
                'mask_len':mask_len,
                'anchors':torch.as_tensor(self.anchor_db[image_file], dtype=torch.float32),
            }
        else:
            target = {
                'image':torch.tensor(image, dtype=torch.float32),
                'size': torch.tensor(list(self.image_size)),
                'orig_size': torch.tensor([width, height]),
                'image_id': torch.tensor([db_rec['image_id']], dtype=torch.int64),
                'bboxes': torch.as_tensor(bboxes, dtype=torch.float32),
                'labels': np.arange(1),
                'masks':mask.astype(float),
                'mask_len':mask_len,
                'anchors':torch.as_tensor(self.anchor_db[image_file], dtype=torch.float32),
                'image_file':image_file,
                'image_file':image_file_arr
            }

        return target

    def __len__(self) -> int:
        return len(self.db)

    def _get_db(self):
        # use ground truth bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db
    
    
    def _load_coco_person_detection_results(self):
        import json
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            print('=> Load %s fail!' % self.bbox_file)
            return None

        print('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            index = det_res['image_id']
            img_name = self.image_path_from_index(index)
            box = det_res['bbox']
            score = det_res['score']
            area = box[2] * box[3]

            if score < self.image_thre or area < 32**2:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image_id': index,
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,  # Try this score for evaluation (with COCOEval)
                'area': area,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        print('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        file_name = im_ann['file_name']

        annIds = self.coco.getAnnIds(imgIds=index)
        objs = self.coco.loadAnns(annIds)

        rec = []
        temp = []
        for obj in objs:
            temp.append(np.array(obj['bbox']))

        

        rec.append({
            'image_id': index,
            'image': self.image_path_from_index(index),
            'objs': temp,
            'file_name': file_name,
            
        })

        return rec
    
    def get_anchors(self,image_file,keypoints):
        img = cv2.imread(image_file)
        orig = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #FOR LOOP
        anchor_x = np.array([])
        anchor_y = np.array([])
        for i,j in keypoints:
            temp = self.get_anchor_point((i,j),img,image_file)
            assert temp != None
            anchor_x = np.append(anchor_x,temp[0])
            anchor_y = np.append(anchor_y,temp[1])

        anchor_x = anchor_x.astype(np.uint32)
        anchor_y = anchor_y.astype(np.uint32)
        return anchor_x,anchor_y


    def get_anchor_point(self,ground_point,image,image_file):
        if(ground_point ==(0,0)):
            return (0,0)
        image = image.astype(np.int32)
        r_start = 5.0
        r_delta = 0.1
        thetha = np.pi/6
        tolerance =30
        angles = np.arange(0,2*np.pi+0.0001,thetha)
        cosines = np.cos(angles)
        sines = np.sin(angles)
        rs = np.arange(r_start,0,-r_delta)
        for r in rs:
            x,y = ground_point
            x_new = x+ r*cosines 
            y_new = y + r*sines
            x_new = x_new.astype(np.uint32)
            y_new = y_new.astype(np.uint32)
            g_pixel = image[y][x]
            
            for i,j in zip(x_new,y_new):
                if(j>=image.shape[0] or i>=image.shape[1] or j<0 or i<0):
                    continue
                pixel = image[j][i]
                if((np.absolute(g_pixel-pixel)<=tolerance).all()):
                    return i,j

        return None

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        # PPP Tight bbox
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        root = Path(self.root)
        file_name = '%012d.jpg' % index
        image_path = root / f"{self.image_set}2017" / file_name

        return image_path


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    PATHS = {
        "train": ('/home/vp.shivasan/data/data/new_train/anno/combined_line_anno.json'),#('/home/vp.shivasan/data/data/ChartOCR_lines/line/annotations_cleaned/clean_instancesLine(1023)_train2019.json'),
        "val":('/home/vp.shivasan/data/data/ChartOCR_lines/line/annotations_cleaned/temp.json')  #('/home/vp.shivasan/data/data/ChartOCR_lines/line/annotations_cleaned/clean_instancesLine(1023)_val2019.json'),
        #('/home/vp.shivasan/data/data/ChartOCR_lines/line/annotations_cleaned/temp.json')
    }

    ann_file = PATHS[image_set]

    dataset = CocoLine(root, ann_file, image_set,
                         is_train=(image_set == 'train'),
                         input_size=args.input_size, scale_factor=args.scale_factor)
    return dataset


# array([list([93.0, 259.0, 116.0, 244.0, 139.0, 222.0, 162.0, 245.0, 185.0, 256.0, 208.0, 248.0, 231.0, 203.0, 253.0, 236.0, 276.0, 240.0, 299.0, 232.0, 322.0, 227.0, 345.0, 222.0, 367.0, 232.0, 390.0, 253.0, 413.0, 260.0, 436.0, 208.0, 459.0, 185.0, 482.0, 218.0, 504.0, 241.0, 527.0, 254.0, 550.0, 255.0, 573.0, 254.0, 596.0, 244.0, 619.0, 240.0]),
#        list([93.0, 257.0, 116.0, 243.0, 139.0, 237.0, 162.0, 238.0, 185.0, 246.0, 208.0, 241.0, 231.0, 166.0, 253.0, 240.0, 276.0, 232.0, 299.0, 232.0, 322.0, 237.0, 345.0, 251.0, 367.0, 258.0, 390.0, 251.0, 413.0, 263.0, 436.0, 218.0, 459.0, 194.0, 482.0, 244.0, 504.0, 227.0, 527.0, 260.0, 550.0, 251.0, 573.0, 256.0, 596.0, 237.0, 619.0, 237.0]),
#        list([93.0, 199.0, 116.0, 227.0, 139.0, 156.0, 162.0, 232.0, 185.0, 239.0, 208.0, 218.0, 231.0, 256.0, 253.0, 236.0, 276.0, 213.0, 299.0, 204.0, 322.0, 208.0, 345.0, 217.0, 367.0, 238.0, 390.0, 242.0, 413.0, 283.0, 436.0, 280.0, 459.0, 274.0, 482.0, 282.0, 504.0, 270.0, 527.0, 283.0, 550.0, 270.0, 573.0, 260.0, 596.0, 262.0, 619.0, 268.0]),
#        list([93.0, 57.0, 116.0, 189.0, 139.0, 161.0, 162.0, 199.0, 185.0, 199.0, 208.0, 190.0, 231.0, 117.0, 253.0, 222.0, 276.0, 175.0, 299.0, 204.0, 322.0, 203.0, 345.0, 194.0, 390.0, 232.0, 413.0, 259.0, 436.0, 267.0, 459.0, 269.0, 482.0, 267.0, 504.0, 265.0, 527.0, 253.0, 550.0, 281.0, 573.0, 256.0, 596.0, 262.0, 619.0, 260.0])],
#       dtype=object)