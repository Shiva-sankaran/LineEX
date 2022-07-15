from numpy import delete
import torch
import torch.nn.functional as F
from util.misc import (NestedTensor, accuracy, get_world_size,
                       is_dist_avail_and_initialized)
from torch import nn
import numpy as np
from scipy.spatial import distance

    
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,alp):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.alpha = alp

    # @torch.no_grad()
    # def loss_cardinality(self, outputs, targets, indices, num_boxes):
    #     """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
    #     This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
    #     """
    #     pred_logits = outputs['pred_logits']
    #     device = pred_logits.device
    #     tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
    #     # Count the number of predictions that are NOT "no-object" (which is the last class)
    #     card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {'cardinality_error': card_err}
    #     return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the keypoints: the L1 regression loss.
           Targets dicts must contain the key "keypoints" containing a tensor of dim [nb_target_boxes, 2]
           The target keypoints are expected in format (x, y), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).type(torch.float64)
        anchor_boxes = torch.cat([t['anchors'][i] for t, (_, i) in zip(targets, indices)], dim=0).type(torch.float64)
        ang_loss = self.angular_loss(src_boxes,target_boxes,anchor_boxes)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum(axis=1)
        l1_loss = loss_bbox
        loss_bbox = loss_bbox*self.alpha + (1-self.alpha)*ang_loss
        ten = torch.as_tensor([0,0],dtype=torch.float64).cuda()
        t=0
        for i in target_boxes:
            if((i==ten).any()):
                t+=1
        t = len(target_boxes)-t
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / (t)
        losses['l1_loss'] = l1_loss.sum()/(t)
        losses['ang_loss'] = ang_loss.sum()/(t)
        #print(losses['loss_bbox'])
        # if(losses['loss_bbox']>20):
        #     dummy=1

        return losses

    def angular_loss(self,pred_boxes,ground_boxes,anchor_boxes):
        ground_vector = (ground_boxes-anchor_boxes) 
        pred_vector = (pred_boxes - anchor_boxes)  
        # ground_vector = torch.unsqueeze(ground_vector,0)
        # pred_vector = torch.unsqueeze(pred_vector,0)
        cos = nn.CosineSimilarity(dim = 1,eps =1e-6)
        ang_loss = 1-cos(ground_vector,pred_vector)
        return ang_loss

    def get_anchor_point(ground_point):
        r = 5
        thetha = np.pi/6
        angles = np.arange(0,2*np.pi+0.0001,thetha)
        cosines = np.cos(angles)
        sines = np.sin(angles)
        x,y = ground_point
        x_new = x+ r*cosines 
        y_new = y + r*sines

        
        return x_new,y_new
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # def _get_tgt_permutation_idx(self, indices):
    #     # permute targets following indices
    #     batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    #     tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    #     return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            # 'labels': self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            'keypoints': self.loss_keypoints,
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        pred_boxes = outputs_without_aux['pred_boxes']
        l = []
        for i in range(len(targets)):
            l.append(targets[i]['mask'].cpu())
        np_l = [t.numpy() for t in l]
        mask_t = torch.as_tensor(np.array(np_l))
        mask_t = mask_t.cuda()
        outputs_without_aux['pred_boxes'] = (mask_t*pred_boxes).double()
        outputs['pred_boxes'] = (mask_t*pred_boxes).double()
        # print("!!!!!!!!!!!!",targets[0]["bboxes"].shape)
        # Retrieve the matching between the outputs of the last layer and the targets
        # ten = torch.as_tensor([0,0],dtype=torch.float64).cuda()
        
        # for i in range(42):
        #     t=0
        #     t1=0
        #     t2=0
        #     for j in outputs["pred_boxes"][i]:
        #         if((j==ten).all()):
        #             t+=1
        #     for j in targets[i]["bboxes"]:
        #         if((j==ten).all()):
        #             t1+=1
        #     for j in outputs_without_aux["pred_boxes"][i]:
        #         if((j==ten).all()):
        #             t2+=1
        #     if(t1!=t or t1!=t2 or t!=t2):
        #         print(t,t1,t2)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses



