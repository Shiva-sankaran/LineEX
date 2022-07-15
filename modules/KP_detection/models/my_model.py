from operator import mod
from models import potr
import torch, sys
import torch.nn as nn
from .transformer_vit import build_transformer_vit
import torch.nn.functional as F
from util.misc import (NestedTensor, accuracy, get_world_size,
                       is_dist_avail_and_initialized)


class VITDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        num_points = 2  # 2 for keypoints, 4 for bounding boxes
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_points, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)



    def forward(self, samples,return_attn = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if type(samples) is NestedTensor:
            samples = samples.tensors
        # features, pos = self.backbone(samples)
        # src, mask = features[-1].decompose()
        # assert mask is not None
        hs,_,attn_list = self.transformer(samples, self.query_embed.weight,return_attn)
        # TODO: For inference only the last transformer block is used (hs[-1])
        # For calculating the loss all blocks are needed.
        # It will save some flops to only pass the MLPs from the last block for inference

        # CLASS REMOVED
        # outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_boxes': outputs_coord[-1]}
       
        return out
        
        


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.args =args
        transformer = build_transformer_vit(args)
        self.model = VITDETR(
        transformer,
        num_queries=args.num_queries
    )
    def forward(self, x,return_attn = False):
        return self.model(x,return_attn)
    