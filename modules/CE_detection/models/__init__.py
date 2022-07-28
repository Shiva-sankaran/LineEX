# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_det_model(args):
    return build(args)
