import config

import torch
from .cut_mix import CutMixCriterion
from .FocalLoss import FocalLoss
from pytorch_metric_learning import losses


def get_loss(name, cutmix=False):
    if cutmix:
        criterion = CutMixCriterion(reduction='mean')
    elif name == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif name == "focal":
        criterion = FocalLoss()
    elif name == "ArcFaceLoss":
        criterion = losses.ArcFaceLoss(
            num_classes=config.class_num, embedding_size=config.class_num
        )
    return criterion
