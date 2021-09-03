import config

import torch
from .cut_mix import CutMixCriterion
from .FocalLoss import FocalLoss
from pytorch_metric_learning import losses
# from loss_functions import AngularPenaltySMLoss
from .label_smoothing import LabelSmoothingLoss

def get_loss(name, cutmix=False, class_num=None):
    if cutmix:
        criterion = CutMixCriterion(reduction='mean', class_num=class_num, loss=name)
        print(f'loss function is {name}!!')
    elif name == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif name == "focal":
        criterion = FocalLoss()
    elif name == "LabelSmoothing":
        criterion = LabelSmoothingLoss()
    elif name is None:
        exit()
    # elif name == "ArcFaceLoss":
    #     criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface')

    return criterion
