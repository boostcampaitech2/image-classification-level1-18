import torch
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses

from torchensemble.bagging import BaggingClassifier
from torchensemble.utils.logging import set_logger

import os
import wandb
from datetime import datetime

from data_set import MaskDataset
from model import PretrainedModel
from utils import Label
from . import k_fold
import config
from loss_set import CutMixCriterion, get_loss
from utils import transformation

def feature_train(train_df, test_df, feature, model_name, model_dir):
    print(f"{feature}, {model_name}")

    train_dataset = MaskDataset(
        train_df, config.train_dir, feature=feature, transforms=transformation
    )

    class_num = 18 if config.merge_feature else len(getattr(Label, feature))

    device = torch.device("cuda:0")
    if len(config.pretrained_path) == 0:
        load_model = False
    else:
        load_model = True
    model = PretrainedModel(model_name, class_num, load_model=load_model).model
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
    criterion = get_loss(config.loss, cutmix=True)
    model_config = {
        'class_num': class_num,
        'device': device,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'k_split': config.k_split,
        'feature': feature,
        'epoch': config.NUM_EPOCH,
        'batch_size': config.BATCH_SIZE,
        'model_dir': model_dir,
        'model_name': model_name,
        'cut_mix': config.cutmix,
        'cut_mix_alpha': config.cutmix_alpha,
        'cut_mix_vertical': config.curmix_vertical,
        'cut_mix_vertical_half': config.cutmix_vertical_half
    }

    kt = k_fold.KFoldTrainer(model_config)
    kt.train(train_dataset)

