import torch

from data_set import MaskDataset
from model import PretrainedModel
from utils import Label
from . import k_fold
import config
from utils import transformation


def feature_train(train_df, test_df, feature, model_name, model_dir):
    print(f"{feature}, {model_name}")

    train_dataset = MaskDataset(
        train_df, config.train_dir, feature=feature, transforms=transformation
    )

    class_num = 18 if config.merge_feature else len(getattr(Label, feature))

    device = torch.device("cuda:0")

    pretrained_path = None
    # Load pretrained model
    if len(config.pretrained_path) != 0:
        for path in config.pretrained_path:
            if feature in path:
                pretrained_path = path

    model = PretrainedModel(model_name, class_num, pretrained_path=pretrained_path).model
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                           first_cycle_steps=200,
    #                                           cycle_mult=1.0,
    #                                           max_lr=0.1,
    #                                           min_lr=0.001,
    #                                           warmup_steps=50,
    #                                           gamma=1.0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    model_config = {
        'class_num': class_num,
        'device': device,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'k_split': config.k_split,
        'feature': feature,
        'epoch': config.NUM_EPOCH,
        'batch_size': config.BATCH_SIZE,
        'model_dir': model_dir,
        'model_name': model_name,
        'cut_mix': config.cutmix,
        'cut_mix_alpha': config.cutmix_alpha,
        'cut_mix_vertical': config.curmix_vertical,
        'cut_mix_vertical_half': config.cutmix_vertical_half,
        'loss': config.loss
    }

    kt = k_fold.KFoldTrainer(model_config)
    kt.train(train_dataset)

