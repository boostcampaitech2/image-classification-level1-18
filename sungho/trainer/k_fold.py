from sklearn.model_selection import KFold
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from trainer import BaseTrainer
from loss_set import CutMixCollator, CutMixCriterion

from loss_set import get_loss

from utils import test_transformation
class KFoldTrainer:
    def __init__(self,model_config) -> None:
        self.config = model_config
        self.trainer = BaseTrainer(self.config)

    def train(self, dataset) -> float:
        if self.config['cut_mix']:
            collator = CutMixCollator(self.config['cut_mix_alpha'],
                                      vertical=self.config['cut_mix_vertical'], vertical_half=self.config['cut_mix_vertical_half'])
        else:
            collator = torch.utils.data.dataloader.default_collate

        train_size = int(0.95 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        test_dataset.transforms = test_transformation

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=2,
            shuffle=True,
            collate_fn=collator,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=2,
        )

        self.trainer.train(train_dataloader, test_dataloader)

    def validate(self, test_dataset) -> list:
        valid_acc_list = []
        kfold = KFold(n_splits=self.k_split, shuffle=True)

        for fold, (train_idx, validate_idx) in enumerate(kfold.split(test_dataset)):
            if self.cutmix:
                collator = CutMixCollator(self.cutmix_alpha, vertical=True)
            else:
                collator = torch.utils.data.dataloader.default_collate

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            validate_subsampler = torch.utils.data.SubsetRandomSampler(
                validate_idx
            )

            print(f"Start train with {fold} fold")
            train_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                sampler=train_subsampler,
                num_workers=0,
                collate_fn=collator,
            )
            validate_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                sampler=validate_subsampler,
                num_workers=0,
            )

            _, valid_acc = self.trainer.train(
                train_dataloader, validate_dataloader, self.feature, self.epoch
            )
            valid_acc_list.append(valid_acc)

        return valid_acc_list
