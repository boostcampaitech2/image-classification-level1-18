import os
import time
import argparse

import torchvision.models
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn import model_selection as sk_model_selection
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from datasets import MaskDataset
from utils import set_seed
from models import make_model

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--dataset', type=str, default='base')
parser.add_argument('--save_path', type=str, default='saved')
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--label_type', type=str, default='all')

# parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--test_size', type=float, default=0.2)

parser.add_argument('--model', type=str, default='efficientnet-b2')
parser.add_argument('--age', type=int, default=60)
parser.add_argument('--T_max', type=int, default=60)

args = parser.parse_args()


class Trainer:
    def __init__(self, name, args, model, device, optimizer, lr_scheduler, criterion):
        self.name = name
        self.args = args
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        self.best_valid_score = 0
        self.n_patience = 0
        self.last_model = None

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            print(f"EPOCH: {n_epoch}")
            train_loss, train_time, train_acc, train_f1 = self.train_epoch(train_loader)
            valid_loss, valid_time, val_acc, val_f1 = self.valid_epoch(valid_loader)

            wandb.log({
                'train_loss': train_loss,
                'val_loss': valid_loss,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

            print(f"[Epoch Train: {n_epoch}] loss: {train_loss:.4f}, acc: {train_acc:.2f}, f1: {train_f1:.2f}, time: {train_time:.2f} s")
            print(f"[Epoch Valid: {n_epoch}] loss: {valid_loss:.4f}, acc: {val_acc:.2f}, f1: {val_f1:.2f}, time: {valid_time:.2f} s")

            if self.best_valid_score <= val_f1:
                self.save_model(self.name, n_epoch, save_path, valid_loss, val_f1)
                print(f"val_f1 improved from {self.best_valid_score:.4f} to {val_f1:.4f}. Saved model to '{self.last_model}'")
                self.best_valid_score = val_f1
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                print(f"\nValid loss didn't improve last {patience} epochs.")
                break

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze(1)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            loss.backward()

            sum_loss += loss.detach().item()
            self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            y_all.extend(labels.tolist())
            outputs_all.extend(preds.tolist())
            print(f'Train Step {step}/{len(train_loader)}, train_loss: {sum_loss / step:.4f}')
        f1 = f1_score(y_all, outputs_all, average='micro')
        acc = accuracy_score(y_all, outputs_all)
        return sum_loss / len(train_loader), int(time.time() - t), acc, f1

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step,  (inputs, labels) in enumerate(valid_loader, 1):
            with torch.no_grad():
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs).squeeze(1)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                sum_loss += loss.detach().item()
                y_all.extend(labels.tolist())
                outputs_all.extend(preds.tolist())
            print(f'Valid Step {step}/{len(valid_loader)}, valid_loss: {sum_loss / step:.4f}')

        f1 = f1_score(y_all, outputs_all , average='micro')
        acc = accuracy_score(y_all, outputs_all)
        return sum_loss / len(valid_loader), int(time.time() - t), acc, f1

    def save_model(self, name, n_epoch, save_path, loss, f1):
        self.last_model = f"{save_path}/s{args.seed}_{name}_{args.model}_e{n_epoch}_f1{f1:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.last_model)


def main():
    wandb.init(project="NaverMask")
    run_name = wandb.run.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    train_df = pd.read_csv(f"{os.path.join(args.data_dir, args.dataset)}/train/train/train.csv")
    dataset_length = len(train_df)

    if args.label_type == 'age':
        train_df['pre_age'] = [0 if i < 30 else 1 if 30 <= i < args.age else 2 for i in list(train_df['age'])]
        labels = train_df['pre_age']
        num_classes = 3
    elif args.label_type == 'gender':
        train_df['pre_gender'] = [0 if i else 1 for i in list(train_df['gender'] == 'male')]
        labels = train_df['pre_gender']
        num_classes = 2
    elif args.label_type == 'mask':
        labels = train_df['mask']
        num_classes = 3
    elif args.label_type == 'all':
        train_df['pre_gender'] = [0 if i else 1 for i in list(train_df['gender'] == 'male')]
        train_df['pre_age'] = [0 if i < 30 else 1 if 30 <= i < args.age else 2 for i in list(train_df['age'])]
        train_df['all'] = train_df['mask'] * 6 + train_df['pre_gender'] * 3 + train_df['pre_age']
        labels = train_df['all']
        num_classes = 18

    df_train, df_valid = sk_model_selection.train_test_split(train_df, test_size=args.test_size, random_state=args.seed, stratify=labels)

    transform = A.Compose([
        # transforms.Resize(args.img_size),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
        # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # A.RandomContrast(limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    train_dataset = MaskDataset(base_path=os.path.join(args.data_dir, args.dataset), indexes=df_train.index, label_type=args.label_type, split='train', transform=transform, age=args.age)
    valid_dataset = MaskDataset(base_path=os.path.join(args.data_dir, args.dataset), indexes=df_valid.index, label_type=args.label_type, split='train', transform=transform, age=args.age)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = make_model(model_name=args.model, num_classes=num_classes)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.T_max)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(run_name, args, model, device, optimizer, cosine_scheduler, criterion)
    trainer.fit(epochs=args.epochs, train_loader=train_loader, valid_loader=valid_loader, save_path=args.save_path, patience=args.epochs)


if __name__ == '__main__':
    main()