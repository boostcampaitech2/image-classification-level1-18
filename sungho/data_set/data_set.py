from torch.utils.data import Dataset, DataLoader
from utils import get_test_img_path
from utils import Label
from PIL import Image
import numpy as np
import torch
import cv2
import albumentations as A
import albumentations.pytorch

from utils import test_transformation


class MaskDataset(Dataset):
    def __init__(
        self, data_df, image_dir, transforms=None, feature=None, train=True
    ):
        self.data_df = data_df
        self.image_dir = image_dir
        self.label = Label()
        if train:
            self.classes = self.label.get_class_num(feature)
        self.transforms = transforms
        self.train = train
        self.feature = feature

        if not train:
            # system path list of test images
            self.data_df = get_test_img_path(self.data_df, self.image_dir)

        if self.transforms is None:
            self.transforms = test_transformation

    def __len__(self):
        if isinstance(self.data_df, list):
            return len(self.data_df)
        else:
            return self.data_df.shape[0]

    def __getitem__(self, idx):
        if self.train:
            target_path = self.data_df.iloc[idx]["system_path"]
            label = self.label.get_label(target_path, self.feature)
        else:
            target_path = self.data_df[idx]
            label = target_path

        # img = np.array(Image.open(target_path).convert("RGB"))
        img = cv2.imread(target_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)
            img = img["image"].type(torch.FloatTensor)
        return img, label
