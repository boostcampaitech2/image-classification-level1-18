from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from time import time
import torch.utils.data as data


import get_transforms



DATA_DIR = "/opt/ml/input/data/train"
IMAGE_PATH = '/opt/ml/input/data/train/images'
LABELED_DATA_PATH = '/opt/ml/code/labeled_train_data.csv'
mean = (0.560,0.524,0.501)
std = (0.233,0.243,0.246)

class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    def __init__(self, image_dir, transform=None):
        """
        MaskBaseDataset을 initialize 합니다.

        Args:
            img_dir: 학습 이미지 폴더의 root directory 입니다.
            img_name: 학습 이미지 파일 name
            transform: Augmentation을 하는 함수입니다.
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(LABELED_DATA_PATH)
        self.image_name = self.data['imageName']
        self.label = self.data['label']
        self.mean = mean
        self.std = std
        self.transform = transform
        

    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        # 이미지를 불러옵니다.
        image_path = self.image_dir + '/' + self.image_name[index]
        image = Image.open(image_path)
        
        # 레이블을 불러옵니다.
        class_label = self.label[index]
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, class_label

    def __len__(self):
        return len(self.data)

def start():
    transform = get_transforms.start(mean=mean, std=std)
    dataset = MaskBaseDataset(image_dir=IMAGE_PATH)

    # train dataset과 validation dataset을 8:2 비율로 나눕니다.
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

    # 각 dataset에 augmentation 함수를 설정합니다.
    train_dataset.dataset.set_transform(transform['train'])
    val_dataset.dataset.set_transform(transform['val'])

    return train_dataset, val_dataset