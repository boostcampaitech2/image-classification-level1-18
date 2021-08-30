import make_image_name
import make_label
import make_new_train_data
import dataset
import train
import traine
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


IMAGE_PATH = '/opt/ml/input/data/train/images'


if __name__ == '__main__':
    set_seed(451)
    print('start making image path')
    image_names = []
    image_names = make_image_name.start(IMAGE_PATH)
    labels = make_label.start(image_names)
    make_new_train_data.start(image_names,labels)
    print('finish making labeled data')
    print('load dataset')
    train_dataset, val_dataset = dataset.start()
    print('load finished')
    #train.start(train_dataset,val_dataset)
    traine.start(train_dataset,val_dataset)
    