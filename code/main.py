import make_image_name
import make_label
import make_new_train_data
import dataset
import traine


import numpy as np
import torch


IMAGE_PATH = '/opt/ml/input/data/train/new_imgs'
#IMAGE_PATH = '/opt/ml/input/data/train/images'


if __name__ == '__main__':
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
    