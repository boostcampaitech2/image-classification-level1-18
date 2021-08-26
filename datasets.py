import os
import PIL.Image as Image

import torch
import pandas as pd
import cv2


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, indexes, label_type='gender', split='train', transform=None, target_transform=None, age=59):
        self.base_path = base_path
        self.split = split
        self.indexes = indexes
        self.label_type = label_type
        self.transform = transform
        self.target_transform = target_transform
        self.age = age

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        raw_data = pd.read_csv(os.path.join(self.base_path, 'train', self.split, 'train.csv'))
        data = raw_data.loc[item]
        # img = Image.open(os.path.join(self.base_path, 'train', self.split, 'images', data['path']))
        img = cv2.imread(os.path.join(self.base_path, 'train', self.split, 'images', data['path']))

        if self.transform:
            img = self.transform(image=img)['image']
        elif self.target_transform:
            raise NotImplementedError

        if self.label_type == 'gender':
            if data['gender'] == 'male':
                return img, torch.tensor(0)
            else:
                return img, torch.tensor(1)
        elif self.label_type == 'age':
            if data['age'] < 30:
                return img, torch.tensor(0)
            elif 30 <= data['age'] < self.age:
                return img, torch.tensor(1)
            elif 60 <= data['age']:
                return img, torch.tensor(2)
        elif self.label_type == 'mask':
            label = data['mask']
            return img, label
        elif self.label_type == 'all':
            label = 0

            label += data['mask'] * 6

            if data['gender'] == 'male':
                label += 0
            else:
                label += 3

            if data['age'] < 30:
                label += 0
            elif 30 <= data['age'] < self.age:
                label += 1
            elif 60 <= data['age']:
                label += 2

            return img, label


def preprocessing_dataset_path(path):
    orig = pd.read_csv(path)
    data = pd.DataFrame(columns=['id', 'gender', 'race', 'age', 'mask', 'path'])

    for row in orig.iterrows():
        raw_data = row[1]
        filenames = os.listdir(os.path.join('data', 'base', 'train', 'train', 'images', row[1]['path']))
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 1, 'path': raw_data['path'] + '/' + filenames[0]}, ignore_index=True)
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 0, 'path': raw_data['path'] + '/' + filenames[1]}, ignore_index=True)
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 0, 'path': raw_data['path'] + '/' + filenames[2]}, ignore_index=True)
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 0, 'path': raw_data['path'] + '/' + filenames[3]}, ignore_index=True)
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 0, 'path': raw_data['path'] + '/' + filenames[4]}, ignore_index=True)
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 0, 'path': raw_data['path'] + '/' + filenames[5]}, ignore_index=True)
        data = data.append(
            {'id': raw_data['id'], 'gender': raw_data['gender'], 'race': raw_data['race'], 'age': raw_data['age'],
             'mask': 2, 'path': raw_data['path'] + '/' + filenames[6]}, ignore_index=True)
        print(raw_data[0])
    data.to_csv(path)
    # return data


if __name__ == '__main__':
    # import sklearn.model_selection as sk_model_selection
    # train_df = pd.read_csv(f"data/base/train/train/train.csv")
    # labels = train_df['mask']
    # num_classes = 3
    # # if args.label_type == 'age':
    # #     train_df['pre_age'] = [0 if i < 30 else 1 if 30 <= i < 60 else 2 for i in list(train_df['age'])]
    # #     labels = train_df['pre_age']
    # #     num_classes = 3
    # # elif args.label_type == 'gender':
    # #     train_df['pre_gender'] = [0 if i else 1 for i in list(train_df['gender'] == 'female')]
    # #     labels = train_df['pre_gender']
    # #     num_classes = 2
    # # elif args.label_type == 'mask':
    # #     labels = train_df['mask']
    # #     num_classes = 3
    # # elif args.label_type == 'all':
    # #     pass
    #
    # df_train, df_valid = sk_model_selection.train_test_split(train_df, test_size=0.1, random_state=42, stratify=labels)
    #
    #
    # print()
    preprocessing_dataset_path(f"data/base/train/train/train.csv")