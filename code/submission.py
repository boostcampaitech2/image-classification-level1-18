import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import ColorJitter, Resize, ToTensor, Normalize, RandomAffine
import my_emodel
test_dir = '/opt/ml/input/data/eval'
#SAVE_PATH='/opt/ml/model/emodelf_b4crop.pt'
#SAVE_PATHA='/opt/ml/model/emodela_b4e20.pt'
#SAVE_PATH='/opt/ml/model/emodelf_b4crop.pt'
SAVE_PATHF='/opt/ml/model/emodelf_b4_1249_t1.pt'
SAVE_PATHA='/opt/ml/model/emodela_b4_537.pt'

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'new_imgs')
#mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])
"""
tta1 = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    RandomAffine(30),
    ToTensor(),
    Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])
tta2 = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    ColorJitter(brightness=.5,hue=.3),
    ToTensor(),
    Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])
"""
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)
"""
tta1_data = TestDataset(image_paths, tta1)

tta1_loader = DataLoader(
    tta1_data,
    shuffle=False
)

tta2_data = TestDataset(image_paths, tta2)

tta2_loader = DataLoader(
    tta2_data,
    shuffle=False
)
"""
    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')

new_model = my_emodel.start(18)
new_model.load_state_dict(torch.load(SAVE_PATHF))
model = new_model.to(device)
model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
#for i1,i2,i3 in zip(loader,tta1_loader,tta2_loader):
for i1 in loader:
    with torch.no_grad():
        i1 = i1.to(device)
        #i2 = i2.to(device)
        #i3 = i3.to(device)
        #pred1 = model(i1) / 3
        #pred2 = model(i2) / 3
        #pred3 = model(i3) / 3
        pred = model(i1)
        #pred = pred1+pred2+pred3
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission_f_b4_1249_91.csv'), index=False)
print('test inference is done!')