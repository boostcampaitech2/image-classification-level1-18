import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import GaussianBlur, ColorJitter, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation
import my_emodel
test_dir = '/opt/ml/input/data/eval'
SAVE_PATHF1249='/opt/ml/model/emodelf_b4_1249.pt'
SAVE_PATHF451='/opt/ml/model/emodelf_b4_final_451.pt'
SAVE_PATHF12499='/opt/ml/model/emodelf_b4_1249_t1.pt'
#SAVE_PATHA='/opt/ml/model/emodela_b4_537.pt'

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
e1 = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])

e2 = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])
e3 = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])

d1 = TestDataset(image_paths, e1)
l1 = DataLoader(
    d1,
    shuffle=False
)

d2 = TestDataset(image_paths, e2)
l2 = DataLoader(
    d2,
    shuffle=False
)

d3 = TestDataset(image_paths, e3)
l3 = DataLoader(
    d3,
    shuffle=False
)
    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')

new_model1 = my_emodel.start(18)
new_model1.load_state_dict(torch.load(SAVE_PATHF1249))
model1 = new_model1.to(device)
model1.eval()

new_model2 = my_emodel.start(18)
new_model2.load_state_dict(torch.load(SAVE_PATHF451))
model2 = new_model2.to(device)
model2.eval()

new_model3 = my_emodel.start(18)
new_model3.load_state_dict(torch.load(SAVE_PATHF12499))
model3 = new_model3.to(device)
model3.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for i1,i2,i3 in zip(l1,l2,l3):
    with torch.no_grad():
        i1 = i1.to(device)
        i2 = i2.to(device)
        i3 = i3.to(device)
        pred1 = model1(i1) / 3
        pred2 = model2(i2) / 3
        pred3 = model3(i3) / 3
        pred = pred1+pred2+pred3
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission_b4_ense.csv'), index=False)
print('test inference is done!')