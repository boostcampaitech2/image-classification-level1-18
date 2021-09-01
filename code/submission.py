import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize, ToTensor, Normalize
import my_emodel
test_dir = '/opt/ml/input/data/eval'
#SAVE_PATH='/opt/ml/model/emodelf_b4crop.pt'
#SAVE_PATHA='/opt/ml/model/emodela_b4e20.pt'
#SAVE_PATH='/opt/ml/model/emodelf_b4crop.pt'
SAVE_PATHF='/opt/ml/model/emodelf_b4_final.pt'
SAVE_PATHA='/opt/ml/model/emodela_b4_final.pt'

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

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((224, 244), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')

new_model = my_emodel.start(18)
new_model.load_state_dict(torch.load(SAVE_PATHF))
model = new_model.to(device)
model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission_fcb_b4.csv'), index=False)
print('test inference is done!')