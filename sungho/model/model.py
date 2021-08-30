# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
import math

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
# from transformers import BeitFeatureExtractor, BeitForImageClassification

from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cait import CaiT
import timm
from model import volo
from tlt.utils import load_pretrained_weights
import config


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class PretrainedModel:
    """
    Generate pre-trainned model.
    Downlaod model, append layer, and init weight and bias.
    """

    def __init__(self, name, class_num, load_model=False) -> None:
        self.name = name
        print("load model... class num is", class_num)
        if name == 'test':
            self.model = BaseModel(class_num)

        elif name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(
                in_features=512, out_features=class_num, bias=True
            )
            print("네트워크 필요 입력 채널 개수", self.model.conv1.weight.shape[1])
            print(
                "네트워크 출력 채널 개수 (예측 class type 개수)",
                self.model.fc.weight.shape[0],
            )

            self.init_weight(self.model.fc)
        elif name == "mobilenetv2":
            self.model = timm.create_model('mobilenetv2_100', pretrained=True)
            # print(self.model)
            self.model.classifier = torch.nn.Linear(
                in_features=1280, out_features=class_num, bias=True
            )
            self.init_weight(self.model.classifier)

        elif name == "efficientnet-b4":
            self.model = EfficientNet.from_pretrained(
                "efficientnet-b4", num_classes=class_num
            )
            self.reset_parameters(self.model._fc)
        elif name == "efficientnet-b7":
            self.model = EfficientNet.from_pretrained(
                "efficientnet-b7", num_classes=class_num
            )
            self.reset_parameters(self.model._fc)
        elif name == "volod3":
            self.model = volo.volo_d1()
            load_pretrained_weights(
                model=self.model,
                checkpoint_path="/opt/ml/downloads/d1_224_84.2.pth.tar",
                use_ema=False,
                strict=False,
                num_classes=class_num,
            )
        elif name == "BiT":
            self.model = timm.create_model(
                "resnetv2_101x1_bitm", pretrained=True, num_classes=class_num,
            )
        elif name == 'ViT':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=class_num)
        elif name == "deit":
            self.model = torch.hub.load(
                "facebookresearch/deit:main",
                "deit_base_patch16_224",
                pretrained=True,
            )
            self.model.head = torch.nn.Linear(
                in_features=768, out_features=class_num, bias=True
            )
            torch.nn.init.xavier_normal_(self.model.head.weight)
            stdv = 1.0 / math.sqrt(self.model.head.weight.size(1))
            self.model.head.bias.data.uniform_(-stdv, stdv)
        elif name == "CaiT":
            # https://github.com/lucidrains/vit-pytorch
            self.model = CaiT(
                image_size=224,
                patch_size=32,
                num_classes=class_num,
                dim=1024,
                depth=12,  # depth of transformer for patch to patch attention only
                cls_depth=2,  # depth of cross attention of CLS tokens to patch
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
                layer_dropout=0.05,  # randomly dropout 5% of the layers
            )

        if load_model:
            self.model.load_state_dict(torch.load(config.pretrained_path))
            print('load custom pretrained model!!', config.pretrained_path)

    def reset_parameters(self, layer):
        bound = 1 / math.sqrt(layer.weight.size(1))
        torch.nn.init.xavier_uniform_(layer.weight)
        # torch.nn.init.uniform_(layer.weight, -bound, bound)
        if layer.bias is not None:
            torch.nn.init.uniform_(layer.bias, -bound, bound)

    def init_weight(self, layer):
        torch.nn.init.xavier_uniform_(layer.weight)
        stdv = 1.0 / math.sqrt(layer.weight.size(1))
        layer.bias.data.uniform_(-stdv, stdv)
