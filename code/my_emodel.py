import timm
import torch
from efficientnet_pytorch import EfficientNet

class MaskClassifier(torch.nn.Module):
    def __init__(self,model_name,class_num,pretrained=False):
        super().__init__()
        self.name = model_name
        #self.model = EfficientNet.from_pretrained("efficientnet-b7",num_classes = class_num)
        self.model = timm.create_model(model_name,num_classes=class_num, pretrained=pretrained)
        #self.arcface = ArcMarginProduct(1000,class_num)
        #self.reset_parameters(self.model._fc)

    def forward(self,x):
        x=self.model(x)
        #x = self.arcface(x,True)
        return x


def start(classes):
    mmodel = MaskClassifier("efficientnet_b4",classes,pretrained=True)
    

    return mmodel
    