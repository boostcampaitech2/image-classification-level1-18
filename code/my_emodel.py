import timm
import torch
class MaskClassifier(torch.nn.Module):
    def __init__(self,model_name,class_num,pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name,num_classes=class_num,pretrained=pretrained)

    def forward(self,x):
        x=self.model(x)
        return x


def start(classes):
    mmodel = MaskClassifier("efficientnet_b4",classes,pretrained=True)
    

    return mmodel
    