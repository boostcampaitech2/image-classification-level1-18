import torch
import torchvision
import timm

def Mask_Classifier(classes):
    mmodel = torchvision.models.resnet18(pretrained=True)
    mmodel.fc = torch.nn.Linear(in_features=512,out_features=classes, bias=True)
    
    print("network input channel : ",mmodel.conv1.weight.shape[1])
    print("network output channel : ", mmodel.fc.weight.shape[0])

    return mmodel
    