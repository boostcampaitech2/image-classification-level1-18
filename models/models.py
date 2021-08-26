import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def make_model(model_name, num_classes):
    class Model(nn.Module):
        def __init__(self, model='efficientnet-b0', num_classes=num_classes):
            super().__init__()
            self.net = EfficientNet.from_pretrained(model, num_classes=num_classes)

        def forward(self, x):
            out = self.net(x)
            return out

    return Model(model_name)


if __name__ == '__main__':
    model = make_model(model_name='efficientnet-b0', num_classes=1)
    noise = torch.randn(2, 3, 256, 256)
    output = model(noise)
    print(f'output {output}')
    print(output.shape)