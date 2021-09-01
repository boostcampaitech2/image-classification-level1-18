import torch
import timm
from torch.nn import functional as F
import pytorch_lightning as pl


class CustomModel(pl.LightningModule):
    def __init__(self, num_classes=18):
        super().__init__()

        self.model = timm.create_model('efficientnet-b0',pretrained=True,num_classes=18)

        self.save_hyperparameters()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)      # Loss 수정
        self.log('val_loss', loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(x)
        loss = F.mse_loss(x_hat, x)   # Loss 수정
        self. log('val_loss', loss)