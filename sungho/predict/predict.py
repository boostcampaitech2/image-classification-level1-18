from sys import path
from tqdm import tqdm

import torch


class Predictor:
    def __init__(self, model, epochs, device, batch_size, ensemble=False):
        self.model = model
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.start_epoch = 1
        self.ensemble = ensemble

    def predict(self, dataloader, feature):
        result = []

        self.model.eval()

        with tqdm(dataloader, unit="batch") as tepoch:
            len(dataloader)
            for ind, (images, paths) in enumerate(tepoch):
                tepoch.set_description(f"{feature}")
                if not self.ensemble:
                    images = images.to(self.device)
                with torch.no_grad():
                    if not self.ensemble:
                        logits = self.model(images)
                    else:
                        logits = self.model.predict(images)
                    _, preds = torch.max(logits, 1)

                    path_list = [path.split("/")[-1] for path in paths]

                    result.append([path_list, preds.tolist()])

        merge_result = []
        for epoch_pred in result:
            for path, pred_class in zip(epoch_pred[0], epoch_pred[1]):
                merge_result.append([path, pred_class])

        return merge_result

