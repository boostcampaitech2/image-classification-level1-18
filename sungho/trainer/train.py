from tqdm import tqdm

import torch
import wandb
import config
from sklearn.metrics import f1_score
from ray import tune
from .early_stopping import EarlyStopping
from loss_set import get_loss, CutMixCriterion

class BaseTrainer:
    def __init__(self, model_config):
        self.config = model_config
        self.model = self.config['model']
        self.device = self.config['device']
        self.optimizer = self.config['optimizer']
        # self.criterion = self.config['criterion']
        self.scheduler = self.config['scheduler']
        self.criterion = get_loss(self.config['loss'], cutmix=config.cutmix, class_num=self.config['class_num'])
        self.fp16 = config.fp16
        self.wandb_tag = [self.config['feature'], self.config['model_name']]
        if self.config['cut_mix'] and self.config['cut_mix_vertical']:
            self.wandb_tag.append('CutMix-Vertical')
        elif self.config['cut_mix'] and not self.config['cut_mix_vertical']:
            self.wandb_tag.append('CutMix')

        if self.config['cut_mix']:
            self.val_criterion = get_loss('focal', cutmix=False)

    def train(self, train_dataloader, val_dataloader):
        self._forward(train_dataloader, val_dataloader)

    def _forward(self, train_dataloader, val_dataloader, patience=7):
        run = wandb.init(
            project="aistage-mask", entity="naem1023",
            tags=self.wandb_tag
        )
        wandb.config.learning_rate = config.LEARNING_RATE
        wandb.config.batch_size = config.BATCH_SIZE
        wandb.config.epoch = config.NUM_EPOCH
        wandb.config.k_fold = config.k_split
        wandb.watch(self.model)

        scaler = torch.cuda.amp.GradScaler()

        early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=self.config['model_dir'], feature=self.config['feature'],
            model_name=self.config['model_name']
        )

        for epoch in range(self.config['epoch']):
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0
            pred_target_list = []
            if self.config['cut_mix']:
                target_list = [[], []]
                sum_f1_score = 0.0
                count = 0
            else:
                target_list = []

            print(f"{self.config['feature']}: Epoch {epoch}")
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for ind, (images, targets) in enumerate(tepoch):
                    tepoch.set_description(f"{self.config['feature']}{epoch}")
                    images = images.to(self.device)

                    # CutMix
                    if isinstance(targets, (tuple, list)):
                        targets1, targets2, lam = targets
                        targets = (targets1.to(self.device), targets2.to(self.device), lam)
                        target_list[0] += targets[0].tolist()
                        target_list[1] += targets[1].tolist()
                    # Normal
                    else:
                        targets = targets.to(self.device)
                        target_list += targets.tolist()

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)

                        if self.config['loss'] == 'LabelSmoothing':
                            preds = logits
                        elif self.config['model_name'] in ["volo", "CaiT"]:
                            preds = torch.argmax(logits, dim=1)
                        else:
                            # _, preds = torch.max(logits, 1)
                            preds = torch.nn.functional.softmax(logits, dim=-1)
                            # finally get the index of the prediction with highest score
                            # topk_scores, preds = torch.topk(scores, k=1, dim=-1)

                        loss = self.criterion(preds, targets)

                    # scaler.scale(loss).backward(retain_graph=True)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    # loss.backward()
                    # self.optimizer.step()

                    self.scheduler.step()

                    running_loss += loss.item()

                    pred_target = torch.argmax(preds, dim=1)
                    num = images.size(0)
                    if isinstance(targets, (tuple, list)):
                        targets1, targets2, lam = targets
                        correct1 = pred_target.eq(targets1).sum().item()
                        correct2 = pred_target.eq(targets2).sum().item()
                        accuracy = (lam * correct1 + (1 - lam) * correct2) / num
                        pred_target_list += pred_target.tolist()
                    else:
                        correct_ = pred_target.eq(targets).sum().item()
                        accuracy = correct_ / num
                        # Append inferenced label and real label for f1 score
                        pred_target_list += pred_target.tolist()

                    running_acc += accuracy

                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=accuracy
                    )

                    if self.config['cut_mix']:
                        count += 1
                        sum_f1_score += f1_score(target_list[0], pred_target_list, average="macro") * lam + \
                            f1_score(target_list[1], pred_target_list, average="macro") * (1 - lam)

            ##################
            # validation
            running_val_loss = 0.0
            running_val_acc = 0.0
            val_pred_target_list = []
            val_target_list = []
            with torch.no_grad():
                self.model.eval()

                with tqdm(val_dataloader, unit="batch") as tepoch:
                    for ind, (images, targets) in enumerate(tepoch):
                        tepoch.set_description(f"val{epoch}")

                        images = images.to(self.device)

                        targets = targets.to(self.device)
                        val_target_list += targets.tolist()

                        logits = self.model(images)

                        # if self.config['model_name'] in ['ViT', "BiT", "deit", "efficientnet-b4", "efficientnet-b7", "resnet18","mobilenetv2"]:

                            # finally get the index of the prediction with highest score
                            # topk_scores, preds = torch.topk(scores, k=1, dim=-1)
                        if self.config['model_name'] in ["volo", "CaiT"]:
                            preds = torch.argmax(logits, dim=1)
                        else:
                            # _, preds = torch.max(logits, 1)
                            preds = torch.nn.functional.softmax(logits, dim=-1)

                        if self.config['cut_mix']:
                            val_loss = self.val_criterion(preds, targets)
                        else:
                            val_loss = self.criterion(preds, targets)

                        running_val_loss += val_loss.item()

                        pred_target = torch.argmax(preds, dim=1)
                        num = images.size(0)
                        correct_ = pred_target.eq(targets).sum().item()
                        accuracy = correct_ / num
                        val_pred_target_list += pred_target.tolist()

                        running_val_acc += accuracy

                        tepoch.set_postfix(
                            loss=val_loss.item(), accuracy=accuracy
                        )

            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = running_acc / len(train_dataloader)

            epoch_val_loss = running_val_loss / len(val_dataloader)
            epoch_val_acc = running_val_acc / len(val_dataloader)

            if self.config['cut_mix']:
                epoch_f1 = sum_f1_score / count
            else:
                epoch_f1 = f1_score(target_list, pred_target_list, average="macro")
            epoch_val_f1 = f1_score(val_target_list, val_pred_target_list, average="macro")

            if config.ray_tune:
                tune.report(loss=epoch_val_loss, accuracy=epoch_val_acc, f1_score=epoch_val_f1)

            wandb.log({
                "accuracy": epoch_acc,
                "loss": epoch_loss,
                "f1_score": epoch_f1,
                "val_acc": epoch_val_acc,
                "val_loss": epoch_val_loss,
                "val_f1_score": epoch_val_f1,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            # if epoch > int(self.config['epoch'] * 0.6):
            # Check loss
            # If loss is decreased, save model.
            early_stopping(epoch_f1, self.model)

            if early_stopping.early_stop:
                print('Early Stopping!!')
                break

            print(
                f"epoch-{epoch} val loss: {epoch_val_loss:.3f}, val acc: {epoch_val_acc:.3f}, val_f1_score: {epoch_val_f1:.3f}"
            )
            print(
                f"epoch-{epoch} loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}, f1_score: {epoch_f1:.3f}"
            )

        run.finish()
