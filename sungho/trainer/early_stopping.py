import numpy as np
import torch
import os
from datetime import datetime
import wandb

class EarlyStopping:
    """주어진 patience 이후로 지표가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='.' + os.sep, check='max', feature=None, model_name=''):
        """
        Args:
            patience (int): 지표가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 지표의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
            check (str): validatinod 지표가 max로 개선되는지, min으로 개선되는지 결정
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.check = check
        if self.check == 'max':
            self.val_check = -np.Inf
        elif self.check == 'min':
            self.val_check = np.Inf
        self.delta = delta
        self.path = path
        self.feature = feature
        self.model_name = model_name

    def check_val(self, score):
        if self.check == 'max':
            if score > self.best_score - self.delta:
                return True
            else:
                return False
        elif self.check == 'min':
            if score < self.best_score + self.delta:
                return True
            else:
                return False

    def __call__(self, val, model):
        if self.check == 'max':
            score = val
        elif self.check == 'min':
            score = -val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif not self.check_val(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val, model):
        '''validation 지표가 개선되면 모델을 저장한다.'''
        model_name = f"{self.model_name}-{self.feature}-{wandb.run.name}-.pt"
        if self.verbose:
            print(f'Good training! ({self.val_check:.6f} --> {val:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, model_name))
        self.val_check = val