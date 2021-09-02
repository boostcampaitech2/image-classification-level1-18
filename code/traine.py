import torch
import torch.utils.data as data
import os
import numpy as np
import math
import tqdm
from sklearn.metrics import f1_score
import torch.nn.functional
import focalloss
from pytorch_metric_learning import losses
import random
import wandb
import randbox

import my_emodel
#/opt/ml/input/data/train/new_imgs
SAVE_PATHF='/opt/ml/model/emodelf_b4_1249_t1.pt'
SAVE_PATHA='/opt/ml/model/emodela_b4_1249_t1.pt'

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def start(train_dataset, val_dataset):
    wandb.init(project='mask_classification', entity='jaehyung25')
    set_seed(1249)
    
    # b4
    BATCH_SIZE = 32
    #b7
    #BATCH_SIZE = 16

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    ## 2. mnist train 데이터 셋을 resnet18 모델에 학습하기

    CLASS_NUM = 18
    my_mmodel = my_emodel.start(CLASS_NUM)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음

    print(f"{device} is using!")

    my_mmodel.to(device) # Resnent 18 네트워크의 Tensor들을 GPU에 올릴지 Memory에 올릴지 결정함

    LEARNING_RATE = 0.0001 # 학습 때 사용하는 optimizer의 학습률 옵션 설정
    #b4
    NUM_EPOCH = 60 # 학습 때 mnist train 데이터 셋을 얼마나 많이 학습할지 결정하는 옵션
    #b7
    #NUM_EPOCH = 40

    #loss_fn = torch.nn.CrossEntropyLoss() # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용 - https://en.wikipedia.org/wiki/Cross_entropy
    #loss_fn = losses.ArcFaceLoss(512,18,loss_type='arcface')
    loss_fn = focalloss.FocalLoss(gamma=2)
    #in_features = 512
    #out_features = CLASS_NUM
    #loss_fn = AngularPenaltySMLoss(in_features,out_features,loss_type='arcface')
    optimizer = torch.optim.Adam(my_mmodel.parameters(), lr=LEARNING_RATE) # weight 업데이트를 위한 optimizer를 Adam으로 사용함

    dataloaders = {
        "train" : train_dataloader,
        "test" : test_dataloader
    }

    ### 학습 코드 시작
    best_test_accuracy = 0.
    best_test_loss = 9999.
    best_test_f1 = 0.
    early_stop_patience = 7
    early_stop_cnt = 0
    #earlystopping = EarlyStopping(patience = 7,verbose = True)
    wandb.watch(my_mmodel)
    for epoch in range(NUM_EPOCH):
        running_train_loss = 0.
        running_train_acc = 0.
        running_train_f1 = 0.
        n_train_iter = 0

        my_mmodel.train()

        for ind, (images, labels) in enumerate(tqdm.tqdm(dataloaders["train"],leave=False)):
            # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
            # hint - with, pbar
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함
            
            if np.random.random()>0.5: #cutmix를 하는 경우
                lam = np.random.beta(1, 1)
                rand_index = torch.randperm(images.size()[0]).to(device)
                target_a = labels # 원본 이미지 label
                target_b = labels[rand_index] # 패치 이미지 label       
                bbx1, bby1, bbx2, bby2 = randbox.rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                logits = my_mmodel(images)
                loss = loss_fn(logits, target_a) * lam + loss_fn(logits, target_b) * (1. - lam) # 패치 이미지와 원본 이미지의 비율에 맞게 
            
            else: #cutmix 안하는 경우
                logits = my_mmodel(images)
                loss = loss_fn(logits, labels)

            _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함        
                
            loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
            optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

            running_train_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
            running_train_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장
            running_train_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            n_train_iter += 1

            # 한 epoch이 모두 종료되었을 때,
        epoch_train_loss = running_train_loss / len(dataloaders['train'].dataset)
        epoch_train_acc = running_train_acc / len(dataloaders['train'].dataset)
        epoch_train_f1 = running_train_f1 / n_train_iter
        wandb.log({"epoch":epoch, "train_loss":epoch_train_loss, "train_acc":epoch_train_acc, "train_f1":epoch_train_f1})
        print(f"현재 epoch-{epoch}의 train-데이터 셋에서 평균 Loss : {epoch_train_loss:.3f}, 평균 Accuracy : {epoch_train_acc:.3f}, 평균 f1 : {epoch_train_f1:.3f}")
        
        running_test_loss = 0.
        running_test_acc = 0.
        running_test_f1 = 0.
        n_test_iter = 0

        with torch.no_grad():
            my_mmodel.eval()

            for ind, (images, labels) in enumerate(tqdm.tqdm(dataloaders['test'],leave=False)):
                # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
                # hint - with, pbar
                images = images.to(device)
                labels = labels.to(device)

                logits = my_mmodel(images)
                _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
    
                loss = loss_fn(logits, labels)
                    
                running_test_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
                running_test_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장
                running_test_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                n_test_iter += 1

        epoch_test_loss = running_test_loss / len(dataloaders['test'].dataset)
        epoch_test_acc = running_test_acc / len(dataloaders['test'].dataset)
        epoch_test_f1 = running_test_f1 / n_test_iter

        wandb.log({"epoch":epoch, "test_loss":epoch_test_loss, "test_acc":epoch_test_acc, "test_f1":epoch_test_f1})
        print(f"현재 epoch-{epoch}의 test-데이터 셋에서 평균 Loss : {epoch_test_loss:.3f}, 평균 Accuracy : {epoch_test_acc:.3f}, 평균 f1 : {epoch_test_f1:.3f}")
        if best_test_accuracy < epoch_test_acc: # phase가 test일 때, best accuracy 계산
            best_test_accuracy = epoch_test_acc
            torch.save(my_mmodel.state_dict(),SAVE_PATHA)
        if best_test_loss > epoch_test_loss: # phase가 test일 때, best loss 계산
            best_test_loss = epoch_test_loss
        if best_test_f1 < epoch_test_f1: # phase가 test일 때, best f1 계산
            best_test_f1 = epoch_test_f1
            torch.save(my_mmodel.state_dict(),SAVE_PATHF)
            early_stop_cnt = 0
        else:    
            early_stop_cnt += 1

        if early_stop_cnt == early_stop_patience:
            print('Early stopping')
            break
        #earlystopping(epoch_test_loss,my_mmodel)
        #if earlystopping.early_stop:
        #    print("Early stopping")
        #    break

            
    print("학습 종료!")
    print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}, 최고 높은 f1 : {best_test_f1}")

