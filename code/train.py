import torch
import torch.utils.data as data
import math
import tqdm
from sklearn.metrics import f1_score
import my_model
SAVE_PATH='/opt/ml/model/model1.pt'
SAVE_PATHA='/opt/ml/model/modela.pt'

def start(train_dataset, val_dataset):
    
   
    # Mnist Dataset을 DataLoader에 붙이기
    BATCH_SIZE = 32
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    ## 2. mnist train 데이터 셋을 resnet18 모델에 학습하기

    CLASS_NUM = 18
    my_mmodel = my_model.Mask_Classifier(CLASS_NUM)
    torch.nn.init.xavier_uniform_(my_mmodel.fc.weight)
    stdv = 1. / math.sqrt(my_mmodel.fc.weight.size(1))
    my_mmodel.fc.bias.data.uniform_(-stdv, stdv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음

    print(f"{device} is using!")

    my_mmodel.to(device) # Resnent 18 네트워크의 Tensor들을 GPU에 올릴지 Memory에 올릴지 결정함

    LEARNING_RATE = 0.0001 # 학습 때 사용하는 optimizer의 학습률 옵션 설정
    NUM_EPOCH = 4 # 학습 때 mnist train 데이터 셋을 얼마나 많이 학습할지 결정하는 옵션

    loss_fn = torch.nn.CrossEntropyLoss() # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용 - https://en.wikipedia.org/wiki/Cross_entropy
    optimizer = torch.optim.Adam(my_mmodel.parameters(), lr=LEARNING_RATE) # weight 업데이트를 위한 optimizer를 Adam으로 사용함

    dataloaders = {
        "train" : train_dataloader,
        "test" : test_dataloader
    }

    ### 학습 코드 시작
    best_test_accuracy = 0.
    best_test_loss = 9999.
    best_test_f1 = 0.

    for epoch in range(NUM_EPOCH):
        for phase in ["train", "test"]:
            running_loss = 0.
            running_acc = 0.
            running_f1 = 0.
            n_iter = 0

            if phase == "train":
                my_mmodel.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
            elif phase == "test":
                my_mmodel.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함

            for ind, (images, labels) in enumerate(tqdm.tqdm(dataloaders[phase],leave=False)):
            # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
            # hint - with, pbar
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

                with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                    logits = my_mmodel(images)
                    _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
                    
                    loss = loss_fn(logits, labels)

                    if phase == "train":
                        loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                        optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

                running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
                running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장
                running_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                n_iter += 1

            # 한 epoch이 모두 종료되었을 때,
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)
            epoch_f1 = running_f1 / n_iter

            print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, 평균 f1 : {epoch_f1:.3f}")
            if phase == "test" and best_test_accuracy < epoch_acc: # phase가 test일 때, best accuracy 계산
                best_test_accuracy = epoch_acc
                torch.save(my_mmodel.state_dict(),SAVE_PATHA)
            if phase == "test" and best_test_loss > epoch_loss: # phase가 test일 때, best loss 계산
                best_test_loss = epoch_loss
            if phase == "test" and best_test_f1 < epoch_f1: # phase가 test일 때, best f1 계산
                best_test_f1 = epoch_f1
                torch.save(my_mmodel.state_dict(),SAVE_PATH)
            
    print("학습 종료!")
    print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}, 최고 높은 f1 : {best_test_f1}")
