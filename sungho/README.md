# Image Classification
Focus on how to make pipeline.
Testing many method of CNN, Ensemble and creating simple MLops.
- Model
    - ResNet18
    - EfficientNet-b4, b7
    - ViT
    - BiT
    - CaiT
    - Volo
    - MobilenetV2
- Augmentation
  - CutMix
  - Albumentation 
- Ensemble
  - Voting 
- Method
  - Early stopping
  - Ray tune
  - Wandb
  
## Requirement
```shell
pip install -r requirements.txt
```

## config.py
create config.py on your root directory.

```
test_dir = "D:\\dev\\train\\eval\\images"
train_dir = "D:\\dev\\train\\train\\images"

test_csv = "D:\\dev\\train\\eval\\info.csv"
train_csv = "D:\\dev\\train\\train\\train.csv"
with_system_path_csv = ".\\train-with-system-path.csv"

model_dir = ".\\saved_model"
BATCH_SIZE = 128

NUM_EPOCH = 5
k_split = 2
model_name = "resnet18"
ensemble = False
if model_name == "deit":
    LEARNING_RATE = 0.0005
else:
    LEARNING_RATE = 0.001

ray_tune = False
loss = "focal"
predict_dir = "2021-08-26T22:21:20.254632"
features = [
    "mask",
    "gender",
    "age",
]

wandb_api_key = ''
```

## Run
```shell
# generate train csv file
python train.py -g-path

# train
python train.py
```