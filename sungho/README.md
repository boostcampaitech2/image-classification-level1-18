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
# test_dir = "/opt/ml/input/data/eval/images"
# train_dir = "/opt/ml/input/data/train/images"
train_dir = '/opt/ml/train_crop_images'
test_dir = '/opt/ml/eval_crop_images'

test_csv = "/opt/ml/input/data/eval/info.csv"
train_csv = "/opt/ml/input/data/train/train.csv"

with_system_path_csv = "/opt/ml/crop-train-with-system-path.csv"
# with_system_path_csv = "/opt/ml/crop-train-with-system-path.csv"

model_dir = "/opt/ml/repo/sungho/saved_model"
BATCH_SIZE = 256

NUM_EPOCH = 100
k_split = 1
model_name = "resnet18"
ensemble = False

if model_name == "deit":
    LEARNING_RATE = 0.0005
else:
    LEARNING_RATE = 0.001

ray_tune = False
loss = "LabelSmoothing"
predict_dir = "2021-08-30T19-32-30.131418"
features = [
    "mask",
    "gender",
    "age",
]

pretrained_path = ''

merge_feature = False
merge_feature_name = 'merged_feature'
cutmix = True
curmix_vertical = True
cutmix_vertical_half = True
cutmix_alpha = 1.0
wandb_api_key = ''
```

## Run
```shell
# generate train csv file
python train.py -g-path

# train
python train.py
```