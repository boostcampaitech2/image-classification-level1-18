# Image Classification 18 team

## Requirement
```shell
pip install -r req.txt
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
BATCH_SIZE = 128

NUM_EPOCH = 1
k_split = 1
model_name = "resnet18"
ensemble = False

if model_name == "deit":
    LEARNING_RATE = 0.0005
else:
    LEARNING_RATE = 0.001

ray_tune = False
loss = "focal"
predict_dir = "2021-08-29T10-13-16.359788"
features = [
    "mask",
    "gender",
    "age",
]

pretrained_path = [
]
fp16 = True

merge_feature = False
merge_feature_name = 'merged_feature'
cutmix = True
curmix_vertical = True
cutmix_vertical_half = False
cutmix_alpha = 1.0
wandb_api_key = ''

tta = True
```

## Run
```shell
# generate train csv file
python train.py -g-path

# train
python train.py
```