import albumentations as A
import albumentations.pytorch
import numpy as np

test_transformation = A.Compose(
    [
        # A.CenterCrop(350, 300, p=1),
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

tta_set = [
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    ),
    albumentations.pytorch.transforms.ToTensorV2()
]
tta_possible_set = [
    A.GaussNoise(var_limit=(20.0, 60.0),p=1),
    A.MedianBlur(blur_limit=9, p=1),
    A.Blur(blur_limit=9, p=1),
    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=40,p=1),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50,p=1),
    A.ChannelDropout(p=1),
    A.ChannelShuffle(p=1),
    A.CoarseDropout(p=1),
    # A.ToGray(always_apply=True, p=1.0),
    # A.ToSepia(always_apply=True, p=1.0),
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,p=1),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5,p=1),
    A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=10,
            border_mode=0,
            p=1,
    ),
    # A.Sharpen(p=1),
    # A.HorizontalFlip(p=1),
]

import copy
def tta_augmentation():
    # for possible in tta_possible_set:
    #     temp = copy.deepcopy(tta_set)
    #     temp.insert(1, possible)
    #     print(temp)
    #     yield A.Compose(temp)

    for i in range(20):
        temp = copy.deepcopy(tta_set)
        # temp.insert(1, np.random.choice(tta_possible_set, np.random.randint(len(tta_possible_set)), replace=False).tolist())

        temp[1:1] = np.random.choice(tta_possible_set, np.random.randint(len(tta_possible_set)), replace=False).tolist()
        print(temp)
        yield A.Compose(temp)


transformation = A.Compose(
    [
        A.Resize(224, 224),
        # A.CenterCrop(300, 256, p=1),
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.GaussNoise()], p=0.4),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.Blur(blur_limit=3, p=0.2),
            ],
            p=1,
        ),
        A.OneOf(
            [
                # A.CLAHE(clip_limit=2, p=0.5),
                # A.Sharpen(p=0.5),
                # A.Emboss(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.5),
                A.ChannelShuffle(p=0.5),
            ],
            p=1,
        ),
        # A.ShiftScaleRotate(
        #     shift_limit=0.2,
        #     scale_limit=0.2,
        #     rotate_limit=10,
        #     border_mode=0,
        #     p=0.4,
        # ),
        A.CoarseDropout(p=0.5),
        A.ColorJitter(p=0.3),
        A.RandomBrightnessContrast(p=0.7),
        # A.Rotate(limit=(-10, 10), p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)
