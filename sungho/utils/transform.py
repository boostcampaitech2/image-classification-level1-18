import albumentations as A
import albumentations.pytorch

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
transformation = A.Compose(
    [
        A.Resize(224, 224),
        # A.CenterCrop(300, 256, p=1),
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.GaussNoise()], p=0.2),
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
                # A.RGBShift(p=0.5),
                # A.ChannelShuffle(p=0.5),
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
        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.8),
        # A.Rotate(limit=(-10, 10), p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)
