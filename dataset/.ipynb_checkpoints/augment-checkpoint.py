from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomResizedCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, GaussNoise,
    RGBShift, RandomRain, RandomSnow, RandomShadow, RandomFog, ElasticTransform, SmallestMaxSize, RandomCrop, Normalize,PiecewiseAffine,Sharpen,Emboss
)
from albumentations.pytorch import ToTensorV2

def strong_aug(p=0.5, crop_size=(512, 512)):
    return Compose([
        RandomResizedCrop(crop_size[0], crop_size[1], scale=(0.3, 1.0), ratio=(0.75, 1.3), interpolation=4, p=1.0),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
#             IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.8),
        OneOf([
            MotionBlur(p=0.5),
            MedianBlur(blur_limit=3, p=0.5),
            Blur(blur_limit=3, p=0.5),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=180, p=0.8),
        OneOf([
            OpticalDistortion(p=0.5),
            GridDistortion(p=0.5),
#             IAAPiecewiseAffine(p=0.5),
            PiecewiseAffine(p=0.5),
            ElasticTransform(p=0.5),
        ], p=0.3),
        OneOf([
            CLAHE(clip_limit=2),
#             IAASharpen(),
#             IAAEmboss(),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        OneOf([
            GaussNoise(),
            RandomRain(p=0.2, brightness_coefficient=0.9, drop_width=1, blur_value=5),
            RandomSnow(p=0.4, brightness_coeff=0.5, snow_point_lower=0.1, snow_point_upper=0.3),
            RandomShadow(p=0.2, num_shadows_lower=1, num_shadows_upper=1,
                        shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1)),
            RandomFog(p=0.5, fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1)
        ], p=0.3),
        RGBShift(),
        HueSaturationValue(p=0.9),
        # ToTensorV2() #add by xxy
    ], p=p)
    
    # train_transform = Compose([
    #     SmallestMaxSize(max_size=160),
    #     ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    #     RandomCrop(height=128, width=128),
    #     RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    #     RandomBrightnessContrast(p=0.5),
    #     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ])
    # return train_transform
