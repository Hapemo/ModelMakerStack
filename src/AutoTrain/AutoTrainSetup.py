''' This file will contain function to turn off auto augmentation of training images '''
from ultralytics.data import YOLODataset
from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    v8_transforms,
)

autoaugment = True
backup_build_transforms = YOLODataset.build_transforms

# NOTE: IF you want to enable certain auto augmentation specifically, v8_transforms consists of the following transformations chornologically.
# pre_transform
# MixUp
# Albumentations
# RandomHSV
# RandomFlip horizontal
# RandomFlip vertical
# So you can remove them in build_transforms function instead of removing v8_transforms totally.


def build_transforms(self, hyp=None):
    """Builds and appends transforms to the list."""
    global autoaugment
    if autoaugment and self.augment: # Only apply auto augmentation if autoaugment is True and it's train set.
        hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
        hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
        transforms = v8_transforms(self, self.imgsz, hyp)
    else:
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
    transforms.append(
        Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            return_obb=self.use_obb,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio,
            mask_overlap=hyp.overlap_mask,
            bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
        )
    )
    return transforms

def ToggleAutoAugmentation(status: bool):
    ''' status is True if you want auto augmentation of the yolo dataset. Otherwise false.
    Utilizes monkey patch technique.
    '''
    return
    global autoaugment
    autoaugment = status
    YOLODataset.build_transforms = build_transforms


