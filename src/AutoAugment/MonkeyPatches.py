''' This file contains all the monkey patches required for this project '''
from ultralytics.data import YOLODataset
from Logger import Logger

from ultralytics.data.base import BaseDataset
from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    Albumentations,
)
from ultralytics.utils.checks import check_version
from ultralytics.utils import DEFAULT_CFG, colorstr

import RandAugmentGenerator as RandAugmentGenerator
import albumentations as A
import torch
from albumentations.core.bbox_utils import normalize_bboxes, check_bboxes
from albumentations.core.utils import ShapeType
from typing import Literal

# from albumentations.core.types import np.ndarray
from typing import Sequence
import numpy as np

def CustomGetItem(self, index):
    if not self.augment:
        return self.transforms(self.get_image_and_label(index))
    
    # else:
    #     data = self.transforms(self.get_image_and_label(index))
    #     print("\n\ndata: ", data)
    #     return data
    
    # Prepare augmentations
    # Logger.LogCustom("temp", f"self.augmentationList while in custom get item: {self.augmentationList}")
    currentAugmentationList = RandAugmentGenerator.GetRandomAugments(self.n, self.augmentationList)
    
    albumentations = Albumentations(t = currentAugmentationList, p = 1.0)
    compose = Compose([albumentations, self.formatTransform])
    # print("albumentations transforms: ", albumentations.transform)
    # print("compose: ", compose)

    transformed = compose(self.get_image_and_label(index))

    # Convert the cls number to float and reshape
    # This is to fix "RuntimeError: Tensors must have same number of dimensions: got 1 and 2" when batch size more than 1
    transformed["cls"] = transformed["cls"].float().view(-1, 1)

    # Save the array to a text file (Uncomment this portion if you want to save the transformed images during training phase in local storage)
    
    # '''
    # import os
    # import cv2
    # import random
    # filename = f"{os.getcwd()}/tempimg/temp{random.randint(0,100000)}"
    # with open(f"{filename}.txt", "w") as f:
    #     for b in transformed["bboxes"]:
    #         tensor_cpu = b.cpu()
    #         array = tensor_cpu.numpy()
    #         array_str = " ".join(map(str, array))
    #         f.write(f"0 {array_str}\n")
    # for b in transformed["bboxes"]:
    #     tensor_cpu = b.cpu()
    #     array = tensor_cpu.numpy()

    # img_np = transformed["img"].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]  # Convert tensor to numpy array, transpose to HWC, and convert RGB to BGR 
    # cv2.imwrite(f"{filename}.jpg", img_np)
    # '''
    # print("\n\ndata: ", transformed)

    return transformed

# The below functions are done at initialization time, so should not affect the multithreading of dataset.
def CustomAlbumentationInit(self, t, p=1.0):
    """
    Initialize the Albumentations transform object for YOLO bbox formatted parameters.

    This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
    conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
    contrast, RandomGamma, and image quality reduction through compression.

    Args:
        p (float): Probability of applying the augmentations. Must be between 0 and 1.

    Attributes:
        p (float): Probability of applying the augmentations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial transformations.

    Raises:
        ImportError: If the Albumentations package is not installed.
        Exception: For any other errors during initialization.

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
        >>> augmented_image = augmented["image"]
        >>> augmented_bboxes = augmented["bboxes"]

    Notes:
        - Requires Albumentations version 1.0.3 or higher.
        - Spatial transforms are handled differently to ensure bbox compatibility.
        - Some transforms are applied with very low probability (0.01) by default.
    """
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")

    try:
        import albumentations as A

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement

        # List of possible spatial transforms
        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

        # Transforms
        T = t

        # Compose transforms
        self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
        self.transform = (
            A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True, min_visibility=0.5))
            if self.contains_spatial
            else A.Compose(T)
        )
        # Logger.LogInfo(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        Logger.LogError(f"{prefix}{e}")

def CustomBuildTransforms(self, hyp=None):
    """Builds and appends transforms to the list.
    Also remove the augmetations for training, since it will be modified later on.
    """
    
    transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

    self.formatTransform = Format(
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
    transforms.append(self.formatTransform)
    
    return transforms

from albumentations.core.bbox_utils import BboxProcessor, filter_bboxes, check_bboxes

def custom_convert_bboxes_to_albumentations(
    bboxes: np.ndarray,
    source_format: Literal["coco", "pascal_voc", "yolo"],
    shape: ShapeType,
    check_validity: bool = False,
) -> np.ndarray:
    """Convert bounding boxes from a specified format to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in the form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
        source_format: Format of the input bounding boxes. Should be 'coco', 'pascal_voc', or 'yolo'.
        shape: Image shape (height, width).
        check_validity: Check if all boxes are valid boxes.

    Returns:
        np.ndarray: An array of bounding boxes in albumentations format with shape (num_bboxes, 4+).

    Raises:
        ValueError: If `source_format` is not 'coco', 'pascal_voc', or 'yolo'.
        ValueError: If in YOLO format, any coordinates are not in the range (0, 1].
    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )

    bboxes = bboxes.copy().astype(np.float32)
    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns

    if source_format == "coco":
        converted_bboxes[:, 0] = bboxes[:, 0]  # x_min
        converted_bboxes[:, 1] = bboxes[:, 1]  # y_min
        converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max
        converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max
    elif source_format == "yolo":
        if check_validity and np.any((bboxes[:, :4] <= 0) | (bboxes[:, :4] > 1)):
            raise ValueError(f"In YOLO format all coordinates must be float and in range (0, 1], got {bboxes}")

        w_half, h_half = bboxes[:, 2] / 2, bboxes[:, 3] / 2
        converted_bboxes[:, 0] = bboxes[:, 0] - w_half  # x_min
        converted_bboxes[:, 1] = bboxes[:, 1] - h_half  # y_min
        converted_bboxes[:, 2] = bboxes[:, 0] + w_half  # x_max
        converted_bboxes[:, 3] = bboxes[:, 1] + h_half  # y_max
    else:  # pascal_voc
        converted_bboxes[:, :4] = bboxes[:, :4]

    if source_format != "yolo":
        converted_bboxes[:, :4] = normalize_bboxes(converted_bboxes[:, :4], shape)

    # old = converted_bboxes
    if np.any(converted_bboxes[:, 0] < 0):
        # Logger.LogWarning(f"Albumentation transformation error: x_min < 0. Clipping values to 0")
        converted_bboxes[:, 0] = np.clip(converted_bboxes[:, 0], 0, None)
    if np.any(converted_bboxes[:, 1] < 0):
        # Logger.LogWarning(f"Albumentation transformation error: y_min < 0. Clipping values to 0")
        converted_bboxes[:, 1] = np.clip(converted_bboxes[:, 1], 0, None)
    if np.any(converted_bboxes[:, 2] > 1):
        # Logger.LogWarning(f"Albumentation transformation error: x_max > 1. Clipping values to 1")
        converted_bboxes[:, 2] = np.clip(converted_bboxes[:, 2], None, 1)
    if np.any(converted_bboxes[:, 3] > 1):
        # Logger.LogWarning(f"Albumentation transformation error: y_max > 1. Clipping values to 1")
        converted_bboxes[:, 3] = np.clip(converted_bboxes[:, 3], None, 1)
    
    if check_validity:
        check_bboxes(converted_bboxes)

    return converted_bboxes

def custom_convert_to_albumentations(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
    if self.params.clip:
        data_np = custom_convert_bboxes_to_albumentations(data, self.params.format, shape, check_validity=False)
        data_np = filter_bboxes(data_np, shape, min_area=0, min_visibility=0.4, min_width=0, min_height=0)
        check_bboxes(data_np)
        return data_np

    return custom_convert_bboxes_to_albumentations(data, self.params.format, shape, check_validity=True)

# Make BaseDataset init to include augmentation list
oldDatasetInit = BaseDataset.__init__
def CustomDatasetInit(
    self,
    img_path,
    imgsz=640,
    cache=False,
    augment=True,
    hyp=DEFAULT_CFG,
    prefix="",
    rect=False,
    batch_size=16,
    stride=32,
    pad=0.5,
    single_cls=False,
    classes=None,
    fraction=1.0,
):
    oldDatasetInit(
        self,
        img_path,
        imgsz,
        cache,
        augment,
        hyp,
        prefix,
        rect,
        batch_size,
        stride,
        pad,
        single_cls,
        classes,
        fraction
    )
    
    self.n = RandAugmentGenerator.globalN
    self.augmentationList = RandAugmentGenerator.globalAugmentationList
    # Logger.LogCustom("temp", f"RandAugmentFinder.globalAugmentationList2: {self.augmentationList}")
    pass
class MonkeyPatch:
    def __init__(self):
        self.albumentationInit = Albumentations.__init__
        self.buildTransforms = YOLODataset.build_transforms
        self.getItem = BaseDataset.__getitem__

    def ApplyMonkeyPatch(self):
        Albumentations.__init__ = CustomAlbumentationInit
        YOLODataset.build_transforms = CustomBuildTransforms
        BaseDataset.__getitem__ = CustomGetItem
        BaseDataset.__init__ = CustomDatasetInit
        BboxProcessor.convert_to_albumentations = custom_convert_to_albumentations

    def RevertMonkeyPatch(self):
        Albumentations.__init__ = self.albumentationInit
        YOLODataset.build_transforms = self.buildTransforms
        BaseDataset.__getitem__ = self.getItem
        

MonkeyPatch().ApplyMonkeyPatch()
