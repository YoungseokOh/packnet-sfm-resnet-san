# Copyright 2020 Toyota Research Institute.  All rights reserved.
from functools import partial
from packnet_sfm.datasets.augmentations import resize_image, resize_sample, resize_depth, \
    duplicate_sample, colorjitter_sample, to_tensor_sample, crop_sample, crop_sample_input, resize_depth_preserve
from packnet_sfm.utils.misc import parse_crop_borders

# 🆕 Advanced augmentation import (선택적)
try:
    from packnet_sfm.datasets.augmentations_kitti_compatible import (
        KITTIAdvancedTrainTransform, KITTIAdvancedValTransform
    )
    ADVANCED_AUGMENTATION_AVAILABLE = True
except ImportError:
    ADVANCED_AUGMENTATION_AVAILABLE = False

########################################################################################################################

def train_transforms(sample, image_shape, jittering, crop_train_borders):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    crop_train_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_train_borders) > 0:
        borders = parse_crop_borders(crop_train_borders, sample['rgb'].size[::-1])
        sample = crop_sample(sample, borders)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    sample = to_tensor_sample(sample)
    return sample

def validation_transforms(sample, image_shape, crop_eval_borders):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    crop_eval_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample['rgb'].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        if 'input_depth' in sample:
            sample['input_depth'] = resize_depth_preserve(sample['input_depth'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape, crop_eval_borders):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample['rgb'].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        if 'input_depth' in sample:
            sample['input_depth'] = resize_depth(sample['input_depth'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape=(), jittering=(), crop_train_borders=(),
                   crop_eval_borders=(), **kwargs):
    """
    Get data augmentation transformations for each split
    """
    
    # 🆕 Advanced augmentation 검사 (kwargs에서 확인)
    augmentation = kwargs.get('augmentation', None)
    
    if augmentation is not None:
        # Advanced augmentation이 활성화되어 있는지 확인
        has_advanced = any([
            augmentation.get('randaugment', {}).get('enabled', False),
            augmentation.get('random_erasing', {}).get('enabled', False),
            augmentation.get('mixup', {}).get('enabled', False),
            augmentation.get('cutmix', {}).get('enabled', False),
        ])
        
        if has_advanced:
            print(f"🎨 Advanced augmentation activated for {mode}")
            # 🆕 실제로 Advanced Transform 사용
            if ADVANCED_AUGMENTATION_AVAILABLE:
                if mode == 'train':
                    return KITTIAdvancedTrainTransform(augmentation)
                else:
                    return KITTIAdvancedValTransform(augmentation)
            else:
                print(f"⚠️ Advanced augmentation requested but not available")
    
    # 기존 코드 그대로 유지
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       crop_train_borders=crop_train_borders)
    elif mode == 'validation':
        return partial(validation_transforms,
                       crop_eval_borders=crop_eval_borders,
                       image_shape=image_shape)
    elif mode == 'test':
        return partial(test_transforms,
                       crop_eval_borders=crop_eval_borders,
                       image_shape=image_shape)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

