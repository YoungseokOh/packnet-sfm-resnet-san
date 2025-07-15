"""
KITTI 최적화 데이터셋과 호환되는 Advanced Augmentation 시스템
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps

from packnet_sfm.datasets.augmentations import resize_sample, to_tensor_sample, duplicate_sample
from packnet_sfm.utils.misc import filter_dict

########################################################################################################################
#### RANDAUGMENT (기존과 동일)
########################################################################################################################

class RandAugment:
    """RandAugment implementation for depth estimation"""
    
    def __init__(self, n=9, m=0.5):
        self.n = n
        self.m = m
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.rotate, 0, 30),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
        ]

    def __call__(self, img):
        """Apply RandAugment to PIL image"""
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

    def auto_contrast(self, pil_img, level):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, level):
        return ImageOps.equalize(pil_img)

    def rotate(self, pil_img, level):
        degrees = int(level)
        if random.random() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def color(self, pil_img, level):
        return ImageEnhance.Color(pil_img).enhance(level)

    def contrast(self, pil_img, level):
        return ImageEnhance.Contrast(pil_img).enhance(level)

    def brightness(self, pil_img, level):
        return ImageEnhance.Brightness(pil_img).enhance(level)

    def sharpness(self, pil_img, level):
        return ImageEnhance.Sharpness(pil_img).enhance(level)

########################################################################################################################
#### RANDOM ERASING (텐서용)
########################################################################################################################

class RandomErasing:
    """Random Erasing augmentation for tensors"""
    
    def __init__(self, probability=0.1, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        return img

########################################################################################################################
#### KITTI 호환 Transform 클래스
########################################################################################################################

class KITTIAdvancedTrainTransform:
    """KITTI Advanced Training Transform with RandAugment, RandomErasing, MixUp, CutMix"""
    
    def __init__(self, augmentation_config):
        self.augmentation_config = augmentation_config
        
        # 🔧 image_shape 처리 개선
        image_shape = augmentation_config.get('image_shape', ())
        if not image_shape or len(image_shape) == 0:
            # 기본값으로 KITTI 표준 크기 사용
            self.image_shape = (352, 1216)
            print(f"⚠️ Empty image_shape detected, using default: {self.image_shape}")
        else:
            self.image_shape = tuple(image_shape)
        
        # 기타 설정들
        self.jittering = augmentation_config.get('jittering', (0.2, 0.2, 0.2, 0.05))
        self.crop_train_borders = augmentation_config.get('crop_train_borders', ())
        
        # Advanced augmentation 설정들
        self.randaugment_config = augmentation_config.get('randaugment', {})
        self.random_erasing_config = augmentation_config.get('random_erasing', {})
        self.mixup_config = augmentation_config.get('mixup', {})
        self.cutmix_config = augmentation_config.get('cutmix', {})
        
        print(f"🎨 Advanced Train Transform initialized:")
        print(f"   - RandAugment: {self.randaugment_config.get('enabled', False)}")
        print(f"   - RandomErasing: {self.random_erasing_config.get('enabled', False)}")
        print(f"   - MixUp: {self.mixup_config.get('enabled', False)}")
        print(f"   - CutMix: {self.cutmix_config.get('enabled', False)}")

    def __call__(self, sample):
        """Apply advanced training transforms"""
        
        # 1. 기본 변환 (crop, resize)
        if len(self.crop_train_borders) > 0:
            from packnet_sfm.utils.misc import parse_crop_borders
            from packnet_sfm.datasets.augmentations import crop_sample
            borders = parse_crop_borders(self.crop_train_borders, sample['rgb'].size[::-1])
            sample = crop_sample(sample, borders)
        
        # 2. 이미지 리사이즈 (image_shape가 유효한 경우에만)
        if len(self.image_shape) == 2:
            from packnet_sfm.datasets.augmentations import resize_sample
            sample = resize_sample(sample, self.image_shape)
        
        # 3. Color jittering
        if len(self.jittering) > 0:
            from packnet_sfm.datasets.augmentations import colorjitter_sample
            sample = colorjitter_sample(sample, self.jittering)
        
        # 4. RandAugment (자체 구현 사용)
        if self.randaugment_config.get('enabled', False):
            try:
                prob = self.randaugment_config.get('prob', 0.5)
                if torch.rand(1).item() < prob:
                    n = self.randaugment_config.get('n', 9)
                    m = self.randaugment_config.get('m', 0.5)
                    
                    # 자체 구현된 RandAugment 사용
                    randaug = RandAugment(n=n, m=m)
                    sample['rgb'] = randaug(sample['rgb'])
                    
            except Exception as e:
                print(f"⚠️ RandAugment failed: {e}")
        
        # 5. RandomErasing은 tensor 변환 후 적용
        from packnet_sfm.datasets.augmentations import to_tensor_sample
        sample = to_tensor_sample(sample)
        
        if self.random_erasing_config.get('enabled', False):
            try:
                prob = self.random_erasing_config.get('probability', 0.1)
                if torch.rand(1).item() < prob:
                    sl = self.random_erasing_config.get('sl', 0.02)
                    sh = self.random_erasing_config.get('sh', 0.4)
                    r1 = self.random_erasing_config.get('r1', 0.3)
                    mean = self.random_erasing_config.get('mean', [0.485, 0.456, 0.406])
                    
                    # 자체 구현된 RandomErasing 사용
                    erasing = RandomErasing(probability=1.0, sl=sl, sh=sh, r1=r1, mean=mean)
                    sample['rgb'] = erasing(sample['rgb'])
                    
            except Exception as e:
                print(f"⚠️ RandomErasing failed: {e}")
        
        return sample


class KITTIAdvancedValTransform:
    """KITTI Advanced Validation Transform (validation/test용)"""
    
    def __init__(self, augmentation_config):
        self.augmentation_config = augmentation_config
        
        # image_shape 처리
        image_shape = augmentation_config.get('image_shape', ())
        if not image_shape or len(image_shape) == 0:
            self.image_shape = ()
        else:
            self.image_shape = tuple(image_shape)
        
        self.crop_eval_borders = augmentation_config.get('crop_eval_borders', ())

    def __call__(self, sample):
        """Apply validation transforms (no advanced augmentation)"""
        
        # 1. Crop 적용 (validation용)
        if len(self.crop_eval_borders) > 0:
            from packnet_sfm.utils.misc import parse_crop_borders
            from packnet_sfm.datasets.augmentations import crop_sample_input
            borders = parse_crop_borders(self.crop_eval_borders, sample['rgb'].size[::-1])
            sample = crop_sample_input(sample, borders)
        
        # 2. 이미지 리사이즈 (image_shape가 유효한 경우에만)
        if len(self.image_shape) == 2:
            from packnet_sfm.datasets.augmentations import resize_sample
            sample = resize_sample(sample, self.image_shape)
        
        # 3. Tensor 변환
        from packnet_sfm.datasets.augmentations import to_tensor_sample
        sample = to_tensor_sample(sample)
        
        return sample

########################################################################################################################
#### MIXUP & CUTMIX (배치 레벨)
########################################################################################################################

class MixUp:
    """MixUp augmentation for batches"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch):
        if self.alpha <= 0:
            return batch
            
        batch_size = batch['rgb'].size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(batch_size)
        
        # RGB 이미지 믹싱
        batch['rgb'] = lam * batch['rgb'] + (1 - lam) * batch['rgb'][indices]
        
        # Depth 믹싱 (둘 다 유효한 경우만)
        if 'depth' in batch:
            valid_mask = (batch['depth'].sum(dim=[1,2,3]) > 0) & \
                        (batch['depth'][indices].sum(dim=[1,2,3]) > 0)
            
            mixed_depth = batch['depth'].clone()
            mixed_depth[valid_mask] = lam * batch['depth'][valid_mask] + \
                                     (1 - lam) * batch['depth'][indices][valid_mask]
            batch['depth'] = mixed_depth
            
        return batch

class CutMix:
    """CutMix augmentation for batches"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        if self.alpha <= 0:
            return batch
            
        batch_size = batch['rgb'].size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        _, _, H, W = batch['rgb'].shape
        
        # 랜덤 박스 생성
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        indices = torch.randperm(batch_size)
        
        # CutMix 적용
        batch['rgb'][:, :, bby1:bby2, bbx1:bbx2] = batch['rgb'][indices, :, bby1:bby2, bbx1:bbx2]
        
        return batch

def create_kitti_advanced_collate_fn(augmentation_config):
    """KITTI용 advanced collate function 생성"""
    from packnet_sfm.models.model_utils import prep_dataset
    
    # 배치 레벨 augmentation 초기화
    batch_mixup = None
    batch_cutmix = None
    
    mixup_config = augmentation_config.get('mixup', {})
    if mixup_config.get('enabled', False):
        batch_mixup = MixUp(
            alpha=mixup_config.get('alpha', 0.2)
        )
    
    cutmix_config = augmentation_config.get('cutmix', {})
    if cutmix_config.get('enabled', False):
        batch_cutmix = CutMix(
            alpha=cutmix_config.get('alpha', 1.0)
        )
    
    def collate_fn(batch):
        # 표준 collation
        batch = prep_dataset(batch)
        
        # 배치 레벨 augmentation 적용
        if batch_mixup and random.random() < mixup_config.get('prob', 0.5):
            batch = batch_mixup(batch)
        
        if batch_cutmix and random.random() < cutmix_config.get('prob', 0.5):
            batch = batch_cutmix(batch)
            
        return batch
    
    return collate_fn