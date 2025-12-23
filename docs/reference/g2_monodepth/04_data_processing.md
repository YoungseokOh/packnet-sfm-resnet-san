# G2-MonoDepth 데이터 처리 상세

## 1. 개요

G2-MonoDepth의 데이터 처리는 **다양한 센서 특성을 시뮬레이션**하는 것이 핵심입니다.

### 1.1 핵심 아이디어

다양한 depth 센서는 각각 다른 특성을 가집니다:

| 센서 | Sparsity | 노이즈 | Artifacts |
|------|----------|--------|-----------|
| LiDAR | 매우 sparse (~5%) | 낮음 | 가끔 hole |
| Stereo | Dense (~70%) | 중간 | occlusion blur |
| SfM | 매우 sparse (~1%) | 높음 | outlier |
| ToF | Dense (~90%) | 낮음 | edge blur |
| RGB-only | 0% | - | - |

**G2-MonoDepth는 모든 sparsity (0%~100%)에서 학습**하여 일반화 능력을 확보합니다.

---

## 2. 입력 데이터 구조

### 2.1 5-Channel 입력

```python
# 입력 텐서 구성
input_tensor = torch.cat([
    rgb,          # [B, 3, H, W]: RGB 이미지
    point,        # [B, 1, H, W]: sparse depth 값
    hole_point    # [B, 1, H, W]: depth가 있는 위치 마스크
], dim=1)  # → [B, 5, H, W]
```

### 2.2 각 채널의 의미

| Channel | 이름 | 범위 | 설명 |
|---------|------|------|------|
| 0-2 | RGB | [0, 1] | 정규화된 RGB 이미지 |
| 3 | point | [0, max_depth] | sparse depth 값 (없으면 0) |
| 4 | hole_point | {0, 1} | depth 유무 binary 마스크 |

### 2.3 hole_point의 역할

```
GT Depth:           point (sparse):      hole_point:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 5.2  5.1  4.9│    │ 5.2  0.0  0.0│    │  1    0    0 │
│ 5.0  4.8  4.7│ →  │ 0.0  0.0  0.0│    │  0    0    0 │
│ 2.1  2.0  1.9│    │ 0.0  2.0  0.0│    │  0    1    0 │
└──────────────┘    └──────────────┘    └──────────────┘
```

- 네트워크가 "어디에 depth 정보가 있는지" 알 수 있음
- Sparsity 0% (RGB-only) 시: hole_point = all zeros

---

## 3. Data Augmentation

### 3.1 기본 Augmentation (data_tools.py)

```python
class TrainAugmentation:
    def __init__(self, config):
        self.img_size = config.img_size  # (H, W)
        
        # 확률 기반 augmentation
        self.p_flip = 0.5           # horizontal flip
        self.p_color = 0.5          # color jitter
        self.p_point_hole = 0.5     # depth에 구멍 생성
        self.p_point_noise = 0.5    # depth에 노이즈 추가
        self.p_point_blur = 0.5     # depth에 blur 적용
        
        # Sparsity 범위
        self.random_point_percentages = list(range(0, 101))  # 0% ~ 100%
```

### 3.2 Color Jitter

```python
def color_jitter(self, rgb):
    """
    RGB 이미지에 color augmentation 적용
    """
    if random.random() < self.p_color:
        # brightness, contrast, saturation, hue
        transforms = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        rgb = transforms(rgb)
    return rgb
```

### 3.3 Horizontal Flip

```python
def horizontal_flip(self, rgb, depth, point, hole_point):
    """
    모든 데이터에 동일하게 좌우 반전 적용
    """
    if random.random() < self.p_flip:
        rgb = torch.flip(rgb, dims=[-1])
        depth = torch.flip(depth, dims=[-1])
        point = torch.flip(point, dims=[-1])
        hole_point = torch.flip(hole_point, dims=[-1])
    return rgb, depth, point, hole_point
```

---

## 4. Sparse Depth Augmentation (핵심!)

### 4.1 Random Sparsity

```python
def apply_random_sparsity(self, depth):
    """
    GT depth에서 랜덤 sparsity로 sparse depth 생성
    
    Returns:
        point: sparse depth values
        hole_point: binary mask
    """
    # 랜덤 sparsity 선택 (0% ~ 100%)
    percentage = random.choice(self.random_point_percentages)
    
    # 유효한 depth 픽셀 위치
    valid_mask = (depth > 0)
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)
    num_valid = len(valid_indices[0])
    
    # 샘플링할 개수
    num_samples = int(num_valid * percentage / 100)
    
    # 랜덤 샘플링
    if num_samples > 0:
        sample_idx = random.sample(range(num_valid), num_samples)
        
        point = torch.zeros_like(depth)
        hole_point = torch.zeros_like(depth)
        
        for idx in sample_idx:
            h, w = valid_indices[0][idx], valid_indices[1][idx]
            point[0, h, w] = depth[0, h, w]
            hole_point[0, h, w] = 1.0
    else:
        # 0% sparsity: RGB-only mode
        point = torch.zeros_like(depth)
        hole_point = torch.zeros_like(depth)
    
    return point, hole_point
```

### 4.2 Sparsity 시각화

```
100% Sparsity:        50% Sparsity:         5% Sparsity:          0% Sparsity:
(Dense)               (Medium)              (Sparse like LiDAR)   (RGB-only)
┌──────────┐          ┌──────────┐          ┌──────────┐          ┌──────────┐
│●●●●●●●●●●│          │●  ●  ● ● │          │●         │          │          │
│●●●●●●●●●●│          │ ●●  ●  ● │          │    ●     │          │          │
│●●●●●●●●●●│   →      │●  ●●   ● │   →      │      ●   │   →      │          │
│●●●●●●●●●●│          │ ● ●  ●●  │          │  ●       │          │          │
│●●●●●●●●●●│          │●●  ●   ● │          │        ● │          │          │
└──────────┘          └──────────┘          └──────────┘          └──────────┘
```

---

## 5. Depth Artifacts 시뮬레이션

### 5.1 Point Hole (구멍 생성)

센서가 특정 영역에서 depth를 못 읽는 상황 시뮬레이션:

```python
def apply_point_hole(self, point, hole_point):
    """
    Sparse depth에 랜덤 구멍 생성
    LiDAR가 반사율 낮은 표면에서 측정 실패하는 것을 시뮬레이션
    """
    if random.random() < self.p_point_hole:
        # 랜덤 크기의 직사각형 영역을 0으로
        h, w = point.shape[-2:]
        
        # 구멍 크기: 이미지의 5%~20%
        hole_h = random.randint(int(h * 0.05), int(h * 0.2))
        hole_w = random.randint(int(w * 0.05), int(w * 0.2))
        
        # 랜덤 위치
        start_h = random.randint(0, h - hole_h)
        start_w = random.randint(0, w - hole_w)
        
        # 구멍 적용
        point[:, start_h:start_h+hole_h, start_w:start_w+hole_w] = 0
        hole_point[:, start_h:start_h+hole_h, start_w:start_w+hole_w] = 0
    
    return point, hole_point
```

### 5.2 Point Noise (노이즈 추가)

센서 측정 오차 시뮬레이션:

```python
def apply_point_noise(self, point, hole_point):
    """
    Sparse depth에 가우시안 노이즈 추가
    ToF 센서의 측정 오차를 시뮬레이션
    """
    if random.random() < self.p_point_noise:
        # 노이즈 강도: depth 값의 1%~5%
        noise_ratio = random.uniform(0.01, 0.05)
        
        # 가우시안 노이즈
        noise = torch.randn_like(point) * point * noise_ratio
        
        # hole_point가 1인 곳에만 적용
        point = point + noise * hole_point
        
        # 음수 방지
        point = torch.clamp(point, min=0)
    
    return point, hole_point
```

### 5.3 Point Blur (블러 적용)

Edge에서의 depth bleeding 시뮬레이션:

```python
def apply_point_blur(self, point, hole_point):
    """
    Sparse depth에 가우시안 블러 적용
    ToF 센서의 flying pixels 현상을 시뮬레이션
    """
    if random.random() < self.p_point_blur:
        # 커널 크기: 3, 5, 7 중 랜덤
        kernel_size = random.choice([3, 5, 7])
        sigma = kernel_size / 3.0
        
        # 가우시안 블러
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        
        # 블러 적용
        blurred_point = blur(point)
        
        # hole 영역에만 블러 적용 (hole 바깥은 0 유지)
        point = blurred_point * (hole_point > 0).float()
    
    return point, hole_point
```

---

## 6. StandardizeData (Robust Normalization)

### 6.1 왜 Robust Standardization인가?

일반 standardization:
$$z = \frac{x - \mu}{\sigma}$$

Depth 데이터의 문제:
- Sky (무한대) → 매우 큰 값
- 가까운 객체 → 작은 값
- 분포가 heavily skewed

### 6.2 구현

```python
class StandardizeData(torch.nn.Module):
    """
    Masked Mean + MAD (Mean Absolute Deviation) standardization
    """
    def __init__(self):
        super(StandardizeData, self).__init__()
    
    @staticmethod
    def __masked_mean_robust_standardization__(depth, mask, eps=1e-6):
        """
        Args:
            depth: [B, 1, H, W]
            mask: [B, 1, H, W] - 유효한 depth 위치
        
        Returns:
            mean: [B, 1, 1, 1]
            std: [B, 1, 1, 1]
        """
        # 유효 픽셀 수
        mask_num = torch.sum(mask, dim=(1, 2, 3))  # [B]
        mask_num[mask_num == 0] = eps  # 0 나누기 방지
        
        # Masked Mean
        depth_mean = torch.sum(depth * mask, dim=(1, 2, 3)) / mask_num
        depth_mean = depth_mean.view(depth.shape[0], 1, 1, 1)
        
        # MAD (Mean Absolute Deviation)
        # 일반 std: sqrt(E[(x-μ)²])
        # MAD: E[|x-μ|]  ← outlier에 더 강건
        depth_std = torch.sum(
            torch.abs((depth - depth_mean) * mask), 
            dim=(1, 2, 3)
        ) / mask_num
        depth_std = depth_std.view(depth.shape[0], 1, 1, 1) + eps
        
        return depth_mean, depth_std
    
    def forward(self, depth, gt, mask_hole):
        """
        Standardize both prediction and GT to relative domain
        """
        t_d, s_d = self.__masked_mean_robust_standardization__(depth, mask_hole)
        t_g, s_g = self.__masked_mean_robust_standardization__(gt, mask_hole)
        
        sta_depth = (depth - t_d) / s_d
        sta_gt = (gt - t_g) / s_g
        
        return sta_depth, sta_gt
```

### 6.3 Standardization 효과

```
Original Depth Distribution:
        │
 count  │    ╭╮
        │   ╭╯│
        │  ╭╯ │  (skewed)
        │ ╭╯  ╰╮
        │╭╯    ╰───────
        └─────────────────→ depth
          0   5   10  50

After Standardization:
        │
 count  │      ╭─╮
        │     ╭╯ ╰╮
        │    ╭╯   ╰╮ (centered, scaled)
        │   ╭╯     ╰╮
        │───╯       ╰───
        └─────────────────→ z
         -2  -1   0   1   2
```

---

## 7. Dataset 클래스

### 7.1 NYUv2 Dataset

```python
class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: 데이터 루트 경로
            split: 'train' or 'test'
            transform: augmentation 함수
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # 파일 리스트 로드
        self.rgb_paths = sorted((self.root_dir / split / 'rgb').glob('*.png'))
        self.depth_paths = sorted((self.root_dir / split / 'depth').glob('*.png'))
    
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        # RGB 로드
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = T.ToTensor()(rgb)  # [3, H, W], [0, 1]
        
        # Depth 로드 (16-bit PNG)
        depth = cv2.imread(str(self.depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 1000.0  # mm → m
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        
        # Augmentation 적용
        if self.transform:
            rgb, depth, point, hole_point = self.transform(rgb, depth)
        
        return {
            'rgb': rgb,              # [3, H, W]
            'depth': depth,          # [1, H, W]
            'point': point,          # [1, H, W]
            'hole_point': hole_point # [1, H, W]
        }
```

### 7.2 DataLoader 구성

```python
def create_dataloader(config, split='train'):
    """
    Create DataLoader with augmentation
    """
    if split == 'train':
        transform = TrainAugmentation(config)
        shuffle = True
    else:
        transform = None  # 검증/테스트는 augmentation 없음
        shuffle = False
    
    dataset = NYUv2Dataset(
        root_dir=config.data_root,
        split=split,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
```

---

## 8. 우리 프로젝트와의 비교

### 8.1 데이터 처리 비교

| 항목 | G2-MonoDepth | PackNet-SfM (우리) |
|------|--------------|--------------------|
| 입력 | RGB + sparse depth | RGB only     |
| Channels | 5 (RGB + point + hole) | 3 (RGB) |
| Sparsity Aug | 0~100% | ❌ |
| Depth Artifacts | hole, noise, blur | ❌ |
| Color Aug | jitter | jitter |
| Geometric Aug | flip | flip, rotation |

### 8.2 적용 가능한 요소

1. **Robust Standardization**:
   - 우리 SSI Loss에서도 유사한 개념 사용 중
   - MAD 기반 normalization 검토 가능

2. **Color Jitter 강화**:
   - 현재보다 더 다양한 color augmentation
   - 야간/조명 변화에 대한 강건성

3. **개념적 참고**:
   - hole_point 같은 "정보 유무 마스크"
   - 우리는 mask를 GT 유효성에만 사용 중

### 8.3 우리 데이터셋 특성

```
NCDB Dataset:
├── Train: 5,072장
├── Validation: 634장
└── Test: 635장

특성:
- 주/야간 혼합
- Fisheye 카메라
- LiDAR-based distance GT
- 640×384 해상도
```

---

## 9. 핵심 인사이트

### 9.1 G2-MonoDepth의 데이터 전략

1. **Sparsity Generalization**: 모든 sparsity에서 학습하여 어떤 입력에도 대응
2. **Artifact Robustness**: 다양한 센서 특성 시뮬레이션
3. **Scale Invariance**: Robust standardization으로 스케일 변화 대응

### 9.2 우리 프로젝트에의 시사점

- RGB-only이므로 sparsity augmentation은 직접 적용 불가
- 하지만 **data augmentation 강화**는 참고 가능
- **Robust normalization** 개념은 loss 설계에 참고 가능

---

## 다음 문서

- [05_training_process.md](05_training_process.md) - 학습 프로세스 상세
