# Direct Depth Output Implementation Plan

## 🎯 목표
- Sigmoid 제거하고 Linear Depth를 직접 출력
- INT8 양자화 친화적 설계 (±28mm 균일 오류)

## 📋 수정 사항

### 1. ResNetSAN01.py 수정

#### **Option A: depth_transform 파라미터 추가 (추천)** ⭐
```python
class ResNetSAN01(nn.Module):
    def __init__(self, min_depth=0.5, max_depth=15.0, 
                 depth_transform='bounded_inverse',  # 'linear', 'log', 'bounded_inverse'
                 **kwargs):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_transform = depth_transform
        
        # Decoder는 그대로 (sigmoid 출력)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
    
    def run_network(self, rgb, input_depth=None):
        # Decoder sigmoid 출력
        outputs = self.decoder(skip_features)  # [0, 1]
        sigmoid = outputs[("disp", 0)]
        
        # Transform sigmoid to depth
        if self.depth_transform == 'linear':
            depth = self.min_depth + (self.max_depth - self.min_depth) * sigmoid
        elif self.depth_transform == 'log':
            log_range = torch.log(torch.tensor(self.max_depth / self.min_depth))
            depth = self.min_depth * torch.exp(log_range * sigmoid)
        elif self.depth_transform == 'bounded_inverse':
            inv_min = 1.0 / self.max_depth
            inv_max = 1.0 / self.min_depth
            inv_depth = inv_min + (inv_max - inv_min) * sigmoid
            depth = 1.0 / inv_depth
        
        return depth
```

#### **장점**:
- ✅ 기존 checkpoint 호환 (sigmoid 가중치 재사용)
- ✅ YAML에서 depth_transform만 변경
- ✅ Bounded Inverse / Linear / Log 모두 테스트 가능
- ✅ Decoder 구조 수정 불필요

#### **단점**:
- Sigmoid → Linear 변환 오버헤드 (negligible)

---

#### **Option B: Direct Linear Output Head** (더 깔끔하지만 재학습 필요)
```python
class ResNetSAN01(nn.Module):
    def __init__(self, min_depth=0.5, max_depth=15.0, 
                 use_sigmoid=False,  # NEW: False for direct depth
                 **kwargs):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_sigmoid = use_sigmoid
        
        if use_sigmoid:
            # Original: Sigmoid output
            self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        else:
            # NEW: Direct Depth output
            self.depth_head = nn.Sequential(
                nn.Conv2d(self.decoder.num_ch_dec[0], 1, kernel_size=1),
                nn.ReLU(),  # Ensure non-negative
            )
    
    def run_network(self, rgb, input_depth=None):
        features = self.decoder_network(skip_features)  # No sigmoid
        
        if self.use_sigmoid:
            # Original path
            depth_logits = self.sigmoid_head(features)
            sigmoid = torch.sigmoid(depth_logits)
            # Transform to depth...
        else:
            # NEW: Direct depth
            depth_logits = self.depth_head(features)
            depth = torch.clamp(depth_logits, min=self.min_depth, max=self.max_depth)
        
        return depth
```

#### **장점**:
- ✅ 가장 깔끔한 구조
- ✅ Sigmoid 변환 없음 (faster)
- ✅ INT8 양자화 최적화

#### **단점**:
- ❌ 기존 checkpoint 사용 불가 (재학습 필수)
- ❌ DepthDecoder 구조 수정 필요

---

### 2. Loss 계산 수정

#### 현재 구조:
```python
# SemiSupCompletionModel.py
def forward(self, batch):
    inv_depths = self.depth_net(rgb)  # Sigmoid outputs [0, 1]
    
    # Convert to depth for loss
    pred_depth = inv2depth(inv_depths)
    
    # Loss (SSI + Silog)
    loss = self.loss_fn(inv_depths, gt_inv_depth)  # SSI in inv_depth
```

#### 수정 (Option A 기준):
```python
# SemiSupCompletionModel.py
def forward(self, batch):
    depths = self.depth_net(rgb)  # Direct depth [0.5, 15.0]m
    
    # Loss (SSI + Silog)
    # SSI는 내부에서 inv_depth 변환
    loss = self.loss_fn(depths, gt_depth)
```

#### ssi_silog_loss.py 수정:
```python
class SSISilogLoss:
    def forward(self, pred_depth, gt_depth, mask=None):
        # ✅ SSI: Convert to inv_depth internally
        pred_inv = 1.0 / pred_depth
        gt_inv = 1.0 / gt_depth
        ssi_loss = self.compute_ssi_loss_inv(pred_inv, gt_inv, mask)
        
        # ✅ Silog: Use depth directly
        silog_loss = self.compute_silog_loss(pred_depth, gt_depth, mask)
        
        return self.ssi_weight * ssi_loss + self.silog_weight * silog_loss
```

---

### 3. YAML 설정

#### train_resnet_san_ncdb_640x384_direct_linear.yaml:
```yaml
model:
  arch: ResNetSAN01
  version: 18A
  min_depth: 0.5
  max_depth: 15.0
  depth_transform: 'linear'  # NEW! 'linear', 'log', 'bounded_inverse'
  use_film: false
  
  loss:
    supervised_method: 'sparse-ssi-silog'
    ssi_weight: 0.7
    silog_weight: 0.3
    min_depth: 0.5
    max_depth: 15.0
```

---

### 4. INT8 Quantization (ONNX)

#### FP32 ONNX Export:
```python
# scripts/export_onnx.py
model = ResNetSAN01(depth_transform='linear', min_depth=0.5, max_depth=15.0)
model.eval()

dummy_input = torch.randn(1, 3, 384, 640)
torch.onnx.export(
    model,
    dummy_input,
    "resnetsan_linear_depth.onnx",
    input_names=['rgb'],
    output_names=['depth'],  # [0.5, 15.0]m range
    opset_version=11
)
```

#### NPU INT8 Quantization:
```python
# Quantization parameters
scale = (15.0 - 0.5) / 255  # 0.056863
zero_point = 0

# FP32 → INT8
depth_fp32 = model(rgb)  # [0.5, 15.0]m
int8_value = ((depth_fp32 - 0.5) / scale).to(torch.uint8)  # [0, 255]

# INT8 → FP32 (NPU dequantization)
depth_reconstructed = 0.5 + scale * int8_value.to(torch.float32)

# Error: ±28mm (uniform)
```

---

## 🔧 Implementation Strategy

### **Phase 1: 기존 Checkpoint 활용 (즉시 테스트)**

1. **ResNetSAN01.py 수정 (Option A)**:
   - `depth_transform` 파라미터 추가
   - `run_network()`에서 Linear 변환 추가

2. **YAML 생성**:
   - `train_resnet_san_ncdb_640x384_linear.yaml`
   - `depth_transform: 'linear'` 설정

3. **테스트**:
   ```bash
   python scripts/eval.py --checkpoint checkpoints/resnetsan_linear_05_15.ckpt \
                          --config configs/eval_resnet_san_kitti.yaml \
                          --depth_transform linear
   ```

4. **예상 결과**:
   - abs_rel: 0.030 (PyTorch FP32)
   - 성능 변화 확인 (Linear vs Bounded Inverse)

---

### **Phase 2: 재학습 (Linear Depth 최적화)**

1. **Full Training**:
   ```bash
   python scripts/train.py configs/train_resnet_san_ncdb_640x384_linear.yaml
   ```

2. **목표**:
   - Training = Inference (완벽한 일치)
   - INT8 양자화 친화적 학습

---

### **Phase 3: INT8 Quantization**

1. **ONNX Export**:
   ```bash
   python scripts/export_onnx.py --checkpoint checkpoints/linear_depth.ckpt \
                                  --output onnx/resnetsan_linear_int8.onnx
   ```

2. **NPU INT8 Quantization**:
   - scale=0.056863, zero_point=0
   - 예상: abs_rel < 0.035 (vs 0.114 Bounded Inverse)

---

## 📊 Expected Results

| Method | Training | Inference | INT8 Error @ 15m | abs_rel (Expected) |
|--------|----------|-----------|------------------|-------------------|
| **Bounded Inverse** (현재) | Inv-Depth | Sigmoid→Inv | 853mm | 0.114 (NPU INT8) |
| **Linear (Phase 1)** | Inv-Depth | Sigmoid→Linear | 28mm | 0.040 (예상) |
| **Linear (Phase 2)** | Linear | Linear | 28mm | 0.032 (예상) |

---

## ✅ Action Items

1. ✅ INT8 분석 완료 (±28mm 균일 오류)
2. 🔄 ResNetSAN01.py 수정 (Option A)
3. 🔄 ssi_silog_loss.py 수정
4. 🔄 YAML 설정 생성
5. 🔄 Phase 1 테스트 (기존 checkpoint)
6. ⏸️ Phase 2 재학습 (선택)
7. ⏸️ Phase 3 INT8 Quantization

