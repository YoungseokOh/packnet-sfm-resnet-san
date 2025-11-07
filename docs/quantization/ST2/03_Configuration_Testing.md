# 3. 설정 및 테스트

## YAML Configuration

### 3.1. Single-Head (기존 - Baseline)

```yaml
# configs/train_resnet_san_ncdb_640x384.yaml
model:
    name: 'SemiSupCompletionModel'
    depth_net:
        name: 'ResNetSAN01'
        version: '18A'
        use_dual_head: false  # Single-Head (기존)
        use_film: false
        use_enhanced_lidar: false
    params:
        min_depth: 0.5
        max_depth: 15.0
```

### 3.2. Dual-Head (신규 - Experimental)

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
model:
    name: 'SemiSupCompletionModel'
    loss:
        supervised_method: 'sparse-l1'  # Dual-Head loss 자동 선택됨
        supervised_num_scales: 1
        supervised_loss_weight: 1.0
    depth_net:
        name: 'ResNetSAN01'
        version: '18A'
        use_dual_head: true   # 🆕 Dual-Head 활성화
        use_film: false       # FiLM 비활성화 (단순화)
        use_enhanced_lidar: false
    params:
        min_depth: 0.5
        max_depth: 15.0       # Integer head 범위
```

### 3.3. Dual-Head + FiLM (하이브리드)

```yaml
# configs/train_resnet_san_ncdb_dual_head_film_640x384.yaml
model:
    name: 'SemiSupCompletionModel'
    depth_net:
        name: 'ResNetSAN01'
        version: '18A'
        use_dual_head: true   # Dual-Head
        use_film: true        # + FiLM
        film_scales: [0]
        use_enhanced_lidar: false
    params:
        min_depth: 0.5
        max_depth: 15.0
```

---

## 테스트 및 검증

### 4.1. 단위 테스트

**테스트 스크립트**: `tests/test_dual_head_integration.py`

```bash
cd /workspace/packnet-sfm

# Test 1: Decoder만 테스트
python -c "
from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
import torch

decoder = DualHeadDepthDecoder([64, 64, 128, 256, 512], max_depth=15.0)
features = [torch.randn(1, c, 96//(2**i), 160//(2**i)) for i, c in enumerate([64, 64, 128, 256, 512])]
outputs = decoder(features)
assert ('integer', 0) in outputs and ('fractional', 0) in outputs
print('✅ Decoder test passed')
"

# Test 2: Helper functions
python -c "
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth, decompose_depth
import torch

depth = torch.tensor([[[[5.7]]]])
integer_gt, frac_gt = decompose_depth(depth, 15.0)
depth_recon = dual_head_to_depth(integer_gt, frac_gt, 15.0)
assert torch.allclose(depth, depth_recon)
print('✅ Helper functions test passed')
"
```

### 4.2. 통합 테스트

**전체 모델 로딩 테스트**:

```bash
# Single-Head (기존)
python -c "
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
import torch

model = ResNetSAN01(version='18A', use_dual_head=False, max_depth=15.0)
rgb = torch.randn(1, 3, 384, 640)
output = model.run_network(rgb)
print('✅ Single-Head integration test passed')
"

# Dual-Head (신규)
python -c "
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
import torch

model = ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
rgb = torch.randn(1, 3, 384, 640)
outputs, _ = model.run_network(rgb)
assert all(('integer', i) in outputs or ('disp', i) in outputs for i in range(4))
print('✅ Dual-Head integration test passed')
"
```

### 4.3. Backward Compatibility 검증

```bash
# 기존 checkpoint 로딩 테스트
python scripts/eval.py \
    --checkpoint checkpoints/resnetsan01_640x384_linear_05_15/epoch_29.ckpt \
    --config configs/train_resnet_san_ncdb_640x384.yaml

# 예상 결과: 정상 로딩 및 평가 (use_dual_head=false가 기본값)
```

---

## 상세 테스트 케이스

### Test Case 1: Decoder 출력 검증

```python
# tests/test_dual_head_decoder_detailed.py
import torch
import pytest
from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder

class TestDualHeadDecoder:
    @pytest.fixture
    def decoder(self):
        num_ch_enc = [64, 64, 128, 256, 512]
        return DualHeadDepthDecoder(num_ch_enc, max_depth=15.0, scales=[0])
    
    @pytest.fixture
    def dummy_features(self):
        return [
            torch.randn(2, 64, 96, 160),
            torch.randn(2, 64, 48, 80),
            torch.randn(2, 128, 24, 40),
            torch.randn(2, 256, 12, 20),
            torch.randn(2, 512, 6, 10),
        ]
    
    def test_output_keys(self, decoder, dummy_features):
        outputs = decoder(dummy_features)
        assert ("integer", 0) in outputs
        assert ("fractional", 0) in outputs
    
    def test_output_shapes(self, decoder, dummy_features):
        outputs = decoder(dummy_features)
        assert outputs[("integer", 0)].shape == (2, 1, 96, 160)
        assert outputs[("fractional", 0)].shape == (2, 1, 96, 160)
    
    def test_output_ranges(self, decoder, dummy_features):
        outputs = decoder(dummy_features)
        integer_out = outputs[("integer", 0)]
        fractional_out = outputs[("fractional", 0)]
        
        assert torch.all(integer_out >= 0.0) and torch.all(integer_out <= 1.0)
        assert torch.all(fractional_out >= 0.0) and torch.all(fractional_out <= 1.0)
    
    def test_gradient_flow(self, decoder, dummy_features):
        outputs = decoder(dummy_features)
        loss = outputs[("integer", 0)].sum() + outputs[("fractional", 0)].sum()
        loss.backward()
        
        # Check gradients exist
        assert any(p.grad is not None for p in decoder.parameters())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Test Case 2: Helper Functions 검증

```python
# tests/test_helper_functions_detailed.py
import torch
import pytest
from packnet_sfm.networks.layers.resnet.layers import (
    dual_head_to_depth, decompose_depth, dual_head_to_inv_depth
)

class TestHelperFunctions:
    @pytest.mark.parametrize("depth_value,max_depth", [
        (5.7, 15.0),
        (12.3, 15.0),
        (0.8, 15.0),
        (14.99, 15.0),
        (0.01, 15.0),
    ])
    def test_decompose_reconstruct(self, depth_value, max_depth):
        depth = torch.tensor([[[[depth_value]]]])
        integer_gt, frac_gt = decompose_depth(depth, max_depth)
        reconstructed = dual_head_to_depth(integer_gt, frac_gt, max_depth)
        
        assert torch.allclose(depth, reconstructed, atol=1e-5)
    
    def test_batch_processing(self):
        batch_depth = torch.tensor([
            [[[5.7, 12.3]]],
            [[[0.8, 14.5]]],
        ])
        max_depth = 15.0
        
        integer_gt, frac_gt = decompose_depth(batch_depth, max_depth)
        reconstructed = dual_head_to_depth(integer_gt, frac_gt, max_depth)
        
        assert torch.allclose(batch_depth, reconstructed, atol=1e-5)
    
    def test_edge_cases(self):
        max_depth = 15.0
        
        # Test exact integers
        depth_integers = torch.tensor([[[[0.0, 1.0, 5.0, 15.0]]]])
        integer_gt, frac_gt = decompose_depth(depth_integers, max_depth)
        
        # Fractional part should be 0 for exact integers
        assert torch.allclose(frac_gt, torch.zeros_like(frac_gt), atol=1e-5)
    
    def test_inv_depth_conversion(self):
        integer_sig = torch.tensor([[[[0.333]]]])
        frac_sig = torch.tensor([[[[0.5]]]])
        max_depth = 15.0
        min_depth = 0.5
        
        inv_depth = dual_head_to_inv_depth(integer_sig, frac_sig, max_depth, min_depth)
        
        # Expected: 0.333*15 + 0.5 = 5.5m → inv = 1/5.5 = 0.1818
        expected_inv = 1.0 / 5.5
        assert torch.allclose(inv_depth, torch.tensor([[[[expected_inv]]]]), atol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Test Case 3: 통합 테스트

```python
# tests/test_dual_head_integration_detailed.py
import torch
import pytest
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01

class TestDualHeadIntegration:
    @pytest.fixture
    def single_head_model(self):
        return ResNetSAN01(version='18A', use_dual_head=False, max_depth=15.0)
    
    @pytest.fixture
    def dual_head_model(self):
        return ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
    
    @pytest.fixture
    def dummy_rgb(self):
        return torch.randn(2, 3, 384, 640)
    
    def test_single_head_forward(self, single_head_model, dummy_rgb):
        single_head_model.eval()
        with torch.no_grad():
            output, _ = single_head_model.run_network(dummy_rgb)
        
        assert output.shape == (2, 1, 384, 640)
    
    def test_dual_head_forward(self, dual_head_model, dummy_rgb):
        dual_head_model.eval()
        with torch.no_grad():
            output, _ = dual_head_model.run_network(dummy_rgb)
        
        # In eval mode, returns ("disp", 0)
        assert output.shape == (2, 1, 384, 640)
    
    def test_dual_head_training_mode(self, dual_head_model, dummy_rgb):
        dual_head_model.train()
        inv_depths, features = dual_head_model.run_network(dummy_rgb)
        
        # In training mode, returns list of 4 scales
        assert len(inv_depths) == 4
        assert inv_depths[0].shape == (2, 1, 384, 640)
    
    def test_backward_compatibility(self, single_head_model, dual_head_model, dummy_rgb):
        # Both models should have similar output shapes
        single_head_model.eval()
        dual_head_model.eval()
        
        with torch.no_grad():
            single_out, _ = single_head_model.run_network(dummy_rgb)
            dual_out, _ = dual_head_model.run_network(dummy_rgb)
        
        assert single_out.shape == dual_out.shape
    
    def test_parameter_count(self, single_head_model, dual_head_model):
        single_params = sum(p.numel() for p in single_head_model.parameters())
        dual_params = sum(p.numel() for p in dual_head_model.parameters())
        
        # Dual-head should have slightly more parameters (2 output heads)
        # But difference should be minimal (only final conv layers)
        param_diff = dual_params - single_params
        print(f"Parameter difference: {param_diff:,}")
        
        # Expected: ~2x final conv layers (negligible compared to total)
        assert param_diff < single_params * 0.01  # Less than 1% increase

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 테스트 실행 가이드

### 전체 테스트 실행

```bash
cd /workspace/packnet-sfm

# 1. Unit tests (개별 컴포넌트)
pytest tests/test_dual_head_decoder_detailed.py -v
pytest tests/test_helper_functions_detailed.py -v

# 2. Integration tests (전체 시스템)
pytest tests/test_dual_head_integration_detailed.py -v

# 3. 모든 테스트 한 번에 실행
pytest tests/test_dual_head_*.py -v

# 4. Coverage 리포트와 함께 실행
pytest tests/test_dual_head_*.py --cov=packnet_sfm --cov-report=html
```

### 빠른 통합 테스트 (CLI)

```bash
# Decoder 단독 테스트
python -c "
from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
import torch

decoder = DualHeadDepthDecoder([64, 64, 128, 256, 512], max_depth=15.0)
features = [torch.randn(1, c, 96//(2**i), 160//(2**i)) for i, c in enumerate([64, 64, 128, 256, 512])]
outputs = decoder(features)
print(f'✅ Decoder outputs: {list(outputs.keys())}')
print(f'✅ Integer shape: {outputs[(\"integer\", 0)].shape}')
print(f'✅ Fractional shape: {outputs[(\"fractional\", 0)].shape}')
"

# Helper functions 테스트
python -c "
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth, decompose_depth
import torch

depth = torch.tensor([[[[5.7, 12.3, 0.8]]]])
integer_gt, frac_gt = decompose_depth(depth, 15.0)
reconstructed = dual_head_to_depth(integer_gt, frac_gt, 15.0)
error = torch.abs(depth - reconstructed).max().item()
print(f'✅ Reconstruction error: {error:.6f}m (should be < 1e-5)')
assert error < 1e-5
"

# 전체 모델 테스트
python -c "
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
import torch

# Single-Head
single_model = ResNetSAN01(version='18A', use_dual_head=False, max_depth=15.0)
print(f'✅ Single-Head initialized: is_dual_head={single_model.is_dual_head}')

# Dual-Head
dual_model = ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
print(f'✅ Dual-Head initialized: is_dual_head={dual_model.is_dual_head}')

rgb = torch.randn(1, 3, 384, 640)
dual_model.eval()
with torch.no_grad():
    output, _ = dual_model.run_network(rgb)
print(f'✅ Dual-Head forward pass: output shape={output.shape}')
"
```

---

## 성공 기준

### 단위 테스트 통과 조건

- ✅ **Decoder Test**: 모든 출력 키 존재, 올바른 shape, 값 범위 [0, 1]
- ✅ **Helper Functions Test**: Decompose → Reconstruct 오차 < 1e-5
- ✅ **Integration Test**: Single/Dual-Head 모두 forward pass 성공

### 통합 테스트 통과 조건

- ✅ **Backward Compatibility**: 기존 checkpoint 정상 로딩
- ✅ **YAML Configuration**: Single/Dual-Head 설정으로 모델 생성 성공
- ✅ **Gradient Flow**: Loss backpropagation 정상 동작

### 실제 학습 전 체크리스트

- [ ] 모든 단위 테스트 통과 (pytest)
- [ ] 통합 테스트 통과 (실제 모델 forward/backward)
- [ ] YAML config 로딩 테스트
- [ ] Dummy data로 1 epoch 학습 테스트
- [ ] Loss 값이 NaN이 아님 확인
- [ ] 메모리 사용량 확인 (기존 대비 큰 차이 없음)

→ **다음**: [학습 및 평가](04_Training_Evaluation.md)
