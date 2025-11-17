#!/usr/bin/env python3
"""
PTQ Dual-Head 범위 조절 분석: 학습 코드에서 실제 계산

다양한 max_depth, min_depth 설정에서:
1. Integer/Fractional 분해 방식
2. 손실 함수 계산
3. PTQ 양자화 효과

를 수치로 비교합니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def decompose_depth_train(depth_gt, max_depth, min_depth):
    """
    학습 단계: Ground Truth 분해
    Integer: floor(depth_gt) / max_depth → [0, 1]
    Fractional: depth_gt - floor(depth_gt) → [0, 1]
    """
    integer_meters = torch.floor(depth_gt)
    integer_gt = integer_meters / max_depth  # Normalize to [0, 1]
    fractional_gt = depth_gt - integer_meters  # [0, 1]m
    
    return integer_gt, fractional_gt


def decompose_depth_ptq(depth_gt, max_depth, min_depth, n_levels=256):
    """
    PTQ 단계: 이산 양자화 기반 분해
    Integer: quantize(depth_gt / max_depth) × max_depth
    Fractional: 나머지
    """
    # Integer: 256 discrete levels
    int_levels = torch.round((depth_gt / max_depth) * (n_levels - 1))
    int_levels = torch.clamp(int_levels, 0, n_levels - 1)
    
    # Fractional: 나머지
    integer_meters = (int_levels / (n_levels - 1)) * max_depth
    fractional_gt = depth_gt - integer_meters
    fractional_gt = torch.clamp(fractional_gt, 0, 1.0)
    
    return int_levels / (n_levels - 1), fractional_gt


def dual_head_to_depth_train(integer_sigmoid, fractional_sigmoid, max_depth):
    """학습 단계: Integer와 Fractional 조합"""
    integer_part = integer_sigmoid * max_depth
    fractional_part = fractional_sigmoid
    depth = integer_part + fractional_part
    return depth


def dual_head_to_depth_ptq(integer_sigmoid, fractional_sigmoid, max_depth, n_levels=256):
    """PTQ 단계: 양자화된 값 조합"""
    integer_part = integer_sigmoid * max_depth
    fractional_part = fractional_sigmoid
    depth = integer_part + fractional_part
    return depth


def analyze_single_depth(depth_value, max_depth, min_depth, config_name):
    """
    단일 깊이 값에 대한 분석
    
    Returns:
        dict: 분석 결과
    """
    depth = torch.tensor([[[[depth_value]]]])
    
    # ===== 학습 단계 =====
    int_train, frac_train = decompose_depth_train(depth, max_depth, min_depth)
    reconstructed_train = dual_head_to_depth_train(int_train, frac_train, max_depth)
    
    # ===== PTQ 단계 =====
    int_ptq, frac_ptq = decompose_depth_ptq(depth, max_depth, min_depth, n_levels=256)
    reconstructed_ptq = dual_head_to_depth_ptq(int_ptq, frac_ptq, max_depth, n_levels=256)
    
    # 계산 결과
    results = {
        'depth': depth_value,
        'max_depth': max_depth,
        'min_depth': min_depth,
        'config': config_name,
        
        # Training phase
        'train_integer': int_train.item(),
        'train_integer_meters': (int_train * max_depth).item(),
        'train_fractional': frac_train.item(),
        'train_reconstructed': reconstructed_train.item(),
        'train_error': abs(reconstructed_train.item() - depth_value),
        
        # PTQ phase
        'ptq_integer_level': (int_ptq * 255).item(),  # [0, 255]
        'ptq_integer': int_ptq.item(),
        'ptq_integer_meters': (int_ptq * max_depth).item(),
        'ptq_fractional': frac_ptq.item(),
        'ptq_fractional_mm': (frac_ptq * 1000).item(),
        'ptq_reconstructed': reconstructed_ptq.item(),
        'ptq_error': abs(reconstructed_ptq.item() - depth_value),
        
        # 양자화 효과
        'int_quantization_interval': max_depth / 255,
        'frac_quantization_interval': 1.0 / 255 * 1000,  # mm
    }
    
    return results


def analyze_config(max_depth, min_depth, config_name, test_depths=None):
    """
    특정 config에 대한 전체 분석
    """
    if test_depths is None:
        test_depths = [
            min_depth,  # 최소값
            (max_depth + min_depth) / 2,  # 중간값
            max_depth * 0.5,  # 50%
            max_depth * 0.75,  # 75%
            max_depth * 0.9,  # 90%
            max_depth,  # 최대값
        ]
    
    results = []
    for depth in test_depths:
        result = analyze_single_depth(depth, max_depth, min_depth, config_name)
        results.append(result)
    
    return results


# ============================================================================
# 분석할 설정들
# ============================================================================

configs = [
    # 기존 설정
    {'max_depth': 15.0, 'min_depth': 0.5, 'name': 'Original (15m)'},
    
    # 더 짧은 범위 (실내 근거리)
    {'max_depth': 5.0, 'min_depth': 0.1, 'name': 'Short Range (5m)'},
    
    # 더 긴 범위 (장거리 실외)
    {'max_depth': 30.0, 'min_depth': 0.5, 'name': 'Long Range (30m)'},
    
    # 매우 긴 범위 (극장거리)
    {'max_depth': 80.0, 'min_depth': 0.5, 'name': 'Very Long Range (80m)'},
    
    # 중간 범위
    {'max_depth': 50.0, 'min_depth': 0.3, 'name': 'Medium Range (50m)'},
]

print("=" * 90)
print("PTQ Dual-Head 범위 조절 분석: 학습 코드에서의 실제 계산")
print("=" * 90)
print()

all_results = {}

for config in configs:
    max_d = config['max_depth']
    min_d = config['min_depth']
    name = config['name']
    
    print(f"\n{'='*90}")
    print(f"설정: {name}")
    print(f"  Max Depth: {max_d}m, Min Depth: {min_d}m")
    print(f"{'='*90}\n")
    
    results = analyze_config(max_d, min_d, name)
    all_results[name] = results
    
    # 테이블 출력
    print(f"{'Depth (m)':<12} {'Train Int':<12} {'Train Frac':<12} {'PTQ Int Lvl':<13} {'PTQ Frac (mm)':<15} {'Train Error':<12} {'PTQ Error':<12}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['depth']:<12.3f} "
              f"{r['train_integer']:<12.4f} "
              f"{r['train_fractional']:<12.4f} "
              f"{r['ptq_integer_level']:<13.1f} "
              f"{r['ptq_fractional_mm']:<15.2f} "
              f"{r['train_error']:<12.6f} "
              f"{r['ptq_error']:<12.6f}")
    
    print()
    
    # 상세 정보
    print("📊 양자화 효과:")
    print(f"  Integer 양자화 간격:    {results[0]['int_quantization_interval']:.4f}m ({results[0]['int_quantization_interval']*1000:.2f}mm)")
    print(f"  Fractional 양자화 간격: {results[0]['frac_quantization_interval']:.2f}mm")
    print()
    
    # 정밀도 통계
    train_errors = [r['train_error'] for r in results]
    ptq_errors = [r['ptq_error'] for r in results]
    
    print(f"📈 정밀도 통계:")
    print(f"  Train 평균 오차: {np.mean(train_errors):.4f}m ({np.mean(train_errors)*1000:.2f}mm)")
    print(f"  PTQ 평균 오차:   {np.mean(ptq_errors):.4f}m ({np.mean(ptq_errors)*1000:.2f}mm)")
    print(f"  Train 최악 오차: {np.max(train_errors):.4f}m ({np.max(train_errors)*1000:.2f}mm)")
    print(f"  PTQ 최악 오차:   {np.max(ptq_errors):.4f}m ({np.max(ptq_errors)*1000:.2f}mm)")


# ============================================================================
# 비교 분석
# ============================================================================

print(f"\n{'='*90}")
print("🔍 범위별 비교 분석")
print(f"{'='*90}\n")

print(f"{'Config':<25} {'Int Interval (mm)':<20} {'Frac Interval (mm)':<20} {'Total Levels':<15}")
print("-" * 80)

for name, results in all_results.items():
    r = results[0]  # 첫 번째 결과 사용
    int_interval = r['int_quantization_interval'] * 1000
    frac_interval = r['frac_quantization_interval']
    total_levels = 256 * 256
    
    print(f"{name:<25} {int_interval:<20.2f} {frac_interval:<20.2f} {total_levels:<15,}")

print()
print("✅ 정밀도 비교 (같은 깊이값 5.0m에서):")
print("-" * 90)

depth_test = 5.0
print(f"\n테스트 깊이: {depth_test}m")
print()
print(f"{'Config':<25} {'Train Integer':<15} {'PTQ Int Level':<15} {'Frac Recon':<15}")
print("-" * 70)

for name, results in all_results.items():
    # 5.0m과 가장 가까운 결과 찾기
    closest_result = min(results, key=lambda x: abs(x['depth'] - depth_test))
    
    print(f"{name:<25} "
          f"{closest_result['train_integer']:<15.4f} "
          f"{closest_result['ptq_integer_level']:<15.1f} "
          f"{closest_result['ptq_fractional_mm']:<15.2f}mm")


# ============================================================================
# 시각화
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('PTQ Dual-Head Range Analysis: Training vs PTQ Quantization', fontsize=16, fontweight='bold')

configs_to_plot = list(all_results.keys())[:5]  # 처음 5개 설정만 시각화

for idx, (name, results) in enumerate([(k, all_results[k]) for k in configs_to_plot]):
    ax = axes[idx // 3, idx % 3]
    
    depths = [r['depth'] for r in results]
    train_errors = [r['train_error'] * 1000 for r in results]  # mm
    ptq_errors = [r['ptq_error'] * 1000 for r in results]  # mm
    
    x = np.arange(len(depths))
    width = 0.35
    
    ax.bar(x - width/2, train_errors, width, label='Train', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, ptq_errors, width, label='PTQ', alpha=0.8, color='coral')
    
    ax.set_xlabel('Depth', fontsize=10)
    ax.set_ylabel('Error (mm)', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.1f}m' for d in depths], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

# 마지막 서브플롯: 범위별 비교
ax = axes[1, 2]
names = list(all_results.keys())[:5]
int_intervals = [all_results[name][0]['int_quantization_interval']*1000 for name in names]
frac_intervals = [all_results[name][0]['frac_quantization_interval'] for name in names]

x_pos = np.arange(len(names))
ax.bar(x_pos - 0.2, int_intervals, 0.4, label='Integer', alpha=0.8, color='steelblue')
ax.bar(x_pos + 0.2, frac_intervals, 0.4, label='Fractional', alpha=0.8, color='coral')
ax.set_xlabel('Configuration', fontsize=10)
ax.set_ylabel('Quantization Interval (mm)', fontsize=10)
ax.set_title('Quantization Intervals by Config', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' Range', '') for n in names], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/packnet-sfm/outputs/ptq_range_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ 시각화 저장됨: /workspace/packnet-sfm/outputs/ptq_range_analysis.png")

# ============================================================================
# 결론
# ============================================================================

print(f"\n{'='*90}")
print("📋 결론 및 권장사항")
print(f"{'='*90}\n")

print("""
1️⃣ 범위 조절의 영향

   max_depth를 늘리면:
   ✓ 더 먼 거리 측정 가능
   ✗ Integer 양자화 간격 증가 (정밀도 감소)
   
   예시:
   - 5m:  Integer 간격 = 19.6mm  (정밀, 근거리)
   - 15m: Integer 간격 = 58.8mm  (중간, 기본)
   - 30m: Integer 간격 = 117.6mm (저정밀, 장거리)
   - 80m: Integer 간격 = 313.7mm (아주 낮은, 극장거리)


2️⃣ Fractional Head의 중요성

   Fractional 양자화 간격은 항상 3.92mm (고정)
   
   따라서:
   - 짧은 범위 (5m):  정밀도 주도 = Fractional
   - 긴 범위 (30m+):  정밀도 = Integer에 지배됨
   
   → Fractional의 가중치를 높게 설정하는 이유!


3️⃣ 최적 설정 선택

   사용 케이스별 추천:
   
   📱 실내 / 근거리:
      max_depth = 5m,  min_depth = 0.1m
      정밀도: ~20mm (Fractional 주도)
      
   🚗 자율주행 (KITTI):
      max_depth = 80m, min_depth = 0.5m
      정밀도: ~314mm (Integer 주도, Fractional 보완)
      
   🏢 중간거리:
      max_depth = 30m, min_depth = 0.3m
      정밀도: ~118mm (균형)


4️⃣ PTQ 배포 시

   각 설정별 8-bit 양자화:
   
   Integer Head:
   - 출력 범위 [0, max_depth] → 8-bit [0, 255]
   - 각 레벨 = max_depth / 255
   
   Fractional Head:
   - 출력 범위 [0, 1]m → 8-bit [0, 255]
   - 각 레벨 = 3.92mm (고정)
   
   → 범위를 늘릴수록 Integer 양자화가 거칠어짐
      따라서 Fractional이 더욱 중요해짐!
""")

print(f"\n{'='*90}")
print("✅ 분석 완료!")
print(f"{'='*90}\n")
