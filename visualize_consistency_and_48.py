#!/usr/bin/env python3
"""
Consistency Weight와 48 Levels 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# 한글 폰트 설정
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. Consistency Weight 영향 시각화
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Consistency Weight 및 48 Levels 완전 설명', fontsize=16, fontweight='bold')

# ============================================================================
# 서브플롯 1: Consistency Weight별 손실 구성
# ============================================================================
ax1 = axes[0, 0]

weights = ['0.0\n(협력 무시)', '0.25\n(약한 협력)', '0.5\n(균형 협력)\n★현재', '1.0\n(강한 협력)']
integer_contrib = np.array([8.3, 9.3, 9.1, 8.3])  # %
fractional_contrib = np.array([91.7, 86.9, 87.6, 90.9])  # %
consistency_contrib = np.array([0, 3.8, 3.3, 0.8])  # %

x = np.arange(len(weights))
width = 0.6

p1 = ax1.bar(x, integer_contrib, width, label='Integer Loss (1.0×)', color='#FF6B6B')
p2 = ax1.bar(x, fractional_contrib, width, bottom=integer_contrib, 
             label='Fractional Loss (10.0×)', color='#4ECDC4')
p3 = ax1.bar(x, consistency_contrib, width, 
             bottom=integer_contrib+fractional_contrib,
             label='Consistency Loss', color='#95E1D3')

ax1.set_ylabel('Loss Contribution (%)', fontweight='bold')
ax1.set_title('Consistency Weight별 손실 기여도', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(weights)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim([0, 100])

# 주석 추가
ax1.text(2, 50, '추천\n설정', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# 서브플롯 2: Integer vs Fractional 정보 용량
# ============================================================================
ax2 = axes[0, 1]

components = ['Integer Head\n(48 levels)', 'Fractional Head\n(256 levels)', 'PTQ 배포 후\nInteger & Frac\n(각 256 levels)']
bits = [5.58, 8.0, 8.0]
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

bars = ax2.barh(components, bits, color=colors)
ax2.set_xlabel('정보 용량 (bits)', fontweight='bold')
ax2.set_title('정보 용량 비교', fontweight='bold')
ax2.set_xlim([0, 9])

# 값 레이블 추가
for i, (bar, bit) in enumerate(zip(bars, bits)):
    ax2.text(bit + 0.1, i, f'{bit:.2f} bits', va='center', fontweight='bold')

# 간격 정보 추가
ax2.text(4, -0.8, 'Integer: 15m÷48=0.31m', fontsize=8, ha='center', style='italic')
ax2.text(4, -1.2, 'Fractional: 1m÷255=3.9mm', fontsize=8, ha='center', style='italic')

# ============================================================================
# 서브플롯 3: 48 레벨의 의미
# ============================================================================
ax3 = axes[1, 0]
ax3.axis('off')

# 텍스트 박스로 설명
explanation_text = """
【 48 레벨이란? 】

1️⃣  ResNet 구조
   입력: 640 × 384 해상도
   ├─ 8배 축소 (1/8): 80 × 48 ← Integer/Fractional 출력
   ├─ 16배 축소 (1/16): 40 × 24
   ├─ 32배 축소 (1/32): 20 × 12
   └─ 64배 축소 (1/64): 10 × 6

2️⃣  정보 용량
   Integer: 15m ÷ 48 = 0.31m (312mm 간격)
   → log₂(48) = 5.58 bits 정보
   
3️⃣  설계 의도
   Integer: 정수부 (범위) 담당 [0~15m]
   Fractional: 소수부 (정밀도) 담당 [0~1m]
   
   → 역할 분담으로 최적화!

4️⃣  PTQ 배포
   훈련: Float32 (Integer 48 → Fractional 256)
   배포: Int8 (Integer 256 → Fractional 256)
   → 양쪽 다 256 레벨로 통일
"""

ax3.text(0.05, 0.95, explanation_text, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# 서브플롯 4: 손실 함수 구조
# ============================================================================
ax4 = axes[1, 1]

depth_range = np.linspace(0, 15, 100)

# 각 weight별 총 손실 계산 (시뮬레이션)
loss_components = {
    'L_int': 0.1 * np.exp(-depth_range / 5),      # Integer 손실
    'L_frac': 0.05 * np.ones_like(depth_range),   # Fractional 손실
    'L_cons': 0.02 * np.exp(-depth_range / 8)     # Consistency 손실
}

# 다양한 consistency_weight별 총 손실
total_loss_0 = 1.0 * loss_components['L_int'] + 10.0 * loss_components['L_frac']
total_loss_05 = 1.0 * loss_components['L_int'] + 10.0 * loss_components['L_frac'] + 0.5 * loss_components['L_cons']
total_loss_1 = 1.0 * loss_components['L_int'] + 10.0 * loss_components['L_frac'] + 1.0 * loss_components['L_cons']

ax4.plot(depth_range, total_loss_0, 'o-', label='consistency_weight=0.0', linewidth=2, markersize=3)
ax4.plot(depth_range, total_loss_05, 's-', label='consistency_weight=0.5 ★', linewidth=2.5, markersize=3, color='green')
ax4.plot(depth_range, total_loss_1, '^-', label='consistency_weight=1.0', linewidth=2, markersize=3)

ax4.set_xlabel('Depth (m)', fontweight='bold')
ax4.set_ylabel('Total Loss', fontweight='bold')
ax4.set_title('Consistency Weight별 총 손실 곡선', fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/packnet-sfm/outputs/consistency_weight_and_48_levels.png', dpi=150, bbox_inches='tight')
print("✅ 시각화 저장: outputs/consistency_weight_and_48_levels.png")

# ============================================================================
# 추가 상세 시각화: 48 레벨 vs 256 레벨
# ============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Integer (48 레벨) vs Fractional (256 레벨) 상세 비교', 
              fontsize=14, fontweight='bold')

# ============================================================================
# Integer: 48 레벨 분포
# ============================================================================
ax_int = axes2[0]

integer_levels = 48
integer_values = np.linspace(0, 15, integer_levels + 1)
integer_interval = 15 / integer_levels

# 히스토그램 그리기
ax_int.bar(range(integer_levels), np.ones(integer_levels), 
          color='#FF6B6B', alpha=0.7, edgecolor='darkred', linewidth=0.5)
ax_int.set_xlabel('Integer Level', fontweight='bold')
ax_int.set_ylabel('Count', fontweight='bold')
ax_int.set_title(f'Integer Head: 48 Levels\nInterval: {integer_interval:.3f}m (312mm)',
                fontweight='bold', fontsize=11)
ax_int.set_ylim([0, 1.2])

# 정보 표시
info_text_int = f"""
정보 용량: log₂(48) = 5.58 bits
범위: 0 ~ 15m
간격: 15m ÷ 48 = 0.3125m
역할: 정수부 (범위 위치)

훈련 중: Sigmoid [0, 1] × 15
배포 시: 0~255 (256 레벨로 양자화)
"""
ax_int.text(0.98, 0.97, info_text_int, transform=ax_int.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.2))

# ============================================================================
# Fractional: 256 레벨 분포
# ============================================================================
ax_frac = axes2[1]

fractional_levels = 256
fractional_values = np.linspace(0, 1, fractional_levels + 1)
fractional_interval = 1.0 / fractional_levels

# 대표 샘플만 표시 (너무 많으니까)
sample_frac_levels = 32
ax_frac.bar(range(sample_frac_levels), np.ones(sample_frac_levels),
           color='#4ECDC4', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
ax_frac.set_xlabel('Fractional Level (샘플: 32/256)', fontweight='bold')
ax_frac.set_ylabel('Count', fontweight='bold')
ax_frac.set_title(f'Fractional Head: 256 Levels\nInterval: {fractional_interval*1000:.2f}mm (3.92mm)',
                 fontweight='bold', fontsize=11)
ax_frac.set_ylim([0, 1.2])

# 정보 표시
info_text_frac = f"""
정보 용량: 8 bits (2⁸ = 256)
범위: 0 ~ 1m
간격: 1m ÷ 255 = 3.92mm
역할: 소수부 (정밀도)

훈련 중: Sigmoid [0, 1] × 1m
배포 시: 0~255 (8-bit 유지)
"""
ax_frac.text(0.98, 0.97, info_text_frac, transform=ax_frac.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.2))

plt.tight_layout()
plt.savefig('/workspace/packnet-sfm/outputs/integer_vs_fractional_levels.png', dpi=150, bbox_inches='tight')
print("✅ 시각화 저장: outputs/integer_vs_fractional_levels.png")

# ============================================================================
# 추가 상세 시각화: Loss 구성 요소 분해
# ============================================================================

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Loss Function 구성 요소 상세 분석', fontsize=14, fontweight='bold')

# Ground truth와 예측 설정
gt_depths = np.array([0.5, 2.0, 5.0, 8.5, 12.0, 14.5])
pred_depths = gt_depths + np.array([-0.1, 0.05, -0.2, 0.15, 0.1, -0.05])

max_depth = 15.0

# Integer와 Fractional 분해
gt_int = (gt_depths / max_depth).astype(int)
gt_frac = gt_depths % 1.0

pred_int = (pred_depths / max_depth).astype(int)
pred_frac = pred_depths % 1.0

# ============================================================================
# 서브플롯 1: 깊이값별 Integer 손실
# ============================================================================
ax3_1 = axes3[0, 0]

int_losses = np.abs((pred_int / max_depth) - (gt_int / max_depth))
bars1 = ax3_1.bar(range(len(gt_depths)), int_losses, color='#FF6B6B', alpha=0.7, edgecolor='darkred')
ax3_1.set_xlabel('Test Depth Index', fontweight='bold')
ax3_1.set_ylabel('Integer Loss (L1)', fontweight='bold')
ax3_1.set_title('깊이값별 Integer Loss', fontweight='bold')
ax3_1.set_xticks(range(len(gt_depths)))
ax3_1.set_xticklabels([f'{d:.1f}m' for d in gt_depths], rotation=45)

# 값 레이블
for i, (bar, loss) in enumerate(zip(bars1, int_losses)):
    ax3_1.text(i, loss + 0.001, f'{loss:.4f}', ha='center', fontsize=8)

# ============================================================================
# 서브플롯 2: 깊이값별 Fractional 손실
# ============================================================================
ax3_2 = axes3[0, 1]

frac_losses = np.abs(pred_frac - gt_frac)
bars2 = ax3_2.bar(range(len(gt_depths)), frac_losses, color='#4ECDC4', alpha=0.7, edgecolor='darkblue')
ax3_2.set_xlabel('Test Depth Index', fontweight='bold')
ax3_2.set_ylabel('Fractional Loss (L1)', fontweight='bold')
ax3_2.set_title('깊이값별 Fractional Loss', fontweight='bold')
ax3_2.set_xticks(range(len(gt_depths)))
ax3_2.set_xticklabels([f'{d:.1f}m' for d in gt_depths], rotation=45)

# 값 레이블
for i, (bar, loss) in enumerate(zip(bars2, frac_losses)):
    ax3_2.text(i, loss + 0.01, f'{loss:.3f}', ha='center', fontsize=8)

# ============================================================================
# 서브플롯 3: 깊이값별 Consistency 손실
# ============================================================================
ax3_3 = axes3[1, 0]

cons_losses = np.abs(pred_depths - gt_depths)
bars3 = ax3_3.bar(range(len(gt_depths)), cons_losses, color='#95E1D3', alpha=0.7, edgecolor='darkgreen')
ax3_3.set_xlabel('Test Depth Index', fontweight='bold')
ax3_3.set_ylabel('Consistency Loss (L1)', fontweight='bold')
ax3_3.set_title('깊이값별 Consistency Loss (최종 복원 깊이)', fontweight='bold')
ax3_3.set_xticks(range(len(gt_depths)))
ax3_3.set_xticklabels([f'{d:.1f}m' for d in gt_depths], rotation=45)

# 값 레이블
for i, (bar, loss) in enumerate(zip(bars3, cons_losses)):
    ax3_3.text(i, loss + 0.01, f'{loss:.3f}', ha='center', fontsize=8)

# ============================================================================
# 서브플롯 4: 총 손실 (다양한 consistency_weight)
# ============================================================================
ax3_4 = axes3[1, 1]

total_loss_w0 = 1.0 * int_losses + 10.0 * frac_losses
total_loss_w05 = 1.0 * int_losses + 10.0 * frac_losses + 0.5 * cons_losses
total_loss_w1 = 1.0 * int_losses + 10.0 * frac_losses + 1.0 * cons_losses

x_pos = np.arange(len(gt_depths))
width = 0.25

ax3_4.bar(x_pos - width, total_loss_w0, width, label='w_cons=0.0', color='lightcoral', alpha=0.8)
ax3_4.bar(x_pos, total_loss_w05, width, label='w_cons=0.5 ★', color='lightgreen', alpha=0.8)
ax3_4.bar(x_pos + width, total_loss_w1, width, label='w_cons=1.0', color='lightyellow', alpha=0.8)

ax3_4.set_xlabel('Test Depth Index', fontweight='bold')
ax3_4.set_ylabel('Total Loss', fontweight='bold')
ax3_4.set_title('Consistency Weight별 총 손실', fontweight='bold')
ax3_4.set_xticks(x_pos)
ax3_4.set_xticklabels([f'{d:.1f}m' for d in gt_depths], rotation=45)
ax3_4.legend(loc='upper right')
ax3_4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/packnet-sfm/outputs/loss_components_analysis.png', dpi=150, bbox_inches='tight')
print("✅ 시각화 저장: outputs/loss_components_analysis.png")

print("\n" + "="*80)
print("📊 모든 시각화 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. consistency_weight_and_48_levels.png")
print("  2. integer_vs_fractional_levels.png")
print("  3. loss_components_analysis.png")

