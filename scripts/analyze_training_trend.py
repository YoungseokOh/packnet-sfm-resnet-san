#!/usr/bin/env python3
"""
학습 추이 분석 및 수렴값 예측
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 데이터 로드
results_dir = Path('checkpoints/resnetsan01_adaptive_multi_domain/default_config-train_resnet_san_ncdb_adaptive_loss-2025.10.22-07h46m55s/evaluation_results')

epochs = []
abs_rel = []
rmse = []
a1 = []

for epoch in range(14):  # 0-13
    file_path = results_dir / f'epoch_{epoch}_results.json'
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
            epochs.append(epoch)
            abs_rel.append(data['ncdb-cls-640x384-combined_val-abs_rel'])
            rmse.append(data['ncdb-cls-640x384-combined_val-rmse'])
            a1.append(data['ncdb-cls-640x384-combined_val-a1'])

epochs = np.array(epochs)
abs_rel = np.array(abs_rel)
rmse = np.array(rmse)
a1 = np.array(a1)

print("=" * 80)
print("📊 TRAINING TREND ANALYSIS (Epochs 0-13)")
print("=" * 80)

# abs_rel 분석
print("\n🎯 abs_rel (Lower is Better)")
print(f"   Epoch 0:  {abs_rel[0]:.6f}")
print(f"   Epoch 5:  {abs_rel[5]:.6f}  (Δ {abs_rel[5]-abs_rel[0]:+.6f}, {(abs_rel[5]-abs_rel[0])/abs_rel[0]*100:+.1f}%)")
print(f"   Epoch 10: {abs_rel[10]:.6f}  (Δ {abs_rel[10]-abs_rel[5]:+.6f}, {(abs_rel[10]-abs_rel[5])/abs_rel[5]*100:+.1f}%)")
print(f"   Epoch 13: {abs_rel[13]:.6f}  (Δ {abs_rel[13]-abs_rel[10]:+.6f}, {(abs_rel[13]-abs_rel[10])/abs_rel[10]*100:+.1f}%)")

# 최근 5개 epoch의 평균 개선율
recent_improvements = np.diff(abs_rel[-5:])
avg_improvement = np.mean(recent_improvements)
print(f"\n   📉 Recent trend (last 5 epochs): {avg_improvement:.6f} per epoch")

# 지수 감소 가정으로 수렴값 예측
# 최근 추세를 기반으로 30 epoch까지 외삽
remaining_epochs = 30 - 13
if avg_improvement < 0:  # 개선 중
    # 감소율이 점점 줄어든다고 가정 (지수 감쇠)
    decay_factor = 0.9  # 매 epoch마다 개선량이 10% 감소
    predicted_abs_rel = abs_rel[13]
    for i in range(remaining_epochs):
        improvement = avg_improvement * (decay_factor ** i)
        predicted_abs_rel += improvement
    
    print(f"   🔮 Predicted at Epoch 30: {predicted_abs_rel:.6f}")
    print(f"   📊 Expected improvement: {(predicted_abs_rel - abs_rel[13]):.6f} ({(predicted_abs_rel - abs_rel[13])/abs_rel[13]*100:+.1f}%)")
else:
    print(f"   ⚠️  No recent improvement detected")

# RMSE 분석
print("\n📏 RMSE (Lower is Better)")
print(f"   Epoch 0:  {rmse[0]:.4f}")
print(f"   Epoch 13: {rmse[13]:.4f}  (Δ {rmse[13]-rmse[0]:.4f}, {(rmse[13]-rmse[0])/rmse[0]*100:+.1f}%)")

recent_rmse_improvements = np.diff(rmse[-5:])
avg_rmse_improvement = np.mean(recent_rmse_improvements)
predicted_rmse = rmse[13]
for i in range(remaining_epochs):
    improvement = avg_rmse_improvement * (decay_factor ** i)
    predicted_rmse += improvement

print(f"   🔮 Predicted at Epoch 30: {predicted_rmse:.4f}")

# a1 분석
print("\n✅ a1 Accuracy (Higher is Better)")
print(f"   Epoch 0:  {a1[0]:.4f} ({a1[0]*100:.2f}%)")
print(f"   Epoch 13: {a1[13]:.4f} ({a1[13]*100:.2f}%)")

recent_a1_improvements = np.diff(a1[-5:])
avg_a1_improvement = np.mean(recent_a1_improvements)
predicted_a1 = min(a1[13] + avg_a1_improvement * remaining_epochs * 0.5, 0.999)  # cap at 99.9%

print(f"   🔮 Predicted at Epoch 30: {predicted_a1:.4f} ({predicted_a1*100:.2f}%)")

print("\n" + "=" * 80)
print("📈 CONVERGENCE ANALYSIS")
print("=" * 80)

# 변화율 계산 (최근 3개 epoch)
recent_changes = abs_rel[-3:] - abs_rel[-4:-1]
change_rate = np.mean(np.abs(recent_changes))

if change_rate < 0.001:
    print("✅ Model is converging (change rate < 0.001)")
    print(f"   Current change rate: {change_rate:.6f}")
elif change_rate < 0.003:
    print("⚠️  Model is still improving moderately")
    print(f"   Current change rate: {change_rate:.6f}")
else:
    print("🔄 Model is still in active training phase")
    print(f"   Current change rate: {change_rate:.6f}")

print("\n" + "=" * 80)
print("🎯 FINAL PREDICTIONS (Epoch 30)")
print("=" * 80)
print(f"abs_rel: {predicted_abs_rel:.6f} (current best: {min(abs_rel):.6f} at epoch {np.argmin(abs_rel)})")
print(f"rmse:    {predicted_rmse:.4f} (current best: {min(rmse):.4f} at epoch {np.argmin(rmse)})")
print(f"a1:      {predicted_a1:.4f} = {predicted_a1*100:.2f}% (current best: {max(a1):.4f} at epoch {np.argmax(a1)})")

print("\n📊 Comparison with targets:")
print(f"   Target abs_rel < 0.050: {'✅ ACHIEVED' if predicted_abs_rel < 0.050 else f'❌ Need {(0.050-predicted_abs_rel)*1000:.2f}ms improvement'}")
print(f"   Target a1 > 0.965:      {'✅ ACHIEVED' if predicted_a1 > 0.965 else f'❌ Need {(0.965-predicted_a1)*100:.2f}% improvement'}")

print("\n" + "=" * 80)
