#!/usr/bin/env python3
"""
Direct Depth: PyTorch FP32 vs NPU INT8 비교 표 생성
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def create_fp32_vs_int8_table(fp32_metrics, int8_metrics, output_path):
    """
    PyTorch FP32 vs NPU INT8 비교 표 생성
    """
    # 메트릭 이름과 표시 형식
    metric_info = [
        ('abs_rel', 'Abs Rel', '{:.4f}', 'lower'),
        ('sq_rel', 'Sq Rel', '{:.4f}', 'lower'),
        ('rmse', 'RMSE (m)', '{:.4f}', 'lower'),
        ('rmse_log', 'RMSE log', '{:.4f}', 'lower'),
        ('a1', 'δ < 1.25', '{:.4f}', 'higher'),
        ('a2', 'δ < 1.25²', '{:.4f}', 'higher'),
        ('a3', 'δ < 1.25³', '{:.4f}', 'higher')
    ]
    
    # Figure 생성
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    fig.suptitle('Direct Depth Model: PyTorch FP32 vs NPU INT8 Comparison\n(91 test images, depth range: 0.5~15m, NO GT MEDIAN SCALING)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    ax.axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    
    # 헤더
    header = ['Model'] + [metric_name for _, metric_name, _, _ in metric_info]
    table_data.append(header)
    
    # PyTorch FP32 행
    fp32_row = ['PyTorch FP32 (Baseline)']
    for metric_key, _, fmt, _ in metric_info:
        val = fp32_metrics[metric_key]
        fp32_row.append(fmt.format(val))
    table_data.append(fp32_row)
    
    # NPU INT8 행
    int8_row = ['NPU INT8 (Quantized)']
    for metric_key, _, fmt, _ in metric_info:
        val = int8_metrics[metric_key]
        int8_row.append(fmt.format(val))
    table_data.append(int8_row)
    
    # Degradation 행
    degradation_row = ['Degradation (INT8 vs FP32)']
    for metric_key, _, fmt, better in metric_info:
        fp32_val = fp32_metrics[metric_key]
        int8_val = int8_metrics[metric_key]
        diff = int8_val - fp32_val
        
        # 차이 계산
        if better == 'lower':
            # Lower is better (abs_rel, rmse, etc.)
            pct_change = (int8_val - fp32_val) / fp32_val * 100
            marker = '✓' if diff < 0 else '✗'
            degradation_str = fmt.format(diff) + f' ({pct_change:+.1f}% {marker})'
        else:
            # Higher is better (a1, a2, a3)
            pct_change = (int8_val - fp32_val) / fp32_val * 100
            marker = '✓' if diff > 0 else '✗'
            degradation_str = fmt.format(diff) + f' ({pct_change:+.1f}% {marker})'
        
        degradation_row.append(degradation_str)
    table_data.append(degradation_row)
    
    # Ratio 행
    ratio_row = ['Ratio (INT8/FP32)']
    for metric_key, _, fmt, better in metric_info:
        fp32_val = fp32_metrics[metric_key]
        int8_val = int8_metrics[metric_key]
        
        if fp32_val != 0:
            ratio = int8_val / fp32_val
            
            # 색상 코딩을 위한 마커
            if better == 'lower':
                # Lower is better: ratio > 1 is bad
                marker = '✓' if ratio < 1.0 else '✗'
            else:
                # Higher is better: ratio < 1 is bad
                marker = '✓' if ratio > 1.0 else '✗'
            
            ratio_row.append(f'{ratio:.3f}x {marker}')
        else:
            ratio_row.append('∞')
    table_data.append(ratio_row)
    
    # 컬럼 너비
    num_cols = len(header)
    col_widths = [0.22] + [0.78 / (num_cols - 1)] * (num_cols - 1)
    
    # 테이블 그리기
    table = ax.table(cellText=table_data, 
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.2)
    
    # 헤더 행 스타일
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            
            # Model 열
            if j == 0:
                cell.set_facecolor('#D9D9D9')
                cell.set_text_props(weight='bold')
            else:
                # PyTorch FP32 행 (baseline)
                if i == 1:
                    cell.set_facecolor('#CCE5FF')  # Light blue (baseline)
                # NPU INT8 행
                elif i == 2:
                    cell.set_facecolor('#FFE5CC')  # Light orange (quantized)
                # Degradation 행
                elif i == 3:
                    text = cell.get_text().get_text()
                    if '✓' in text:
                        cell.set_facecolor('#90EE90')  # Light green for improvement
                    elif '✗' in text:
                        cell.set_facecolor('#FFB6C1')  # Light red for degradation
                    else:
                        cell.set_facecolor('#FFFFFF')
                # Ratio 행
                elif i == 4:
                    text = cell.get_text().get_text()
                    if '✓' in text:
                        cell.set_facecolor('#E0FFE0')  # Very light green
                    elif '✗' in text:
                        cell.set_facecolor('#FFE0E0')  # Very light red
                    else:
                        cell.set_facecolor('#F2F2F2')
    
    # 범례 추가
    legend_elements = [
        mpatches.Patch(facecolor='#CCE5FF', label='PyTorch FP32 (Baseline)'),
        mpatches.Patch(facecolor='#FFE5CC', label='NPU INT8 (Quantized)'),
        mpatches.Patch(facecolor='#90EE90', label='✓ Better/Equal'),
        mpatches.Patch(facecolor='#FFB6C1', label='✗ Worse (Degradation)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison table saved to: {output_path}")
    plt.close()


def create_summary_json(fp32_metrics, int8_metrics, output_path):
    """
    JSON 요약 파일 생성
    """
    summary = {
        'model': 'Direct Depth (ResNetSAN01)',
        'checkpoint': 'epoch=29_ncdb-cls-640x384-combined_val-loss=0.000.ckpt',
        'test_set': 'combined_test.json (91 images)',
        'depth_range': '0.5~15.0m',
        'pytorch_fp32': fp32_metrics,
        'npu_int8': int8_metrics,
        'degradation': {}
    }
    
    metric_info = [
        ('abs_rel', 'lower'),
        ('sq_rel', 'lower'),
        ('rmse', 'lower'),
        ('rmse_log', 'lower'),
        ('a1', 'higher'),
        ('a2', 'higher'),
        ('a3', 'higher')
    ]
    
    for key, better in metric_info:
        fp32_val = fp32_metrics[key]
        int8_val = int8_metrics[key]
        
        # Degradation percentage
        if better == 'lower':
            # Lower is better: positive % means worse
            degradation_pct = (int8_val - fp32_val) / fp32_val * 100
        else:
            # Higher is better: negative % means worse
            degradation_pct = (int8_val - fp32_val) / fp32_val * 100
        
        summary['degradation'][key] = {
            'fp32': fp32_val,
            'int8': int8_val,
            'difference': int8_val - fp32_val,
            'ratio': int8_val / fp32_val if fp32_val != 0 else None,
            'degradation_percentage': degradation_pct
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary JSON saved to: {output_path}")


def main():
    # 입력: 수동으로 입력된 메트릭 (검증된 결과)
    fp32_metrics = {
        'abs_rel': 0.043371,
        'sq_rel': 0.034174,
        'rmse': 0.390933,
        'rmse_log': 0.084644,
        'a1': 0.975855,
        'a2': 0.992422,
        'a3': 0.996822
    }
    
    int8_metrics = {
        'abs_rel': 0.113251,
        'sq_rel': 0.254719,
        'rmse': 0.740760,
        'rmse_log': 0.163539,
        'a1': 0.923948,
        'a2': 0.957999,
        'a3': 0.974594
    }
    
    # 출력 파일
    output_dir = Path('outputs/direct_depth_comparison')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    table_output = output_dir / 'fp32_vs_int8_comparison_table.png'
    json_output = output_dir / 'fp32_vs_int8_comparison_metrics.json'
    
    print("="*80)
    print("🚀 Direct Depth: PyTorch FP32 vs NPU INT8 Comparison Table Generator")
    print("="*80)
    
    # 메트릭 출력
    print(f"\n📊 PyTorch FP32 (Baseline):")
    print(f"   abs_rel:  {fp32_metrics['abs_rel']:.4f}")
    print(f"   RMSE:     {fp32_metrics['rmse']:.4f}m")
    print(f"   δ < 1.25: {fp32_metrics['a1']:.4f}")
    
    print(f"\n📊 NPU INT8 (Quantized):")
    print(f"   abs_rel:  {int8_metrics['abs_rel']:.4f}")
    print(f"   RMSE:     {int8_metrics['rmse']:.4f}m")
    print(f"   δ < 1.25: {int8_metrics['a1']:.4f}")
    
    # Degradation 계산
    print(f"\n📈 Degradation Summary:")
    abs_rel_degradation = (int8_metrics['abs_rel'] - fp32_metrics['abs_rel']) / fp32_metrics['abs_rel'] * 100
    rmse_degradation = (int8_metrics['rmse'] - fp32_metrics['rmse']) / fp32_metrics['rmse'] * 100
    a1_degradation = (int8_metrics['a1'] - fp32_metrics['a1']) / fp32_metrics['a1'] * 100
    
    print(f"   abs_rel:  {abs_rel_degradation:+.1f}% (ratio: {int8_metrics['abs_rel']/fp32_metrics['abs_rel']:.2f}x)")
    print(f"   RMSE:     {rmse_degradation:+.1f}% (ratio: {int8_metrics['rmse']/fp32_metrics['rmse']:.2f}x)")
    print(f"   δ < 1.25: {a1_degradation:+.1f}% (ratio: {int8_metrics['a1']/fp32_metrics['a1']:.3f}x)")
    
    # JSON 저장
    print(f"\n💾 Saving JSON summary...")
    create_summary_json(fp32_metrics, int8_metrics, json_output)
    
    # 비교 표 생성
    print(f"\n🎨 Creating comparison table...")
    create_fp32_vs_int8_table(fp32_metrics, int8_metrics, table_output)
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == '__main__':
    main()
