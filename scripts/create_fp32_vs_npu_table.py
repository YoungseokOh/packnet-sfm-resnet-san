#!/usr/bin/env python3
"""
Dual-Head Model: PyTorch FP32 vs NPU 비교 표 생성
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def create_fp32_vs_npu_table(fp32_metrics, npu_metrics, output_path):
    """
    Dual-Head: PyTorch FP32 vs NPU 비교 표 생성
    """
    # 메트릭 이름과 표시 형식
    metric_info = [
        ('abs_rel', 'Abs Rel ↓', '{:.4f}'),
        ('sq_rel', 'Sq Rel ↓', '{:.4f}'),
        ('rmse', 'RMSE (m) ↓', '{:.3f}'),
        ('rmse_log', 'RMSE log ↓', '{:.4f}'),
        ('a1', 'δ < 1.25 ↑', '{:.4f}'),
        ('a2', 'δ < 1.25² ↑', '{:.4f}'),
        ('a3', 'δ < 1.25³ ↑', '{:.4f}')
    ]
    
    # Figure 생성
    fig, ax = plt.subplots(1, 1, figsize=(22, 4))  # 충분한 높이 확보
    
    # 제목
    title_text = 'Dual-Head Model: PyTorch FP32 vs NPU Comparison'
    subtitle_text = 'NCDB test set: 91 images | Depth range: 0.5~15m | NO GT MEDIAN SCALING'
    fig.text(0.5, 0.92, title_text, ha='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.86, subtitle_text, ha='center', fontsize=13, style='italic')
    
    ax.axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    
    # 헤더
    header = ['Model'] + [metric_name for _, metric_name, _ in metric_info]
    table_data.append(header)
    
    # PyTorch FP32 행
    fp32_row = ['PyTorch FP32']
    for metric_key, _, fmt in metric_info:
        val = fp32_metrics[metric_key]
        fp32_row.append(fmt.format(val))
    table_data.append(fp32_row)
    
    # NPU 행
    npu_row = ['NPU']
    for metric_key, _, fmt in metric_info:
        val = npu_metrics[metric_key]
        npu_row.append(fmt.format(val))
    table_data.append(npu_row)
    
    # Delta 행 (NPU - FP32)
    delta_row = ['Δ (NPU - FP32)']
    for i, (metric_key, metric_name, fmt) in enumerate(metric_info):
        fp32_val = fp32_metrics[metric_key]
        npu_val = npu_metrics[metric_key]
        diff = npu_val - fp32_val
        
        delta_str = fmt.format(diff)
        delta_row.append(delta_str)
    table_data.append(delta_row)
    
    # 컬럼 너비
    num_cols = len(header)
    col_widths = [0.18] + [0.82 / (num_cols - 1)] * (num_cols - 1)
    
    # 테이블 그리기
    table = ax.table(cellText=table_data, 
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3.5)
    
    # 헤더 행 스타일
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E5090')  # Dark blue
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # 데이터 행 스타일
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            
            # Model 열
            if j == 0:
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold', fontsize=11)
            else:
                # PyTorch FP32 행
                if i == 1:
                    cell.set_facecolor('#D6E9F8')  # Light blue
                # NPU 행
                elif i == 2:
                    cell.set_facecolor('#FFF4CC')  # Light yellow
                # Delta 행 (색상 코딩)
                elif i == 3:
                    text = cell.get_text().get_text()
                    
                    # Extract metric name from header
                    metric_name = header[j]
                    is_lower_better = '↓' in metric_name
                    
                    # Parse the value
                    try:
                        value_str = text.split()[0]  # Get first part before (
                        value = float(value_str)
                        
                        # Color coding based on direction
                        if is_lower_better:
                            # Lower is better: negative diff is good
                            if value < 0:
                                cell.set_facecolor('#C6EFCE')  # Green
                            else:
                                cell.set_facecolor('#FFC7CE')  # Red
                        else:
                            # Higher is better: positive diff is good
                            if value > 0:
                                cell.set_facecolor('#C6EFCE')  # Green
                            else:
                                cell.set_facecolor('#FFC7CE')  # Red
                    except:
                        cell.set_facecolor('#F5F5F5')
    
    # 범례를 표 아래에
    legend_elements = [
        mpatches.Patch(facecolor='#D6E9F8', label='PyTorch FP32 (Baseline)'),
        mpatches.Patch(facecolor='#FFF4CC', label='NPU (Dual-Head)'),
        mpatches.Patch(facecolor='#C6EFCE', label='Better Performance'),
        mpatches.Patch(facecolor='#FFC7CE', label='Worse Performance')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, 0.12), ncol=4, frameon=False, fontsize=10)
    
    # Note를 범례 아래에
    note_text = 'Note: ↓ = lower is better, ↑ = higher is better\nDual-Head: Integer head (0-15m) + Fractional head (0-1m) for enhanced precision'
    fig.text(0.5, 0.01, note_text, ha='center', fontsize=8, 
             style='italic', color='#555555')
    
    # 간격 조정
    plt.subplots_adjust(top=0.85, bottom=0.12, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison table saved to: {output_path}")
    plt.close()


def main():
    """메인 함수"""
    # FP32 metrics (from pytorch_fp32_official_pipeline)
    fp32_metrics = {
        'abs_rel': 0.041400480829596796,
        'sq_rel': 0.15042918384699908,
        'rmse': 1.568,
        'rmse_log': 0.07028696685116632,
        'a1': 0.9791706994968653,
        'a2': 0.9952664315350087,
        'a3': 0.9983699866415188
    }
    
    # NPU metrics (from resnetsan01_dual_head_ncdb_640x384)
    npu_metrics = {
        'abs_rel': 0.05797538568607849,
        'sq_rel': 0.23651698990910732,
        'rmse': 1.827,
        'rmse_log': 0.09175873552719252,
        'a1': 0.9629890362161595,
        'a2': 0.9906451221388877,
        'a3': 0.9969149843886747
    }
    
    # Output 디렉토리 생성
    output_dir = Path("outputs/fp32_vs_npu_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 테이블 생성
    output_path = output_dir / "comparison_table_final.png"
    create_fp32_vs_npu_table(fp32_metrics, npu_metrics, output_path)
    
    # JSON 요약도 저장
    summary = {
        'model': 'Dual-Head (Integer + Fractional)',
        'test_set': 'NCDB combined_test.json (91 images)',
        'depth_range': '0.5~15.0m',
        'pytorch_fp32': fp32_metrics,
        'npu': npu_metrics,
        'delta': {k: npu_metrics[k] - fp32_metrics[k] for k in fp32_metrics.keys()}
    }
    
    json_path = output_dir / "comparison_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary JSON saved to: {json_path}")


if __name__ == '__main__':
    main()
