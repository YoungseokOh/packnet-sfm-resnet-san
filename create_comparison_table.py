#!/usr/bin/env python3
"""
PyTorch vs NPU 비교 표 생성
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def parse_eval_log(log_path):
    """
    공식 eval.py 로그 파일에서 메트릭 추출
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 테이블에서 메트릭 추출
    lines = content.split('\n')
    
    metrics = {}
    
    for i, line in enumerate(lines):
        if '| DEPTH ' in line and 'MAIN (BOUNDED INVERSE DEPTH)' in lines[i-2]:
            # NO GT SCALE 라인
            parts = line.split('|')
            if len(parts) >= 9:
                metrics['no_scale'] = {
                    'abs_rel': float(parts[2].strip()),
                    'sq_rel': float(parts[3].strip()),
                    'rmse': float(parts[4].strip()),
                    'rmse_log': float(parts[5].strip()),
                    'a1': float(parts[6].strip()),
                    'a2': float(parts[7].strip()),
                    'a3': float(parts[8].strip())
                }
        
        if '| DEPTH_GT ' in line:
            # WITH GT SCALE 라인
            parts = line.split('|')
            if len(parts) >= 9:
                metrics['with_scale'] = {
                    'abs_rel': float(parts[2].strip()),
                    'sq_rel': float(parts[3].strip()),
                    'rmse': float(parts[4].strip()),
                    'rmse_log': float(parts[5].strip()),
                    'a1': float(parts[6].strip()),
                    'a2': float(parts[7].strip()),
                    'a3': float(parts[8].strip())
                }
    
    return metrics


def parse_npu_log(log_path):
    """
    NPU 평가 로그에서 메트릭 추출
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    metrics = {'no_scale': {}, 'with_scale': {}}
    in_no_scale = False
    in_with_scale = False
    
    for line in lines:
        line = line.strip()
        
        if '📊 NO GT SCALE:' in line:
            in_no_scale = True
            in_with_scale = False
            continue
        elif '📊 WITH GT SCALE:' in line:
            in_no_scale = False
            in_with_scale = True
            continue
        elif 'Total evaluated:' in line or '========' in line:
            if in_with_scale:  # WITH SCALE 섹션 끝
                break
            continue
        
        # 메트릭 파싱 (공백 기준)
        if not (in_no_scale or in_with_scale):
            continue
        
        if line.startswith('abs_rel'):
            val = float(line.split(':')[1].strip())
            if in_no_scale:
                metrics['no_scale']['abs_rel'] = val
            elif in_with_scale:
                metrics['with_scale']['abs_rel'] = val
        elif line.startswith('sq_rel'):
            val = float(line.split(':')[1].strip())
            if in_no_scale:
                metrics['no_scale']['sq_rel'] = val
            elif in_with_scale:
                metrics['with_scale']['sq_rel'] = val
        elif line.startswith('rmse') and 'rmse_log' not in line:
            val = float(line.split(':')[1].strip().rstrip('m'))
            if in_no_scale:
                metrics['no_scale']['rmse'] = val
            elif in_with_scale:
                metrics['with_scale']['rmse'] = val
        elif line.startswith('rmse_log'):
            val = float(line.split(':')[1].strip())
            if in_no_scale:
                metrics['no_scale']['rmse_log'] = val
            elif in_with_scale:
                metrics['with_scale']['rmse_log'] = val
        elif line.startswith('a1'):
            val = float(line.split(':')[1].strip())
            if in_no_scale:
                metrics['no_scale']['a1'] = val
            elif in_with_scale:
                metrics['with_scale']['a1'] = val
        elif line.startswith('a2'):
            val = float(line.split(':')[1].strip())
            if in_no_scale:
                metrics['no_scale']['a2'] = val
            elif in_with_scale:
                metrics['with_scale']['a2'] = val
        elif line.startswith('a3'):
            val = float(line.split(':')[1].strip())
            if in_no_scale:
                metrics['no_scale']['a3'] = val
            elif in_with_scale:
                metrics['with_scale']['a3'] = val
    
    return metrics


def create_comparison_table(pytorch_metrics, npu_metrics, output_path):
    """
    PyTorch vs NPU 비교 표 생성 (행: GPU/NPU, 열: Metrics)
    """
    # 메트릭 이름과 표시 형식
    metric_info = [
        ('abs_rel', 'Abs Rel', '{:.4f}'),
        ('sq_rel', 'Sq Rel', '{:.4f}'),
        ('rmse', 'RMSE (m)', '{:.3f}'),
        ('rmse_log', 'RMSE log', '{:.4f}'),
        ('a1', 'δ < 1.25', '{:.4f}'),
        ('a2', 'δ < 1.25²', '{:.4f}'),
        ('a3', 'δ < 1.25³', '{:.4f}')
    ]
    
    # 단일 subplot (NO GT SCALE만)
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    fig.suptitle('PyTorch FP32 vs NPU INT8 Comparison (91 images, 0.5~15m)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    ax.axis('off')
    
    # 테이블 데이터 준비 (전치: 행=모델, 열=메트릭)
    table_data = []
    
    # 헤더: Model + 각 메트릭
    header = ['Model'] + [metric_name for _, metric_name, _ in metric_info]
    table_data.append(header)
    
    # PyTorch 행
    pt_row = ['PyTorch FP32']
    for metric_key, _, fmt in metric_info:
        pt_val = pytorch_metrics['no_scale'][metric_key]
        pt_row.append(fmt.format(pt_val))
    table_data.append(pt_row)
    
    # NPU 행
    npu_row = ['NPU INT8']
    for metric_key, _, fmt in metric_info:
        npu_val = npu_metrics['no_scale'][metric_key]
        npu_row.append(fmt.format(npu_val))
    table_data.append(npu_row)
    
    # Difference 행
    diff_row = ['Difference']
    for metric_key, _, fmt in metric_info:
        pt_val = pytorch_metrics['no_scale'][metric_key]
        npu_val = npu_metrics['no_scale'][metric_key]
        diff = npu_val - pt_val
        
        # 차이에 따라 색상 표시용 마커
        if metric_key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']:
            # Lower is better
            marker = '↓' if diff < 0 else '↑'
            color_marker = '🟢' if diff < 0 else '🔴'
        else:
            # Higher is better (a1, a2, a3)
            marker = '↑' if diff > 0 else '↓'
            color_marker = '🟢' if diff > 0 else '🔴'
        
        diff_str = fmt.format(diff) + f' {marker}'
        diff_row.append(diff_str)
    table_data.append(diff_row)
    
    # Ratio 행
    ratio_row = ['Ratio (NPU/PT)']
    for metric_key, _, fmt in metric_info:
        pt_val = pytorch_metrics['no_scale'][metric_key]
        npu_val = npu_metrics['no_scale'][metric_key]
        
        if pt_val != 0:
            ratio = npu_val / pt_val
            ratio_row.append(f'{ratio:.2f}x')
        else:
            ratio_row.append('∞')
    table_data.append(ratio_row)
    
    # 컬럼 너비 계산
    num_cols = len(header)
    col_widths = [0.15] + [0.85 / (num_cols - 1)] * (num_cols - 1)
    
    # 테이블 그리기
    table = ax.table(cellText=table_data, 
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.0)
    
    # 헤더 행 스타일
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            
            # Model 열 (첫 번째 열)
            if j == 0:
                cell.set_facecolor('#D9D9D9')
                cell.set_text_props(weight='bold')
            else:
                # PyTorch 행
                if i == 1:
                    cell.set_facecolor('#D9E2F3')
                # NPU 행
                elif i == 2:
                    cell.set_facecolor('#FCE4D6')
                # Difference 행
                elif i == 3:
                    text = cell.get_text().get_text()
                    if '↑' in text and j >= 1 and j <= 4:  # Error metrics
                        cell.set_facecolor('#FFCCCC')  # Red for worse
                    elif '↓' in text and j >= 1 and j <= 4:
                        cell.set_facecolor('#CCFFCC')  # Green for better
                    elif '↑' in text and j >= 5:  # Accuracy metrics
                        cell.set_facecolor('#CCFFCC')  # Green for better
                    elif '↓' in text and j >= 5:
                        cell.set_facecolor('#FFCCCC')  # Red for worse
                    else:
                        cell.set_facecolor('#FFFFFF')
                # Ratio 행
                elif i == 4:
                    cell.set_facecolor('#F2F2F2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison table saved to: {output_path}")
    plt.close()


def create_summary_json(pytorch_metrics, npu_metrics, output_path):
    """
    JSON 요약 파일 생성
    """
    summary = {
        'pytorch_fp32': pytorch_metrics,
        'npu_int8': npu_metrics,
        'comparison': {
            'no_scale': {},
            'with_scale': {}
        }
    }
    
    for scale_type in ['no_scale', 'with_scale']:
        for key in pytorch_metrics[scale_type].keys():
            pt_val = pytorch_metrics[scale_type][key]
            npu_val = npu_metrics[scale_type][key]
            
            summary['comparison'][scale_type][key] = {
                'pytorch': pt_val,
                'npu': npu_val,
                'difference': npu_val - pt_val,
                'ratio': npu_val / pt_val if pt_val != 0 else None
            }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary JSON saved to: {output_path}")


def main():
    # 입력 파일
    pytorch_log = 'outputs/eval_official.log'
    npu_log = 'outputs/npu_91_evaluation.log'
    
    # 출력 파일
    output_dir = Path('outputs/comparison')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    table_output = output_dir / 'pytorch_vs_npu_comparison_table.png'
    json_output = output_dir / 'pytorch_vs_npu_metrics.json'
    
    print("="*80)
    print("PyTorch vs NPU Comparison Table Generator")
    print("="*80)
    
    # PyTorch 메트릭 파싱
    print(f"\n📊 Parsing PyTorch results from: {pytorch_log}")
    pytorch_metrics = parse_eval_log(pytorch_log)
    print(f"   NO SCALE  → abs_rel: {pytorch_metrics['no_scale']['abs_rel']:.4f}, "
          f"δ<1.25: {pytorch_metrics['no_scale']['a1']:.4f}")
    print(f"   WITH SCALE → abs_rel: {pytorch_metrics['with_scale']['abs_rel']:.4f}, "
          f"δ<1.25: {pytorch_metrics['with_scale']['a1']:.4f}")
    
    # NPU 메트릭 파싱
    print(f"\n📊 Parsing NPU results from: {npu_log}")
    npu_metrics = parse_npu_log(npu_log)
    print(f"   NO SCALE  → abs_rel: {npu_metrics['no_scale']['abs_rel']:.4f}, "
          f"δ<1.25: {npu_metrics['no_scale']['a1']:.4f}")
    print(f"   WITH SCALE → abs_rel: {npu_metrics['with_scale']['abs_rel']:.4f}, "
          f"δ<1.25: {npu_metrics['with_scale']['a1']:.4f}")
    
    # JSON 저장
    print(f"\n💾 Saving JSON summary...")
    create_summary_json(pytorch_metrics, npu_metrics, json_output)
    
    # 비교 표 생성
    print(f"\n🎨 Creating comparison table...")
    create_comparison_table(pytorch_metrics, npu_metrics, table_output)
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == '__main__':
    main()
