#!/usr/bin/env python3
"""
Bounded Inverse vs Direct Depth NPU 비교 표 생성
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def parse_npu_eval_log(log_path):
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
        
        # Updated patterns to match new log format
        if 'WITHOUT GT MEDIAN SCALING:' in line or '🔹 NO GT SCALE:' in line or '📊 NO GT SCALE:' in line:
            in_no_scale = True
            in_with_scale = False
            continue
        elif 'WITH GT MEDIAN SCALING:' in line or '🔹 WITH GT SCALE:' in line or '📊 WITH GT SCALE:' in line:
            in_no_scale = False
            in_with_scale = True
            continue
        elif 'Total evaluated:' in line or ('========' in line and in_with_scale):
            if in_with_scale:  # WITH SCALE 섹션 끝
                break
            continue
        
        # 메트릭 파싱
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


def create_comparison_table(bounded_inv_metrics, direct_depth_metrics, output_path):
    """
    Bounded Inverse vs Direct Depth 비교 표 생성 (NO GT SCALE)
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
    
    # Figure 생성
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    fig.suptitle('NPU INT8: Bounded Inverse vs Direct Depth Comparison\n(91 images, 0.5~15m, NO GT SCALE)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    ax.axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    
    # 헤더
    header = ['Model'] + [metric_name for _, metric_name, _ in metric_info]
    table_data.append(header)
    
    # Bounded Inverse 행
    bounded_row = ['Bounded Inverse (Old)']
    for metric_key, _, fmt in metric_info:
        val = bounded_inv_metrics['no_scale'][metric_key]
        bounded_row.append(fmt.format(val))
    table_data.append(bounded_row)
    
    # Direct Depth 행
    direct_row = ['Direct Depth (NEW)']
    for metric_key, _, fmt in metric_info:
        val = direct_depth_metrics['no_scale'][metric_key]
        direct_row.append(fmt.format(val))
    table_data.append(direct_row)
    
    # Improvement 행
    improvement_row = ['Improvement']
    for metric_key, _, fmt in metric_info:
        bounded_val = bounded_inv_metrics['no_scale'][metric_key]
        direct_val = direct_depth_metrics['no_scale'][metric_key]
        diff = direct_val - bounded_val
        
        # 차이에 따라 색상 표시용 마커
        if metric_key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']:
            # Lower is better
            marker = '✓' if diff < 0 else '✗'
            pct_improvement = abs((bounded_val - direct_val) / bounded_val * 100)
            improvement_str = fmt.format(diff) + f' ({pct_improvement:.1f}% {marker})'
        else:
            # Higher is better (a1, a2, a3)
            marker = '✓' if diff > 0 else '✗'
            pct_improvement = abs((direct_val - bounded_val) / bounded_val * 100)
            improvement_str = fmt.format(diff) + f' (+{pct_improvement:.1f}% {marker})'
        
        improvement_row.append(improvement_str)
    table_data.append(improvement_row)
    
    # Ratio 행
    ratio_row = ['Ratio (Direct/Bounded)']
    for metric_key, _, fmt in metric_info:
        bounded_val = bounded_inv_metrics['no_scale'][metric_key]
        direct_val = direct_depth_metrics['no_scale'][metric_key]
        
        if bounded_val != 0:
            ratio = direct_val / bounded_val
            ratio_row.append(f'{ratio:.2f}x')
        else:
            ratio_row.append('∞')
    table_data.append(ratio_row)
    
    # 컬럼 너비
    num_cols = len(header)
    col_widths = [0.18] + [0.82 / (num_cols - 1)] * (num_cols - 1)
    
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
            
            # Model 열
            if j == 0:
                cell.set_facecolor('#D9D9D9')
                cell.set_text_props(weight='bold')
            else:
                # Bounded Inverse 행
                if i == 1:
                    cell.set_facecolor('#FFCCCC')  # Red tint (old method)
                # Direct Depth 행
                elif i == 2:
                    cell.set_facecolor('#CCFFCC')  # Green tint (new method)
                # Improvement 행
                elif i == 3:
                    text = cell.get_text().get_text()
                    if '✓' in text:
                        cell.set_facecolor('#90EE90')  # Light green for improvement
                    elif '✗' in text:
                        cell.set_facecolor('#FFB6C1')  # Light red for worse
                    else:
                        cell.set_facecolor('#FFFFFF')
                # Ratio 행
                elif i == 4:
                    cell.set_facecolor('#F2F2F2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison table saved to: {output_path}")
    plt.close()


def create_summary_json(bounded_inv_metrics, direct_depth_metrics, output_path):
    """
    JSON 요약 파일 생성
    """
    summary = {
        'bounded_inverse_npu_int8': bounded_inv_metrics,
        'direct_depth_npu_int8': direct_depth_metrics,
        'improvement': {
            'no_scale': {},
            'with_scale': {}
        }
    }
    
    for scale_type in ['no_scale', 'with_scale']:
        for key in bounded_inv_metrics[scale_type].keys():
            bounded_val = bounded_inv_metrics[scale_type][key]
            direct_val = direct_depth_metrics[scale_type][key]
            
            # Improvement percentage
            if key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']:
                # Lower is better
                improvement_pct = (bounded_val - direct_val) / bounded_val * 100
            else:
                # Higher is better
                improvement_pct = (direct_val - bounded_val) / bounded_val * 100
            
            summary['improvement'][scale_type][key] = {
                'bounded_inverse': bounded_val,
                'direct_depth': direct_val,
                'difference': direct_val - bounded_val,
                'ratio': direct_val / bounded_val if bounded_val != 0 else None,
                'improvement_percentage': improvement_pct
            }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary JSON saved to: {output_path}")


def main():
    # 입력 파일
    bounded_inv_log = 'outputs/npu_91_evaluation.log'
    direct_depth_log = 'outputs/npu_direct_depth_evaluation.log'
    
    # 출력 파일
    output_dir = Path('outputs/direct_depth_comparison')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    table_output = output_dir / 'bounded_inv_vs_direct_depth_table.png'
    json_output = output_dir / 'bounded_inv_vs_direct_depth_metrics.json'
    
    print("="*80)
    print("🚀 Bounded Inverse vs Direct Depth Comparison Table Generator")
    print("="*80)
    
    # Bounded Inverse 메트릭 파싱
    print(f"\n📊 Parsing Bounded Inverse NPU results from: {bounded_inv_log}")
    bounded_inv_metrics = parse_npu_eval_log(bounded_inv_log)
    print(f"   NO SCALE  → abs_rel: {bounded_inv_metrics['no_scale']['abs_rel']:.4f}, "
          f"rmse: {bounded_inv_metrics['no_scale']['rmse']:.3f}m, "
          f"δ<1.25: {bounded_inv_metrics['no_scale']['a1']:.4f}")
    
    # Direct Depth 메트릭 파싱
    print(f"\n📊 Parsing Direct Depth NPU results from: {direct_depth_log}")
    direct_depth_metrics = parse_npu_eval_log(direct_depth_log)
    print(f"   NO SCALE  → abs_rel: {direct_depth_metrics['no_scale']['abs_rel']:.4f}, "
          f"rmse: {direct_depth_metrics['no_scale']['rmse']:.3f}m, "
          f"δ<1.25: {direct_depth_metrics['no_scale']['a1']:.4f}")
    
    # Improvement 계산
    print(f"\n📈 Improvement Summary:")
    abs_rel_improvement = (bounded_inv_metrics['no_scale']['abs_rel'] - 
                           direct_depth_metrics['no_scale']['abs_rel']) / bounded_inv_metrics['no_scale']['abs_rel'] * 100
    rmse_improvement = (bounded_inv_metrics['no_scale']['rmse'] - 
                       direct_depth_metrics['no_scale']['rmse']) / bounded_inv_metrics['no_scale']['rmse'] * 100
    a1_improvement = (direct_depth_metrics['no_scale']['a1'] - 
                     bounded_inv_metrics['no_scale']['a1']) / bounded_inv_metrics['no_scale']['a1'] * 100
    
    print(f"   abs_rel:  {abs_rel_improvement:+.1f}%")
    print(f"   RMSE:     {rmse_improvement:+.1f}%")
    print(f"   δ<1.25:   {a1_improvement:+.1f}%")
    
    # JSON 저장
    print(f"\n💾 Saving JSON summary...")
    create_summary_json(bounded_inv_metrics, direct_depth_metrics, json_output)
    
    # 비교 표 생성
    print(f"\n🎨 Creating comparison table...")
    create_comparison_table(bounded_inv_metrics, direct_depth_metrics, table_output)
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == '__main__':
    main()
