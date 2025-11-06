#!/usr/bin/env python3
"""
PyTorch FP32 Direct Depth vs NPU INT8 Direct Depth 비교 표 생성
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_pytorch_metrics(json_path):
    """
    PyTorch FP32 평가 결과 JSON에서 메트릭 추출
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # GT scaling 적용 여부에 따라 메트릭 분리
    metrics = {
        'no_scale': {
            'abs_rel': data['ncdb-cls-640x384-combined_val-abs_rel_lin'],
            'sq_rel': data['ncdb-cls-640x384-combined_val-sqr_rel_lin'],
            'rmse': data['ncdb-cls-640x384-combined_val-rmse_lin'],
            'rmse_log': data['ncdb-cls-640x384-combined_val-rmse_log_lin'],
            'a1': data['ncdb-cls-640x384-combined_val-a1_lin'],
            'a2': data['ncdb-cls-640x384-combined_val-a2_lin'],
            'a3': data['ncdb-cls-640x384-combined_val-a3_lin']
        },
        'with_scale': {
            'abs_rel': data['ncdb-cls-640x384-combined_val-abs_rel_lin_gt'],
            'sq_rel': data['ncdb-cls-640x384-combined_val-sqr_rel_lin_gt'],
            'rmse': data['ncdb-cls-640x384-combined_val-rmse_lin_gt'],
            'rmse_log': data['ncdb-cls-640x384-combined_val-rmse_log_lin_gt'],
            'a1': data['ncdb-cls-640x384-combined_val-a1_lin_gt'],
            'a2': data['ncdb-cls-640x384-combined_val-a2_lin_gt'],
            'a3': data['ncdb-cls-640x384-combined_val-a3_lin_gt']
        }
    }
    
    return metrics


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
        
        if '🔹 NO GT SCALE:' in line or '📊 NO GT SCALE:' in line:
            in_no_scale = True
            in_with_scale = False
            continue
        elif '🔹 WITH GT SCALE:' in line or '📊 WITH GT SCALE:' in line:
            in_no_scale = False
            in_with_scale = True
            continue
        elif 'Total evaluated:' in line or '========' in line:
            if in_with_scale:
                break
            continue
        
        if not (in_no_scale or in_with_scale):
            continue
        
        # 메트릭 파싱
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


def create_comparison_table(pytorch_fp32, npu_int8, output_path):
    """
    PyTorch FP32 vs NPU INT8 비교 표 생성 (Direct Depth 모델)
    """
    metric_info = [
        ('abs_rel', 'Abs Rel', '{:.4f}'),
        ('sq_rel', 'Sq Rel', '{:.4f}'),
        ('rmse', 'RMSE (m)', '{:.3f}'),
        ('rmse_log', 'RMSE log', '{:.4f}'),
        ('a1', 'δ < 1.25', '{:.4f}'),
        ('a2', 'δ < 1.25²', '{:.4f}'),
        ('a3', 'δ < 1.25³', '{:.4f}')
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    fig.suptitle('Direct Depth Model: PyTorch FP32 vs NPU INT8 Comparison\n(91 images, 0.5~15m, NO GT SCALE)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    ax.axis('off')
    
    # 테이블 데이터
    table_data = []
    
    # 헤더
    header = ['Model'] + [metric_name for _, metric_name, _ in metric_info]
    table_data.append(header)
    
    # PyTorch FP32 행
    fp32_row = ['PyTorch FP32 (Direct Depth)']
    for metric_key, _, fmt in metric_info:
        val = pytorch_fp32['no_scale'][metric_key]
        fp32_row.append(fmt.format(val))
    table_data.append(fp32_row)
    
    # NPU INT8 행
    int8_row = ['NPU INT8 (Direct Depth)']
    for metric_key, _, fmt in metric_info:
        val = npu_int8['no_scale'][metric_key]
        int8_row.append(fmt.format(val))
    table_data.append(int8_row)
    
    # Degradation 행
    degradation_row = ['INT8 Degradation']
    for metric_key, _, fmt in metric_info:
        fp32_val = pytorch_fp32['no_scale'][metric_key]
        int8_val = npu_int8['no_scale'][metric_key]
        diff = int8_val - fp32_val
        
        if metric_key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']:
            # Lower is better - degradation if increased
            pct_change = abs(diff / fp32_val * 100)
            marker = '↑' if diff > 0 else '↓'
            degradation_str = fmt.format(diff) + f' ({pct_change:.1f}% {marker})'
        else:
            # Higher is better - degradation if decreased
            pct_change = abs(diff / fp32_val * 100)
            marker = '↓' if diff < 0 else '↑'
            degradation_str = fmt.format(diff) + f' ({pct_change:.1f}% {marker})'
        
        degradation_row.append(degradation_str)
    table_data.append(degradation_row)
    
    # Ratio 행
    ratio_row = ['Ratio (INT8/FP32)']
    for metric_key, _, fmt in metric_info:
        fp32_val = pytorch_fp32['no_scale'][metric_key]
        int8_val = npu_int8['no_scale'][metric_key]
        
        if fp32_val != 0:
            ratio = int8_val / fp32_val
            ratio_row.append(f'{ratio:.2f}x')
        else:
            ratio_row.append('∞')
    table_data.append(ratio_row)
    
    # 컬럼 너비
    num_cols = len(header)
    col_widths = [0.20] + [0.80 / (num_cols - 1)] * (num_cols - 1)
    
    # 테이블 그리기
    table = ax.table(cellText=table_data, 
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.0)
    
    # 헤더 스타일
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            
            if j == 0:
                cell.set_facecolor('#D9D9D9')
                cell.set_text_props(weight='bold')
            else:
                # PyTorch FP32 (Reference)
                if i == 1:
                    cell.set_facecolor('#CCE5FF')  # Light blue
                # NPU INT8
                elif i == 2:
                    cell.set_facecolor('#FFE5CC')  # Light orange
                # Degradation
                elif i == 3:
                    text = cell.get_text().get_text()
                    # 에러 메트릭이 증가 or 정확도 메트릭이 감소 → 나쁨
                    if ('↑' in text and j >= 1 and j <= 4) or ('↓' in text and j >= 5):
                        cell.set_facecolor('#FFB6C1')  # Light red (worse)
                    # 에러 메트릭이 감소 or 정확도 메트릭이 증가 → 좋음 (없을 것)
                    elif ('↓' in text and j >= 1 and j <= 4) or ('↑' in text and j >= 5):
                        cell.set_facecolor('#90EE90')  # Light green (better)
                    else:
                        cell.set_facecolor('#FFFFFF')
                # Ratio
                elif i == 4:
                    cell.set_facecolor('#F2F2F2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison table saved to: {output_path}")
    plt.close()


def create_summary_json(pytorch_fp32, npu_int8, output_path):
    """JSON 요약 파일"""
    summary = {
        'pytorch_fp32_direct_depth': pytorch_fp32,
        'npu_int8_direct_depth': npu_int8,
        'int8_degradation': {
            'no_scale': {},
            'with_scale': {}
        }
    }
    
    for scale_type in ['no_scale', 'with_scale']:
        for key in pytorch_fp32[scale_type].keys():
            fp32_val = pytorch_fp32[scale_type][key]
            int8_val = npu_int8[scale_type][key]
            
            if key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']:
                degradation_pct = (int8_val - fp32_val) / fp32_val * 100
            else:
                degradation_pct = (int8_val - fp32_val) / fp32_val * 100
            
            summary['int8_degradation'][scale_type][key] = {
                'pytorch_fp32': fp32_val,
                'npu_int8': int8_val,
                'difference': int8_val - fp32_val,
                'ratio': int8_val / fp32_val if fp32_val != 0 else None,
                'degradation_percentage': degradation_pct
            }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary JSON saved to: {output_path}")


def main():
    # 입력 파일
    pytorch_json = 'checkpoints/resnetsan01_direct_depth_05_15/default_config-train_resnet_san_ncdb_640x384_direct_depth-2025.11.05-09h12m03s/evaluation_results/epoch_29_results.json'
    npu_log = 'outputs/npu_direct_depth_evaluation.log'
    
    # 출력 파일
    output_dir = Path('outputs/direct_depth_fp32_vs_int8')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    table_output = output_dir / 'direct_depth_fp32_vs_int8_table.png'
    json_output = output_dir / 'direct_depth_fp32_vs_int8_metrics.json'
    
    print("="*80)
    print("🚀 Direct Depth: PyTorch FP32 vs NPU INT8 Comparison")
    print("="*80)
    
    # PyTorch FP32 메트릭 로드
    print(f"\n📊 Loading PyTorch FP32 results from: {pytorch_json}")
    pytorch_fp32 = load_pytorch_metrics(pytorch_json)
    print(f"   NO SCALE  → abs_rel: {pytorch_fp32['no_scale']['abs_rel']:.4f}, "
          f"rmse: {pytorch_fp32['no_scale']['rmse']:.3f}m, "
          f"δ<1.25: {pytorch_fp32['no_scale']['a1']:.4f}")
    print(f"   WITH SCALE → abs_rel: {pytorch_fp32['with_scale']['abs_rel']:.4f}, "
          f"rmse: {pytorch_fp32['with_scale']['rmse']:.3f}m, "
          f"δ<1.25: {pytorch_fp32['with_scale']['a1']:.4f}")
    
    # NPU INT8 메트릭 파싱
    print(f"\n📊 Parsing NPU INT8 results from: {npu_log}")
    npu_int8 = parse_npu_eval_log(npu_log)
    print(f"   NO SCALE  → abs_rel: {npu_int8['no_scale']['abs_rel']:.4f}, "
          f"rmse: {npu_int8['no_scale']['rmse']:.3f}m, "
          f"δ<1.25: {npu_int8['no_scale']['a1']:.4f}")
    print(f"   WITH SCALE → abs_rel: {npu_int8['with_scale']['abs_rel']:.4f}, "
          f"rmse: {npu_int8['with_scale']['rmse']:.3f}m, "
          f"δ<1.25: {npu_int8['with_scale']['a1']:.4f}")
    
    # INT8 Degradation 계산
    print(f"\n📉 INT8 Quantization Degradation (NO GT SCALE):")
    abs_rel_deg = (npu_int8['no_scale']['abs_rel'] - 
                   pytorch_fp32['no_scale']['abs_rel']) / pytorch_fp32['no_scale']['abs_rel'] * 100
    rmse_deg = (npu_int8['no_scale']['rmse'] - 
                pytorch_fp32['no_scale']['rmse']) / pytorch_fp32['no_scale']['rmse'] * 100
    a1_deg = (npu_int8['no_scale']['a1'] - 
              pytorch_fp32['no_scale']['a1']) / pytorch_fp32['no_scale']['a1'] * 100
    
    print(f"   abs_rel:  {abs_rel_deg:+.2f}%  (FP32: {pytorch_fp32['no_scale']['abs_rel']:.4f} → INT8: {npu_int8['no_scale']['abs_rel']:.4f})")
    print(f"   RMSE:     {rmse_deg:+.2f}%  (FP32: {pytorch_fp32['no_scale']['rmse']:.3f}m → INT8: {npu_int8['no_scale']['rmse']:.3f}m)")
    print(f"   δ<1.25:   {a1_deg:+.2f}%  (FP32: {pytorch_fp32['no_scale']['a1']:.4f} → INT8: {npu_int8['no_scale']['a1']:.4f})")
    
    print(f"\n💡 INT8 Quantization Impact:")
    print(f"   Theoretical error: ±28.4mm (uniform across all depths)")
    print(f"   Actual RMSE increase: {rmse_deg:.2f}%")
    print(f"   Conclusion: {'✅ Acceptable degradation' if abs(rmse_deg) < 100 else '⚠️ Significant degradation'}")
    
    # JSON 저장
    print(f"\n💾 Saving JSON summary...")
    create_summary_json(pytorch_fp32, npu_int8, json_output)
    
    # 비교 표 생성
    print(f"\n🎨 Creating comparison table...")
    create_comparison_table(pytorch_fp32, npu_int8, table_output)
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == '__main__':
    main()
