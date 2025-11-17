#!/usr/bin/env python3
"""
거리별 NPU vs GT 메트릭 평가 표 생성
Distance ranges: 0.1-0.5m, 0.5-1m, 1-3m, 3-5m, 5m+
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import os


def load_gt_depth(gt_path):
    """Load GT depth from 16-bit PNG (value/256 = meters)"""
    gt_img = Image.open(gt_path)
    gt_depth = np.array(gt_img, dtype=np.float32) / 256.0
    return gt_depth


def load_npu_depth(int_path, frac_path, max_depth=15.0):
    """Load and compose NPU dual-head depth"""
    integer_sigmoid = np.load(int_path).squeeze()
    fractional_sigmoid = np.load(frac_path).squeeze()
    depth = integer_sigmoid * max_depth + fractional_sigmoid
    return depth


def compute_metrics(gt, pred, valid_mask=None):
    """
    Compute depth evaluation metrics
    Returns: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    if valid_mask is None:
        valid_mask = np.ones_like(gt, dtype=bool)
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    # Only consider positive depths
    mask = (gt_valid > 0) & (pred_valid > 0)
    gt_valid = gt_valid[mask]
    pred_valid = pred_valid[mask]
    
    if len(gt_valid) == 0:
        return None
    
    # Metrics
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    # Accuracy (delta < threshold)
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'count': len(gt_valid)
    }


def main():
    """메인 함수"""
    
    # 디렉토리 설정
    gt_dir = Path('Fin_Test_Set_ncdb/GT')
    npu_base_dir = Path('Fin_Test_Set_ncdb/npu/resnetsan_dual_head_seperate_static')
    npu_int_dir = npu_base_dir / 'integer_sigmoid'
    npu_frac_dir = npu_base_dir / 'fractional_sigmoid'
    output_dir = Path('outputs/fp32_vs_npu_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 거리별 범위 정의
    distance_ranges = [
        (0.5, 1.0, '0.5-1.0m'),
        (1.0, 3.0, '1.0-3.0m'),
        (3.0, 5.0, '3.0-5.0m'),
        (5.0, 15.0, '5.0m+')
    ]
    
    # GT 파일 목록
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    
    print(f"Processing {len(gt_files)} samples for distance-based metrics...")
    
    # 거리별 누적 메트릭
    distance_metrics = {
        range_name: {
            'abs_rel_list': [],
            'sq_rel_list': [],
            'rmse_list': [],
            'rmse_log_list': [],
            'a1_list': [],
            'a2_list': [],
            'a3_list': [],
            'count': 0
        }
        for _, _, range_name in distance_ranges
    }
    
    # 각 샘플 처리
    for idx, gt_file in enumerate(gt_files):
        base_name = gt_file.replace('.png', '')
        
        # 경로 구성
        gt_path = gt_dir / gt_file
        npu_int_path = npu_int_dir / f'{base_name}.npy'
        npu_frac_path = npu_frac_dir / f'{base_name}.npy'
        
        # 파일 확인
        if not (npu_int_path.exists() and npu_frac_path.exists()):
            print(f"⚠️  Skipping {base_name}: missing NPU files")
            continue
        
        try:
            # 데이터 로드
            gt_depth = load_gt_depth(gt_path)
            npu_depth = load_npu_depth(npu_int_path, npu_frac_path, max_depth=15.0)
            
            # 유효한 픽셀 (GT depth > 0)
            valid_mask = gt_depth > 0
            
            # 각 거리 범위별로 메트릭 계산
            for min_d, max_d, range_name in distance_ranges:
                # 거리 범위 마스크
                distance_mask = (gt_depth >= min_d) & (gt_depth < max_d) & valid_mask
                
                if np.sum(distance_mask) == 0:
                    continue
                
                # 메트릭 계산
                metrics = compute_metrics(gt_depth, npu_depth, distance_mask)
                
                if metrics is not None:
                    distance_metrics[range_name]['abs_rel_list'].append(metrics['abs_rel'])
                    distance_metrics[range_name]['sq_rel_list'].append(metrics['sq_rel'])
                    distance_metrics[range_name]['rmse_list'].append(metrics['rmse'])
                    distance_metrics[range_name]['rmse_log_list'].append(metrics['rmse_log'])
                    distance_metrics[range_name]['a1_list'].append(metrics['a1'])
                    distance_metrics[range_name]['a2_list'].append(metrics['a2'])
                    distance_metrics[range_name]['a3_list'].append(metrics['a3'])
                    distance_metrics[range_name]['count'] += metrics['count']
        
        except Exception as e:
            print(f"⚠️  Error processing {base_name}: {str(e)}")
            continue
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx+1}/{len(gt_files)} samples...")
    
    # 각 거리별 평균 메트릭 계산
    summary_metrics = {}
    for range_name in [r[2] for r in distance_ranges]:
        if distance_metrics[range_name]['abs_rel_list']:
            summary_metrics[range_name] = {
                'abs_rel': float(np.mean(distance_metrics[range_name]['abs_rel_list'])),
                'sq_rel': float(np.mean(distance_metrics[range_name]['sq_rel_list'])),
                'rmse': float(np.mean(distance_metrics[range_name]['rmse_list'])),
                'rmse_log': float(np.mean(distance_metrics[range_name]['rmse_log_list'])),
                'a1': float(np.mean(distance_metrics[range_name]['a1_list'])),
                'a2': float(np.mean(distance_metrics[range_name]['a2_list'])),
                'a3': float(np.mean(distance_metrics[range_name]['a3_list'])),
                'count': int(distance_metrics[range_name]['count'])
            }
    
    # 거리별 평가 테이블 생성
    create_distance_metrics_table(summary_metrics, distance_ranges, output_dir / "distance_based_metrics_table.png")
    
    # JSON 저장
    json_path = output_dir / "distance_based_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"✅ Distance-based metrics JSON saved to: {json_path}")
    
    print("\n✅ Distance-based metrics calculation complete!")


def create_distance_metrics_table(summary_metrics, distance_ranges, output_path):
    """
    거리별 메트릭을 테이블로 시각화
    """
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
    fig, ax = plt.subplots(1, 1, figsize=(24, 6))
    
    # 제목
    title_text = 'Distance-Based Metrics: NPU vs GT (NCDB 91 test images)'
    subtitle_text = 'Depth range: 0.5~15m | NO GT MEDIAN SCALING'
    fig.text(0.5, 0.92, title_text, ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.88, subtitle_text, ha='center', fontsize=11, style='italic')
    
    ax.axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    
    # 헤더
    header = ['Distance Range'] + [metric_name for _, metric_name, _ in metric_info]
    table_data.append(header)
    
    # 각 거리 범위별 행
    for _, _, range_name in distance_ranges:
        if range_name not in summary_metrics:
            continue
        
        row = [range_name]
        metrics = summary_metrics[range_name]
        
        for metric_key, _, fmt in metric_info:
            val = metrics[metric_key]
            row.append(fmt.format(val))
        
        table_data.append(row)
    
    # 컬럼 너비
    num_cols = len(header)
    col_widths = [0.15] + [0.85 / (num_cols - 1)] * (num_cols - 1)
    
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
            
            # Distance Range 열
            if j == 0:
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold', fontsize=11)
            else:
                # 메트릭별로 색상 구분 (거리별로는 같은 색 유지)
                cell.set_facecolor('#F0F8FF')  # Light blue
    
    # Note 추가
    note_text = 'Note: ↓ = lower is better, ↑ = higher is better\nMetrics computed on valid GT pixels within each distance range'
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, 
             style='italic', color='#555555')
    
    plt.subplots_adjust(top=0.85, bottom=0.10, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Distance-based metrics table saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
