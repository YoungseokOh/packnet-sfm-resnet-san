#!/usr/bin/env python3
"""
깊이 추정 메트릭 시각화 스크립트
CSV 파일을 읽어 matplotlib으로 종합적인 대시보드를 생성합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
import os

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_data(metrics_path='outputs/ResNet-SAN_0.05to100_results/metrics'):
    """CSV 파일 로드"""
    summary_csv = os.path.join(metrics_path, 'summary.csv')
    distance_csv = os.path.join(metrics_path, 'summary_by_distance.csv')
    
    if not os.path.exists(summary_csv) or not os.path.exists(distance_csv):
        print(f"❌ Error: CSV files not found in {metrics_path}")
        print(f"   Looking for: {summary_csv}")
        print(f"   Looking for: {distance_csv}")
        sys.exit(1)
    
    summary_df = pd.read_csv(summary_csv)
    distance_df = pd.read_csv(distance_csv)
    return summary_df, distance_df

def create_distance_table_subplot(ax, distance_df, class_name, color):
    """서브플롯에 거리별 메트릭 표 생성"""
    ax.axis('tight')
    ax.axis('off')
    
    data = distance_df[distance_df['Class'] == class_name].copy()
    
    # 표 데이터 준비
    table_data = []
    table_data.append(['Range', 'Pixels', 'abs_rel', 'rmse', 'rmse_log', 'a1 (%)'])
    
    for _, row in data.iterrows():
        table_data.append([
            row['Range'],
            f"{int(row['Pixels']):,}",
            f"{row['abs_rel']:.4f}",
            f"{row['rmse']:.3f}",
            f"{row['rmse_log']:.4f}",
            f"{row['a1']*100:.2f}"
        ])
    
    # 표 생성 - 너비 확장
    table = ax.table(cellText=table_data, 
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.16, 0.16, 0.17, 0.14, 0.17, 0.17])  # 전체적으로 균등하게 확장
    
    # 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.4)  # 표 높이를 낮춰 제목 공간 확보
    
    # 헤더 스타일 - 크게
    for j in range(len(table_data[0])):
        cell = table[(0, j)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', fontsize=12)  # 11 -> 12 (헤더 더 크게)
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
    
    # 데이터 행 스타일 - 모두 연한 회색으로 통일
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor('#F5F5F5')  # 연한 회색으로 통일
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(0.5)
            
            # 데이터 행 폰트 - 헤더보다 작게
            if j >= 2:
                cell.set_text_props(weight='bold', fontsize=9)  # 10 -> 9 (메트릭 값)
            else:
                cell.set_text_props(fontsize=9)  # Range, Pixels도 작게
    
    # 제목
    ax.text(
        0.5,
        1.10,
        f'{class_name.upper()} - Performance by Distance Range',
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold',
        color='black'
    )

def create_dashboard(summary_df, distance_df, output_path='outputs/ResNet-SAN_0.05to100_results/metrics'):
    """종합 대시보드 생성
    
    Args:
        summary_df: 전체 요약 데이터프레임
        distance_df: 거리별 메트릭 데이터프레임
        output_path: 대시보드 저장 경로 (기본값: outputs/ResNet-SAN_0.05to100_results/metrics)
    """
    
    # 출력 경로 생성
    os.makedirs(output_path, exist_ok=True)
    
    # Figure 생성 (충분한 세로 공간 확보)
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('Depth Estimation Performance Dashboard - ResNet-SAN (0.05-100m)', 
                 fontsize=20, fontweight='bold', y=0.999)
    
    # GridSpec으로 레이아웃 구성 (6행 4열)
    gs = GridSpec(6, 4, figure=fig, hspace=0.55, wspace=0.40, 
                  left=0.06, right=0.94, top=0.96, bottom=0.04)
    
    # 색상 팔레트
    colors = {
        'car': '#FF6B6B',        # 빨강
        'road': '#FFE66D',       # 연노란색
        'car+road': '#FFB380',   # 주황색 (빨강 + 노란색 혼합)
        'ALL': '#95E1D3'         # 청록색
    }
    
    # ========== 1. 상단: 전체 요약 메트릭 (표 형식) ==========
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    # 요약 표 데이터 준비
    summary_table_data = []
    summary_table_data.append(['Class', 'Count', 'abs_rel', 'rmse', 'rmse_log', 'a1 (%)', 'a2 (%)', 'a3 (%)'])
    
    for _, row in summary_df.iterrows():
        summary_table_data.append([
            row['Class'],
            f"{int(row['Count']):,}",
            f"{row['abs_rel']:.4f}",
            f"{row['rmse']:.3f}",
            f"{row['rmse_log']:.4f}",
            f"{row['a1']*100:.2f}",
            f"{row['a2']*100:.2f}",
            f"{row['a3']*100:.2f}"
        ])
    
    # 표 생성 (가운데 정렬)
    table = ax_summary.table(cellText=summary_table_data,
                            loc='center',
                            cellLoc='center',
                            colWidths=[0.12, 0.13, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    
    # 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3.0)
    
    # 헤더 스타일
    for j in range(len(summary_table_data[0])):
        cell = table[(0, j)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # 데이터 행 스타일 - 연한 회색으로 통일
    for i in range(1, len(summary_table_data)):
        for j in range(len(summary_table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor('#F5F5F5')  # 연한 회색으로 통일
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(0.5)
            
            # Class 열과 메트릭 열 강조
            if j == 0:  # Class 열
                cell.set_text_props(weight='bold', fontsize=13)
            elif j >= 2:  # 메트릭 열
                cell.set_text_props(weight='bold', fontsize=12)
    
    # 제목
    ax_summary.set_title('Overall Performance Summary', fontsize=16, fontweight='bold', 
                        pad=20, color='black')
    
    # ========== 2. 좌측: 클래스별 abs_rel 비교 (막대 그래프) ==========
    ax1 = fig.add_subplot(gs[1, 0:2])
    classes = summary_df[summary_df['Class'] != 'ALL']['Class'].values
    abs_rels = summary_df[summary_df['Class'] != 'ALL']['abs_rel'].values
    
    bars = ax1.barh(classes, abs_rels, color=[colors[c] for c in classes], 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Absolute Relative Error (abs_rel)', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Relative Error by Class', fontsize=14, fontweight='bold', pad=10, color='black')
    ax1.set_xlim(0, max(abs_rels) * 1.2)
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, abs_rels)):
        ax1.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    
    ax1.grid(axis='x', alpha=0.3)
    
    # ========== 3. 우측: 클래스별 정확도 메트릭 (표 형식) ==========
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.axis('tight')
    ax2.axis('off')
    
    # 정확도 메트릭 표 데이터 준비
    accuracy_table_data = []
    accuracy_table_data.append(['Class', 'a1 (%)\nδ<1.25', 'a2 (%)\nδ<1.56', 'a3 (%)\nδ<1.95'])
    
    for _, row in summary_df[summary_df['Class'] != 'ALL'].iterrows():
        accuracy_table_data.append([
            row['Class'],
            f"{row['a1']*100:.2f}",
            f"{row['a2']*100:.2f}",
            f"{row['a3']*100:.2f}"
        ])
    
    # 표 생성
    table = ax2.table(cellText=accuracy_table_data,
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.20, 0.27, 0.27, 0.27])
    
    # 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 4.0)
    
    # 헤더 스타일
    header_color = '#2C3E50'
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # 데이터 행 스타일 (클래스별 색상 적용)
    class_colors_light = {
        'car': '#FFE5E5',      # 연한 빨강
        'road': '#FFF9E5',     # 연한 노란색
        'car+road': '#FFE8D9'  # 연한 주황색
    }
    
    for i in range(1, len(accuracy_table_data)):
        class_name = accuracy_table_data[i][0]
        row_color = class_colors_light.get(class_name, '#ECF0F1')
        
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('#BDC3C7')
            cell.set_linewidth(0.5)
            
            if j == 0:  # Class 열
                cell.set_text_props(weight='bold', fontsize=13)
            else:  # 메트릭 값
                cell.set_text_props(weight='bold', fontsize=13, color='#2C3E50')
    
    # 제목
    ax2.set_title('Accuracy Metrics by Class\n(Near-perfect accuracy: 98-100%)', 
                 fontsize=13, fontweight='bold', pad=20, color='black')
    
    # ========== 4. 좌측: Car 거리별 abs_rel (선 그래프) ==========
    ax3 = fig.add_subplot(gs[2, 0:2])
    car_dist = distance_df[distance_df['Class'] == 'car'].copy()
    car_dist['Range_num'] = range(len(car_dist))
    
    ax3.plot(car_dist['Range_num'], car_dist['abs_rel'], 
            marker='o', linewidth=3, markersize=10, color=colors['car'],
            markeredgecolor='black', markeredgewidth=1.5, label='Car')
    
    ax3.set_xlabel('Distance Range', fontsize=12, fontweight='bold')
    ax3.set_ylabel('abs_rel', fontsize=12, fontweight='bold')
    ax3.set_title('Car: abs_rel by Distance Range', fontsize=14, fontweight='bold', pad=10, color='black')
    ax3.set_xticks(car_dist['Range_num'])
    ax3.set_xticklabels(car_dist['Range'], rotation=15, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=11)
    
    # 값 표시
    for x, y in zip(car_dist['Range_num'], car_dist['abs_rel']):
        ax3.text(x, y + 0.003, f'{y:.4f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # ========== 5. 우측: Road 거리별 abs_rel (선 그래프) ==========
    ax4 = fig.add_subplot(gs[2, 2:])
    road_dist = distance_df[distance_df['Class'] == 'road'].copy()
    road_dist['Range_num'] = range(len(road_dist))
    
    ax4.plot(road_dist['Range_num'], road_dist['abs_rel'], 
            marker='s', linewidth=3, markersize=10, color=colors['road'],
            markeredgecolor='black', markeredgewidth=1.5, label='Road')
    
    ax4.set_xlabel('Distance Range', fontsize=12, fontweight='bold')
    ax4.set_ylabel('abs_rel', fontsize=12, fontweight='bold')
    ax4.set_title('Road: abs_rel by Distance Range', fontsize=14, fontweight='bold', pad=10, color='black')
    ax4.set_xticks(road_dist['Range_num'])
    ax4.set_xticklabels(road_dist['Range'], rotation=15, ha='right')
    ax4.set_ylim(0, max(road_dist['abs_rel']) * 1.3)  # Y축 범위를 30% 더 확장
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=11)
    
    # 값 표시
    for x, y in zip(road_dist['Range_num'], road_dist['abs_rel']):
        ax4.text(x, y + 0.001, f'{y:.4f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # ========== 6. 좌측: RMSE 비교 (막대 그래프) ==========
    ax5 = fig.add_subplot(gs[3, 0:2])
    rmse_vals = summary_df[summary_df['Class'] != 'ALL']['rmse'].values
    
    bars = ax5.bar(classes, rmse_vals, color=[colors[c] for c in classes],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax5.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
    ax5.set_title('Root Mean Square Error by Class', fontsize=14, fontweight='bold', pad=10, color='black')
    ax5.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ========== 7. 우측: Car+Road 거리별 픽셀 분포 (영역 그래프) ==========
    ax6 = fig.add_subplot(gs[3, 2:])
    combined_dist = distance_df[distance_df['Class'] == 'car+road'].copy()
    combined_dist['Range_num'] = range(len(combined_dist))
    
    # 픽셀 수를 천 단위로 변환
    pixels_k = combined_dist['Pixels'] / 1000
    
    ax6.fill_between(combined_dist['Range_num'], pixels_k, 
                     alpha=0.6, color=colors['car+road'], edgecolor='black', linewidth=2)
    ax6.plot(combined_dist['Range_num'], pixels_k, 
            marker='D', linewidth=2, markersize=8, color='darkgreen',
            markeredgecolor='black', markeredgewidth=1)
    
    ax6.set_xlabel('Distance Range', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Pixel Count (×1000)', fontsize=12, fontweight='bold')
    ax6.set_title('Car+Road: Pixel Distribution by Distance', fontsize=14, fontweight='bold', pad=10, color='black')
    ax6.set_xticks(combined_dist['Range_num'])
    ax6.set_xticklabels(combined_dist['Range'], rotation=15, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # 값 표시
    for x, y in zip(combined_dist['Range_num'], pixels_k):
        ax6.text(x, y + 10, f'{y:.0f}K', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # ========== 7-10. 하단: 거리별 상세 메트릭 표 (4개 클래스) ==========
    available_classes = set(distance_df['Class'].unique())

    bottom_layout = [
        ('car', gs[4, 0:2]),
        ('road', gs[4, 2:]),
        ('car+road', gs[5, 0:2]),
        ('ALL', gs[5, 2:])
    ]

    for cls_name, grid_spec in bottom_layout:
        if cls_name in available_classes:
            ax_table = fig.add_subplot(grid_spec)
            create_distance_table_subplot(ax_table, distance_df, cls_name, colors.get(cls_name, '#2C3E50'))
    
    # 저장
    png_path = os.path.join(output_path, 'dashboard.png')
    pdf_path = os.path.join(output_path, 'dashboard.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print("✅ Dashboard saved:")
    print(f"   - {png_path}")
    print(f"   - {pdf_path}")
    
    plt.show()

def main():
    """메인 실행 함수"""
    # 커맨드 라인 인자로 경로 받기
    if len(sys.argv) > 1:
        metrics_path = sys.argv[1]
        print(f"📊 Loading data from: {metrics_path}")
    else:
        metrics_path = 'outputs/ResNet-SAN_0.05to100_results/metrics'
        print(f"📊 Loading data from default path: {metrics_path}")
    
    summary_df, distance_df = load_data(metrics_path)
    
    print("📈 Creating comprehensive dashboard with integrated tables...")
    create_dashboard(summary_df, distance_df, output_path=metrics_path)
    
    print("\n✨ All visualizations complete!")

if __name__ == '__main__':
    main()
