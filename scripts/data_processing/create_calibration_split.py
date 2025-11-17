#!/usr/bin/env python3
# create_calibration_split.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class CalibrationDatasetCreator:
    """Representative Calibration Dataset 생성기"""
    
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)
        print(f"총 {len(self.df)}개의 샘플이 메타데이터에 있습니다.")
    
    def create_stratified_split(self, target_size=300, output_file='outputs/calibration/calibration_split.json',
                                depth_bins=None, sampling_ratios=None):
        """
        Depth 분포에 기반하여 계층화된 샘플링을 수행합니다.
        """
        # 기본값 설정
        if depth_bins is None:
            depth_bins = [0, 3, 8, 15, 100]
        
        if sampling_ratios is None:
            # 근거리(25%), 중거리(40%), 원거리(25%), 초원거리(10%)
            sampling_ratios = [0.25, 0.40, 0.25, 0.10]
        
        # 구간 라벨
        labels = ['near', 'mid', 'far', 'very_far'][:len(depth_bins)-1]
        
        # Depth 범위별로 분류
        self.df['depth_range'] = pd.cut(
            self.df['mean_depth'], 
            bins=depth_bins, 
            labels=labels, 
            right=True
        )
        
        # 각 범위별 데이터 개수 확인
        print("\n" + "="*60)
        print("Depth Range 분포")
        print("="*60)
        range_counts = self.df['depth_range'].value_counts(sort=False)
        print(range_counts)
        print("\n비율:")
        print(self.df['depth_range'].value_counts(normalize=True, sort=False))
        
        # 각 구간별 샘플링 크기 결정
        sampled_dfs = []
        total_sampled = 0
        
        print("\n" + "="*60)
        print("샘플링 계획")
        print("="*60)
        
        for i, label in enumerate(labels):
            available = range_counts.get(label, 0)
            target = int(target_size * sampling_ratios[i])
            actual = min(target, available)
            
            if actual > 0:
                samples = self.df[self.df['depth_range'] == label].sample(
                    n=actual, replace=False, random_state=42
                )
                sampled_dfs.append(samples)
                total_sampled += actual
                print(f"{label:10s} ({depth_bins[i]:>5.1f}-{depth_bins[i+1]:>5.1f}m): "
                      f"목표 {target:3d}, 실제 {actual:3d} (가용 {available:3d})")
        
        # 목표 크기에 미달하면 가장 많은 범위에서 추가 샘플링
        if total_sampled < target_size:
            shortage = target_size - total_sampled
            mid_available = range_counts.get('mid', 0) - int(target_size * sampling_ratios[1])
            if mid_available > 0:
                additional = min(shortage, mid_available)
                already_sampled = sampled_dfs[1] if len(sampled_dfs) > 1 else pd.DataFrame()
                mid_pool = self.df[self.df['depth_range'] == 'mid']
                mid_pool = mid_pool[~mid_pool.index.isin(already_sampled.index)]
                
                if len(mid_pool) >= additional:
                    extra_samples = mid_pool.sample(n=additional, replace=False, random_state=42)
                    sampled_dfs.append(extra_samples)
                    total_sampled += additional
                    print(f"\n중거리에서 {additional}개 추가 샘플링")
        
        print(f"\n총 샘플링: {total_sampled}개")
        print("="*60)
        
        # 최종 데이터셋 병합
        representative_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # JSON 형식으로 변환
        calibration_data = []
        for _, row in representative_df.iterrows():
            calibration_data.append({
                'dataset_root': row['dataset_root'],
                'new_filename': row['filename']
            })
        
        # 출력 디렉토리 생성
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 저장
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n✅ '{output_file}' 생성 완료 ({len(calibration_data)}개 샘플)")
        
        # 시각화
        self.visualize_distribution(self.df, representative_df, output_file)
        
        return representative_df
    
    def visualize_distribution(self, original_df, sampled_df, output_file):
        """원본과 샘플링된 데이터셋의 분포 비교"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 원본 분포 - 히스토그램
        axes[0, 0].hist(original_df['mean_depth'], bins=50, alpha=0.7, 
                        color='blue', edgecolor='black')
        axes[0, 0].set_title(f'Original Dataset (n={len(original_df)})')
        axes[0, 0].set_xlabel('Mean Depth (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 샘플링된 분포 - 히스토그램
        axes[0, 1].hist(sampled_df['mean_depth'], bins=50, alpha=0.7, 
                        color='green', edgecolor='black')
        axes[0, 1].set_title(f'Calibration Dataset (n={len(sampled_df)})')
        axes[0, 1].set_xlabel('Mean Depth (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Depth Range별 비교
        range_labels = ['near\n(0-3m)', 'mid\n(3-8m)', 'far\n(8-15m)', 'very_far\n(15m+)']
        original_counts = original_df['depth_range'].value_counts(sort=False)
        sampled_counts = sampled_df['depth_range'].value_counts(sort=False)
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, original_counts.values, width, 
                       label='Original', alpha=0.7, color='blue')
        axes[1, 0].bar(x + width/2, sampled_counts.values, width, 
                       label='Calibration', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Depth Range')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Depth Range Distribution Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(range_labels)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Scene Type별 비교
        if 'scene_type' in original_df.columns and 'scene_type' in sampled_df.columns:
            scene_orig = original_df['scene_type'].value_counts()
            scene_samp = sampled_df['scene_type'].value_counts()
            
            scene_labels = list(set(scene_orig.index) | set(scene_samp.index))
            x_scene = np.arange(len(scene_labels))
            
            orig_vals = [scene_orig.get(label, 0) for label in scene_labels]
            samp_vals = [scene_samp.get(label, 0) for label in scene_labels]
            
            axes[1, 1].bar(x_scene - width/2, orig_vals, width, 
                           label='Original', alpha=0.7, color='blue')
            axes[1, 1].bar(x_scene + width/2, samp_vals, width, 
                           label='Calibration', alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Scene Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Scene Type Distribution Comparison')
            axes[1, 1].set_xticks(x_scene)
            axes[1, 1].set_xticklabels(scene_labels, rotation=15)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = output_file.replace('.json', '_distribution.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ 분포 비교 그래프가 '{plot_file}'에 저장되었습니다.")
        plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create calibration split from metadata')
    parser.add_argument('--metadata', type=str, required=True,
                        help='메타데이터 CSV 파일 경로')
    parser.add_argument('--target_size', type=int, default=300,
                        help='목표 샘플 개수 (기본: 300)')
    parser.add_argument('--output', type=str, default='outputs/calibration/calibration_split.json',
                        help='출력 JSON 파일명')
    
    args = parser.parse_args()
    
    # Calibration Dataset 생성기 초기화
    creator = CalibrationDatasetCreator(args.metadata)
    
    # Stratified Sampling 수행
    creator.create_stratified_split(
        target_size=args.target_size,
        output_file=args.output
    )
