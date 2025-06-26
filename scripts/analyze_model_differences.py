import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict
import argparse
import os

# PackNet-SfM imports
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
from packnet_sfm.networks.depth.YOLOv8SAN01 import YOLOv8SAN01


class ModelAnalyzer:
    def __init__(self, resnet_ckpt, yolov8_ckpt):
        self.resnet_model = self.load_resnet_model(resnet_ckpt)
        self.yolov8_model = self.load_yolov8_model(yolov8_ckpt)
        
        # 분석 결과 저장용
        self.analysis_results = {}
    
    def load_resnet_model(self, checkpoint_path):
        """ResNet-SAN 모델 로드"""
        print(f"🔄 Loading ResNet-SAN model from: {checkpoint_path}")
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = checkpoint['config']
            state_dict = checkpoint['state_dict']
            
            # ResNet-SAN 모델 생성
            model = ResNetSAN01(
                dropout=config.model.depth_net.get('dropout', 0.5),
                version=config.model.depth_net.get('version', '18A'),
                use_film=config.model.depth_net.get('use_film', False),
                film_scales=config.model.depth_net.get('film_scales', [0]),
                use_enhanced_lidar=config.model.depth_net.get('use_enhanced_lidar', False)
            )
            
            # State dict에서 depth_net 부분만 추출
            depth_state = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('model.depth_net.'):
                    new_key = key.replace('model.depth_net.', '')
                    depth_state[new_key] = value
            
            # 모델 로드 (strict=False로 호환성 문제 해결)
            missing_keys, unexpected_keys = model.load_state_dict(depth_state, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
            
            model.eval()
            print("✅ ResNet-SAN model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load ResNet model: {e}")
            raise
    
    def load_yolov8_model(self, checkpoint_path):
        """YOLOv8-SAN 모델 로드"""
        print(f"🔄 Loading YOLOv8-SAN model from: {checkpoint_path}")
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = checkpoint['config']
            state_dict = checkpoint['state_dict']
            
            # YOLOv8-SAN 모델 생성
            model = YOLOv8SAN01(
                variant=config.model.depth_net.get('variant', 's'),
                dropout=config.model.depth_net.get('dropout', 0.1),
                use_film=config.model.depth_net.get('use_film', False),
                film_scales=config.model.depth_net.get('film_scales', [0]),
                use_enhanced_lidar=config.model.depth_net.get('use_enhanced_lidar', True),
                use_head_features=config.model.depth_net.get('use_head_features', True)
            )
            
            # State dict에서 depth_net 부분만 추출
            depth_state = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('model.depth_net.'):
                    new_key = key.replace('model.depth_net.', '')
                    depth_state[new_key] = value
            
            # 모델 로드 (strict=False로 호환성 문제 해결)
            missing_keys, unexpected_keys = model.load_state_dict(depth_state, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
            
            model.eval()
            print("✅ YOLOv8-SAN model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 model: {e}")
            raise
    
    def analyze_feature_complexity(self):
        """1. Feature Complexity 분석"""
        print("\n🔍 Phase 1: Feature Complexity Analysis")
        
        # 1.1 Model Capacity 비교
        resnet_params = sum(p.numel() for p in self.resnet_model.parameters())
        yolov8_params = sum(p.numel() for p in self.yolov8_model.parameters())
        
        print(f"📊 Model Parameters:")
        print(f"   ResNet18-SAN: {resnet_params:,}")
        print(f"   YOLOv8s-SAN:  {yolov8_params:,}")
        print(f"   Ratio: {yolov8_params/resnet_params:.2f}x")
        
        # 1.2 Feature Map 크기 및 복잡도 비교
        dummy_input = torch.randn(1, 3, 352, 1216)
        
        with torch.no_grad():
            try:
                # ResNet features
                print("📊 Analyzing ResNet features...")
                resnet_features = self.resnet_model.encoder(dummy_input)
                
                print("📊 Analyzing YOLOv8 features...")
                yolov8_features = self.yolov8_model.extract_features(dummy_input)
                
                # Feature 통계
                resnet_stats = self.compute_feature_stats(resnet_features, "ResNet")
                yolov8_stats = self.compute_feature_stats(yolov8_features, "YOLOv8")
                
                complexity_analysis = {
                    'resnet_params': resnet_params,
                    'yolov8_params': yolov8_params,
                    'param_ratio': yolov8_params/resnet_params,
                    'resnet_feature_stats': resnet_stats,
                    'yolov8_feature_stats': yolov8_stats,
                    'resnet_features': resnet_features,
                    'yolov8_features': yolov8_features
                }
                
                self.analysis_results['complexity'] = complexity_analysis
                return complexity_analysis
                
            except Exception as e:
                print(f"❌ Error during feature analysis: {e}")
                return None
    
    def compute_feature_stats(self, features, model_name):
        """Feature 통계 계산"""
        stats = {}
        
        for i, feat in enumerate(features):
            feat_np = feat.detach().cpu().numpy()
            stats[f'scale_{i}'] = {
                'shape': feat.shape,
                'mean': float(np.mean(feat_np)),
                'std': float(np.std(feat_np)),
                'min': float(np.min(feat_np)),
                'max': float(np.max(feat_np)),
                'variance': float(np.var(feat_np)),
                'sparsity': float(np.mean(feat_np == 0))  # Sparsity ratio
            }
            
            print(f"   {model_name} Scale {i}: {feat.shape}, "
                  f"mean={stats[f'scale_{i}']['mean']:.4f}, "
                  f"std={stats[f'scale_{i}']['std']:.4f}")
        
        return stats
    
    def analyze_rgb_feature_quality(self):
        """2. RGB Feature Quality 분석"""
        print("\n🔍 Phase 2: RGB Feature Quality Analysis")
        
        # 다양한 패턴의 synthetic 이미지로 테스트
        test_scenarios = {
            'uniform': self.create_uniform_image(),      # 균일한 이미지
            'gradient': self.create_gradient_image(),    # 그라디언트
            'checkerboard': self.create_checkerboard(),  # 체커보드 패턴
            'noise': self.create_noise_image(),          # 노이즈
            'edges': self.create_edge_image()            # 강한 에지
        }
        
        rgb_quality_scores = {}
        
        for scenario, image in test_scenarios.items():
            print(f"📊 Testing scenario: {scenario}")
            score = self.measure_feature_quality(image, scenario)
            rgb_quality_scores[scenario] = score
        
        self.analysis_results['rgb_quality'] = rgb_quality_scores
        return rgb_quality_scores
    
    def create_uniform_image(self):
        """균일한 이미지 생성"""
        return torch.ones(1, 3, 352, 1216) * 0.5
    
    def create_gradient_image(self):
        """그라디언트 이미지 생성"""
        img = torch.zeros(1, 3, 352, 1216)
        for i in range(1216):
            img[:, :, :, i] = i / 1216.0
        return img
    
    def create_checkerboard(self):
        """체커보드 패턴 이미지 생성"""
        img = torch.zeros(1, 3, 352, 1216)
        for i in range(0, 352, 16):
            for j in range(0, 1216, 16):
                if ((i//16) + (j//16)) % 2 == 0:
                    img[:, :, i:i+16, j:j+16] = 1.0
        return img
    
    def create_noise_image(self):
        """노이즈 이미지 생성"""
        return torch.randn(1, 3, 352, 1216) * 0.3 + 0.5
    
    def create_edge_image(self):
        """에지가 강한 이미지 생성"""
        img = torch.zeros(1, 3, 352, 1216)
        img[:, :, 176:177, :] = 1.0  # 수평선
        img[:, :, :, 608:609] = 1.0  # 수직선
        return img
    
    def measure_feature_quality(self, image, scenario_name):
        """Feature quality 측정"""
        quality_metrics = {}
        
        with torch.no_grad():
            try:
                # ResNet feature quality
                resnet_features = self.resnet_model.encoder(image)
                resnet_quality = self.compute_feature_quality_metrics(resnet_features, "ResNet")
                
                # YOLOv8 feature quality
                yolov8_features = self.yolov8_model.extract_features(image)
                yolov8_quality = self.compute_feature_quality_metrics(yolov8_features, "YOLOv8")
                
                quality_metrics = {
                    'resnet': resnet_quality,
                    'yolov8': yolov8_quality
                }
                
                print(f"   {scenario_name} - ResNet avg variance: {resnet_quality['avg_variance']:.6f}")
                print(f"   {scenario_name} - YOLOv8 avg variance: {yolov8_quality['avg_variance']:.6f}")
                
            except Exception as e:
                print(f"❌ Error measuring quality for {scenario_name}: {e}")
                quality_metrics = {'error': str(e)}
        
        return quality_metrics
    
    def compute_feature_quality_metrics(self, features, model_name):
        """Feature quality 메트릭 계산"""
        variances = []
        gradients = []
        
        for feat in features:
            feat_np = feat.detach().cpu().numpy()
            
            # Variance (특징 다양성)
            variance = np.var(feat_np)
            variances.append(variance)
            
            # Gradient magnitude (특징 선명도)
            if feat_np.shape[2] > 1 and feat_np.shape[3] > 1:
                grad_x = np.abs(np.diff(feat_np, axis=3))
                grad_y = np.abs(np.diff(feat_np, axis=2))
                gradient_mag = np.mean(grad_x) + np.mean(grad_y)
                gradients.append(gradient_mag)
        
        return {
            'variances': variances,
            'gradients': gradients,
            'avg_variance': np.mean(variances),
            'avg_gradient': np.mean(gradients)
        }
    
    def analyze_lidar_fusion_mechanism(self):
        """3. LiDAR Fusion 메커니즘 분석"""
        print("\n🔍 Phase 3: LiDAR Fusion Mechanism Analysis")
        
        try:
            # 3.1 Fusion Weight 분석
            resnet_weights = self.resnet_model.weight.data.cpu().numpy()
            yolov8_weights = self.yolov8_model.weight.data.cpu().numpy()
            
            resnet_bias = self.resnet_model.bias.data.cpu().numpy()
            yolov8_bias = self.yolov8_model.bias.data.cpu().numpy()
            
            print(f"📊 Fusion Weights Analysis:")
            print(f"   ResNet weights: {resnet_weights}")
            print(f"   YOLOv8 weights: {yolov8_weights}")
            print(f"   ResNet bias: {resnet_bias}")
            print(f"   YOLOv8 bias: {yolov8_bias}")
            
            # 3.2 Weight distribution 분석
            weight_analysis = self.analyze_weight_distribution(resnet_weights, yolov8_weights)
            
            fusion_analysis = {
                'resnet_fusion_weights': resnet_weights,
                'yolov8_fusion_weights': yolov8_weights,
                'resnet_fusion_bias': resnet_bias,
                'yolov8_fusion_bias': yolov8_bias,
                'weight_distribution': weight_analysis
            }
            
            self.analysis_results['fusion'] = fusion_analysis
            return fusion_analysis
            
        except Exception as e:
            print(f"❌ Error during fusion analysis: {e}")
            return None
    
    def analyze_weight_distribution(self, resnet_weights, yolov8_weights):
        """가중치 분포 분석"""
        analysis = {}
        
        # 통계 계산
        analysis['resnet_stats'] = {
            'mean': float(np.mean(resnet_weights)),
            'std': float(np.std(resnet_weights)),
            'min': float(np.min(resnet_weights)),
            'max': float(np.max(resnet_weights))
        }
        
        analysis['yolov8_stats'] = {
            'mean': float(np.mean(yolov8_weights)),
            'std': float(np.std(yolov8_weights)),
            'min': float(np.min(yolov8_weights)),
            'max': float(np.max(yolov8_weights))
        }
        
        # 차이 분석
        weight_diff = np.abs(resnet_weights - yolov8_weights)
        analysis['difference'] = {
            'mean_abs_diff': float(np.mean(weight_diff)),
            'max_abs_diff': float(np.max(weight_diff)),
            'relative_diff': float(np.mean(weight_diff) / (np.mean(np.abs(resnet_weights)) + 1e-8))
        }
        
        print(f"   Weight difference - Mean: {analysis['difference']['mean_abs_diff']:.4f}, "
              f"Max: {analysis['difference']['max_abs_diff']:.4f}, "
              f"Relative: {analysis['difference']['relative_diff']:.4f}")
        
        return analysis
    
    def visualize_results(self, save_dir="analysis_results"):
        """분석 결과 시각화"""
        print(f"\n🎨 Creating visualizations in {save_dir}/")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Model complexity comparison
        if 'complexity' in self.analysis_results:
            self.plot_model_complexity(save_dir)
        
        # 2. Feature quality comparison
        if 'rgb_quality' in self.analysis_results:
            self.plot_feature_quality(save_dir)
        
        # 3. Fusion weight comparison
        if 'fusion' in self.analysis_results:
            self.plot_fusion_weights(save_dir)
        
        print("✅ Visualizations saved!")
    
    def plot_model_complexity(self, save_dir):
        """모델 복잡도 시각화"""
        complexity = self.analysis_results['complexity']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Parameter count comparison
        models = ['ResNet18-SAN', 'YOLOv8s-SAN']
        params = [complexity['resnet_params'], complexity['yolov8_params']]
        ax1.bar(models, params, color=['blue', 'orange'])
        ax1.set_ylabel('Parameters')
        ax1.set_title('Model Parameter Count')
        for i, v in enumerate(params):
            ax1.text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Feature variance comparison
        resnet_vars = [complexity['resnet_feature_stats'][f'scale_{i}']['variance'] 
                      for i in range(len(complexity['resnet_feature_stats']))]
        yolov8_vars = [complexity['yolov8_feature_stats'][f'scale_{i}']['variance'] 
                      for i in range(len(complexity['yolov8_feature_stats']))]
        
        x = range(len(resnet_vars))
        ax2.plot(x, resnet_vars, 'b-o', label='ResNet18-SAN')
        ax2.plot(x, yolov8_vars, 'r-s', label='YOLOv8s-SAN')
        ax2.set_xlabel('Scale')
        ax2.set_ylabel('Feature Variance')
        ax2.set_title('Feature Variance by Scale')
        ax2.legend()
        ax2.grid(True)
        
        # Feature shapes
        resnet_shapes = [complexity['resnet_feature_stats'][f'scale_{i}']['shape'] 
                        for i in range(len(complexity['resnet_feature_stats']))]
        yolov8_shapes = [complexity['yolov8_feature_stats'][f'scale_{i}']['shape'] 
                        for i in range(len(complexity['yolov8_feature_stats']))]
        
        resnet_pixels = [s[2] * s[3] for s in resnet_shapes]
        yolov8_pixels = [s[2] * s[3] for s in yolov8_shapes]
        
        ax3.plot(x, resnet_pixels, 'b-o', label='ResNet18-SAN')
        ax3.plot(x, yolov8_pixels, 'r-s', label='YOLOv8s-SAN')
        ax3.set_xlabel('Scale')
        ax3.set_ylabel('Spatial Resolution (pixels)')
        ax3.set_title('Spatial Resolution by Scale')
        ax3.legend()
        ax3.grid(True)
        
        # Channel counts
        resnet_channels = [s[1] for s in resnet_shapes]
        yolov8_channels = [s[1] for s in yolov8_shapes]
        
        ax4.plot(x, resnet_channels, 'b-o', label='ResNet18-SAN')
        ax4.plot(x, yolov8_channels, 'r-s', label='YOLOv8s-SAN')
        ax4.set_xlabel('Scale')
        ax4.set_ylabel('Channel Count')
        ax4.set_title('Channel Count by Scale')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_complexity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_quality(self, save_dir):
        """Feature quality 시각화"""
        quality = self.analysis_results['rgb_quality']
        
        scenarios = list(quality.keys())
        resnet_variances = []
        yolov8_variances = []
        
        for scenario in scenarios:
            if 'error' not in quality[scenario]:
                resnet_variances.append(quality[scenario]['resnet']['avg_variance'])
                yolov8_variances.append(quality[scenario]['yolov8']['avg_variance'])
            else:
                resnet_variances.append(0)
                yolov8_variances.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Variance comparison
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax1.bar(x - width/2, resnet_variances, width, label='ResNet18-SAN', color='blue', alpha=0.7)
        ax1.bar(x + width/2, yolov8_variances, width, label='YOLOv8s-SAN', color='orange', alpha=0.7)
        ax1.set_xlabel('Test Scenarios')
        ax1.set_ylabel('Average Feature Variance')
        ax1.set_title('Feature Quality: Average Variance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative performance
        relative_performance = []
        for i in range(len(scenarios)):
            if yolov8_variances[i] > 0:
                relative_performance.append(resnet_variances[i] / yolov8_variances[i])
            else:
                relative_performance.append(1.0)
        
        ax2.bar(scenarios, relative_performance, color='green', alpha=0.7)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Equal Performance')
        ax2.set_ylabel('ResNet/YOLOv8 Variance Ratio')
        ax2.set_title('Relative Feature Quality (>1: ResNet better)')
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_quality.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_fusion_weights(self, save_dir):
        """Fusion weight 시각화"""
        fusion = self.analysis_results['fusion']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Weight comparison
        scales = range(len(fusion['resnet_fusion_weights']))
        ax1.plot(scales, fusion['resnet_fusion_weights'], 'b-o', 
                label='ResNet18-SAN', linewidth=2, markersize=8)
        ax1.plot(scales, fusion['yolov8_fusion_weights'], 'r-s', 
                label='YOLOv8s-SAN', linewidth=2, markersize=8)
        ax1.set_xlabel('Feature Scale')
        ax1.set_ylabel('Fusion Weight')
        ax1.set_title('LiDAR-RGB Fusion Weights by Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Weight distribution
        all_weights = np.concatenate([fusion['resnet_fusion_weights'], 
                                     fusion['yolov8_fusion_weights']])
        ax2.hist(fusion['resnet_fusion_weights'], bins=10, alpha=0.7, 
                label='ResNet18-SAN', color='blue')
        ax2.hist(fusion['yolov8_fusion_weights'], bins=10, alpha=0.7, 
                label='YOLOv8s-SAN', color='orange')
        ax2.set_xlabel('Weight Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fusion Weight Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fusion_weights.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, save_dir="analysis_results"):
        """분석 보고서 생성"""
        report_path = f"{save_dir}/analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ResNet18-SAN vs YOLOv8s-SAN Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Model Complexity
            if 'complexity' in self.analysis_results:
                complexity = self.analysis_results['complexity']
                f.write("1. MODEL COMPLEXITY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"ResNet18-SAN Parameters: {complexity['resnet_params']:,}\n")
                f.write(f"YOLOv8s-SAN Parameters:  {complexity['yolov8_params']:,}\n")
                f.write(f"Parameter Ratio (YOLOv8/ResNet): {complexity['param_ratio']:.2f}x\n\n")
                
                f.write("Feature Statistics:\n")
                for i in range(len(complexity['resnet_feature_stats'])):
                    resnet_stat = complexity['resnet_feature_stats'][f'scale_{i}']
                    yolov8_stat = complexity['yolov8_feature_stats'][f'scale_{i}']
                    f.write(f"  Scale {i}:\n")
                    f.write(f"    ResNet: {resnet_stat['shape']}, var={resnet_stat['variance']:.6f}\n")
                    f.write(f"    YOLOv8: {yolov8_stat['shape']}, var={yolov8_stat['variance']:.6f}\n")
                f.write("\n")
            
            # Feature Quality
            if 'rgb_quality' in self.analysis_results:
                f.write("2. RGB FEATURE QUALITY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                quality = self.analysis_results['rgb_quality']
                for scenario, results in quality.items():
                    if 'error' not in results:
                        f.write(f"  {scenario.capitalize()}:\n")
                        f.write(f"    ResNet avg variance: {results['resnet']['avg_variance']:.6f}\n")
                        f.write(f"    YOLOv8 avg variance: {results['yolov8']['avg_variance']:.6f}\n")
                        if results['yolov8']['avg_variance'] > 0:
                            ratio = results['resnet']['avg_variance'] / results['yolov8']['avg_variance']
                            f.write(f"    Ratio (ResNet/YOLOv8): {ratio:.3f}\n")
                        f.write("\n")
            
            # Fusion Weights
            if 'fusion' in self.analysis_results:
                f.write("3. LIDAR FUSION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                fusion = self.analysis_results['fusion']
                f.write("Fusion Weights:\n")
                f.write(f"  ResNet: {fusion['resnet_fusion_weights']}\n")
                f.write(f"  YOLOv8: {fusion['yolov8_fusion_weights']}\n\n")
                
                weight_dist = fusion['weight_distribution']
                f.write("Weight Statistics:\n")
                f.write(f"  ResNet - Mean: {weight_dist['resnet_stats']['mean']:.4f}, "
                       f"Std: {weight_dist['resnet_stats']['std']:.4f}\n")
                f.write(f"  YOLOv8 - Mean: {weight_dist['yolov8_stats']['mean']:.4f}, "
                       f"Std: {weight_dist['yolov8_stats']['std']:.4f}\n")
                f.write(f"  Mean Absolute Difference: {weight_dist['difference']['mean_abs_diff']:.4f}\n")
                f.write(f"  Relative Difference: {weight_dist['difference']['relative_diff']:.4f}\n")
        
        print(f"📄 Analysis report saved to: {report_path}")
    
    def run_full_analysis(self, save_dir="analysis_results"):
        """전체 분석 실행"""
        print("🚀 Starting comprehensive model analysis...")
        
        # 분석 실행
        self.analyze_feature_complexity()
        self.analyze_rgb_feature_quality()
        self.analyze_lidar_fusion_mechanism()
        
        # 결과 저장 및 시각화
        os.makedirs(save_dir, exist_ok=True)
        self.visualize_results(save_dir)
        self.generate_report(save_dir)
        
        print(f"\n✅ Analysis complete! Results saved in: {save_dir}/")
        return self.analysis_results


def parse_args():
    """Command line arguments parsing"""
    parser = argparse.ArgumentParser(description='Analyze differences between ResNet-SAN and YOLOv8-SAN models')
    parser.add_argument('--resnet_checkpoint', type=str, required=True,
                        help='Path to ResNet-SAN checkpoint (.ckpt)')
    parser.add_argument('--yolov8_checkpoint', type=str, required=True,
                        help='Path to YOLOv8-SAN checkpoint (.ckpt)')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Output directory for analysis results')
    
    return parser.parse_args()


def main():
    """Main analysis function"""
    args = parse_args()
    
    # Validate checkpoint files
    if not os.path.exists(args.resnet_checkpoint):
        print(f"❌ ResNet checkpoint not found: {args.resnet_checkpoint}")
        return
    
    if not os.path.exists(args.yolov8_checkpoint):
        print(f"❌ YOLOv8 checkpoint not found: {args.yolov8_checkpoint}")
        return
    
    # Run analysis
    try:
        analyzer = ModelAnalyzer(args.resnet_checkpoint, args.yolov8_checkpoint)
        results = analyzer.run_full_analysis(args.output_dir)
        
        print("\n🎯 Key Findings:")
        if 'complexity' in results:
            param_ratio = results['complexity']['param_ratio']
            print(f"   • YOLOv8 is {param_ratio:.1f}x larger than ResNet in parameters")
        
        if 'fusion' in results:
            fusion = results['fusion']
            weight_diff = fusion['weight_distribution']['difference']['relative_diff']
            print(f"   • Fusion weight difference: {weight_diff:.1%}")
        
        print(f"\n📁 Detailed results and visualizations saved in: {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()