import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import time
import os

# VADAS 기본 캘리브레이션 데이터 (ref_generate_luts 스타일)
DEFAULT_VADAS_CALIB = {
    'k_coeffs': [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391],  # 7개 다항식 계수
    's': 1.0447,           # 스케일 팩터
    'div': 0.0021,         # 디비전 팩터
    'ux': 44.9516,         # 주점 x
    'uy': 2.48822,         # 주점 y
}

class VADASLookupTableGenerator:
    """VADAS 카메라 모델의 역 다항식 계산을 위한 Lookup table 생성기 (ref_generate_luts 방식)"""
    
    def __init__(self, image_size: Tuple[int, int], 
                 calib_data: Optional[Dict] = None,
                 use_cache: bool = True):
        """
        Parameters
        ----------
        image_size : Tuple[int, int]
            이미지 크기 (H, W)
        calib_data : Dict, optional
            VADAS 캘리브레이션 데이터 (기본값 사용 시 None)
        use_cache : bool
            캐싱 사용 여부
        """
        # 캘리브레이션 데이터 설정
        if calib_data is None:
            calib_data = DEFAULT_VADAS_CALIB
        
        self.k = calib_data['k_coeffs']
        self.s = calib_data['s']
        self.div = calib_data['div']
        self.ux = calib_data['ux']
        self.uy = calib_data['uy']
        self.image_size = image_size
        self.use_cache = use_cache
        
        # 캐시 키 생성
        self.cache_key = self._generate_cache_key()
        
        # 픽셀별 LUT 생성 (ref_generate_luts 방식)
        self._generate_pixel_wise_lut()
    
    def _generate_cache_key(self) -> str:
        """캐시 키 생성"""
        config_str = f"{self.k}_{self.s}_{self.div}_{self.ux}_{self.uy}_{self.image_size}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _poly_eval(self, coeffs: List[float], x: float) -> float:
        """다항식 평가 (Horner's method)"""
        result = 0.0
        for coeff in reversed(coeffs):
            result = result * x + coeff
        return result
    
    def _poly_derivative(self, coeffs: List[float]) -> List[float]:
        """다항식의 도함수 계수 계산"""
        n = len(coeffs)
        if n <= 1:
            return [0.0]
        return [i * coeffs[i] for i in range(1, n)]
    
    def _inverse_polynomial_newton(self, r_d: float, max_iter: int = 20, tol: float = 1e-8) -> float:
        """Newton-Raphson 방법을 사용한 7차 다항식 역 계산 (해가 하나만 존재한다는 가정으로 단순화)"""
        if r_d < 1e-8:
            return 0.0
        
        # 초기 추정값: r_d로부터 합리적인 theta 추정
        theta = np.arctan(r_d)
        
        # 도함수 계수 미리 계산
        deriv_coeffs = self._poly_derivative(self.k)
        
        print(f"🔍 Solving for r_d = {r_d:.6f}, initial theta = {np.degrees(theta):.2f}°")
        
        for i in range(max_iter):
            # Forward polynomial 계산
            xd = theta * self.s
            poly_val = self._poly_eval(self.k, xd) / self.div
            
            # 목표값과의 차이
            residual = poly_val - r_d
            
            print(f"   Iteration {i+1}: theta = {np.degrees(theta):.4f}°, poly_val = {poly_val:.6f}, residual = {residual:.8f}")
            
            # 수렴 체크
            if abs(residual) < tol:
                print(f"   ✅ Converged! Final theta = {np.degrees(theta):.4f}°")
                return theta
            
            # Derivative 계산
            deriv_val = self._poly_eval(deriv_coeffs, xd) * self.s / self.div
            
            # Newton step
            if abs(deriv_val) > tol:
                delta_theta = residual / deriv_val
                theta = theta - 0.5 * delta_theta  # 학습률 0.5로 안정화
                print(f"      Delta theta = {np.degrees(delta_theta):.6f}°, new theta = {np.degrees(theta):.4f}°")
            else:
                print(f"   ⚠️ Derivative too small, stopping")
                break
        
        print(f"   ❌ Not converged after {max_iter} iterations, final theta = {np.degrees(theta):.4f}°")
        return theta
    
    def _generate_pixel_wise_lut(self):
        """픽셀별 theta와 angle_maps 생성 (ref_generate_luts 방식)"""
        print("🔧 Generating pixel-wise LUTs (ref_generate_luts style)...")
        print(f"   Image size: {self.image_size}")
        print(f"   VADAS params: k={self.k[:3]}..., s={self.s}, div={self.div}")
        
        # 이미지 좌표 생성 (ref_generate_luts 방식)
        x = np.linspace(0, self.image_size[1] - 1, self.image_size[1])  # width
        y = np.linspace(0, self.image_size[0] - 1, self.image_size[0])  # height
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)
        
        # 카메라 좌표계로 변환 (VADAS 모델에 맞게)
        x_cam = (mesh_x - self.ux) / self.s
        y_cam = (mesh_y - self.uy) / self.div
        
        # 왜곡 반경 계산
        r = np.sqrt(x_cam * x_cam + y_cam * y_cam)
        
        # theta LUT (방향각)
        self.theta_lut = np.arctan2(y_cam, x_cam).astype(np.float32)
        
        # angle LUT (7차 다항식의 역으로 구한 각도)
        self.angle_lut = np.zeros_like(r, dtype=np.float32)
        
        print(f"   Processing {len(r)} pixels...")
        
        # 테스트를 위해 일부 픽셀만 처리 (전체 처리 시 너무 많은 출력)
        test_pixels = [0, 1000, 5000, 10000, 20000, len(r)-1]  # 시작, 중간, 끝 픽셀들
        
        for i, _r in enumerate(r):
            if _r[0] < 1e-8:  # 중심부 픽셀
                self.angle_lut[i] = 0.0
            else:
                try:
                    # Newton-Raphson으로 7차 다항식 역 계산
                    theta = self._inverse_polynomial_newton(_r[0])
                    self.angle_lut[i] = theta
                    
                    # 테스트 픽셀에 대해서만 상세 출력
                    if i in test_pixels:
                        print(f"\n📍 Test Pixel {i}:")
                        print(f"   r_d = {_r[0]:.6f}")
                        print(f"   Final theta = {np.degrees(theta):.4f}°")
                        print(f"   Verification: forward calc = {self._poly_eval(self.k, theta * self.s) / self.div:.6f}")
                        
                except Exception as e:
                    print(f"Warning: Failed to compute inverse for pixel {i}, r={_r[0]}: {e}")
                    self.angle_lut[i] = 0.0
            
            # 진행 상황 출력 (1000픽셀마다)
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(r)} pixels...")
        
        print(f"✅ Pixel-wise LUTs generated: {self.image_size[0]}x{self.image_size[1]} pixels")
        print(f"   Theta range: {np.degrees(self.theta_lut.min()):.2f}° ~ {np.degrees(self.theta_lut.max()):.2f}°")
        print(f"   Angle range: {np.degrees(self.angle_lut.min()):.2f}° ~ {np.degrees(self.angle_lut.max()):.2f}°")
        
        # 최종 통계 출력
        valid_angles = self.angle_lut[self.angle_lut != 0.0]
        print(f"   Valid angles count: {len(valid_angles)}")
        if len(valid_angles) > 0:
            print(f"   Valid angle stats: mean={np.degrees(valid_angles.mean()):.2f}°, std={np.degrees(valid_angles.std()):.2f}°")
    
    def save_lookup_table(self, filepath: Path):
        """LUT를 pickle 파일로 저장 (ref_generate_luts 방식)"""
        lut_data = {
            'theta_lut': self.theta_lut,
            'angle_lut': self.angle_lut,
            'metadata': {
                'k_coeffs': self.k,
                's': self.s,
                'div': self.div,
                'ux': self.ux,
                'uy': self.uy,
                'image_size': self.image_size,
                'cache_key': self.cache_key,
                'timestamp': time.time()
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(lut_data, f, pickle.HIGHEST_PROTOCOL)
        
        print(f"💾 LUT saved to: {filepath}")
        print(f"   File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    
    @classmethod
    def load_lookup_table(cls, filepath: Path) -> 'VADASLookupTableGenerator':
        """저장된 LUT 로드"""
        with open(filepath, 'rb') as f:
            lut_data = pickle.load(f)
        
        # 새 인스턴스 생성
        instance = cls.__new__(cls)
        
        # 데이터 복원
        instance.theta_lut = lut_data['theta_lut']
        instance.angle_lut = lut_data['angle_lut']
        
        # 메타데이터 복원
        meta = lut_data['metadata']
        instance.k = meta['k_coeffs']
        instance.s = meta['s']
        instance.div = meta['div']
        instance.ux = meta['ux']
        instance.uy = meta['uy']
        instance.image_size = meta['image_size']
        instance.cache_key = meta['cache_key']
        
        print(f"📂 LUT loaded from: {filepath}")
        print(f"   Image size: {instance.image_size}")
        print(f"   Cache key: {instance.cache_key}")
        
        return instance
    
    def get_pixel_angle(self, u: int, v: int) -> Tuple[float, float]:
        """특정 픽셀의 theta와 angle 값 반환"""
        if not (0 <= v < self.image_size[0] and 0 <= u < self.image_size[1]):
            return 0.0, 0.0
        
        idx = v * self.image_size[1] + u
        return self.theta_lut[idx, 0], self.angle_lut[idx, 0]

def create_vadas_lookup_table(image_size: Tuple[int, int], 
                             output_path: Path,
                             calib_data: Optional[Dict] = None,
                             use_cache: bool = True) -> VADASLookupTableGenerator:
    """VADAS 캘리브레이션 데이터로부터 픽셀별 LUT 생성 (ref_generate_luts 방식)
    
    Parameters
    ----------
    image_size : Tuple[int, int]
        이미지 크기 (H, W)
    output_path : Path
        출력 파일 경로
    calib_data : Dict, optional
        VADAS 캘리브레이션 데이터 (기본값 사용 시 None)
    use_cache : bool
        캐싱 사용 여부
        
    Returns
    -------
    VADASLookupTableGenerator
        생성된 LUT 생성기
    """
    print("🔧 Creating VADAS pixel-wise LUTs (ref_generate_luts style)...")
    print(f"   Target image size: {image_size}")
    
    # LUT 생성기 생성
    generator = VADASLookupTableGenerator(image_size, calib_data, use_cache)
    
    # LUT 저장
    generator.save_lookup_table(output_path)
    
    return generator

def test_vadas_polynomial():
    """VADAS 다항식 역변환 테스트 함수"""
    print("\n" + "="*60)
    print("🧪 VADAS 다항식 역변환 테스트")
    print("="*60)
    
    # VADAS 계수
    k = [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391]
    s, div = 1.0447, 0.0021
    
    # 테스트할 theta 값들 (각도)
    test_thetas = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5]  # 라디안
    
    print("📊 Forward 다항식 계산 (theta → r_d):")
    print("-" * 40)
    
    for theta in test_thetas:
        xd = theta * s
        rd = 0.0
        for coeff in reversed(k):
            rd = rd * xd + coeff
        rd = rd / div
        
        print(f"   Theta = {np.degrees(theta):.2f}°, r_d = {rd:.6f}")
    
    print("\n🔄 역 다항식 계산 (r_d → theta):")
    print("-" * 40)
    
    # LUT 생성기 생성 (작은 크기로 테스트)
    generator = VADASLookupTableGenerator((10, 10), use_cache=False)
    
    for theta in test_thetas:
        # Forward 계산
        xd = theta * s
        rd_expected = 0.0
        for coeff in reversed(k):
            rd_expected = rd_expected * xd + coeff
        rd_expected = rd_expected / div
        
        # 역 계산
        theta_reconstructed = generator._inverse_polynomial_newton(rd_expected)
        
        error = abs(theta - theta_reconstructed)
        print(f"   r_d = {rd_expected:.6f}, Reconstructed Theta = {np.degrees(theta_reconstructed):.2f}°, Error = {np.degrees(error):.6f}°")

def main():
    parser = argparse.ArgumentParser(description="Create VADAS pixel-wise lookup tables (ref_generate_luts style)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for LUT pickle file")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Image width")
    parser.add_argument("--image_height", type=int, default=384,
                        help="Image height")
    parser.add_argument("--test", action="store_true",
                        help="Run polynomial test before creating LUT")
    
    args = parser.parse_args()
    
    # 테스트 실행 (옵션)
    if args.test:
        test_vadas_polynomial()
    
    # LUT 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    image_size = (args.image_height, args.image_width)
    
    print(f"\n🚀 Creating LUT for image size {image_size}")
    generator = create_vadas_lookup_table(image_size, output_path)
    
    # 최종 테스트
    print("\n🎯 Final LUT Test:")
    print("-" * 30)
    
    # 몇 개의 픽셀 테스트
    test_coords = [(320, 192), (100, 100), (500, 200)]  # center, top-left, right
    for u, v in test_coords:
        if u < image_size[1] and v < image_size[0]:
            theta, angle = generator.get_pixel_angle(u, v)
            print(f"   Pixel ({u}, {v}): Theta = {np.degrees(theta):.2f}°, Angle = {np.degrees(angle):.2f}°")
    
    print("\n✅ LUT 생성 및 테스트 완료!")

if __name__ == "__main__":
    main()