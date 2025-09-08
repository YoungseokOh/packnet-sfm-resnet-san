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
                 use_cache: bool = True,
                 use_roots: bool = True,
                 verbose: bool = False):
        """
        Parameters
        ----------
        image_size : Tuple[int, int]
            이미지 크기 (H, W)
        calib_data : Dict, optional
            VADAS 캘리브레이션 데이터 (기본값 사용 시 None)
        use_cache : bool
            캐싱 사용 여부
        use_roots : bool
            다항식 역산에 numpy.roots 사용 여부 (기본 True, 실패 시 뉴턴법 폴백)
        verbose : bool
            디버그 로깅 여부
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
        self.use_roots = use_roots
        self.verbose = verbose
        
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
    
    def _inverse_polynomial_roots(self, r_d: float) -> float:
        """7차 다항식 P(xd) - r_d*div = 0 을 xd에 대해 np.roots로 풀고 theta=xd/s 반환"""
        # P(xd) = sum k_i * xd^i
        coeffs = list(self.k)  # a0..aN
        coeffs[0] = coeffs[0] - (r_d * self.div)  # 상수항 보정
        # np.roots는 최고차부터 기대하므로 뒤집기
        poly = list(reversed(coeffs))
        roots = np.roots(poly)
        real_roots = np.real(roots[np.isreal(roots)])
        # 물리적으로 의미 있는 해: xd >= 0 중 최소값
        candidates = real_roots[real_roots >= 0.0]
        if candidates.size == 0:
            return np.nan
        xd = float(np.min(candidates))
        theta = xd / self.s
        if theta > np.deg2rad(95):
            return np.nan
        return float(theta)

    

    def _generate_pixel_wise_lut(self):
        """픽셀별 theta와 angle_maps 생성 (ref_generate_luts 방식)"""
        print("🔧 Generating pixel-wise LUTs (ref_generate_luts style)...")
        print(f"   Image size: {self.image_size}")
        print(f"   VADAS params: k={self.k[:3]}..., s={self.s}, div={self.div}")

        H, W = self.image_size
        # 이미지 좌표 생성
        x = np.linspace(0, W - 1, W)  # width
        y = np.linspace(0, H - 1, H)  # height
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)

        # ref_generate_luts 및 projector와 일치하도록 주점에 이미지 중심 보정 적용
        u0 = self.ux + (W / 2.0)
        v0 = self.uy + (H / 2.0)

        # 픽셀 단위 오프셋 및 반경
        dx = (mesh_x - u0)
        dy = (mesh_y - v0)
        r = np.sqrt(dx * dx + dy * dy)

        # theta LUT (방위각)
        self.theta_lut = np.arctan2(dy, dx).astype(np.float32)

        # angle LUT (입사각)
        self.angle_lut = np.zeros_like(r, dtype=np.float32)

        total = len(r)
        print(f"   Processing {total} pixels...")

        # 샘플 디버그 인덱스
        sample_idx = {0, min(5000, total-1), total-1}

        for i, _r in enumerate(r):
            r_d = float(_r[0])
            if r_d < 1e-8:
                self.angle_lut[i] = 0.0
            else:
                try:
                    theta_inc = self._inverse_polynomial_roots(r_d)
                    self.angle_lut[i] = theta_inc
                    if self.verbose and i in sample_idx:
                        fwd = self._poly_eval(self.k, theta_inc * self.s) / self.div
                        print(f"   [#{i}] r_d={r_d:.4f} -> θ={np.degrees(theta_inc):.3f}°, fwd={fwd:.4f}")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: inverse failed at {i}, r_d={r_d}: {e}")
                    self.angle_lut[i] = np.nan

            if i % 50000 == 0 and i > 0:
                print(f"   Processed {i}/{total} pixels...")

        print(f"✅ Pixel-wise LUTs generated: {H}x{W} pixels")
        print(f"   Theta range: {np.degrees(np.nanmin(self.theta_lut)):.2f}° ~ {np.degrees(np.nanmax(self.theta_lut)):.2f}°")
        print(f"   Angle range: {np.degrees(np.nanmin(self.angle_lut)):.2f}° ~ {np.degrees(np.nanmax(self.angle_lut)):.2f}°")

        valid_angles = self.angle_lut[~np.isnan(self.angle_lut) & (self.angle_lut > 0.0)]
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
    generator = VADASLookupTableGenerator(image_size, calib_data, use_cache, use_roots=True, verbose=False)
    
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
        theta_reconstructed = generator._inverse_polynomial_roots(rd_expected)
        
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