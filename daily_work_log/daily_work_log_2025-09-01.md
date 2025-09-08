## 2025-09-01 작업 로그

### VADAS 룩업 테이블 생성 스크립트 (`create_vadas_lookup_table.py`) 테스트 및 수정

**1. 초기 테스트 (1920x1536, `np.clip` 상한 90도 유지)**
- **명령:** `python /workspace/packnet-sfm/scripts/create_vadas_lookup_table.py --image_width 1920 --image_height 1536 --output /workspace/packnet-sfm/temp_vadas_lut_1920_1536.pkl`
- **결과:**
    - 이미지 크기: 1536x1920 픽셀
    - Angle 범위: 0.08° ~ 90.00°
    - 유효 Angle 통계: 평균 67.44°, 표준편차 21.72°
    - 픽셀 (320, 192): Angle = 90.00°
    - 픽셀 (100, 100): Angle = 90.00°
    - 픽셀 (500, 200): Angle = 79.86°
- **분석:** `np.clip`에 의해 Angle 값이 90도로 제한됨을 확인.

**2. `np.clip` 상한 제한 제거 후 테스트 (1920x1536)**
- **수정 내용:**
    - `/workspace/packnet-sfm/scripts/create_vadas_lookup_table.py` 파일 내 `_inverse_polynomial_roots` 및 `_inverse_polynomial_newton` 함수에서 `np.clip`의 상한 (`np.pi / 2`) 제거.
        - `return float(np.clip(theta, 0.0, np.pi / 2))` -> `return float(theta)`
        - `theta = float(np.clip(theta, 0.0, np.pi / 2))` -> `theta = float(np.clip(theta, 0.0, np.inf))`
        - `theta = float(np.clip(theta - 0.5 * delta, 0.0, np.pi / 2))` -> `theta = float(np.clip(theta - 0.5 * delta, 0.0, np.inf))`
- **명령:** `python /workspace/packnet-sfm/scripts/create_vadas_lookup_table.py --image_width 1920 --image_height 1536 --output /workspace/packnet-sfm/temp_vadas_lut_1920_1536_unclipped.pkl`
- **결과:**
    - 이미지 크기: 1536x1920 픽셀
    - Angle 범위: 0.08° ~ 4618.12°
    - 유효 Angle 통계: 평균 69.67°, 표준편차 24.69°
    - 픽셀 (320, 192): Angle = 91.74°
    - 픽셀 (100, 100): Angle = 112.12°
    - 픽셀 (500, 200): Angle = 79.86°
- **분석:** `np.clip` 제한 제거 후 Angle 값이 90도를 훨씬 초과하여 물리적으로 의미 없는 큰 값(4618.12°)까지 계산됨을 확인. 이는 VADAS 캘리브레이션 모델이 90도 이상의 입사각에 대해 정확하게 정의되지 않았거나 다항식의 특성상 해당 범위에서 발산하는 경향이 있음을 시사. 190도 이상의 화각(95도 이상의 입사각)이 필요하다면, VADAS 캘리브레이션 모델 자체를 해당 화각에 맞게 재조정하거나, 캘리브레이션 계수가 해당 범위에서 유효한지 확인해야 함. 현재 상태로는 90도 이상의 Angle 값은 신뢰하기 어려움.

**3. 가장 왜곡이 심한 부분 테스트 및 검증 기준 설정**
- **검증 기준:**
    1.  **물리적 타당성:** 입사각(Angle)이 물리적 한계(일반적으로 90도 미만, 190도 FoV의 경우 95도 미만)를 넘지 않아야 함.
    2.  **수치적 안정성:** 역 다항식 계산이 수렴해야 하며, `ValueError` 등의 예외가 발생하지 않아야 함.
    3.  **단조성 및 부드러움:** 이미지 중심에서 멀어질수록 입사각이 단조롭게 증가하고, 변화가 부드러워야 함.
- **수정 내용:**
    - `create_vadas_lookup_table.py` 파일 내 `VADASLookupTableGenerator`의 `verbose` 인자를 `True`로 설정.
    - `main` 함수 내 `test_coords`를 이미지의 네 모서리 및 각 변의 중앙 픽셀들로 업데이트.
- **명령:** `python /workspace/packnet-sfm/scripts/create_vadas_lookup_table.py --image_width 1920 --image_height 1536 --output /workspace/packnet-sfm/temp_vadas_lut_1920_1536_extreme_unclipped.pkl`
- **결과:**
    - `_inverse_polynomial_roots` 함수가 "No non-negative real root for xd." 오류로 자주 실패하고, `_inverse_polynomial_newton` 함수도 많은 경우 수렴하지 못함 (`❌ Not converged`).
    - Angle 범위: 0.08° ~ 4618.12° (여전히 비현실적인 큰 값).
    - 가장자리 픽셀 테스트 결과 (Angle 값):
        - Pixel (0, 0): 158.43°
        - Pixel (1919, 0): 119.71°
        - Pixel (0, 1535): 133.10°
        - Pixel (1919, 1535): 119.23°
        - Pixel (960, 0): 80.76°
        - Pixel (960, 1535): 80.22°
        - Pixel (0, 768): 101.08°
        - Pixel (1919, 768): 93.25°
- **분석:** `np.clip` 제한 제거 시 VADAS 캘리브레이션 모델의 다항식 역변환이 90도 이상의 입사각에 대해 물리적으로 타당하지 않은 매우 큰 값을 생성하며 수치적으로 불안정하게 동작함을 재확인. 현재 캘리브레이션 계수로는 190도 이상의 화각을 정확하게 모델링하기 어려움.

**4. `np.clip` 제한 및 테스트 설정 복원**
- **수정 내용:**
    - `/workspace/packnet-sfm/scripts/create_vadas_lookup_table.py` 파일 내 `np.clip` 상한 제한을 `np.pi / 2`로 복원.
    - `VADASLookupTableGenerator`의 `verbose` 인자를 `False`로 복원.
    - `main` 함수 내 `test_coords`를 원래의 기본값으로 복원.
- **결론:** 현재 VADAS 캘리브레이션 모델로는 190도 이상의 화각을 정확하게 처리하기 어려우며, 이를 위해서는 캘리브레이션 데이터 자체의 재조정 또는 모델 변경이 필요함.

**5. `np.roots`만 사용, `np.clip` 제한 제거, 95도 초과 Angle NaN 처리**
- **수정 내용:**
    - `_inverse_polynomial_newton` 함수 제거.
    - `_generate_pixel_wise_lut` 함수에서 `_inverse_polynomial_newton`으로 폴백하는 로직 제거.
    - `_inverse_polynomial_roots` 함수에서 해를 찾지 못하면 `np.nan` 반환하도록 수정.
    - `_generate_pixel_wise_lut` 함수에서 `np.nan` 값을 통계 계산에서 제외하도록 수정.
    - `_inverse_polynomial_roots` 함수 내에서 `theta` 값이 `np.deg2rad(95)`를 초과하면 `np.nan`을 반환하도록 수정.
- **명령:** `python /workspace/packnet-sfm/scripts/create_vadas_lookup_table.py --image_width 1920 --image_height 1536 --output /workspace/packnet-sfm/temp_vadas_lut_1920_1536_roots_clipped_95deg.pkl --test`
- **결과:**
    - Angle 범위: 0.08° ~ 95.00° (최대 Angle 값이 95도로 제한됨).
    - Valid angles count: 2497161 (95도를 초과하는 Angle 값이 `np.nan`으로 처리되어 감소).
    - 픽셀 (320, 192): Angle = 91.74°
    - 픽셀 (100, 100): Angle = nan° (이전 112.12°였던 값이 95도를 초과하여 `np.nan`으로 처리됨)
    - 픽셀 (500, 200): Angle = 79.86°
- **분석:** `np.roots`를 사용하고 95도 초과 Angle을 `np.nan`으로 처리함으로써 물리적으로 타당한 Angle 값 범위를 유지할 수 있게 됨. 하지만 여전히 많은 픽셀에서 `np.nan`이 발생하며, 이는 다항식 모델 자체의 한계이거나 캘리브레이션 계수가 190도 화각을 안정적으로 모델링하기에 부적합하다는 것을 의미함.
- **결론:** 현재 VADAS 캘리브레이션 모델과 계수로는 190도 화각을 완벽하게 커버하면서 모든 픽셀에 대해 안정적이고 물리적으로 타당한 Angle 값을 얻기 어려움. `np.nan`으로 처리된 픽셀은 유효한 깊이 정보를 얻을 수 없는 영역으로 간주하고 후속 처리에서 마스킹하거나 무시하는 방식으로 진행해야 함. 모든 픽셀에 대해 유효한 Angle 값을 얻는 것이 필수적이라면, 캘리브레이션 데이터 자체의 재조정 또는 모델 변경을 고려해야 함.

**6. `np.roots`만 사용, `np.clip` 제한 제거, 95도 초과 Angle NaN 처리 (최종 확인)**
- **수정 내용:** `_inverse_polynomial_roots` 함수에서 `np.real(roots[np.isreal(roots)])`를 통해 실수 해만 필터링하고 `xd >= 0`인 최소 실수 해를 선택하는 로직은 이미 구현되어 있었음. `_inverse_polynomial_newton` 함수 참조 제거 및 `test_vadas_polynomial` 함수 내 `_inverse_polynomial_newton` 호출 제거는 이미 완료되었음.
- **명령:** `python /workspace/packnet-sfm/scripts/create_vadas_lookup_table.py --image_width 1920 --image_height 1536 --output /workspace/packnet-sfm/temp_vadas_lut_1920_1536_roots_clipped_95deg_final.pkl --test`
- **결과:** 이전과 동일한 결과가 나왔으며, 95도 초과 Angle은 `np.nan`으로 처리되고 유효한 Angle 값 범위는 0.08° ~ 95.00°로 유지됨.
- **분석:** `np.roots`를 사용하고 95도 초과 Angle을 `np.nan`으로 처리하는 현재 방식이 물리적 타당성을 유지하면서 왜곡이 심한 부분의 비현실적인 값을 처리하는 데 적합함.
- **결론:** 현재 `create_vadas_lookup_table.py` 코드는 사용자님의 요구사항을 충족시키면서 VADAS 카메라 모델의 한계를 고려한 최적의 상태임. `np.nan`으로 처리된 픽셀은 후속 처리에서 마스킹하거나 무시해야 함.