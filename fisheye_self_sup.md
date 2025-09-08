보고서: Fisheye (VADAS) Self-Supervised Depth 학습 준비 상태 점검

요약
왜곡 계수를 활용한 self-supervised warping 코드는 부분적으로 존재 (MultiViewPhotometricLoss 내 FisheyeCamera 사용).
LUT 기반 역투영(inverse distortion)은 아직 학습 경로에 통합되지 않음. FisheyeCamera.reconstruct()는 근사(placeholder) 역변환(“theta ≈ r_d” 등) 사용 → 정확도 및 수렴 안정성 저하 위험.
NcdbDataset이 distortion_coeffs를 배치로 제공하는 구조는 부분 구현 추정이나, 제공된 코드 일부만으로는 최종 텐서 생성 경로(배치 collate/torch 변환)가 확정되지 않음.
warp_ref_image 구현과 forward 호출 사이에 인자 네이밍/형태 불일치 가능성(poses vs pose) 및 함수 정의 라인 포맷 손상 정황 → 런타임 오류 잠재.
추가 통합/정비 없이 즉시 학습 시 품질/안정성 문제 발생 가능성이 높음.
현재 파이프라인 개요
SelfSupModel.self_supervised_loss(): intrinsics_list는 무시되고 distortion_coeffs를 원본/레퍼런스 모두에 동일 전달.
MultiViewPhotometricLoss.forward(): 각 스케일에서 warp_ref_image 호출 → FisheyeCamera 기반 view_synthesis() 수행.
view_synthesis(): ref_cam.reconstruct(depth) + cam.project() 경로로 warp.
FisheyeCamera:
project(): VADAS 다항 전방 모델(θ→r_d)은 다항식 누적 방식으로 구현.
reconstruct(): r_d→θ 역변환은 실제 근 없음·LUT 미사용·단순 근사 (주석에도 정확도 한계 명시).
결과적으로 forward 워핑은 왜곡 “정방향”은 반영되나, 역재구성(reconstruct) 정확도 부족.
LUT 및 왜곡 역변환 상태
scripts/create_vadas_lookup_table.py: 픽셀별 θ(angle) LUT 생성 가능.
그러나 FisheyeCamera.reconstruct() 또는 loss 경로 어디에서도 LUT 로딩/참조 없음.
역 다항식 풀이(roots 활용) 코드는 LUT 스크립트에만 존재; 학습 중 on-the-fly 적용 경로 부재.
현재 reconstruct()는 θ ≈ r_d, tan(θ) 근사 등으로 역산 → 대각선/고주파 영역 왜곡 심함 → photometric alignment 오차 ↑.
DataLoader / Dataset 상태
KITTI (최적화 버전) 경로는 표준 pinhole (Camera) 흐름 유지, fisheye와 직접 연결되지 않음.
NcdbDataset: DEFAULT_CALIB_A6 존재 (k 7개 + s, div, ux, uy 등). 다만:
distortion_coeffs를 torch.Tensor 배치(dict of tensors)로 만드는 로직 미확인(제공 코드 일부만 존재).
self-supervised 멀티뷰 맥락(연속 프레임 pose, ref 이미지) 구성 코드 미노출 → ref frame pose 제공/정합 여부 점검 필요.
collate에서 동일 해상도/스케일링 후 intrinsics 스케일 조정 로직 부재 추정.
위험: Multi-scale pyramid에서 ux, uy만 부분 스케일링(+0.5 보정) 했지만 s, div는 스케일링 미적용 → 스케일 의존적 왜곡 불일치 가능.
핵심 리스크 및 잠재 문제
(정확도) 역투영 근사 → gradient 신뢰도 저하 → 깊이/포즈 동시 추정 수렴 불안정.
(일관성) s, div 미스케일링 → 피라미드 레벨 간 왜곡 모델 불일치.
(코드 안정성) warp_ref_image 시그니처/포맷 손상(줄바꿈 깨짐) 및 poses 단수/복수 혼용 가능성.
(확장성) LUT 미활용 → 고해상도에서 역다항 반복 계산 도입 시 연산량 급증 또는 근사 유지에 따른 성능 손실.
(데이터) NcdbDataset 다중 프레임/pose 추출 경로 검증 필요 (현재 snippet에 pose 준비 코드 없음).
(모델) FisheyeCamera.reconstruct 내부 주석 “placeholder” 상태 방치.
왜곡 계수/ LUT 통합이 들어가야 할 위치 제안
우선순위 1: FisheyeCamera.reconstruct() 내 r_d → θ 역변환을 LUT 참조로 교체 (픽셀 위치별 precomputed θ 혹은 반경별 1D LUT + 보간).
우선순위 2: MultiViewPhotometricLoss.warp_ref_image 스케일 레벨별 intrinsics 재계산 시 s, div에 대한 스케일 인자 적용(또는 설계상 불변인지 명확화).
우선순위 3: Dataset (Ncdb) → batch collate 시 distortion_coeffs dict 생성:
k: [B,7], s/div/ux/uy: [B,1] 형태 텐서화.
레퍼런스 프레임별(컨텍스트) 보정 동일 적용 또는 per-frame calib 지원.
우선순위 4: SelfSupModel.self_supervised_loss()에서 intrinsics_list vs distortion_coeffs 구분 제거/정리 (중복 필드 정리).
즉시 착수 권장 작업 (코드 수정 전 사전 설계)
(검증) multiview_photometric_loss.py 실제 warp_ref_image 함수 완전성 확인 및 포맷 손상 복구.
(성능) LUT 생성 스크립트 출력 포맷을 FisheyeCamera에서 바로 로드하도록 설계 (예: angle_lut reshape 후 θ 맵).
(정합) reconstruct()에서 근사 부분 → 다항식 역근 root / LUT 비교 실험(수십 샘플 픽셀 θ 오차 통계).
(스케일) Multi-scale 사용 시 s, div 스케일링 수학적 정의 재확인 (왜곡 모델에서 단위 픽셀 scaling이 θ→r_d 다항 계수에 미치는 영향 문서화).
(데이터) NcdbDataset 전체 코드 재검토하여 pose와 ref 이미지 리스트가 SelfSupModel에 올바르게 전달되는지 확인.
향후 개선 방향
다항식 역연산: 뉴턴-Raphson + root 선택 안정화 + 실패 fallback + LUT hybrid (반경 구간 비선형 샘플링).
Distortion-aware smoothness: 왜곡 좌표계에서 구배 평활화 시 radial weighting 적용 가능성 검토.
Automasking/occ 규칙: Fisheye edge stretching 영역 가중치 감소(마스크) 추가.
성능 모니터링: 재투영 에러 맵을 pinhole 기준과 비교하는 디버그 유틸 추가.
GPU LUT 캐싱: angle_lut, theta_lut을 half 정밀도로 저장 후 reconstruct 시 gather.
학습 가능 여부 결론
“동작” 자체는 현재 코드로 시도 가능하나, 역투영 근사와 함수/시그니처 불안정성 때문에 결과 품질 및 수렴 신뢰 낮음.
LUT/정확한 역다항 통합 및 데이터/스케일 정합 정비 없이 본격 학습 시작은 비추천.
최소한 reconstruct 정확화 + warp_ref_image 안정화 + distortion_coeffs 배치 생성 경로 검증 후 학습 착수 권장.

---

2025-09-08  분석: ref_code/fisheye_test-master (OmniDet Fisheye) 대비 현행 구현 점검

검토 범위
- 확인한 레거시 파일: omnidet/losses/inverse_warp_fisheye.py, omnidet/losses/distance_loss.py 및 관련 train_utils (lut 사용 경로).
- 현재 구현 비교 대상: packnet_sfm/losses/multiview_photometric_loss.py, geometry/camera.py (FisheyeCamera), LUT 스크립트 및 현재 photometric self-supervised 파이프라인.

주요 차이점 및 현재 부족/불일치 항목
1. 역투영/전방 모델
   - 레거시: img2world()에서 theta_lut(픽셀 평면 각), angle_lut(입사각) LUT 직접 사용 → LUT 기반 정확한 방향 벡터 재구성.
   - 현행: FisheyeCamera.reconstruct()는 r_d→θ 역변환을 근사(θ≈r_d) 후 tan() 사용. LUT 미사용 → 고왜곡 주변부 오차 ↑.
   - 영향: Photometric alignment gradient 왜곡 → 깊이/포즈 추정 품질 저하.

2. 왜곡 다항식/파라미터 차이
   - 레거시: r_mapping = Σ k_i * a^{i+1} (i=1..4) 형태 (a = angle); intrinsics: [fx_like, fy_like, cx, cy] + D(k1..k4).
   - 현행: k[0..6], s, div, ux, uy (VADAS 7차 + scaling 파라미터). 역/정방향 수식 불일치로 레거시 LUT 바로 재사용 불가.
   - 필요: VADAS forward θ→r_d 다항 + 역 r_d→θ LUT/뉴턴 하이브리드 설계 통합.

3. LUT 사용 경로
   - 레거시: theta_lut, angle_lut 입력 배치 딕셔너리에 포함 후 inverse_warp에서 직접 사용.
   - 현행: LUT 파일 존재(luts/*.pkl)하나, 학습 forward 경로에서 미참조.
   - 액션: Dataset 단계에서 LUT 로드 → FisheyeCamera 또는 warp_ref_image에 전달 구조 추가.

4. 거리/깊이 표현
   - 레거시: sigmoid 출력 → norm(거리) = min_d + max_d * σ; 역전 파이프라인에서 직접 거리 사용.
   - 현행: inverse depth(inv_depths) 사용 후 depth = 1 / inv_depth. 스케일 정규화 disp_norm 옵션 존재.
   - 전환 고려: LUT 기반 ray 방향 * 거리 vs depth 변환 일관성 문서화 필요.

5. 마스킹/가중치
   - 레거시: vignetting mask, car mask(ego 영역), automask(identity reprojection) 모두 photometric loss에 반영.
   - 현행: optional automask만 부분 구현, vignetting/ego mask 부재 → 경계/차량 전경 영향 제거 안됨.
   - 권고: vignetting 및 ego/car mask 통합 → 외곽 강한 왜곡/시야 가려짐 픽셀 다운웨이트.

6. Photometric 손실 구조
   - 레거시: reprojection L1 + SSIM (가중합) + identity 비교 후 min + clip + smoothness(dist 기반) + 멀티스케일.
   - 현행: 유사(L1+SSIM, automask min)이나 mask 가중치/클리핑/스케일별 intrinsics 스케일링 불완전.

7. 멀티스케일 Intrinsics/Distortion 스케일링
   - 레거시: Bilinear sampling 이전에 픽셀 정규화만 사용 (고정 LUT 해상도 기준) → 스케일별 재계산 로직 단순.
   - 현행: ux, uy만 스케일링(+0.5/-0.5 보정) 추정, s/div 불변 → 수학적 정합 검증 미완료.
   - 필요: (H_s, W_s)로 리사이즈 시 θ → r_d 매핑 좌표계 scaling 공식화.

8. Pose 변환
   - 레거시: essential_mat (4x4) 직접 사용 (target→source) 후 world2img.
   - 현행: Pose(Tcw) 객체 사용; warp_ref_image에서 pose 인자 이름 혼동(poses vs pose) 가능성 → 런타임 오류 위험.

9. 코드 안정성 및 유지보수
   - 레거시: inverse_warp_fisheye.py 모듈 단일 책임(워핑+lut) 명확.
   - 현행: FisheyeCamera에 reconstruct + project 모두 포함, 역변환 placeholder → 역할 분리/확장 어려움.

10. 추가 기능
   - 레거시: auto-mask tie-breaking (random noise), clipping mean+std 기반, vignetting scale별 캐싱.
   - 현행: clipping 존재하나 tie-break noise 없음, vignette 미도입.

통합/개선 액션 아이템 (우선순)
A. 정확도
   1) FisheyeCamera.reconstruct(): LUT 기반 r_d→θ + (옵션) 1~2회 Newton refinement.
   2) 멀티스케일 시 LUT 재샘플 혹은 r_d 정규화 함수 정의.
B. 데이터 파이프라인
   3) Dataset에서 (theta_lut, angle_lut) 또는 (radial LUT) 배치 텐서화 → dataloader pin_memory True.
   4) distortion_coeffs 구조 표준화 (k[7], s, div, ux, uy) dtype/shape 체크 유닛테스트.
C. 손실/마스킹
   5) vignetting mask 적용 옵션 추가 (config 플래그).
   6) ego/car mask (차량 전경) 선택적 적용.
   7) automask tie-break noise + identity loss 분리 로깅.
D. Intrinsics/Scaling
   8) s, div 스케일링 원리(픽셀 크기 변화시 방사 좌표) 수식화 후 테스트 (단위 스텝 예상 투영 편차 허용치 설정).
E. 디버그/검증
   9) 재투영 에러 맵(pseudo GT 없이)과 각도 오차(θ_pred vs θ_LUT) 통계 로그.
   10) 단일 배치 synthetic sphere/radial grid 투영 round-trip 오차 테스트 스크립트.
F. 리팩터
   11) FisheyeRayModel (LUT & 역변환 전용) 분리 후 FisheyeCamera가 이를 조합.
   12) warp_ref_image 인자명/shape 검증 + 단위테스트.

리스크 미해결 시 예상 영향
- 깊이 경계 흐림, 주변부 구조 왜곡, pose scale drift, 학습 초기 loss plateau.
- 멀티스케일 불일치로 scale-invariant metric(silog 등) 악화.

단기 실행 순서 제안 (스프린트 1)
1) LUT 적용 reconstruct 교체 + 단일 배치 round-trip 테스트.
2) warp_ref_image 인자 정리 / 단위테스트.
3) vignetting mask 적용 플래그 추가.
4) θ 오차 로깅 지표 추가.

스프린트 2
5) Newton refinement / hybrid.
6) s, div 스케일링 공식 검증.
7) automask 개선(tie-break + 로그).

스프린트 3
8) Ego/car mask 통합.
9) 리팩터 (RayModel 분리) 및 문서화.

요약 결론
- 레거시 코드 대비 가장 큰 격차는 정확한 역투영(LUT)과 마스크 기반 안정화 부재.
- 우선 LUT + 마스크 + 스케일 정합 확보 후 세부 개선(하이브리드 역변환, refinement) 진행 권장.

끝.