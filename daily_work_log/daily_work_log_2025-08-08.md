
# 2025-08-08

## 작업 1: 학습 스크립트 `TypeError` 버그 수정

- **시간:** 10:00 AM
- **요청:** `debug.txt`에 기록된 `TypeError: forward() got an unexpected keyword argument 'masks'` 에러 해결
- **작업 내용:**
    - `packnet_sfm/models/SfmModel.py` 파일 수정
    - `forward`, `compute_depth_net`, `depth_net_flipping` 함수 시그니처에 `**kwargs`를 추가하여 `masks`와 같은 추가 인자를 받을 수 있도록 변경
    - `depth_net_flipping` 함수에서 `self.depth_net`을 호출할 때 `**kwargs`를 전달하여 `masks` 인자가 최종적으로 네트워크 모델에 전달되도록 수정
- **결과:**
    - 학습 시 발생하던 `TypeError` 문제 해결
    - 이제 모델이 `masks` 인자를 정상적으로 처리하여 학습이 원활하게 진행될 것으로 예상됨
