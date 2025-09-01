# PM 업무 기록 (2025년 8월 29일)

## KITTI vs NCDB Dataset Context 처리 비교 분석

### 1. **KITTI Dataset의 Context 처리 방식**

#### **KITTI의 장점들**:
- **사전 검증 (Pre-validation)**: 실제 파일 존재 확인으로 런타임 에러 방지
- **체계적인 Context 관리**: `backward_context`, `forward_context` 파라미터로 명확한 설정
- **파일 기반 인덱싱**: 파일명에서 직접 인덱스 추출 (`'000001.png' → 1`)
- **캐싱 시스템**: 폴더별 파일 수 캐싱으로 반복 작업 최적화
- **견고한 에러 처리**: 경계 체크 및 파일 존재 확인

#### **KITTI의 핵심 코드 구조**:
```python
def _get_sample_context(self, sample_name, backward_context, forward_context, stride=1):
    """
    KITTI의 Context 검증 방식
    """
    base, ext = os.path.splitext(os.path.basename(sample_name))
    f_idx = int(base)  # 파일명 기반 인덱싱
    
    # 캐싱된 파일 수 확인
    if parent_folder in self._cache:
        max_num_files = self._cache[parent_folder]
    else:
        max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
        self._cache[parent_folder] = max_num_files
    
    # 경계 체크
    if (f_idx - backward_context * stride) < 0 or \
       (f_idx + forward_context * stride) >= max_num_files:
        return None, None
    
    # 실제 파일 존재 확인
    for offset in range(-backward_context, forward_context + 1):
        if offset != 0:
            context_idx = f_idx + offset * stride
            filename = self._get_next_file(context_idx, sample_name)
            if os.path.exists(filename):  # 실제 파일 존재 확인
                # 유효한 Context만 추가
```

### 2. **NCDB Dataset의 현재 문제점**

#### **NCDB의 단점들**:
- **단순 인덱스 기반 접근**: `context_idx = idx + offset`만으로 범위 체크
- **런타임 에러 처리**: 실제 파일 존재 확인 없이 로드 시도
- **캐싱 시스템 부재**: 매번 파일 존재 확인
- **유연하지만 불안정한 구조**: JSON 기반 메타데이터 활용

#### **NCDB의 현재 Context 처리**:
```python
# 범위 체크만 하고 실제 파일 존재 확인 안 함
for offset in [-2, -1, 1, 2]:
    context_idx = idx + offset
    if 0 <= context_idx < len(self.data_entries):
        try:
            context_img = load_image(context_idx)
            # 에러 시 그냥 건너뜀
        except Exception as e:
            continue
```

### 3. **주요 차이점 비교**

| 측면 | KITTI 방식 | NCDB 현재 방식 | 차이점 |
|------|------------|----------------|--------|
| **파일 검증** | 실제 파일 존재 확인 | 범위만 체크 | KITTI가 더 안전 |
| **Context 저장** | 별도 리스트에 저장 | 실시간 계산 | KITTI가 더 효율적 |
| **에러 처리** | 사전 검증 | 런타임 에러 | KITTI가 더 견고 |
| **인덱싱** | 파일명 기반 | 리스트 인덱스 | KITTI가 더 정확 |
| **캐싱** | 폴더별 파일 수 캐싱 | 없음 | KITTI가 더 빠름 |

### 4. **KITTI 방식의 장점**

#### **1. 사전 검증으로 안정성 확보**:
```python
# 데이터 로드 전에 Context 유효성 검증
if self.with_context:
    paths_with_context = []
    for stride in strides:
        for idx, file in enumerate(self.paths):
            backward_context, forward_context = self._get_sample_context(
                file, self.backward_context, self.forward_context, stride)
            if backward_context is not None:  # 유효한 Context만 추가
                paths_with_context.append(file)
    self.paths = paths_with_context  # 유효한 파일만 남김
```

#### **2. 캐싱으로 효율성 향상**:
```python
# 폴더별 파일 수 캐싱
if parent_folder in self._cache:
    max_num_files = self._cache[parent_folder]
else:
    max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
    self._cache[parent_folder] = max_num_files
```

#### **3. 파일 기반 정확한 인덱싱**:
```python
# 파일명에서 직접 인덱스 추출
base, ext = os.path.splitext(os.path.basename(sample_name))
f_idx = int(base)  # '000001.png' → 1
```

### 5. **NCDB Dataset 개선 제안**

#### **KITTI 방식 적용 방안**:
```python
# ncdb_dataset.py - KITTI 방식으로 개선
class NcdbDataset(Dataset):
    def __init__(self, ..., back_context=0, forward_context=0, strides=(1,)):
        # 기존 초기화...
        
        # KITTI 방식의 Context 관리 추가
        self.backward_context = back_context
        self.forward_context = forward_context
        self.with_context = (back_context != 0 or forward_context != 0)
        self.strides = strides
        
        # Context 경로 캐싱
        self.backward_context_paths = []
        self.forward_context_paths = []
        
        # 파일 존재 캐시
        self._file_cache = {}
        
        # Context가 있는 경우 사전 검증
        if self.with_context:
            self._filter_paths_with_context()
    
    def _filter_paths_with_context(self):
        """KITTI 방식: Context가 유효한 경로만 필터링"""
        paths_with_context = []
        
        for stride in self.strides:
            for idx, entry in enumerate(self.data_entries):
                backward_context, forward_context = self._get_sample_context(
                    idx, self.backward_context, self.forward_context, stride)
                
                if backward_context is not None:
                    paths_with_context.append(entry)
                    self.backward_context_paths.append(backward_context)
                    self.forward_context_paths.append(forward_context)
        
        self.data_entries = paths_with_context
```

### 6. **결론 및 추천**

#### **KITTI 방식의 우수성**:
- ✅ **안전성**: 실제 파일 존재 확인으로 런타임 에러 방지
- ✅ **효율성**: 사전 검증으로 불필요한 연산 감소
- ✅ **견고성**: 캐싱으로 반복 작업 최적화
- ✅ **정확성**: 파일명 기반 인덱싱으로 정확한 Context 선택

#### **NCDB에 KITTI 방식 적용의 이점**:
- **데이터 품질 향상**: 유효하지 않은 Context 제거
- **학습 안정성**: 런타임 에러 감소
- **성능 향상**: 캐싱으로 로딩 속도 개선
- **확장성**: 다양한 stride 지원

#### **실행 계획**:
1. **Phase 1**: NCDB Dataset에 KITTI 방식의 기본 구조 추가
2. **Phase 2**: 사전 검증 로직 구현
3. **Phase 3**: 캐싱 시스템 도입
4. **Phase 4**: 테스트 및 검증

**결론**: KITTI의 Context 처리 방식을 NCDB에 적용하는 것이 **더 나은 선택**입니다. 현재 NCDB 방식은 너무 단순해서 에러가 많이 발생할 수 있고, KITTI 방식은 검증된 Self-supervised learning에 적합한 구조를 가지고 있습니다.

---

## 다음 단계 작업 계획

### 1. **NCDB Dataset 개선 구현**
- KITTI 방식의 Context 처리 로직 적용
- 사전 검증 시스템 추가
- 캐싱 메커니즘 구현

### 2. **테스트 및 검증**
- 개선된 NCDB Dataset으로 Self-supervised 학습 테스트
- Context 처리 정확도 검증
- 성능 비교 분석

### 3. **문서화 및 공유**
- 개선 사항 문서화
- 팀 내 공유 및 피드백 수집