
import json
import random
from pathlib import Path

def create_combined_splits(dataset_a_root, dataset_b_root, output_dir,
                           train_ratio=0.95, val_ratio=0.025, test_ratio=0.025):
    """
    두 개의 데이터셋 mapping_data.json을 통합하여 train/val/test 스플릿을 생성합니다.

    Parameters
    ----------
    dataset_a_root : str
        첫 번째 데이터셋의 루트 디렉토리 경로 (예: '/workspace/packnet-sfm/ncdb-cls/synced_data')
    dataset_b_root : str
        두 번째 데이터셋의 루트 디렉토리 경로 (예: '/workspace/packnet-sfm/ncdb-cls-sample/synced_data')
    output_dir : str
        새로운 스플릿 JSON 파일이 저장될 디렉토리 경로 (예: '/workspace/packnet-sfm/splits')
    train_ratio : float
        학습 데이터 비율
    val_ratio : float
        검증 데이터 비율
    test_ratio : float
        테스트 데이터 비율
    """
    dataset_a_root = Path(dataset_a_root)
    dataset_b_root = Path(dataset_b_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Dataset A 로드 및 root_dir 추가
    mapping_a_path = dataset_a_root / 'mapping_data.json'
    if not mapping_a_path.exists():
        raise FileNotFoundError(f"Mapping file not found at {mapping_a_path}")
    with open(mapping_a_path, 'r') as f:
        data_a = json.load(f)
        for item in data_a:
            item['dataset_root'] = str(dataset_a_root)
            all_data.append(item)

    # Dataset B 로드 및 root_dir 추가
    mapping_b_path = dataset_b_root / 'mapping_data.json'
    if not mapping_b_path.exists():
        raise FileNotFoundError(f"Mapping file not found at {mapping_b_path}")
    with open(mapping_b_path, 'r') as f:
        data_b = json.load(f)
        for item in data_b:
            item['dataset_root'] = str(dataset_b_root)
            all_data.append(item)

    # 데이터 섞기
    random.shuffle(all_data)

    # 스플릿 비율 계산
    total_samples = len(all_data)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # 데이터 분할
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    # 새로운 JSON 파일로 저장
    with open(output_dir / 'combined_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(output_dir / 'combined_val.json', 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(output_dir / 'combined_test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"Combined splits created in {output_dir}:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")

if __name__ == '__main__':
    # 예시 사용법 (실제 경로에 맞게 수정하세요)
    dataset_a_path = '/workspace/data/ncdb-cls/2025-07-11_15-00-27_410410_A/synced_data' # 예시 경로, 실제 경로로 변경 필요
    dataset_b_path = '/workspace/data/ncdb-cls/2025-07-11_15-39-30_243127_B/synced_data' # 예시 경로, 실제 경로로 변경 필요
    output_splits_dir = '/workspace/data/ncdb-cls/splits' # 예시 경로, 실제 경로로 변경 필요

    create_combined_splits(dataset_a_path, dataset_b_path, output_splits_dir)
