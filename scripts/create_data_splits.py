import json
import random
from pathlib import Path

def split_dataset():
    """
    Splits the mapping_data.json into train, validation, and test sets.
    """
    base_dir = Path('/workspace/packnet-sfm/ncdb-cls-sample/synced_data')
    input_json_path = base_dir / 'mapping_data.json'
    
    print(f"Reading data from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 데이터 구조 확인
    if isinstance(data, dict) and "image_a6" in data and "pcd" in data:
        print("Dictionary format detected with 'image_a6' and 'pcd' keys")
        # 딕셔너리 형식 처리 - 인덱스를 셔플
        image_paths = data["image_a6"]
        pcd_paths = data["pcd"]
        
        # 이미지와 PCD 경로 쌍의 인덱스 리스트 생성
        indices = list(range(min(len(image_paths), len(pcd_paths))))
        random.shuffle(indices)
        
        # 분할 크기 계산
        total_size = len(indices)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        
        # 인덱스 분할
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 분할 데이터 생성
        train_data = {"image_a6": [image_paths[i] for i in train_indices], 
                    "pcd": [pcd_paths[i] for i in train_indices]}
        val_data = {"image_a6": [image_paths[i] for i in val_indices], 
                   "pcd": [pcd_paths[i] for i in val_indices]}
        test_data = {"image_a6": [image_paths[i] for i in test_indices], 
                   "pcd": [pcd_paths[i] for i in test_indices]}
    
    elif isinstance(data, list):
        print("List format detected")
        # 리스트 형식 처리 - 원래 로직 사용
        random.shuffle(data)
        
        # 분할 크기 계산
        total_size = len(data)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        
        # 데이터 분할
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
    
    else:
        raise ValueError("Unexpected format in mapping_data.json. Expected either a dictionary with 'image_a6' and 'pcd' keys or a list of dictionaries.")
    
    # 출력 경로 정의
    train_json_path = base_dir / 'train.json'
    val_json_path = base_dir / 'val.json'
    test_json_path = base_dir / 'test.json'
    
    # 분할된 데이터를 새 파일에 쓰기
    if isinstance(data, dict):
        print(f"Writing {len(train_data['image_a6'])} items to {train_json_path}")
        print(f"Writing {len(val_data['image_a6'])} items to {val_json_path}")
        print(f"Writing {len(test_data['image_a6'])} items to {test_json_path}")
    else:
        print(f"Writing {len(train_data)} items to {train_json_path}")
        print(f"Writing {len(val_data)} items to {val_json_path}")
        print(f"Writing {len(test_data)} items to {test_json_path}")
    
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f, indent=4)
    
    with open(test_json_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print("\nDataset splitting complete.")
    if isinstance(data, dict):
        print(f"Total: {total_size} | Train: {len(train_indices)} | Validation: {len(val_indices)} | Test: {len(test_indices)}")
    else:
        print(f"Total: {total_size} | Train: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")

if __name__ == '__main__':
    split_dataset()
