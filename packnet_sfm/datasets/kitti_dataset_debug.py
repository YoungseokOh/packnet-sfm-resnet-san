import os
from pathlib import Path

def debug_kitti_structure(data_path, split_file):
    """KITTI 데이터 구조 디버깅"""
    print(f"🔍 Debugging KITTI structure:")
    print(f"   Data path: {data_path}")
    print(f"   Split file: {split_file}")
    
    # 1. 기본 경로 확인
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return
    
    # 2. Split 파일 확인
    if not os.path.exists(split_file):
        print(f"❌ Split file does not exist: {split_file}")
        return
    
    # 3. Split 파일 내용 확인
    with open(split_file, 'r') as f:
        lines = f.readlines()[:5]  # 처음 5개만
    
    print(f"📋 First 5 lines from split file:")
    for i, line in enumerate(lines):
        print(f"   {i+1}: {line.strip()}")
    
    # 4. 실제 데이터 구조 확인
    print(f"\n📁 Actual data structure:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:  # 너무 깊지 않게
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # 처음 3개 파일만
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... and {len(files)-3} more files")
        if level > 4:  # 깊이 제한
            break
    
    # 5. 첫 번째 이미지 경로 실제 확인
    if lines:
        first_line = lines[0].strip().split()
        if len(first_line) >= 1:
            test_path = first_line[0]
            full_path = os.path.join(data_path, test_path)
            print(f"\n🎯 Testing first image path:")
            print(f"   Relative: {test_path}")
            print(f"   Full: {full_path}")
            print(f"   Exists: {os.path.exists(full_path)}")
            
            # 가능한 변형들 테스트
            variants = [
                test_path,
                test_path.replace('/', os.sep),  # OS 경로 구분자
                os.path.join(data_path, test_path.split('/')[0])  # 첫 번째 폴더만
            ]
            
            for variant in variants:
                variant_path = os.path.join(data_path, variant) if not os.path.isabs(variant) else variant
                if os.path.exists(variant_path):
                    print(f"   ✅ Found variant: {variant}")

# 사용법
debug_kitti_structure('/data/datasets/KITTI_raw', '/workspace/packnet-sfm/configs/train_resnet_san_kitti_tiny.yaml')