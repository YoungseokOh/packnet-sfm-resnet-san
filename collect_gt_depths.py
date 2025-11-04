#!/usr/bin/env python3
"""
NPU 파일명을 기준으로 GT depth 파일 찾아서 복사
"""

import json
import os
import shutil
from pathlib import Path


def find_and_copy_gt_depths():
    """NPU 파일에 해당하는 GT depth 찾아서 복사"""
    
    print("=" * 90)
    print("🔍 NPU 파일에 대한 GT Depth 찾기 및 복사")
    print("=" * 90)
    print()
    
    # 경로 설정
    npu_folder = '/workspace/packnet-sfm/outputs/sigmoid_prediction_from_aiwbin_npu'
    splits_dir = '/workspace/data/ncdb-cls-640x384/splits'
    output_folder = '/workspace/packnet-sfm/outputs/sigmoid_prediction_GT'
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Output folder created: {output_folder}\n")
    
    # NPU 파일 목록 가져오기
    npu_files = sorted([f.name.replace('.npy', '') for f in Path(npu_folder).glob('*.npy')])
    print(f"📋 NPU files ({len(npu_files)}):")
    for i, fname in enumerate(npu_files, 1):
        print(f"   {i:2d}. {fname}")
    print()
    
    # 모든 split JSON 로드 (train, val, test)
    print(f"📖 Loading all split JSONs from: {splits_dir}")
    all_data = {}
    for split_name in ['combined_train.json', 'combined_val.json', 'combined_test.json']:
        split_path = os.path.join(splits_dir, split_name)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                split_data = json.load(f)
            print(f"   • {split_name}: {len(split_data)} entries")
            # new_filename을 키로 딕셔너리에 추가
            for item in split_data:
                all_data[item['new_filename']] = item
    print(f"   ✅ Total loaded: {len(all_data)} entries\n")
    
    # new_filename을 키로 하는 딕셔너리 (all_data로 변경)
    test_dict = all_data
    
    print("🔄 Finding and copying GT depth files...")
    print("-" * 90)
    
    found = 0
    not_found = 0
    copied = 0
    
    for npu_filename in npu_files:
        if npu_filename in test_dict:
            item = test_dict[npu_filename]
            
            # image_path에서 디렉토리 구조 파악
            # /workspace/data/ncdb-cls-640x384/2025-07-11_15-39-30_243127_B/synced_data/image_a6/0000000567.png
            image_path = item['image_path']
            
            # image_a6를 newest_depth_maps로 변경하고 .png를 유지
            gt_path = image_path.replace('/image_a6/', '/newest_depth_maps/')
            
            print(f"✓ {npu_filename}")
            print(f"  Image: {image_path}")
            print(f"  GT   : {gt_path}")
            
            if os.path.exists(gt_path):
                # GT 파일 복사
                output_path = os.path.join(output_folder, f"{npu_filename}.png")
                shutil.copy2(gt_path, output_path)
                print(f"  ✅ Copied to: {output_path}")
                copied += 1
            else:
                print(f"  ❌ GT file not found!")
                not_found += 1
            
            print()
            found += 1
        else:
            print(f"⚠️  {npu_filename}: Not found in test JSON")
            not_found += 1
            print()
    
    print("-" * 90)
    print(f"\n📊 Summary:")
    print(f"   • Total NPU files: {len(npu_files)}")
    print(f"   • Found in JSON: {found}")
    print(f"   • GT files copied: {copied}")
    print(f"   • Not found: {not_found}")
    print()
    
    # 복사된 파일 확인
    copied_files = sorted(list(Path(output_folder).glob('*.png')))
    if copied_files:
        print(f"📁 Copied GT files ({len(copied_files)}):")
        for i, f in enumerate(copied_files, 1):
            file_size = os.path.getsize(f) / 1024  # KB
            print(f"   {i:2d}. {f.name} ({file_size:.1f} KB)")
        print()
    
    print("=" * 90)
    print("✅ GT depth collection complete!")
    print(f"📁 GT files saved to: {output_folder}")
    print("=" * 90)


if __name__ == '__main__':
    find_and_copy_gt_depths()
