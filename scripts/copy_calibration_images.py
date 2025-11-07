#!/usr/bin/env python3
# copy_calibration_images.py
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_calibration_images(
    dataset_root,
    calibration_split_json,
    output_dir='outputs/calibration/calibration_images'
):
    """
    calibration_split.json에 지정된 이미지들을 output_dir로 복사합니다.
    
    Args:
        dataset_root: NCDB 데이터셋 루트 경로
        calibration_split_json: Calibration split JSON 파일 경로
        output_dir: 이미지를 복사할 출력 디렉토리
    
    Returns:
        복사된 이미지 경로 리스트
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # JSON 파일 로드
    with open(calibration_split_json, 'r') as f:
        split_data = json.load(f)
    
    print("\n" + "="*60)
    print(f"Calibration 이미지 복사")
    print("="*60)
    print(f"출력 디렉토리: {output_dir.absolute()}")
    print(f"총 {len(split_data)}개 이미지 복사 중...")
    
    image_list = []
    copied = 0
    failed = 0
    
    for entry in tqdm(split_data):
        base_dir = dataset_root / entry['dataset_root']
        stem = entry['new_filename']
        
        # 문자열로 변환 후 10자리 형식으로 패딩 (예: "1580" -> "0000001580")
        stem_str = str(stem)
        if stem_str.isdigit():
            stem_padded = stem_str.zfill(10)
        else:
            stem_padded = stem_str
        
        src_path = base_dir / 'image_a6' / f"{stem_padded}.png"
        
        # 파일명은 원본 그대로 사용 (10자리 형식 유지)
        dst_filename = f"{stem_padded}.png"
        dst_path = output_dir / dst_filename
        
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                image_list.append({
                    'original_path': str(src_path),
                    'copied_path': str(dst_path),
                    'filename': dst_filename
                })
                copied += 1
            except Exception as e:
                print(f"\n⚠️ 복사 실패: {src_path} -> {e}")
                failed += 1
        else:
            print(f"\n⚠️ 파일 없음: {src_path}")
            failed += 1
    
    # 이미지 경로 리스트 저장 (절대 경로)
    list_file = output_dir / 'image_list.txt'
    with open(list_file, 'w') as f:
        for img in image_list:
            f.write(f"{Path(img['copied_path']).absolute()}\n")
    
    # 간단한 파일명 리스트도 저장 (상대 경로)
    simple_list_file = output_dir / 'image_filenames.txt'
    with open(simple_list_file, 'w') as f:
        for img in image_list:
            f.write(f"{img['filename']}\n")
    
    # 메타데이터 JSON 저장
    meta_file = output_dir / 'calibration_metadata.json'
    with open(meta_file, 'w') as f:
        json.dump(image_list, f, indent=2)
    
    print("\n" + "="*60)
    print("복사 완료!")
    print("="*60)
    print(f"✅ 성공: {copied}개")
    print(f"❌ 실패: {failed}개")
    print(f"\n생성된 파일:")
    print(f"  - 이미지 디렉토리: {output_dir.absolute()}")
    print(f"  - 절대 경로 리스트: {list_file}")
    print(f"  - 파일명 리스트: {simple_list_file}")
    print(f"  - 메타데이터: {meta_file}")
    print("="*60)
    
    return image_list

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Copy calibration images to a separate folder'
    )
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='NCDB 데이터셋 루트 경로')
    parser.add_argument('--calibration_split', type=str, 
                        default='outputs/calibration/calibration_split.json',
                        help='Calibration split JSON 파일')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs/calibration/calibration_images',
                        help='이미지를 복사할 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 이미지 복사 실행
    copy_calibration_images(
        dataset_root=args.dataset_root,
        calibration_split_json=args.calibration_split,
        output_dir=args.output_dir
    )
