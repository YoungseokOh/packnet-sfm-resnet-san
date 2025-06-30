import torch
from ultralytics import YOLO

def check_yolov8_models():
    """YOLOv8 모델 타입별 구조 비교"""
    
    print("🔍 Checking YOLOv8 Model Types")
    print("=" * 60)
    
    # 1. COCO Detection Model
    try:
        det_model = YOLO('yolov8s.pt')
        print("📋 COCO Detection Model (yolov8s.pt):")
        print(f"   Task: {det_model.task}")
        print(f"   Model type: {type(det_model.model)}")
        
        # Backbone features 확인
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            try:
                # YOLOv8 detection model의 backbone 접근
                backbone_features = []
                x = dummy_input
                
                # YOLOv8 detection model 구조 탐색
                for i, layer in enumerate(det_model.model.model[:10]):  # 처음 10개 레이어만
                    x = layer(x)
                    if i in [1, 2, 4, 6, 9]:  # 대략적인 feature extraction points
                        backbone_features.append(x.shape[1])  # 채널 수
                        if len(backbone_features) >= 5:
                            break
                
                print(f"   Backbone channels: {backbone_features}")
                print(f"   Has detection head: Yes")
                
            except Exception as e:
                print(f"   Feature extraction failed: {e}")
        
    except Exception as e:
        print(f"❌ COCO model error: {e}")
    
    print()
    
    # 2. ImageNet Classification Model  
    try:
        cls_model = YOLO('yolov8s-cls.pt')
        print("📋 ImageNet Classification Model (yolov8s-cls.pt):")
        print(f"   Task: {cls_model.task}")
        print(f"   Model type: {type(cls_model.model)}")
        
        # Classification model backbone 확인
        dummy_input = torch.randn(1, 3, 224, 224)  # ImageNet 표준 크기
        with torch.no_grad():
            try:
                backbone_features = []
                x = dummy_input
                
                # Classification model은 더 단순한 구조
                for i, layer in enumerate(cls_model.model.model[:10]):
                    x = layer(x)
                    if i in [1, 2, 4, 6, 9]:
                        backbone_features.append(x.shape[1])
                        if len(backbone_features) >= 5:
                            break
                
                print(f"   Backbone channels: {backbone_features}")
                print(f"   Has detection head: No")
                
            except Exception as e:
                print(f"   Feature extraction failed: {e}")
                
    except Exception as e:
        print(f"❌ ImageNet model error: {e}")

if __name__ == "__main__":
    check_yolov8_models()