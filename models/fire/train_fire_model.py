from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path

def train_fire_detection_model(
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 8,  # 16 -> 8
    dataset_dir: str = 'datasets/fire'
) -> None:

   # 경로 설정
    dataset_path = Path(dataset_dir)
    yaml_path = dataset_path / "data.yaml"
    
    # 모델 생성 - YOLOv8n -> YOLOv8s
    model = YOLO('yolov8s.pt')  # 다른 경량화 버전 참고: yolov8s.pt, yolov8m.pt

    # 파라미터를 구성
    train_args = {
        'data': str(yaml_path),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': 0,
        'name': 'fire_detection_model_s',
        'patience': 20,
        'pretrained': True,
        'save': True,
        'save_period': 10,
        'single_cls': False,  # 화재, 연기 클래스 감지
        'workers': 8,  # 데이터 로딩 워커 수 증가
        'cache': True,  # 데이터 캐싱 활성화
    }

    # 학습 실행
    results = model.train(**train_args)

    # 학습된 모델 평가
    model.val()

    # 결과 출력
    print(f"Training complete! Best mAP50-95: {results.maps[0]:.4f}")
    print(f"Model saved to: {Path('runs/detect/fire_detection_model')}")

if __name__ == '__main__':
    freeze_support()
    train_fire_detection_model()
