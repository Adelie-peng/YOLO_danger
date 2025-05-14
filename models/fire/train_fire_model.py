from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO

def train_fire_detection_model(
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 8,
    dataset_dir: str = 'datasets/fire',
    model_name: str = 'fire_detection_model_s',
) -> None:

   # 경로 설정
    dataset_path = Path(dataset_dir)
    yaml_path = dataset_path / "data.yaml"
    
    # 모델 생성 - YOLOv8s
    model = YOLO('yolov8s.pt')

    # 학습 파라미터 구성
    train_args = {
        'data': str(yaml_path),
        'imgsz': img_size,

        'epochs': epochs,
        'batch': batch_size,
        'workers': 8,
        'cache': True,
        'patience': 20,

        'pretrained': True,
        'single_cls': False,  # 화재, 연기 클래스 감지

        'name': model_name,
        'save': True,
        'save_period': 10,

        'device': 0,
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
