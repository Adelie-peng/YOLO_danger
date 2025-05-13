from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import numpy as np

def test_fire_detection_model(
    weights_path: str = 'runs/detect/fire_detection_model3/weights/best.pt',
    test_data: str = 'datasets/fire/test/images',
    conf_threshold: float = 0.3,  # 화재 감지의 경우 오탐지보다 누락이 더 위험하므로 낮은 값으로 설정
    save_visuals: bool = True,
    batch_size: int = 16
) -> dict:
    """
    화재 감지 모델을 테스트하고 성능을 평가합니다.
    
    Args:
        weights_path: 학습된 모델 가중치 경로
        test_data: 테스트 이미지 경로
        conf_threshold: 신뢰도 임계값 (0-1)
        save_visuals: 시각화 결과 저장 여부
        batch_size: 배치 크기
    
    Returns:
        평가 결과를 담은 딕셔너리
    """
    # 결과 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/fire_test_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    model = YOLO(weights_path)
    
    # 1. 성능 평가 실행
    metrics = model.val(
        data=Path('datasets/fire/data.yaml'),
        split='test',
        imgsz=640,
        batch=batch_size,
        conf=conf_threshold,
        device=0,
        verbose=False
    )
    
    # 2. 시각화 결과 생성
    if save_visuals:
        predictions = model.predict(
            source=test_data,
            conf=conf_threshold,
            save=True,
            project=results_dir,
            name="visuals",
            stream=False,
            verbose=False
        )
        
        # 예측 결과 경로 출력
        pred_path = results_dir / "visuals"
        print(f"시각화 결과가 저장된 경로: {pred_path}")
    
     # 3. 결과 요약 - 배열 처리
    class_names = model.names
    
    # 전체 평균 계산 (클래스별 값의 평균)
    mean_precision = np.mean(metrics.box.p) if isinstance(metrics.box.p, np.ndarray) else metrics.box.p
    mean_recall = np.mean(metrics.box.r) if isinstance(metrics.box.r, np.ndarray) else metrics.box.r
    mean_f1 = np.mean(metrics.box.f1) if isinstance(metrics.box.f1, np.ndarray) else metrics.box.f1
    
    results = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1-score": mean_f1,
        "classes": {i: name for i, name in enumerate(class_names)},
        # 클래스별 결과도 저장
        "class_metrics": {
            "precision": metrics.box.p.tolist() if isinstance(metrics.box.p, np.ndarray) else [metrics.box.p],
            "recall": metrics.box.r.tolist() if isinstance(metrics.box.r, np.ndarray) else [metrics.box.r],
            "f1-score": metrics.box.f1.tolist() if isinstance(metrics.box.f1, np.ndarray) else [metrics.box.f1]
        }
    }
    
    # 4. 주요 지표 출력
    print(f"\n{'='*50}")
    print(f"모델 테스트 결과 요약:")
    print(f"{'='*50}")
    print(f"mAP50: {results['mAP50']:.4f}")
    print(f"mAP50-95: {results['mAP50-95']:.4f}")
    print(f"평균 Precision: {results['precision']:.4f}")
    print(f"평균 Recall: {results['recall']:.4f}")
    print(f"평균 F1-Score: {results['f1-score']:.4f}")
    
    # 클래스별 지표 출력
    print(f"\n클래스별 결과:")
    for i, class_name in enumerate(class_names):
        if i < len(results["class_metrics"]["precision"]):
            print(f"  {class_name}: Precision={results['class_metrics']['precision'][i]:.4f}, "
                  f"Recall={results['class_metrics']['recall'][i]:.4f}, "
                  f"F1-Score={results['class_metrics']['f1-score'][i]:.4f}")
    print(f"{'='*50}")
    
    # 평가 결과를 텍스트 파일로 저장
    with open(results_dir / "metrics.txt", "w") as f:
        f.write(f"mAP50: {results['mAP50']}\n")
        f.write(f"mAP50-95: {results['mAP50-95']}\n")
        f.write(f"평균 Precision: {results['precision']}\n")
        f.write(f"평균 Recall: {results['recall']}\n")
        f.write(f"평균 F1-Score: {results['f1-score']}\n\n")
        
        f.write("클래스별 결과:\n")
        for i, class_name in enumerate(class_names):
            if i < len(results["class_metrics"]["precision"]):
                f.write(f"  {class_name}: Precision={results['class_metrics']['precision'][i]:.4f}, "
                       f"Recall={results['class_metrics']['recall'][i]:.4f}, "
                       f"F1-Score={results['class_metrics']['f1-score'][i]:.4f}\n")
    
    return results

if __name__ == "__main__":
    test_fire_detection_model()
