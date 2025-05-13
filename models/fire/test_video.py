import os
import cv2
import torch
import subprocess
from ultralytics import YOLO
import time

# 설정값 (하드코딩)
VIDEO_URL = "https://www.youtube.com/watch?v=eXdsXemQL_c"
MODEL_PATH = "runs/detect/fire_detection_model3/weights/best.pt"
OUTPUT_FILE = "temp_video.mp4"
RESULT_FILE = f"detection_result_{int(time.time())}.mp4"

# 클래스 이름 매핑
CLASS_NAMES = {
    0: "fire",
    1: "smoke"
}

def main():
    # 1. GPU 확인
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    # 2. 유튜브 영상 다운로드
    print(f"영상 다운로드 중...")
    subprocess.run([
        "yt-dlp", 
        "-f", "best[height<=720]",
        "-o", OUTPUT_FILE,
        VIDEO_URL
    ])
    
    # 3. 모델 로드
    print(f"모델 로드 중...")
    model = YOLO(MODEL_PATH)
    
    # 4. 비디오 열기
    cap = cv2.VideoCapture(OUTPUT_FILE)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return
    
    # 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 5. 결과 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    out = cv2.VideoWriter(RESULT_FILE, fourcc, fps, (frame_width, frame_height))
    print(f"결과 영상을 {RESULT_FILE}에 저장합니다.")
    
    # 6. 프레임 처리 및 감지
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 화재/연기 감지
        results = model(frame, conf=0.3)
        
        # 감지 결과 표시
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 클래스 ID와 신뢰도
                cls_id = int(box.cls.item())
                conf = box.conf.item()

                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 클래스별 색상 설정 (fire는 빨간색, smoke는 파란색)
                color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 클래스 이름과 신뢰도 표시
                class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                label = f"{class_name}: {conf:.2f}"
                
                # 라벨 텍스트 그리기
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # 결과 프레임 저장
        out.write(frame)

        # 결과 화면 표시
        cv2.imshow('화재/연기 감지', frame)
        
        # q 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 임시 파일 삭제
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print(f"감지 결과 저장 완료: {RESULT_FILE}")

if __name__ == "__main__":
    main()
