@echo off
echo YOLO Danger 프로젝트 환경 설정을 시작합니다.

echo PyTorch CUDA 버전 설치 중...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
echo PyTorch 2.1.0, Torchvision 0.16.0, Torchaudio 2.1.0 설치를 완료했습니다.

echo Ultralytics 및 기타 패키지 설치 중...
pip install ultralytics
pip install -r requirements.txt
echo Ultralytics 외에 호환성 패키지를 설치 완료했습니다.

echo CUDA 가용성을 확인합니다.
python -c "import torch; print(f'CUDA 사용 가능 여부: {torch.cuda.is_available()}')"
pause
