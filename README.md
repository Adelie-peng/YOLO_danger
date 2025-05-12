# ⚠ YOLO Danger Detection

YOLO를 활용한 위험물 감지 서비스

※ 해당 프로젝트에서 감지하는 위험물은 총기류, 도검류, 화재 3가지로 지정함.

## 환경 설정

최초 환경설정 시:
```bash
conda env create -f environment.yml
conda activate py39_yolo_danger
conda env export --from-history > environment.yml
# pip install -r requirements.txt 명령어와 유사하나, CUDA 정보가 포함되어 있음
```


## GitHub 사용 가이드 (팀원용)

### 1. 저장소를 로컬에 클론
```bash
git clone https://github.com/Adelie-peng/YOLO_danger.git
```

### 2. 브랜치 생성
작업하고자 하는 기능에 대해 브랜치를 생성하여 작업해 주세요.
```bash
# 새 브랜치 생성 및 전환
git checkout -b [branch-name]

# 예: 새 기능 개발 시
git checkout -b feature/object-detection

# 예: 버그 수정 시
git checkout -b fix/memory-leak
```

### 3. 변경사항 커밋
```bash
# 변경된 파일 확인
git status

# 모든 변경사항을 스테이징
git add .

# 또는 특정 파일만 스테이징
git add [filename]

# 커밋 생성 (메시지 예시)
git commit -m "chore: 총기류 감지 모델 추가"
```

### 4. requirements.txt 갱신
새 패키지를 설치한 경우, PUSH 하기 전에 environment.yml를 갱신해 주세요.
```bash
# 기존 환경 업데이트 (새 패키지만 추가)
conda env update -f environment.yml
```

### 5. 변경사항 PUSH
```bash
# 첫 푸시
git push -u origin [branch-name]

# 이후 푸시
git push
```
