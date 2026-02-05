# GPU 지원 PyTorch 설치 가이드

## 문제 상황
- 현재 Python 3.14.2를 사용 중
- PyTorch는 Python 3.14에 대해 CUDA 버전을 공식 지원하지 않음
- CPU 버전만 설치 가능한 상태

## 해결 방법

### 방법 1: Python 3.12로 다운그레이드 (권장)

1. Python 3.12 설치
   - https://www.python.org/downloads/
   - Python 3.12.x 다운로드 및 설치

2. 새로운 가상환경 생성
   ```powershell
   # 기존 .venv 삭제 또는 이름 변경
   Rename-Item .venv .venv_old
   
   # Python 3.12로 새 가상환경 생성
   python3.12 -m venv .venv
   
   # 가상환경 활성화
   .\.venv\Scripts\Activate.ps1
   
   # 패키지 설치
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   
   # PyTorch GPU 버전 설치 (CUDA 12.1)
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. GPU 확인
   ```powershell
   python check_gpu.py
   ```

### 방법 2: 현재 환경 유지 (CPU 사용)

GPU를 사용하지 않고 CPU로만 작동하도록 코드를 유지합니다.
- 감성 분석 속도는 느리지만 작동은 정상적으로 됩니다.
- 대량의 뉴스 분석 시 시간이 오래 걸릴 수 있습니다.

### 방법 3: 조건부 GPU 사용

코드에서 GPU를 사용할 수 없을 때 자동으로 CPU로 폴백하도록 이미 구현되어 있습니다.
- `sentiment_analyzer.py`: GPU 없으면 키워드 기반 분석 사용
- `news_analysis.py`: GPU 없으면 CPU로 자동 전환

## 참고사항

### PyTorch 공식 지원 Python 버전
- Python 3.9, 3.10, 3.11, 3.12
- Python 3.13, 3.14는 아직 CUDA 버전 미지원

### CUDA 버전 확인
```powershell
nvidia-smi
```
현재 시스템: CUDA 12.6 (PyTorch CUDA 12.1 사용 권장)

### RTX 3060 사양
- VRAM: 12GB
- CUDA Cores: 3584
- 감성 분석 모델 실행에 충분한 성능
