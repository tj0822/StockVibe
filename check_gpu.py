import torch

print("="*50)
print("PyTorch GPU 설정 확인")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("⚠️ CUDA를 사용할 수 없습니다!")
    print("PyTorch가 CPU 전용으로 설치되어 있을 수 있습니다.")
    print("GPU를 사용하려면 PyTorch CUDA 버전을 설치하세요:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("="*50)
