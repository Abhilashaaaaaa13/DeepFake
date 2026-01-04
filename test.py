import torch

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("❌ GPU not detected. PyTorch is using CPU.")


import mediapipe as mp
print(mp.__version__)
print(dir(mp))
