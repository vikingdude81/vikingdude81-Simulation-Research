import torch

print("\n" + "="*70)
print("🔍 GPU AVAILABILITY CHECK")
print("="*70 + "\n")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test tensor on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"\n✅ GPU tensor operations working!")
    print(f"   Test tensor device: {z.device}")
else:
    print("\n⚠️  CUDA not available - will use CPU")
    print("   For GPU acceleration, install: pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*70 + "\n")
