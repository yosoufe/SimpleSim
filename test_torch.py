#!/usr/bin/env python3

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        print(f"✅ Current CUDA device: {torch.cuda.current_device()}")
        print(f"✅ CUDA device name: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA is not available")
    
    # Test basic tensor operations
    x = torch.randn(3, 3)
    print(f"✅ Basic tensor creation: {x.shape}")
    
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        print(f"✅ CUDA tensor creation: {x_cuda.shape}")
        print(f"✅ CUDA tensor device: {x_cuda.device}")
    
    print("✅ PyTorch installation successful!")
    
except ImportError as e:
    print(f"❌ Failed to import torch: {e}")
except Exception as e:
    print(f"❌ Error testing PyTorch: {e}") 