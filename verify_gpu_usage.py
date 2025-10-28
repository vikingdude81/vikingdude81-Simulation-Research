import torch
import time
import numpy as np

print("="*70)
print("🔍 GPU VERIFICATION & BENCHMARK TEST")
print("="*70)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   Device Count: {torch.cuda.device_count()}")
else:
    print("   ❌ No CUDA available - PyTorch will use CPU")
    exit()

# Memory before
print(f"\n2. GPU Memory Before Test:")
print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Test 1: Simple matrix multiplication
print(f"\n3. Running GPU Computation Test...")
print("   Creating large tensors and performing matrix multiplication...")

device = torch.device('cuda')
size = 5000

# CPU test
print("\n   a) CPU Test:")
cpu_start = time.time()
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - cpu_start
print(f"      Time: {cpu_time:.3f} seconds")

# GPU test
print("\n   b) GPU Test:")
torch.cuda.synchronize()  # Wait for GPU to be ready
gpu_start = time.time()
x_gpu = torch.randn(size, size).cuda()
y_gpu = torch.randn(size, size).cuda()
z_gpu = torch.matmul(x_gpu, y_gpu)
torch.cuda.synchronize()  # Wait for computation to complete
gpu_time = time.time() - gpu_start
print(f"      Time: {gpu_time:.3f} seconds")
print(f"      Speedup: {cpu_time/gpu_time:.1f}x faster than CPU!")

# Memory after
print(f"\n4. GPU Memory After Test:")
print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
print(f"   Peak Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

# Test 2: LSTM-like workload
print(f"\n5. Simulating LSTM Training Workload...")
print("   (This should make GPU usage spike in nvidia-smi)")

batch_size = 32
seq_length = 24
input_size = 95
hidden_size = 256

# Create dummy data
X = torch.randn(batch_size, seq_length, input_size).cuda()
y = torch.randn(batch_size, 1).cuda()

# Create LSTM
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True).cuda()
fc = torch.nn.Linear(hidden_size, 1).cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()))

print(f"   Training for 50 iterations (watch nvidia-smi now!)...")
print(f"   Run 'nvidia-smi' in another terminal to see GPU usage spike!\n")

for i in range(50):
    optimizer.zero_grad()
    lstm_out, _ = lstm(X)
    out = fc(lstm_out[:, -1, :])
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 10 == 0:
        print(f"      Iteration {i+1}/50: Loss = {loss.item():.6f}, "
              f"GPU Mem = {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

print(f"\n6. Final GPU Memory Stats:")
print(f"   Currently Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"   Peak Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

# Check GPU utilization
print(f"\n7. GPU Properties:")
props = torch.cuda.get_device_properties(0)
print(f"   Name: {props.name}")
print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
print(f"   Compute Capability: {props.major}.{props.minor}")
print(f"   Multi-Processor Count: {props.multi_processor_count}")

print("\n" + "="*70)
print("✅ GPU VERIFICATION COMPLETE!")
print("="*70)
print("\nIf you saw GPU usage in Task Manager or nvidia-smi, PyTorch is")
print("correctly using your GPU! If not, the PyTorch CPU version may be")
print("installed instead of the CUDA version.")
print("\nTip: Run 'nvidia-smi' in another terminal while this script runs")
print("     to see real-time GPU utilization!")
print("="*70)
