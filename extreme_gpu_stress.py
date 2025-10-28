import torch
import torch.nn as nn
import time

print("="*70)
print("🔥 EXTREME GPU STRESS TEST - Your GPU Will Actually Work Now!")
print("="*70)

if not torch.cuda.is_available():
    print("❌ No GPU available!")
    exit()

device = torch.device('cuda')
print(f"\n✅ Using: {torch.cuda.get_device_name(0)}")
print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n" + "="*70)
print("🎯 TASK: Train MASSIVE LSTM that will use 80-90% of GPU")
print("="*70)
print("\n📊 Open Task Manager → Performance → GPU now!")
print("   Or run in another terminal: nvidia-smi -l 1")
print("\nStarting in 5 seconds...")
time.sleep(5)

# Create HUGE LSTM
class MassiveLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 1024, 4, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(1024, 1024, 4, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out = out2[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        return self.fc3(out)

print("\n🏗️  Building MASSIVE model...")
model = MassiveLSTM().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,} (~{params/1e6:.1f}M)")
print(f"   Model size: ~{params * 4 / 1024**2:.0f} MB")

# Create HUGE dataset
batch_size = 256  # Large batch
seq_length = 128   # Long sequences
input_size = 512   # Many features
num_batches = 50

print(f"\n📦 Creating massive batches:")
print(f"   Batch size: {batch_size}")
print(f"   Sequence length: {seq_length}")
print(f"   Input features: {input_size}")
print(f"   Total batches: {num_batches}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n" + "="*70)
print("🔥🔥🔥 STARTING EXTREME TRAINING - GPU SHOULD HIT 80-90%! 🔥🔥🔥")
print("="*70)
print("\n👀 WATCH YOUR GPU USAGE SPIKE NOW!\n")

start_time = time.time()

for batch_idx in range(num_batches):
    # Generate huge batch
    X = torch.randn(batch_size, seq_length, input_size).to(device)
    y = torch.randn(batch_size, 1).to(device)
    
    # Forward pass
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Stats
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"Batch {batch_idx+1:2d}/{num_batches}: Loss={loss.item():.4f}, "
          f"GPU Mem={mem_allocated:.2f} GB / {mem_reserved:.2f} GB reserved", 
          end="\r")
    
    if (batch_idx + 1) % 10 == 0:
        print()  # New line every 10 batches

elapsed = time.time() - start_time

print("\n\n" + "="*70)
print("✅ STRESS TEST COMPLETE!")
print("="*70)
print(f"⏱️  Time: {elapsed:.1f} seconds")
print(f"💾 Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
print(f"🌡️  Check GPU temperature in nvidia-smi!")
print("="*70)

print("\n🎯 DID YOU SEE GPU USAGE SPIKE TO 80-90%?")
print("\nIf not, your GPU is TOO POWERFUL! 🚀")
print("That means:")
print("  ✅ Your LSTM Bitcoin predictor will train SUPER FAST")
print("  ✅ You could train WAY bigger models")
print("  ✅ GPU is working perfectly, just underutilized")
print("\n💡 For Bitcoin predictor, 3-4% usage is NORMAL and FINE!")
print("   The model completes in milliseconds per batch.")
