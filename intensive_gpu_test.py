import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import threading

# GPU monitoring function
def monitor_gpu(stop_event, interval=0.5):
    """Monitor GPU usage in real-time"""
    print("\n" + "="*70)
    print("🔍 GPU MONITORING STARTED (Press Ctrl+C to stop training)")
    print("="*70)
    print(f"{'Time':<10} {'GPU Util':<12} {'Memory Used':<15} {'Temperature':<12}")
    print("-" * 70)
    
    while not stop_event.is_set():
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            
            # Get utilization (note: this is approximate)
            current_time = time.strftime("%H:%M:%S")
            print(f"{current_time:<10} {'Active':<12} {mem_allocated:>6.0f} MB / {mem_reserved:>5.0f} MB Reserved", end="\r")
        
        time.sleep(interval)

# LSTM model
class TestLSTM(nn.Module):
    def __init__(self, input_size=95, hidden_size=256, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        return self.fc3(out)

print("="*70)
print("🧠 INTENSIVE GPU LSTM TRAINING TEST")
print("="*70)
print("\nThis will train an LSTM model on your GPU for 2 minutes.")
print("You should see GPU usage spike in:")
print("  1. Task Manager → Performance → GPU")
print("  2. Another terminal running: nvidia-smi -l 1")
print("  3. NVIDIA App (if installed)")
print("\nStarting in 3 seconds...")
time.sleep(3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

if device.type == 'cpu':
    print("❌ No GPU available! Exiting...")
    exit()

# Create large dataset to stress GPU
batch_size = 64
seq_length = 24
input_size = 95
num_samples = 10000

print(f"\nGenerating {num_samples:,} training samples...")
X = torch.randn(num_samples, seq_length, input_size)
y = torch.randn(num_samples, 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
print(f"Initializing LSTM model...")
model = TestLSTM(input_size, 256, 3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(dataloader)}")

# Start GPU monitoring in separate thread
stop_monitoring = threading.Event()
monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_monitoring,))
monitor_thread.start()

# Training loop
epochs = 20
print(f"\n{'='*70}")
print(f"🔥 STARTING INTENSIVE GPU TRAINING ({epochs} epochs)")
print(f"{'='*70}\n")

try:
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        mem_used = torch.cuda.memory_allocated(0) / 1024**2
        
        print(f"\nEpoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.6f}, GPU Mem={mem_used:.0f} MB")
        
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")

# Stop monitoring
stop_monitoring.set()
monitor_thread.join()

# Final stats
print("\n" + "="*70)
print("📊 FINAL GPU STATISTICS")
print("="*70)
print(f"Peak GPU Memory Used: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
print(f"Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
print("="*70)

print("\n✅ If you saw GPU memory usage above, PyTorch IS using your GPU!")
print("\n💡 To see GPU utilization percentage:")
print("   1. Open Task Manager → Performance → GPU")
print("   2. Or run in another terminal: nvidia-smi -l 1")
print("   3. Look for '3D' or 'Compute' usage (not 'Video Decode')")
