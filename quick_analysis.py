"""
Quick Model Analysis - Train a lightweight version and extract attention patterns
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔍 QUICK ATTENTION PATTERN ANALYSIS")
print("=" * 70)

# ============================================================================
# Load data
# ============================================================================

print("\n📊 Loading data...")
btc = yf.download('BTC-USD', period='90d', interval='1h', progress=False)

df = pd.DataFrame()
df['price'] = btc['Close']
df['returns'] = df['price'].pct_change()
df['volume'] = btc['Volume']
df['volatility'] = df['returns'].rolling(24).std()

df = df.dropna()
print(f"✅ Loaded {len(df)} hours")

# ============================================================================
# Prepare training data
# ============================================================================

print("\n🔧 Preparing sequences...")

seq_length = 48
features = ['returns', 'volume', 'volatility']

X_data = df[features].values
y_data = df['returns'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# Create sequences
X_sequences = []
y_targets = []

for i in range(len(X_scaled) - seq_length - 1):
    X_sequences.append(X_scaled[i:i+seq_length])
    y_targets.append(y_data[i+seq_length])

X_sequences = np.array(X_sequences)
y_targets = np.array(y_targets)

print(f"✅ Created {len(X_sequences)} sequences")
print(f"   Shape: {X_sequences.shape}")

# Convert to tensors
X_train = torch.FloatTensor(X_sequences[:-100])
y_train = torch.FloatTensor(y_targets[:-100]).unsqueeze(1)
X_val = torch.FloatTensor(X_sequences[-100:])
y_val = torch.FloatTensor(y_targets[-100:]).unsqueeze(1)

# ============================================================================
# Simple LSTM with Attention
# ============================================================================

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attention_weights

# ============================================================================
# Quick training
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

model = SimpleLSTM(input_size=len(features), hidden_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\n🚀 Training lightweight model (20 epochs)...")
print("   This will take ~30 seconds...")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    predictions, _ = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            val_pred, _ = model(X_val)
            val_loss = criterion(val_pred, y_val)
        print(f"   Epoch {epoch+1:2d}: Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}")

print("✅ Training complete!")

# ============================================================================
# Extract attention from validation samples
# ============================================================================

print("\n🎯 Extracting attention patterns from validation data...")

model.eval()
all_attentions = []

with torch.no_grad():
    for i in range(min(10, len(X_val))):
        _, attention = model(X_val[i:i+1])
        all_attentions.append(attention.cpu().numpy()[0, :, 0])

# Average attention across samples
avg_attention = np.mean(all_attentions, axis=0)
print(f"✅ Extracted attention from {len(all_attentions)} samples")

# ============================================================================
# Visualize
# ============================================================================

print("\n📊 Creating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('🎯 LSTM Attention Pattern Analysis - Trained Model', 
             fontsize=18, fontweight='bold', y=0.98)

# Time labels
time_labels = [f"{i}h" for i in range(seq_length-1, -1, -1)]

# ============================================================================
# Plot 1: Main attention distribution
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
colors = plt.cm.plasma(avg_attention / avg_attention.max())
bars = ax1.bar(range(seq_length), avg_attention, color=colors, 
               edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.set_xlabel('Hours Ago', fontsize=13, fontweight='bold')
ax1.set_ylabel('Attention Weight', fontsize=13, fontweight='bold')
ax1.set_title('Average Attention Distribution (10 validation samples)', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(range(0, seq_length, 4))
ax1.set_xticklabels([time_labels[i] for i in range(0, seq_length, 4)])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=avg_attention.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {avg_attention.mean():.5f}')
ax1.legend(fontsize=11)

# Highlight top values
top_5_idx = np.argsort(avg_attention)[-5:]
for idx in top_5_idx:
    ax1.text(idx, avg_attention[idx], f'{avg_attention[idx]:.4f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# ============================================================================
# Plot 2: Individual sample attention patterns
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
for i, attn in enumerate(all_attentions[:5]):
    ax2.plot(range(seq_length), attn, alpha=0.6, linewidth=1.5, 
             label=f'Sample {i+1}', marker='o', markersize=2, markevery=6)
ax2.plot(range(seq_length), avg_attention, linewidth=3, color='black',
         label='Average', linestyle='--')
ax2.set_xlabel('Hours Ago', fontsize=11, fontweight='bold')
ax2.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
ax2.set_title('Individual Sample Patterns', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(alpha=0.3)

# ============================================================================
# Plot 3: Cumulative attention
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
cumsum = np.cumsum(avg_attention)
ax3.plot(range(seq_length), cumsum, linewidth=3, color='darkblue',
         marker='o', markersize=4, markevery=6)
ax3.fill_between(range(seq_length), 0, cumsum, alpha=0.3, color='darkblue')
ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='50%')
ax3.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80%')

# Find where we cross thresholds
idx_50 = np.where(cumsum >= 0.5)[0][0] if any(cumsum >= 0.5) else seq_length
idx_80 = np.where(cumsum >= 0.8)[0][0] if any(cumsum >= 0.8) else seq_length

ax3.scatter([idx_50], [0.5], color='green', s=200, zorder=5, 
            edgecolor='black', linewidth=2)
ax3.scatter([idx_80], [0.8], color='orange', s=200, zorder=5,
            edgecolor='black', linewidth=2)

ax3.set_xlabel('Hours Ago', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cumulative Attention', fontsize=11, fontweight='bold')
ax3.set_title('Cumulative Attention Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# ============================================================================
# Plot 4: Heatmap
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])
heatmap_data = np.array(all_attentions[:10])
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Weight'},
            ax=ax4, xticklabels=[time_labels[i] if i % 6 == 0 else '' 
                                 for i in range(seq_length)],
            yticklabels=[f'S{i+1}' for i in range(len(all_attentions[:10]))])
ax4.set_xlabel('Hours Ago', fontsize=11, fontweight='bold')
ax4.set_ylabel('Sample', fontsize=11, fontweight='bold')
ax4.set_title('Attention Heatmap (All Samples)', fontsize=12, fontweight='bold')

# ============================================================================
# Plot 5: Top contributing hours
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])
top_k = 10
top_indices = np.argsort(avg_attention)[-top_k:][::-1]
top_weights = avg_attention[top_indices]
top_labels = [time_labels[i] for i in top_indices]

colors_top = plt.cm.RdYlGn(top_weights / top_weights.max())
bars = ax5.barh(range(top_k), top_weights, color=colors_top,
                edgecolor='black', linewidth=1)
ax5.set_yticks(range(top_k))
ax5.set_yticklabels(top_labels, fontsize=10)
ax5.set_xlabel('Attention Weight', fontsize=11, fontweight='bold')
ax5.set_title(f'Top {top_k} Most Important Hours', fontsize=12, fontweight='bold')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    percentage = (top_weights[i] / avg_attention.sum()) * 100
    ax5.text(width, bar.get_y() + bar.get_height()/2.,
             f' {percentage:.2f}%',
             ha='left', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# Plot 6: Attention distribution by time buckets
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])
buckets = {
    'Recent\n(0-12h)': avg_attention[-12:].sum(),
    'Mid-term\n(12-24h)': avg_attention[-24:-12].sum(),
    'Historical\n(24-48h)': avg_attention[:-24].sum()
}
colors_bucket = ['#ff6b6b', '#ffd93d', '#6bcf7f']
bars = ax6.bar(buckets.keys(), buckets.values(), color=colors_bucket,
               edgecolor='black', linewidth=2, alpha=0.8)
ax6.set_ylabel('Total Attention', fontsize=11, fontweight='bold')
ax6.set_title('Attention by Time Period', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}\n({height*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# Plot 7: Statistical summary
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

stats_text = f"""
📊 STATISTICAL SUMMARY

Total Samples: {len(all_attentions)}
Sequence Length: {seq_length} hours

📈 Attention Statistics:
  • Mean: {avg_attention.mean():.6f}
  • Std: {avg_attention.std():.6f}
  • Max: {avg_attention.max():.6f}
  • Min: {avg_attention.min():.6f}

🏆 Top 3 Hours:
"""

top_3_idx = np.argsort(avg_attention)[-3:][::-1]
for rank, idx in enumerate(top_3_idx, 1):
    hours_ago = seq_length - idx - 1
    weight = avg_attention[idx]
    percentage = (weight / avg_attention.sum()) * 100
    stats_text += f"  {rank}. {hours_ago}h ago: {percentage:.2f}%\n"

recent_key = 'Recent\n(0-12h)'
mid_key = 'Mid-term\n(12-24h)'
hist_key = 'Historical\n(24-48h)'

stats_text += f"""
⏰ Time Distribution:
  • Recent (0-12h): {buckets[recent_key]*100:.1f}%
  • Mid (12-24h): {buckets[mid_key]*100:.1f}%
  • Historical (>24h): {buckets[hist_key]*100:.1f}%

🎯 50% attn at: {seq_length - idx_50 - 1}h ago
🎯 80% attn at: {seq_length - idx_80 - 1}h ago
"""

ax7.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('trained_attention_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: trained_attention_analysis.png")

# ============================================================================
# Console output
# ============================================================================

print("\n" + "=" * 70)
print("📊 ATTENTION ANALYSIS RESULTS")
print("=" * 70)

print(f"\n🏆 TOP 5 MOST IMPORTANT TIME POINTS:")
for rank, idx in enumerate(top_indices[:5], 1):
    hours_ago = seq_length - idx - 1
    weight = avg_attention[idx]
    percentage = (weight / avg_attention.sum()) * 100
    print(f"   {rank}. {hours_ago}h ago: {percentage:.2f}% of total attention")

print(f"\n⏰ TEMPORAL DISTRIBUTION:")
recent_key = 'Recent\n(0-12h)'
mid_key = 'Mid-term\n(12-24h)'
hist_key = 'Historical\n(24-48h)'
print(f"   Recent (0-12h): {buckets[recent_key]*100:.1f}%")
print(f"   Mid-term (12-24h): {buckets[mid_key]*100:.1f}%")
print(f"   Historical (24-48h): {buckets[hist_key]*100:.1f}%")

entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10))
max_entropy = np.log(seq_length)
normalized_entropy = entropy / max_entropy

print(f"\n📈 ATTENTION CONCENTRATION:")
print(f"   Normalized Entropy: {normalized_entropy:.4f}")
print(f"   Concentration Score: {1-normalized_entropy:.4f}")

if normalized_entropy > 0.9:
    print("   → Model has DIFFUSE attention (looks at many hours)")
elif normalized_entropy < 0.5:
    print("   → Model has FOCUSED attention (concentrates on few hours)")
else:
    print("   → Model has BALANCED attention")

print(f"\n🎯 KEY INSIGHTS:")
recent_key = 'Recent\n(0-12h)'
mid_key = 'Mid-term\n(12-24h)'
hist_key = 'Historical\n(24-48h)'

if buckets[recent_key] > 0.5:
    print("   ✓ Model prioritizes RECENT data (last 12 hours)")
if buckets[hist_key] > 0.3:
    print("   ✓ Model considers HISTORICAL patterns (24-48h ago)")
if avg_attention.std() / avg_attention.mean() > 0.15:
    print("   ✓ Model shows SELECTIVE attention (high variance)")
else:
    print("   ✓ Model shows UNIFORM attention (low variance)")

print("\n" + "=" * 70)
print("✅ Analysis complete! Check 'trained_attention_analysis.png'")
print("=" * 70)
