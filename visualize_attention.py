"""
Visualize LSTM Attention Weights

This script helps understand what the attention mechanism is focusing on
when making predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_weights(attention_weights, timestamps=None, save_path='attention_weights.png'):
    """Visualize attention weights to see which timesteps the model focuses on.
    
    Args:
        attention_weights: Tensor or array of shape (batch_size, seq_length, 1) or (seq_length,)
        timestamps: Optional list of timestamp labels
        save_path: Where to save the visualization
    """
    # Convert to numpy if needed
    if hasattr(attention_weights, 'cpu'):
        attention_weights = attention_weights.cpu().detach().numpy()
    
    # Handle different shapes
    if len(attention_weights.shape) == 3:
        # (batch, seq, 1) -> take first sample
        weights = attention_weights[0, :, 0]
    elif len(attention_weights.shape) == 2:
        # (batch, seq) -> take first sample
        weights = attention_weights[0, :]
    else:
        # (seq,) -> use as is
        weights = attention_weights
    
    seq_length = len(weights)
    
    # Create timestamps if not provided
    if timestamps is None:
        timestamps = [f"t-{seq_length-i}" for i in range(seq_length)]
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Bar chart
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(weights / weights.max())
    plt.bar(range(seq_length), weights, color=colors)
    plt.xlabel('Time Step (hours ago)', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title('Attention Weights: What Hours Matter Most?', fontsize=14, fontweight='bold')
    plt.xticks(range(0, seq_length, max(1, seq_length//10)), 
               [timestamps[i] for i in range(0, seq_length, max(1, seq_length//10))],
               rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(weights.reshape(-1, 1), 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Attention Weight'},
                yticklabels=[timestamps[i] if i % max(1, seq_length//10) == 0 else '' 
                            for i in range(seq_length)])
    plt.title('Attention Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Time Step', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Attention visualization saved to: {save_path}")
    
    # Print top 5 most important timesteps
    top_indices = np.argsort(weights)[-5:][::-1]
    print("\n📊 Top 5 Most Important Time Steps:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. {timestamps[idx]}: {weights[idx]:.4f} ({weights[idx]/weights.sum()*100:.1f}%)")
    
    return weights

def compare_attention_patterns(attention_list, labels, save_path='attention_comparison.png'):
    """Compare attention patterns across multiple predictions.
    
    Args:
        attention_list: List of attention weight arrays
        labels: List of labels for each pattern
        save_path: Where to save the visualization
    """
    plt.figure(figsize=(14, 8))
    
    for i, (weights, label) in enumerate(zip(attention_list, labels)):
        if hasattr(weights, 'cpu'):
            weights = weights.cpu().detach().numpy()
        
        if len(weights.shape) > 1:
            weights = weights[0].flatten()
        
        plt.plot(weights, label=label, alpha=0.7, linewidth=2)
    
    plt.xlabel('Time Step (hours ago)', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title('Attention Patterns Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison visualization saved to: {save_path}")

if __name__ == "__main__":
    # Example usage with dummy data
    print("🎯 Attention Visualization Tool")
    print("=" * 50)
    
    # Create dummy attention weights (48 hours)
    seq_length = 48
    
    # Simulate attention focusing on recent hours and some historical pattern
    weights = np.zeros(seq_length)
    weights[-3:] = [0.15, 0.25, 0.30]  # Recent hours important
    weights[24] = 0.10  # 24 hours ago (daily pattern)
    weights[12] = 0.08  # 12 hours ago
    weights[:20] = np.random.rand(20) * 0.02  # Some noise in older data
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    # Create timestamps
    timestamps = [f"{i}h ago" for i in range(seq_length-1, -1, -1)]
    
    # Visualize
    plot_attention_weights(weights, timestamps)
    
    print("\n✅ Demo complete! Use this with your trained LSTM model.")
