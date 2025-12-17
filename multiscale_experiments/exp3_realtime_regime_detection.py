"""
Experiment 3: Real-time Regime Detection
Test real-time regime shifts across timeframes
Measure latency benchmarks (<500ms target)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time
from pathlib import Path
import json
from datetime import datetime

from models.multiscale_predictor import MultiscaleMarketEncoder


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("EXPERIMENT 3: Real-time Regime Detection")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 50
    batch_sizes = [1, 4, 8, 16, 32]
    
    # Create model
    print("\nCreating multiscale encoder...")
    model = MultiscaleMarketEncoder(
        input_dim=input_dim,
        hidden_dim=128,
        num_scales=5,
        output_dim=64
    ).to(device)
    model.eval()
    
    # Warm up
    print("Warming up model...")
    dummy_data = {
        tf: torch.randn(1, input_dim).to(device)
        for tf in ['1H', '4H', '12H', '1D', '1W']
    }
    with torch.no_grad():
        _ = model(dummy_data)
    
    # Benchmark latency
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}...")
        
        # Create test data
        test_data = {
            tf: torch.randn(batch_size, input_dim).to(device)
            for tf in ['1H', '4H', '12H', '1D', '1W']
        }
        
        # Measure latency over multiple runs
        latencies = []
        num_runs = 100
        
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                encoded, _ = model(test_data)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
        
        # Statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        results[f"batch_{batch_size}"] = {
            'mean_ms': float(mean_latency),
            'std_ms': float(std_latency),
            'p50_ms': float(p50_latency),
            'p95_ms': float(p95_latency),
            'p99_ms': float(p99_latency)
        }
        
        print(f"  Mean latency: {mean_latency:.2f} ms")
        print(f"  P95 latency:  {p95_latency:.2f} ms")
        print(f"  P99 latency:  {p99_latency:.2f} ms")
    
    # Save results
    output_dir = Path("outputs/multiscale_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exp3_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    # Check success criteria (batch_size=1 for real-time)
    if 'batch_1' in results:
        realtime_latency = results['batch_1']['p95_ms']
        success = realtime_latency < 500.0
        status = "✅ SUCCESS" if success else "⚠️  ABOVE TARGET"
        print(f"\n{status} - Real-time P95 latency: {realtime_latency:.2f} ms (Target: <500ms)")


if __name__ == "__main__":
    main()
