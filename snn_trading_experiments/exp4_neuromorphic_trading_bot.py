"""
Experiment 4: Neuromorphic Trading Bot
Simulate neuromorphic hardware deployment
Measure power consumption vs GPU and latency benchmarks
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

from models.snn_trading_agent import SpikingTradingAgent


def estimate_power_consumption(
    model: SpikingTradingAgent,
    num_inferences: int = 1000,
    device_type: str = 'cpu'
):
    """
    Estimate power consumption for SNN vs traditional deployment.
    
    Note: This is a simulation. Real neuromorphic hardware power consumption
    would be measured with hardware-specific tools.
    """
    # Simulated power consumption (Watts)
    power_estimates = {
        'gpu': 150.0,  # Typical GPU power
        'cpu': 50.0,   # Typical CPU power
        'neuromorphic': 5.0  # Estimated neuromorphic chip (e.g., Loihi)
    }
    
    # Measure inference time
    model.eval()
    device = next(model.parameters()).device
    
    dummy_input = torch.randn(1, model.input_dim).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, num_steps=10)
    
    # Measure
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_inferences):
            outputs, _ = model(dummy_input, num_steps=10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Calculate energy consumption
    if device.type == 'cuda':
        power = power_estimates['gpu']
    else:
        power = power_estimates['cpu']
    
    energy_joules = power * total_time
    
    # Estimate for neuromorphic hardware
    # SNNs are more efficient on neuromorphic hardware
    neuromorphic_speedup = 10.0  # Estimated speedup factor
    neuromorphic_time = total_time / neuromorphic_speedup
    neuromorphic_energy = power_estimates['neuromorphic'] * neuromorphic_time
    
    return {
        'device_type': device_type,
        'total_time_s': total_time,
        'avg_latency_ms': (total_time / num_inferences) * 1000,
        'power_w': power,
        'energy_j': energy_joules,
        'neuromorphic_time_s': neuromorphic_time,
        'neuromorphic_energy_j': neuromorphic_energy,
        'neuromorphic_power_w': power_estimates['neuromorphic'],
        'energy_savings_pct': ((energy_joules - neuromorphic_energy) / energy_joules) * 100
    }


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("EXPERIMENT 4: Neuromorphic Trading Bot")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    input_dim = 50
    
    print(f"\nRunning on: {device}")
    print("Note: This simulates neuromorphic deployment. ")
    print("Real deployment would use hardware like Intel Loihi.\n")
    
    # Create SNN
    print("Creating SNN trading agent...")
    snn_agent = SpikingTradingAgent(
        input_dim=input_dim,
        hidden_dim=100,
        output_dim=3,  # Buy, Hold, Sell
        num_pathways=3
    ).to(device)
    
    # Benchmark latency
    print("\nBenchmarking inference latency...")
    
    batch_sizes = [1, 4, 8, 16]
    latency_results = {}
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, input_dim).to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = snn_agent(dummy_input, num_steps=10)
        
        # Measure
        latencies = []
        num_runs = 100
        
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                outputs, _ = snn_agent(dummy_input, num_steps=10)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
        
        latency_results[f"batch_{batch_size}"] = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
        
        print(f"  Batch {batch_size}: {np.mean(latencies):.2f} ms (mean), "
              f"{np.percentile(latencies, 95):.2f} ms (p95)")
    
    # Estimate power consumption
    print("\nEstimating power consumption...")
    power_results = estimate_power_consumption(
        snn_agent,
        num_inferences=1000,
        device_type=device_type
    )
    
    print(f"  Standard {device_type.upper()}: {power_results['power_w']:.1f}W, "
          f"{power_results['energy_j']:.2f}J total")
    print(f"  Neuromorphic (simulated): {power_results['neuromorphic_power_w']:.1f}W, "
          f"{power_results['neuromorphic_energy_j']:.2f}J total")
    print(f"  Energy savings: {power_results['energy_savings_pct']:.1f}%")
    
    # Calculate trading bot continuous operation power
    seconds_per_day = 86400
    inferences_per_day = seconds_per_day  # 1 inference per second
    
    daily_energy_standard = (power_results['power_w'] * 
                            power_results['avg_latency_ms'] / 1000 * 
                            inferences_per_day) / 3600  # Convert to Wh
    
    daily_energy_neuromorphic = (power_results['neuromorphic_power_w'] * 
                                 power_results['avg_latency_ms'] / 1000 / 10 *  # Speedup factor
                                 inferences_per_day) / 3600
    
    print(f"\n24/7 Trading Bot Energy Consumption:")
    print(f"  Standard {device_type.upper()}: ~{daily_energy_standard:.2f} Wh/day")
    print(f"  Neuromorphic: ~{daily_energy_neuromorphic:.2f} Wh/day")
    
    # Results
    results = {
        'device': device_type,
        'latency_benchmarks': latency_results,
        'power_consumption': {
            'standard_power_w': power_results['power_w'],
            'neuromorphic_power_w': power_results['neuromorphic_power_w'],
            'energy_savings_pct': power_results['energy_savings_pct'],
            'daily_energy_standard_wh': float(daily_energy_standard),
            'daily_energy_neuromorphic_wh': float(daily_energy_neuromorphic)
        },
        'real_time_capable': latency_results['batch_1']['p95_ms'] < 100.0
    }
    
    # Save results
    output_dir = Path("outputs/snn_trading_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exp4_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    # Check success criteria
    neuro_power = power_results['neuromorphic_power_w']
    success = neuro_power < 5.0 and results['real_time_capable']
    status = "✅ SUCCESS" if success else "⚠️  NEEDS REVIEW"
    
    print(f"\n{status}")
    print(f"  Neuromorphic power: {neuro_power:.1f}W (Target: <5W)")
    print(f"  Real-time capable: {results['real_time_capable']}")
    print(f"  Energy savings: {power_results['energy_savings_pct']:.1f}%")


if __name__ == "__main__":
    main()
