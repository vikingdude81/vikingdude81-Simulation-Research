"""
Experiment 1.1: Basic FFT Analysis on Crypto Data

Objective: Apply Fourier transform to BTC and ETH price data to understand
          frequency characteristics and validate basic FTTF concepts on crypto markets.

Hypothesis: Crypto price data contains dominant frequencies corresponding to
           market cycles (daily, weekly, etc.) that can be identified and used.

Success Criteria:
- Identify 3-5 dominant frequencies in price data
- Reconstruct price from top N frequency components with >80% accuracy
- Visual confirmation of frequency patterns matching known market cycles

Timeline: 2-3 hours
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path to import existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from fetch_data import fetch_ohlcv
except ImportError:
    print("Warning: Could not import fetch_data. Using mock data.")
    fetch_ohlcv = None


class CryptoFFTAnalyzer:
    """Analyze cryptocurrency price data using Fourier transforms."""
    
    def __init__(self, symbol: str = "BTC/USDT", hours: int = 1000):
        """
        Initialize analyzer.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            hours: Number of hours of historical data to analyze
        """
        self.symbol = symbol
        self.hours = hours
        self.prices = None
        self.fft = None
        self.frequencies = None
        self.power_spectrum = None
        
    def load_data(self):
        """Load price data from exchange."""
        print(f"\n=== Loading {self.hours} hours of {self.symbol} data ===")
        
        if fetch_ohlcv is None:
            # Generate mock data for testing
            print("Using mock sinusoidal data for testing...")
            t = np.arange(self.hours)
            # Simulate price with multiple frequency components
            self.prices = (
                50000 +  # Base price
                1000 * np.sin(2 * np.pi * t / 168) +  # Weekly cycle
                500 * np.sin(2 * np.pi * t / 24) +    # Daily cycle
                200 * np.random.randn(self.hours)     # Noise
            )
        else:
            # Load real data
            df = fetch_ohlcv(self.symbol, '1h', limit=self.hours)
            self.prices = df['close'].values
            
        print(f"Loaded {len(self.prices)} price points")
        print(f"Price range: ${self.prices.min():.2f} - ${self.prices.max():.2f}")
        
    def compute_fft(self):
        """Compute FFT of price data."""
        print("\n=== Computing FFT ===")
        
        # Normalize prices (remove mean, scale by std)
        normalized_prices = (self.prices - self.prices.mean()) / self.prices.std()
        
        # Compute FFT
        self.fft = np.fft.fft(normalized_prices)
        
        # Compute frequencies (cycles per hour)
        self.frequencies = np.fft.fftfreq(len(normalized_prices), d=1.0)
        
        # Compute power spectrum
        self.power_spectrum = np.abs(self.fft) ** 2
        
        print(f"FFT shape: {self.fft.shape}")
        print(f"Frequency range: {self.frequencies.min():.6f} to {self.frequencies.max():.6f} cycles/hour")
        
    def analyze_dominant_frequencies(self, top_n: int = 10):
        """
        Identify dominant frequencies in the data.
        
        Args:
            top_n: Number of top frequencies to report
            
        Returns:
            DataFrame with dominant frequencies
        """
        print(f"\n=== Identifying Top {top_n} Dominant Frequencies ===")
        
        # Only consider positive frequencies (FFT is symmetric)
        positive_freq_mask = self.frequencies > 0
        positive_freqs = self.frequencies[positive_freq_mask]
        positive_power = self.power_spectrum[positive_freq_mask]
        
        # Find top N frequencies by power
        top_indices = np.argsort(positive_power)[-top_n:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            freq = positive_freqs[idx]
            power = positive_power[idx]
            period_hours = 1.0 / freq if freq > 0 else np.inf
            
            results.append({
                'rank': i,
                'frequency_cycles_per_hour': freq,
                'period_hours': period_hours,
                'period_days': period_hours / 24,
                'power': power,
                'power_pct': 100 * power / positive_power.sum()
            })
            
            print(f"{i}. Freq: {freq:.6f} cycles/hour | "
                  f"Period: {period_hours:.1f}h ({period_hours/24:.1f} days) | "
                  f"Power: {power:.2e} ({100*power/positive_power.sum():.2f}%)")
        
        return pd.DataFrame(results)
    
    def reconstruct_from_top_frequencies(self, top_n: int = 50):
        """
        Reconstruct price signal from top N frequency components.
        
        Args:
            top_n: Number of frequency components to use
            
        Returns:
            Reconstructed prices array
        """
        print(f"\n=== Reconstructing from Top {top_n} Frequencies ===")
        
        # Create filtered FFT (keep only top N components)
        filtered_fft = np.zeros_like(self.fft)
        
        # Find top N frequencies by power
        top_indices = np.argsort(self.power_spectrum)[-top_n:]
        filtered_fft[top_indices] = self.fft[top_indices]
        
        # Inverse FFT to reconstruct signal
        reconstructed_normalized = np.fft.ifft(filtered_fft).real
        
        # Denormalize
        reconstructed = reconstructed_normalized * self.prices.std() + self.prices.mean()
        
        # Calculate reconstruction quality
        mse = np.mean((self.prices - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.prices - reconstructed))
        r2 = 1 - (np.sum((self.prices - reconstructed)**2) / 
                  np.sum((self.prices - self.prices.mean())**2))
        
        print(f"Reconstruction Quality (top {top_n} components):")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  R²:   {r2:.4f} ({r2*100:.2f}%)")
        
        return reconstructed, {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'top_n': top_n
        }
    
    def plot_analysis(self, reconstructed_prices=None, save_path=None):
        """
        Create comprehensive visualization of FFT analysis.
        
        Args:
            reconstructed_prices: Optional reconstructed signal to overlay
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Original price data
        ax = axes[0, 0]
        ax.plot(self.prices, 'b-', linewidth=1, label='Original')
        if reconstructed_prices is not None:
            ax.plot(reconstructed_prices, 'r--', linewidth=1, alpha=0.7, label='Reconstructed')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{self.symbol} Price Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Power spectrum (positive frequencies only)
        ax = axes[0, 1]
        positive_mask = self.frequencies > 0
        ax.semilogy(self.frequencies[positive_mask], 
                    self.power_spectrum[positive_mask], 
                    'b-', linewidth=1)
        ax.set_xlabel('Frequency (cycles/hour)')
        ax.set_ylabel('Power (log scale)')
        ax.set_title('Power Spectrum')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Power spectrum by period (hours)
        ax = axes[1, 0]
        positive_freqs = self.frequencies[positive_mask]
        positive_power = self.power_spectrum[positive_mask]
        periods = 1.0 / positive_freqs
        # Focus on periods between 1 hour and 1 month
        valid_period_mask = (periods >= 1) & (periods <= 720)
        ax.semilogy(periods[valid_period_mask], 
                    positive_power[valid_period_mask],
                    'g-', linewidth=1)
        ax.set_xlabel('Period (hours)')
        ax.set_ylabel('Power (log scale)')
        ax.set_title('Power Spectrum by Period')
        ax.axvline(24, color='r', linestyle='--', alpha=0.5, label='Daily (24h)')
        ax.axvline(168, color='orange', linestyle='--', alpha=0.5, label='Weekly (168h)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative power vs number of components
        ax = axes[1, 1]
        sorted_power = np.sort(self.power_spectrum)[::-1]
        cumulative_power = np.cumsum(sorted_power) / sorted_power.sum()
        ax.plot(np.arange(1, len(cumulative_power) + 1), cumulative_power, 'purple')
        ax.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
        ax.set_xlabel('Number of Frequency Components')
        ax.set_ylabel('Cumulative Power (fraction)')
        ax.set_title('Cumulative Power vs Components')
        ax.set_xlim(0, min(500, len(cumulative_power)))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        
    def save_results(self, dominant_freqs_df, reconstruction_metrics, save_dir):
        """Save analysis results to JSON."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'experiment': '1.1_crypto_fft_basics',
            'timestamp': timestamp,
            'symbol': self.symbol,
            'data_points': len(self.prices),
            'dominant_frequencies': dominant_freqs_df.to_dict('records'),
            'reconstruction': reconstruction_metrics,
            'summary': {
                'top_period_hours': float(dominant_freqs_df.iloc[0]['period_hours']),
                'top_period_days': float(dominant_freqs_df.iloc[0]['period_days']),
                'top_freq_power_pct': float(dominant_freqs_df.iloc[0]['power_pct']),
                'reconstruction_r2': float(reconstruction_metrics['r2'])
            }
        }
        
        results_path = save_dir / f'experiment_1.1_{self.symbol.replace("/", "_")}_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        return results


def main():
    """Run Experiment 1.1: Basic FFT Analysis."""
    
    print("=" * 70)
    print("EXPERIMENT 1.1: Basic FFT Analysis on Crypto Data")
    print("=" * 70)
    
    # Analyze BTC
    print("\n" + "=" * 70)
    print("ANALYZING BTC/USDT")
    print("=" * 70)
    
    btc_analyzer = CryptoFFTAnalyzer("BTC/USDT", hours=1000)
    btc_analyzer.load_data()
    btc_analyzer.compute_fft()
    
    btc_dominant = btc_analyzer.analyze_dominant_frequencies(top_n=10)
    btc_reconstructed, btc_metrics = btc_analyzer.reconstruct_from_top_frequencies(top_n=50)
    
    # Visualization
    results_dir = Path(__file__).parent.parent / "results" / "01_basic_fft"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    btc_analyzer.plot_analysis(
        reconstructed_prices=btc_reconstructed,
        save_path=results_dir / f'btc_fft_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    
    btc_results = btc_analyzer.save_results(btc_dominant, btc_metrics, results_dir)
    
    # Analyze ETH
    print("\n" + "=" * 70)
    print("ANALYZING ETH/USDT")
    print("=" * 70)
    
    eth_analyzer = CryptoFFTAnalyzer("ETH/USDT", hours=1000)
    eth_analyzer.load_data()
    eth_analyzer.compute_fft()
    
    eth_dominant = eth_analyzer.analyze_dominant_frequencies(top_n=10)
    eth_reconstructed, eth_metrics = eth_analyzer.reconstruct_from_top_frequencies(top_n=50)
    
    eth_analyzer.plot_analysis(
        reconstructed_prices=eth_reconstructed,
        save_path=results_dir / f'eth_fft_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    
    eth_results = eth_analyzer.save_results(eth_dominant, eth_metrics, results_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 1.1 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\nBTC/USDT:")
    print(f"  Top Period: {btc_results['summary']['top_period_hours']:.1f}h "
          f"({btc_results['summary']['top_period_days']:.1f} days)")
    print(f"  Reconstruction R²: {btc_results['summary']['reconstruction_r2']:.4f}")
    
    print(f"\nETH/USDT:")
    print(f"  Top Period: {eth_results['summary']['top_period_hours']:.1f}h "
          f"({eth_results['summary']['top_period_days']:.1f} days)")
    print(f"  Reconstruction R²: {eth_results['summary']['reconstruction_r2']:.4f}")
    
    # Success evaluation
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)
    
    success = True
    
    # Criterion 1: Identify 3-5 dominant frequencies ✓
    print("✓ Identified 10 dominant frequencies for each asset")
    
    # Criterion 2: Reconstruction >80% accuracy
    btc_pass = btc_results['summary']['reconstruction_r2'] > 0.80
    eth_pass = eth_results['summary']['reconstruction_r2'] > 0.80
    
    print(f"{'✓' if btc_pass else '✗'} BTC reconstruction R²: "
          f"{btc_results['summary']['reconstruction_r2']:.4f} "
          f"({'PASS' if btc_pass else 'FAIL'} >0.80 threshold)")
    print(f"{'✓' if eth_pass else '✗'} ETH reconstruction R²: "
          f"{eth_results['summary']['reconstruction_r2']:.4f} "
          f"({'PASS' if eth_pass else 'FAIL'} >0.80 threshold)")
    
    success = success and btc_pass and eth_pass
    
    # Criterion 3: Visual confirmation
    print("✓ Visual plots generated - review for frequency patterns")
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 EXPERIMENT 1.1: SUCCESS! Fourier analysis works on crypto data.")
        print("\nNext Steps:")
        print("  1. Review plots to understand frequency patterns")
        print("  2. Compare dominant periods to known market cycles")
        print("  3. Proceed to Experiment 1.2 (Holographic Memory Test)")
    else:
        print("⚠️  EXPERIMENT 1.1: PARTIAL SUCCESS - Review reconstruction quality")
        print("\nRecommendations:")
        print("  - Try different numbers of frequency components")
        print("  - Analyze why certain assets reconstruct better")
        print("  - Consider data preprocessing (detrending, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
