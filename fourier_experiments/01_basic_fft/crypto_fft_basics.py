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
import time
import argparse
import logging
import traceback
from typing import List, Optional

# Add parent directory to path to import existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

fetch_ohlcv = None  # will attempt to resolve dynamically from fetch_data module
try:
    import fetch_data as _fetch_data_module  # Optional module; function may or may not exist
    fetch_ohlcv = getattr(_fetch_data_module, 'fetch_ohlcv', None)
except Exception:
    print("[INFO] fetch_ohlcv not available – defaulting to internal/simple fetch or mock.")

try:
    import yfinance as yf  # lightweight external fetch for quick prototyping
except Exception:
    yf = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class CryptoFFTAnalyzer:
    """Analyze cryptocurrency price data using Fourier transforms.

    Contract:
    - call load_data() before compute_fft()
    - compute_fft() populates fft, frequencies, power_spectrum
    - reconstruct_from_top_frequencies() requires fft + power_spectrum
    """
    
    def __init__(self, symbol: str = "BTC/USDT", hours: int = 1000, use_real: bool = False, mock: bool = False, quiet: bool = False):
        """
        Initialize analyzer.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            hours: Number of hours of historical data to analyze
        """
        self.symbol = symbol
        self.hours = hours
        self.use_real = use_real
        self.mock = mock
        self.quiet = quiet
        # Core data containers (numpy ndarrays once populated)
        self.prices: Optional[np.ndarray] = None
        self.fft: Optional[np.ndarray] = None
        self.frequencies: Optional[np.ndarray] = None
        self.power_spectrum: Optional[np.ndarray] = None
        
    def _generate_mock(self) -> np.ndarray:
        t = np.arange(self.hours)
        base = 50000 if 'BTC' in self.symbol else 3000
        return (
            base +
            0.02 * base * np.sin(2 * np.pi * t / 168) +  # Weekly cycle
            0.01 * base * np.sin(2 * np.pi * t / 24) +   # Daily cycle
            0.004 * base * np.random.randn(self.hours)
        )

    def _simple_yfinance_fetch(self) -> Optional[np.ndarray]:
        if yf is None:
            return None
        # Map symbol to yfinance ticker (simple heuristic)
        if self.symbol.startswith('BTC'): ticker = 'BTC-USD'
        elif self.symbol.startswith('ETH'): ticker = 'ETH-USD'
        else: ticker = self.symbol.replace('/', '-')
        try:
            # 1000 hours ~ 42 days -> request 60d to be safe
            hist = yf.Ticker(ticker).history(period='60d', interval='1h')
            if hist.empty:
                return None
            closes = hist['Close'].tail(self.hours).values
            closes = np.asarray(closes, dtype=float)
            if len(closes) < self.hours:
                # pad with last value
                closes = np.pad(closes, (self.hours - len(closes), 0), mode='edge')
            return closes
        except Exception:
            return None

    def load_data(self) -> None:
        """Load price data (real if requested & available, else mock)."""
        logging.info(f"Loading {self.hours}h of data for {self.symbol} (real={self.use_real}, mock={self.mock})")

        used_source = "mock"
        if not self.mock and self.use_real:
            # Priority 1: fetch_ohlcv if available
            if fetch_ohlcv is not None:
                try:
                    df = fetch_ohlcv(self.symbol, '1h', limit=self.hours)
                    self.prices = df['close'].values
                    used_source = 'fetch_ohlcv'
                except Exception as e:
                    logging.warning(f"fetch_ohlcv failed ({e}); falling back to yfinance/mock")
            if self.prices is None:
                fetched = self._simple_yfinance_fetch()
                if fetched is not None:
                    self.prices = fetched
                    used_source = 'yfinance'
        if self.prices is None:
            self.prices = self._generate_mock()
            used_source = 'mock'

        # Ensure np.ndarray (guard against pandas ExtensionArray types)
        self.prices = np.asarray(self.prices, dtype=float)
        logging.info(f"Data source: {used_source}; points: {len(self.prices)}; min={float(self.prices.min()):.2f} max={float(self.prices.max()):.2f}")
        
    def compute_fft(self) -> None:
        """Compute FFT of price data."""
        if not self.quiet:
            print("\n=== Computing FFT ===")
        
        if self.prices is None:
            raise RuntimeError("Prices not loaded. Call load_data() before compute_fft().")
        
        # Normalize prices (remove mean, scale by std)
        normalized_prices = (self.prices - self.prices.mean()) / self.prices.std()
        
        # Compute FFT
        self.fft = np.fft.fft(normalized_prices)
        
        # Compute frequencies (cycles per hour)
        self.frequencies = np.fft.fftfreq(len(normalized_prices), d=1.0)
        
        # Compute power spectrum
        self.power_spectrum = np.abs(self.fft) ** 2
        
        if not self.quiet:
            print(f"FFT shape: {self.fft.shape}")
            print(f"Frequency range: {self.frequencies.min():.6f} to {self.frequencies.max():.6f} cycles/hour")
        
    def analyze_dominant_frequencies(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify dominant frequencies in the data.
        
        Args:
            top_n: Number of top frequencies to report
            
        Returns:
            DataFrame with dominant frequencies
        """
        if not self.quiet:
            print(f"\n=== Identifying Top {top_n} Dominant Frequencies ===")
        
        # Only consider positive frequencies (FFT is symmetric)
        if self.frequencies is None or self.power_spectrum is None:
            raise RuntimeError("FFT not computed. Call compute_fft() before analyze_dominant_frequencies().")
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
            if not self.quiet:
                print(f"{i}. Freq: {freq:.6f} cycles/hour | Period: {period_hours:.1f}h ({period_hours/24:.1f} days) | Power: {power:.2e} ({100*power/positive_power.sum():.2f}%)")
        
        return pd.DataFrame(results)
    
    def reconstruct_from_top_frequencies(self, top_n: int = 50):
        """
        Reconstruct price signal from top N frequency components.
        
        Args:
            top_n: Number of frequency components to use
            
        Returns:
            Reconstructed prices array
        """
        if not self.quiet:
            print(f"\n=== Reconstructing from Top {top_n} Frequencies ===")
        
        # Create filtered FFT (keep only top N components)
        if self.fft is None or self.power_spectrum is None:
            raise RuntimeError("FFT not computed. Call compute_fft() before reconstruction.")
        if self.prices is None:
            raise RuntimeError("Prices not loaded.")
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
        
        if not self.quiet:
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
    
    def plot_analysis(self, reconstructed_prices=None, save_path=None) -> None:
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
        if self.frequencies is None or self.power_spectrum is None:
            raise RuntimeError("FFT not computed. Cannot plot without frequencies & power spectrum.")
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
        
        # Close plot instead of showing interactively (avoid blocking)
        plt.close()
        
    def save_results(self, dominant_freqs_df: pd.DataFrame, reconstruction_metrics: dict, save_dir) -> dict:
        """Save analysis results to JSON."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Satisfy type checkers and ensure data present
        assert self.prices is not None, "Prices must be loaded before saving results"

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


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1.1: Basic FFT on crypto data")
    p.add_argument('--symbols', nargs='+', default=['BTC/USDT','ETH/USDT'], help='Symbols to analyze')
    p.add_argument('--hours', type=int, default=1000, help='Number of hours of history')
    p.add_argument('--top-n', type=int, default=10, help='Top N dominant frequencies to list')
    p.add_argument('--components', type=int, default=50, help='Number of components for reconstruction')
    p.add_argument('--use-real', action='store_true', help='Attempt to fetch real market data')
    p.add_argument('--mock', action='store_true', help='Force mock data even if real is available')
    p.add_argument('--quiet', action='store_true', help='Reduce console output')
    p.add_argument('--no-plots', action='store_true', help='Skip generating plot images')
    p.add_argument('--output-dir', default=None, help='Custom output directory for results')
    return p.parse_args()


def run_for_symbol(symbol: str, args, results_dir: Path):
    start = time.time()
    analyzer = CryptoFFTAnalyzer(symbol, hours=args.hours, use_real=args.use_real, mock=args.mock, quiet=args.quiet)
    analyzer.load_data()
    analyzer.compute_fft()
    dom = analyzer.analyze_dominant_frequencies(top_n=args.top_n)
    reconstructed, metrics = analyzer.reconstruct_from_top_frequencies(top_n=args.components)
    plot_path = None
    if not args.no_plots:
        plot_path = results_dir / f'{symbol.replace("/","_").lower()}_fft_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        analyzer.plot_analysis(reconstructed_prices=reconstructed, save_path=plot_path)
    res = analyzer.save_results(dom, metrics, results_dir)
    elapsed = time.time() - start
    if not args.quiet:
        print(f"[TIME] {symbol} completed in {elapsed:.2f}s")
    return res


def main():
    args = parse_args()
    print("=" * 70)
    print("EXPERIMENT 1.1: Basic FFT Analysis on Crypto Data")
    print("=" * 70)
    print(f"Symbols: {args.symbols} | Hours: {args.hours} | Real: {args.use_real} | Mock: {args.mock}")

    results_dir = Path(args.output_dir) if args.output_dir else (Path(__file__).parent.parent / "results" / "01_basic_fft")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    failures: List[str] = []
    for sym in args.symbols:
        print("\n" + "=" * 70)
        print(f"ANALYZING {sym}")
        print("=" * 70)
        try:
            all_results[sym] = run_for_symbol(sym, args, results_dir)
        except Exception as e:
            failures.append(sym)
            print(f"[ERROR] Failed {sym}: {e}\n{traceback.format_exc()}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 1.1 COMPLETE - SUMMARY")
    print("=" * 70)
    success_symbols = []
    for sym, res in all_results.items():
        print(f"\n{sym}:")
        print(f"  Top Period: {res['summary']['top_period_hours']:.1f}h ({res['summary']['top_period_days']:.1f} days)")
        print(f"  Reconstruction R²: {res['summary']['reconstruction_r2']:.4f}")
        if res['summary']['reconstruction_r2'] > 0.80:
            success_symbols.append(sym)

    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)
    if success_symbols:
        print(f"✓ Symbols passing R² > 0.80: {success_symbols}")
    else:
        print("✗ No symbols exceeded R² > 0.80 threshold")
    print("✓ Visual plots generated (unless --no-plots passed)")
    if failures:
        print(f"⚠ Failures: {failures}")

    all_r2 = [res['summary']['reconstruction_r2'] for res in all_results.values()]
    if all_r2:
        avg_r2 = sum(all_r2)/len(all_r2)
        print(f"Average R²: {avg_r2:.4f}")

    print("\n" + "=" * 70)
    if success_symbols:
        print("🎉 EXPERIMENT 1.1: SUCCESS for at least one symbol.")
        print("Next Steps:\n  1. Inspect results JSON & plots\n  2. Proceed to Experiment 1.2 (Holographic Memory Test)\n  3. Consider enabling --use-real if mock was used")
    else:
        print("⚠️  EXPERIMENT 1.1: Review reconstruction quality.")
        print("Suggestions:\n  - Adjust --components or --hours\n  - Try --use-real if using mock\n  - Check for outliers / normalization")
    print("=" * 70)


if __name__ == "__main__":
    main()
