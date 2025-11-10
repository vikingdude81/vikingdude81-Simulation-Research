"""
Experiment 1.3: Frequency Band Analysis

Objective: Identify which frequency bands matter most for crypto price reconstruction.
          Test whether low (trends), mid (cycles), or high (noise) frequencies
          carry the most trading-relevant information.

Hypothesis: Low frequencies (long-term trends) carry most signal, high frequencies
           are mostly noise. This will inform band-specific trading strategies.

Success Criteria:
- Quantify importance of LOW (0-25%), MID (25-75%), HIGH (75-100%) bands
- LOW-only reconstruction R² > 0.80
- MID-only reconstruction R² > 0.40
- HIGH-only reconstruction R² < 0.20
- Identify optimal band combinations for different timeframes

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
from typing import List, Optional, Dict, Tuple

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

fetch_ohlcv = None
try:
    import fetch_data as _fetch_data_module
    fetch_ohlcv = getattr(_fetch_data_module, 'fetch_ohlcv', None)
except Exception:
    print("[INFO] fetch_ohlcv not available – using internal fetch or mock.")

try:
    import yfinance as yf
except Exception:
    yf = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class FrequencyBandAnalyzer:
    """Analyze importance of different frequency bands in crypto price data."""
    
    def __init__(self, symbol: str = "BTC/USDT", hours: int = 1000,
                 use_real: bool = False, mock: bool = False, quiet: bool = False):
        """Initialize analyzer.
        
        Args:
            symbol: Trading pair
            hours: Hours of historical data
            use_real: Fetch real market data
            mock: Force mock data
            quiet: Reduce console output
        """
        self.symbol = symbol
        self.hours = hours
        self.use_real = use_real
        self.mock = mock
        self.quiet = quiet
        
        # Core data
        self.prices: Optional[np.ndarray] = None
        self.fft: Optional[np.ndarray] = None
        self.frequencies: Optional[np.ndarray] = None
        self.power_spectrum: Optional[np.ndarray] = None
        
        # Band definitions (as fraction of total frequencies)
        self.band_definitions = {
            'LOW': (0.0, 0.25),    # 0-25%: Long-term trends
            'MID': (0.25, 0.75),   # 25-75%: Medium cycles
            'HIGH': (0.75, 1.0)    # 75-100%: Short-term noise
        }
        
        # Results storage
        self.band_results: Dict[str, Dict] = {}
        self.combination_results: Dict[str, Dict] = {}
        
    def _generate_mock(self) -> np.ndarray:
        """Generate synthetic price data with known frequency structure."""
        t = np.arange(self.hours)
        base = 50000 if 'BTC' in self.symbol else 3000
        return (
            base +
            0.02 * base * np.sin(2 * np.pi * t / 168) +    # Weekly (LOW)
            0.015 * base * np.sin(2 * np.pi * t / 48) +    # 2-day (MID)
            0.01 * base * np.sin(2 * np.pi * t / 24) +     # Daily (MID)
            0.005 * base * np.sin(2 * np.pi * t / 6) +     # 6-hour (MID)
            0.003 * base * np.random.randn(self.hours)     # Noise (HIGH)
        )
    
    def _simple_yfinance_fetch(self) -> Optional[np.ndarray]:
        """Fetch data using yfinance."""
        if yf is None:
            return None
        
        if self.symbol.startswith('BTC'): ticker = 'BTC-USD'
        elif self.symbol.startswith('ETH'): ticker = 'ETH-USD'
        else: ticker = self.symbol.replace('/', '-')
        
        try:
            hist = yf.Ticker(ticker).history(period='60d', interval='1h')
            if hist.empty:
                return None
            closes = hist['Close'].tail(self.hours).values
            closes = np.asarray(closes, dtype=float)
            if len(closes) < self.hours:
                closes = np.pad(closes, (self.hours - len(closes), 0), mode='edge')
            return closes
        except Exception:
            return None
    
    def load_data(self) -> None:
        """Load price data."""
        logging.info(f"Loading {self.hours}h for {self.symbol}")
        
        used_source = "mock"
        if not self.mock and self.use_real:
            if fetch_ohlcv is not None:
                try:
                    df = fetch_ohlcv(self.symbol, '1h', limit=self.hours)
                    self.prices = df['close'].values
                    used_source = 'fetch_ohlcv'
                except Exception as e:
                    logging.warning(f"fetch_ohlcv failed: {e}")
            
            if self.prices is None:
                fetched = self._simple_yfinance_fetch()
                if fetched is not None:
                    self.prices = fetched
                    used_source = 'yfinance'
        
        if self.prices is None:
            self.prices = self._generate_mock()
            used_source = 'mock'
        
        self.prices = np.asarray(self.prices, dtype=float)
        logging.info(f"Data source: {used_source}; points: {len(self.prices)}")
    
    def compute_fft(self) -> None:
        """Compute FFT of price data."""
        if self.prices is None:
            raise RuntimeError("Prices not loaded. Call load_data() first.")
        
        if not self.quiet:
            print("\n=== Computing FFT ===")
        
        # Normalize prices
        normalized = (self.prices - self.prices.mean()) / self.prices.std()
        
        # Compute FFT
        self.fft = np.fft.fft(normalized)
        self.frequencies = np.fft.fftfreq(len(normalized), d=1.0)
        self.power_spectrum = np.abs(self.fft) ** 2
        
        if not self.quiet:
            print(f"FFT computed: {len(self.fft)} components")
    
    def _get_band_indices(self, band_name: str) -> np.ndarray:
        """Get indices corresponding to a frequency band.
        
        Works with positive frequencies only (first half of FFT),
        then mirrors to negative frequencies for proper reconstruction.
        """
        n = len(self.fft)
        n_half = n // 2
        
        start_frac, end_frac = self.band_definitions[band_name]
        start_idx = int(n_half * start_frac)
        end_idx = int(n_half * end_frac)
        
        # Get positive frequency indices
        positive_indices = np.arange(start_idx, end_idx)
        
        # Mirror to negative frequencies (FFT symmetry)
        negative_indices = n - positive_indices
        negative_indices = negative_indices[negative_indices < n]
        
        # Combine both
        all_indices = np.concatenate([positive_indices, negative_indices])
        
        return all_indices
    
    def reconstruct_from_band(self, band_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Reconstruct signal using only specified frequency band.
        
        Args:
            band_name: 'LOW', 'MID', or 'HIGH'
            
        Returns:
            (reconstructed_prices, metrics)
        """
        if self.fft is None or self.prices is None:
            raise RuntimeError("FFT not computed. Call compute_fft() first.")
        
        if not self.quiet:
            print(f"\n=== Reconstructing from {band_name} Band ===")
        
        # Create filtered FFT (keep only band components)
        filtered_fft = np.zeros_like(self.fft)
        band_indices = self._get_band_indices(band_name)
        filtered_fft[band_indices] = self.fft[band_indices]
        
        # Calculate band statistics
        band_power = self.power_spectrum[band_indices].sum()
        total_power = self.power_spectrum.sum()
        power_pct = 100 * band_power / total_power
        
        # Reconstruct signal
        reconstructed_normalized = np.fft.ifft(filtered_fft).real
        reconstructed = reconstructed_normalized * self.prices.std() + self.prices.mean()
        
        # Calculate metrics
        mse = np.mean((self.prices - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.prices - reconstructed))
        r2 = 1 - (np.sum((self.prices - reconstructed)**2) / 
                  np.sum((self.prices - self.prices.mean())**2))
        
        # Calculate frequency range for this band
        band_freqs = self.frequencies[band_indices]
        positive_freqs = band_freqs[band_freqs > 0]
        if len(positive_freqs) > 0:
            freq_range = (float(positive_freqs.min()), float(positive_freqs.max()))
            period_range = (1.0/freq_range[1], 1.0/freq_range[0])  # (min_period, max_period)
        else:
            freq_range = (0, 0)
            period_range = (0, 0)
        
        metrics = {
            'band': band_name,
            'components': int(len(band_indices)),
            'components_pct': float(100 * len(band_indices) / len(self.fft)),
            'power_pct': float(power_pct),
            'freq_range_cycles_per_hour': freq_range,
            'period_range_hours': period_range,
            'period_range_days': (period_range[0]/24, period_range[1]/24),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }
        
        if not self.quiet:
            print(f"  Components: {len(band_indices)}/{len(self.fft)} ({metrics['components_pct']:.1f}%)")
            print(f"  Power: {power_pct:.1f}% of total")
            print(f"  Period range: {period_range[0]:.1f}h - {period_range[1]:.1f}h ({period_range[0]/24:.1f} - {period_range[1]/24:.1f} days)")
            print(f"  Reconstruction R²: {r2:.4f} ({r2*100:.2f}%)")
            print(f"  RMSE: ${rmse:.2f} | MAE: ${mae:.2f}")
        
        return reconstructed, metrics
    
    def reconstruct_from_combination(self, bands: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Reconstruct signal using combination of frequency bands.
        
        Args:
            bands: List of band names to combine (e.g., ['LOW', 'MID'])
            
        Returns:
            (reconstructed_prices, metrics)
        """
        if self.fft is None or self.prices is None:
            raise RuntimeError("FFT not computed. Call compute_fft() first.")
        
        combo_name = '+'.join(bands)
        if not self.quiet:
            print(f"\n=== Reconstructing from {combo_name} ===")
        
        # Create filtered FFT with all specified bands
        filtered_fft = np.zeros_like(self.fft)
        all_indices = np.array([], dtype=int)
        
        for band in bands:
            band_indices = self._get_band_indices(band)
            filtered_fft[band_indices] = self.fft[band_indices]
            all_indices = np.concatenate([all_indices, band_indices])
        
        # Calculate combined statistics
        combo_power = self.power_spectrum[all_indices].sum()
        total_power = self.power_spectrum.sum()
        power_pct = 100 * combo_power / total_power
        
        # Reconstruct
        reconstructed_normalized = np.fft.ifft(filtered_fft).real
        reconstructed = reconstructed_normalized * self.prices.std() + self.prices.mean()
        
        # Metrics
        mse = np.mean((self.prices - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.prices - reconstructed))
        r2 = 1 - (np.sum((self.prices - reconstructed)**2) / 
                  np.sum((self.prices - self.prices.mean())**2))
        
        metrics = {
            'combination': combo_name,
            'bands': bands,
            'components': int(len(all_indices)),
            'components_pct': float(100 * len(all_indices) / len(self.fft)),
            'power_pct': float(power_pct),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }
        
        if not self.quiet:
            print(f"  Components: {len(all_indices)}/{len(self.fft)} ({metrics['components_pct']:.1f}%)")
            print(f"  Power: {power_pct:.1f}% of total")
            print(f"  Reconstruction R²: {r2:.4f} ({r2*100:.2f}%)")
        
        return reconstructed, metrics
    
    def run_band_analysis(self) -> Dict:
        """Run comprehensive band analysis."""
        if not self.quiet:
            print("\n" + "=" * 70)
            print("FREQUENCY BAND ANALYSIS")
            print("=" * 70)
        
        results = {}
        reconstructions = {}
        
        # Test individual bands
        for band in ['LOW', 'MID', 'HIGH']:
            recon, metrics = self.reconstruct_from_band(band)
            results[band] = metrics
            reconstructions[band] = recon
        
        # Test combinations
        combinations = [
            ['LOW', 'MID'],
            ['LOW', 'HIGH'],
            ['MID', 'HIGH'],
        ]
        
        for combo in combinations:
            combo_name = '+'.join(combo)
            recon, metrics = self.reconstruct_from_combination(combo)
            results[combo_name] = metrics
            reconstructions[combo_name] = recon
        
        self.band_results = results
        return results, reconstructions
    
    def plot_band_comparison(self, reconstructions: Dict, save_path: Optional[Path] = None):
        """Visualize original vs band reconstructions."""
        if self.prices is None:
            raise RuntimeError("No prices loaded")
        
        bands = ['LOW', 'MID', 'HIGH']
        n_plots = len(bands) + 1  # +1 for original
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3 * n_plots))
        
        # Plot original
        ax = axes[0]
        ax.plot(self.prices, 'b-', linewidth=1, label='Original')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{self.symbol} - Original Price Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot each band reconstruction
        for i, band in enumerate(bands, 1):
            ax = axes[i]
            ax.plot(self.prices, 'b-', linewidth=0.5, alpha=0.3, label='Original')
            
            if band in reconstructions:
                recon = reconstructions[band]
                metrics = self.band_results[band]
                r2 = metrics['r2']
                power = metrics['power_pct']
                
                ax.plot(recon, 'r-', linewidth=1, alpha=0.8, 
                       label=f'{band} (R²={r2:.3f}, {power:.1f}% power)')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{self.symbol} - {band} Band Reconstruction')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nBand comparison plot saved to: {save_path}")
        
        plt.close()
    
    def plot_band_metrics(self, save_path: Optional[Path] = None):
        """Plot band importance metrics."""
        if not self.band_results:
            raise RuntimeError("No results. Run run_band_analysis() first.")
        
        # Organize data for individual bands
        bands = ['LOW', 'MID', 'HIGH']
        r2_values = [self.band_results[b]['r2'] for b in bands]
        power_values = [self.band_results[b]['power_pct'] for b in bands]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: R² by band
        ax = axes[0, 0]
        bars = ax.bar(bands, r2_values, color=['blue', 'green', 'red'], alpha=0.7)
        ax.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='80% target')
        ax.set_ylabel('Reconstruction R²')
        ax.set_title('Reconstruction Quality by Band')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 2: Power distribution
        ax = axes[0, 1]
        bars = ax.bar(bands, power_values, color=['blue', 'green', 'red'], alpha=0.7)
        ax.set_ylabel('Power (%)')
        ax.set_title('Power Distribution by Band')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, power_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Period ranges
        ax = axes[1, 0]
        for i, band in enumerate(bands):
            period_range = self.band_results[band]['period_range_days']
            ax.barh(i, period_range[1] - period_range[0], 
                   left=period_range[0], height=0.5,
                   color=['blue', 'green', 'red'][i], alpha=0.7,
                   label=f'{band}: {period_range[0]:.1f}-{period_range[1]:.1f} days')
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands)
        ax.set_xlabel('Period (days)')
        ax.set_title('Frequency Band Period Ranges')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        # Plot 4: Combination performance
        ax = axes[1, 1]
        combos = ['LOW', 'MID', 'HIGH', 'LOW+MID', 'LOW+HIGH', 'MID+HIGH']
        combo_r2 = [self.band_results[c]['r2'] if c in self.band_results else 0 
                   for c in combos]
        
        bars = ax.bar(range(len(combos)), combo_r2, 
                     color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'],
                     alpha=0.7)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=45, ha='right')
        ax.set_ylabel('Reconstruction R²')
        ax.set_title('Single Bands vs Combinations')
        ax.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='80% target')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, combo_r2):
            height = bar.get_height()
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nBand metrics plot saved to: {save_path}")
        
        plt.close()
    
    def save_results(self, save_dir: Path) -> Dict:
        """Save experiment results to JSON."""
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        assert self.prices is not None, "Prices must be loaded"
        
        # Evaluate success criteria
        success_criteria = {
            'LOW_r2_gt_0.80': {
                'target': 0.80,
                'actual': self.band_results['LOW']['r2'],
                'achieved': bool(self.band_results['LOW']['r2'] > 0.80)
            },
            'MID_r2_gt_0.40': {
                'target': 0.40,
                'actual': self.band_results['MID']['r2'],
                'achieved': bool(self.band_results['MID']['r2'] > 0.40)
            },
            'HIGH_r2_lt_0.20': {
                'target': 0.20,
                'actual': self.band_results['HIGH']['r2'],
                'achieved': bool(self.band_results['HIGH']['r2'] < 0.20)
            }
        }
        
        # Identify best band and combination
        best_single_band = max(['LOW', 'MID', 'HIGH'], 
                              key=lambda b: self.band_results[b]['r2'])
        
        results = {
            'experiment': '1.3_frequency_band_analysis',
            'timestamp': timestamp,
            'symbol': self.symbol,
            'data_points': int(len(self.prices)),
            'band_definitions': self.band_definitions,
            'individual_bands': {k: v for k, v in self.band_results.items() 
                               if k in ['LOW', 'MID', 'HIGH']},
            'combinations': {k: v for k, v in self.band_results.items() 
                           if '+' in k},
            'success_criteria': success_criteria,
            'summary': {
                'all_criteria_met': bool(all(c['achieved'] for c in success_criteria.values())),
                'best_single_band': best_single_band,
                'best_single_band_r2': float(self.band_results[best_single_band]['r2']),
                'low_band_dominance': float(self.band_results['LOW']['power_pct']),
                'recommended_filter': 'Remove HIGH band' if self.band_results['HIGH']['r2'] < 0.20 else 'Keep all bands',
            }
        }
        
        results_path = save_dir / f'experiment_1.3_{self.symbol.replace("/", "_")}_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        return results


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1.3: Frequency Band Analysis")
    p.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                   help='Symbols to analyze')
    p.add_argument('--hours', type=int, default=1000, help='Hours of history')
    p.add_argument('--use-real', action='store_true', help='Use real market data')
    p.add_argument('--mock', action='store_true', help='Force mock data')
    p.add_argument('--quiet', action='store_true', help='Reduce output')
    p.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    p.add_argument('--output-dir', default=None, help='Output directory')
    return p.parse_args()


def run_for_symbol(symbol: str, args, results_dir: Path):
    """Run band analysis for one symbol."""
    start = time.time()
    
    analyzer = FrequencyBandAnalyzer(
        symbol=symbol,
        hours=args.hours,
        use_real=args.use_real,
        mock=args.mock,
        quiet=args.quiet
    )
    
    analyzer.load_data()
    analyzer.compute_fft()
    
    results, reconstructions = analyzer.run_band_analysis()
    
    if not args.no_plots:
        # Band comparison plot
        plot1_path = results_dir / f'{symbol.replace("/","_").lower()}_band_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        analyzer.plot_band_comparison(reconstructions, save_path=plot1_path)
        
        # Band metrics plot
        plot2_path = results_dir / f'{symbol.replace("/","_").lower()}_band_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        analyzer.plot_band_metrics(save_path=plot2_path)
    
    final_results = analyzer.save_results(results_dir)
    
    elapsed = time.time() - start
    if not args.quiet:
        print(f"[TIME] {symbol} completed in {elapsed:.2f}s")
    
    return final_results


def main():
    args = parse_args()
    
    print("=" * 70)
    print("EXPERIMENT 1.3: Frequency Band Analysis")
    print("=" * 70)
    print(f"Symbols: {args.symbols} | Hours: {args.hours}")
    print(f"Bands: LOW (0-25%), MID (25-75%), HIGH (75-100%)")
    
    results_dir = Path(args.output_dir) if args.output_dir else \
                  (Path(__file__).parent.parent / "results" / "03_band_analysis")
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
    print("EXPERIMENT 1.3 COMPLETE - SUMMARY")
    print("=" * 70)
    
    for sym, res in all_results.items():
        print(f"\n{sym}:")
        criteria = res['success_criteria']
        for criterion, data in criteria.items():
            status = "✓" if data['achieved'] else "✗"
            print(f"  {status} {criterion}: {data['actual']:.4f} (target: {data['target']:.2f})")
        
        print(f"  Best single band: {res['summary']['best_single_band']} (R²={res['summary']['best_single_band_r2']:.4f})")
        print(f"  LOW band power: {res['summary']['low_band_dominance']:.1f}%")
        print(f"  Recommendation: {res['summary']['recommended_filter']}")
    
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)
    
    all_passed = sum(1 for res in all_results.values() 
                    if res['summary']['all_criteria_met'])
    
    if all_passed > 0:
        print(f"✓ {all_passed}/{len(all_results)} symbols passed all criteria")
    else:
        print("⚠ Not all criteria met")
    
    if failures:
        print(f"⚠ Failures: {failures}")
    
    print("\n" + "=" * 70)
    if all_passed > 0:
        print("🎉 EXPERIMENT 1.3: SUCCESS")
        print("Key Findings:")
        print("  - Frequency band importance quantified")
        print("  - LOW bands carry long-term trends")
        print("  - HIGH bands are mostly noise")
        print("\nNext Steps:")
        print("  1. Review band comparison and metrics plots")
        print("  2. Implement band-specific filters in trading system")
        print("  3. Proceed to Experiment 2.1 (Regime Detection with FFT)")
        print("  4. Consider multi-timeframe strategies based on band decomposition")
    else:
        print("📊 EXPERIMENT 1.3: REVIEW NEEDED")
        print("Suggestions:")
        print("  - Examine which criteria failed")
        print("  - Consider adjusting band definitions")
        print("  - Test with different data periods")
    print("=" * 70)


if __name__ == "__main__":
    main()
