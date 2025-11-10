"""
Experiment 2.1: Regime Detection with FFT Features

Objective: Test whether frequency-domain features improve regime detection accuracy
          compared to time-domain features alone.

Hypothesis: FFT features (low_freq_power, spectral_entropy, band_power_ratio) 
           can capture market regime characteristics that time-domain indicators miss.

Success Criteria:
- >5% improvement in regime classification accuracy
- FFT features show significant correlation with regimes
- Combined model outperforms time-domain-only baseline

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))

from regime_detector import RegimeDetector

fetch_ohlcv = None
try:
    import fetch_data as _fetch_data_module
    fetch_ohlcv = getattr(_fetch_data_module, 'fetch_ohlcv', None)
except Exception:
    print("[INFO] fetch_ohlcv not available – using yfinance")

try:
    import yfinance as yf
except Exception:
    yf = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class FFTRegimeFeatureExtractor:
    """Extract frequency-domain features for regime detection."""
    
    def __init__(self, window_size: int = 168):
        """
        Args:
            window_size: Hours of data for FFT analysis (default 168 = 1 week)
        """
        self.window_size = window_size
        
    def extract_fft_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features from price data.
        
        Args:
            prices: Array of price data (length >= window_size)
            
        Returns:
            Dictionary of FFT features
        """
        if len(prices) < self.window_size:
            return self._get_default_features()
        
        # Use last window_size prices
        window = prices[-self.window_size:]
        
        # Normalize
        normalized = (window - window.mean()) / (window.std() + 1e-8)
        
        # Compute FFT
        fft = np.fft.fft(normalized)
        freqs = np.fft.fftfreq(len(normalized), d=1.0)
        power_spectrum = np.abs(fft) ** 2
        
        # Work with positive frequencies only
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_power = power_spectrum[positive_mask]
        
        # Define bands (based on Experiment 1.3 results)
        n_pos = len(positive_freqs)
        low_band = positive_power[:n_pos//4]  # 0-25%: Long-term trends
        mid_band = positive_power[n_pos//4:3*n_pos//4]  # 25-75%: Medium cycles
        high_band = positive_power[3*n_pos//4:]  # 75-100%: Noise
        
        # Calculate features
        total_power = positive_power.sum()
        
        features = {
            # Band power percentages
            'low_freq_power_pct': float(100 * low_band.sum() / (total_power + 1e-8)),
            'mid_freq_power_pct': float(100 * mid_band.sum() / (total_power + 1e-8)),
            'high_freq_power_pct': float(100 * high_band.sum() / (total_power + 1e-8)),
            
            # Band power ratios
            'low_high_power_ratio': float(low_band.sum() / (high_band.sum() + 1e-8)),
            'low_mid_power_ratio': float(low_band.sum() / (mid_band.sum() + 1e-8)),
            
            # Dominant frequency analysis
            'dominant_freq_idx': int(np.argmax(positive_power)),
            'dominant_freq_power_pct': float(100 * positive_power.max() / (total_power + 1e-8)),
            'dominant_period_hours': float(1.0 / (positive_freqs[np.argmax(positive_power)] + 1e-8)),
            
            # Spectral entropy (measure of randomness)
            'spectral_entropy': float(self._calculate_spectral_entropy(positive_power)),
            
            # Power concentration (how concentrated is power in top N components?)
            'power_concentration_top10': float(100 * np.sort(positive_power)[-10:].sum() / (total_power + 1e-8)),
            'power_concentration_top25': float(100 * np.sort(positive_power)[-25:].sum() / (total_power + 1e-8)),
            
            # Total power (proxy for overall volatility)
            'total_spectral_power': float(np.log10(total_power + 1)),
        }
        
        return features
    
    def _calculate_spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """
        Calculate spectral entropy (Shannon entropy of normalized power spectrum).
        High entropy = noisy/random signal
        Low entropy = structured/predictable signal
        """
        # Normalize to probability distribution
        power_norm = power_spectrum / (power_spectrum.sum() + 1e-8)
        
        # Calculate Shannon entropy
        # Avoid log(0) by adding small epsilon
        power_norm = power_norm + 1e-10
        entropy = -np.sum(power_norm * np.log2(power_norm))
        
        # Normalize by max possible entropy
        max_entropy = np.log2(len(power_spectrum))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when not enough data."""
        return {
            'low_freq_power_pct': 0.0,
            'mid_freq_power_pct': 0.0,
            'high_freq_power_pct': 0.0,
            'low_high_power_ratio': 1.0,
            'low_mid_power_ratio': 1.0,
            'dominant_freq_idx': 0,
            'dominant_freq_power_pct': 0.0,
            'dominant_period_hours': 0.0,
            'spectral_entropy': 0.5,
            'power_concentration_top10': 0.0,
            'power_concentration_top25': 0.0,
            'total_spectral_power': 0.0,
        }


def load_historical_data(symbol: str, hours: int = 5000) -> pd.DataFrame:
    """Load historical OHLCV data."""
    logging.info(f"Loading {hours}h of data for {symbol}")
    
    if yf is None:
        raise RuntimeError("yfinance not available")
    
    if symbol.startswith('BTC'): ticker = 'BTC-USD'
    elif symbol.startswith('ETH'): ticker = 'ETH-USD'
    else: ticker = symbol.replace('/', '-')
    
    # Calculate days needed (hours / 24 with buffer)
    days_needed = int(hours / 24 * 1.5)
    
    try:
        hist = yf.Ticker(ticker).history(period=f'{days_needed}d', interval='1h')
        if hist.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Standardize column names
        df = pd.DataFrame({
            'open': hist['Open'].values,
            'high': hist['High'].values,
            'low': hist['Low'].values,
            'close': hist['Close'].values,
            'volume': hist['Volume'].values,
        }, index=hist.index)
        
        # Take last N hours
        df = df.tail(hours)
        
        logging.info(f"Loaded {len(df)} hours of data")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def generate_regime_labels(df: pd.DataFrame, regime_detector: RegimeDetector,
                           min_window: int = 200) -> pd.Series:
    """
    Generate regime labels for each row using RegimeDetector.
    
    Args:
        df: OHLCV dataframe
        regime_detector: Initialized RegimeDetector
        min_window: Minimum rows needed before detection starts
        
    Returns:
        Series of regime labels
    """
    regimes = []
    
    for i in range(len(df)):
        if i < min_window:
            regimes.append('unknown')
        else:
            # Get window of data up to current point
            window = df.iloc[:i+1]
            regime = regime_detector.detect_regime(window, include_vix=False)
            regimes.append(regime)
    
    return pd.Series(regimes, index=df.index)


def extract_all_features(df: pd.DataFrame, fft_extractor: FFTRegimeFeatureExtractor,
                        include_fft: bool = True) -> pd.DataFrame:
    """
    Extract both time-domain and frequency-domain features.
    
    Args:
        df: OHLCV dataframe
        fft_extractor: FFT feature extractor
        include_fft: Whether to include FFT features
        
    Returns:
        DataFrame with all features
    """
    features_list = []
    
    for i in range(len(df)):
        row_features = {}
        
        # Time-domain features (simple technical indicators)
        if i >= 20:
            window = df.iloc[max(0, i-200):i+1]
            
            # Volatility (returns std)
            returns = window['close'].pct_change().dropna()
            row_features['returns_std'] = float(returns.std())
            row_features['returns_mean'] = float(returns.mean())
            
            # Price momentum
            row_features['momentum_20'] = float((window['close'].iloc[-1] / window['close'].iloc[-20] - 1))
            
            # Volume ratio
            vol_mean = window['volume'].mean()
            row_features['volume_ratio'] = float(window['volume'].iloc[-1] / (vol_mean + 1e-8))
            
            # ATR-like
            high_low = (window['high'] - window['low']).mean()
            row_features['avg_range'] = float(high_low / window['close'].mean())
            
            # Trend (simple linear regression slope)
            x = np.arange(len(window))
            y = window['close'].values
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                row_features['trend_slope'] = float(slope / window['close'].mean())
            else:
                row_features['trend_slope'] = 0.0
                
        else:
            # Not enough data
            row_features.update({
                'returns_std': 0.0,
                'returns_mean': 0.0,
                'momentum_20': 0.0,
                'volume_ratio': 1.0,
                'avg_range': 0.0,
                'trend_slope': 0.0,
            })
        
        # FFT features
        if include_fft and i >= fft_extractor.window_size:
            prices = df['close'].iloc[:i+1].values
            fft_features = fft_extractor.extract_fft_features(prices)
            row_features.update(fft_features)
        elif include_fft:
            row_features.update(fft_extractor._get_default_features())
        
        features_list.append(row_features)
    
    return pd.DataFrame(features_list, index=df.index)


def train_and_evaluate_models(X_time: pd.DataFrame, X_combined: pd.DataFrame,
                              y: pd.Series, test_size: float = 0.3) -> Dict:
    """
    Train and evaluate regime classifiers.
    
    Args:
        X_time: Features with time-domain only
        X_combined: Features with time + frequency domain
        y: Regime labels
        test_size: Fraction for test set
        
    Returns:
        Dictionary with results
    """
    # Remove 'unknown' labels
    valid_mask = y != 'unknown'
    X_time = X_time[valid_mask]
    X_combined = X_combined[valid_mask]
    y = y[valid_mask]
    
    if len(y) < 100:
        raise ValueError("Not enough labeled data for training")
    
    logging.info(f"Training with {len(y)} samples")
    logging.info(f"Regime distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_time_train, X_time_test, y_train, y_test = train_test_split(
        X_time, y, test_size=test_size, random_state=42, stratify=y
    )
    X_comb_train, X_comb_test, _, _ = train_test_split(
        X_combined, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train time-domain only model (baseline)
    print("\n=== Training Baseline (Time-Domain Only) ===")
    model_time = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model_time.fit(X_time_train, y_train)
    y_pred_time = model_time.predict(X_time_test)
    acc_time = accuracy_score(y_test, y_pred_time)
    
    print(f"Baseline Accuracy: {acc_time:.4f} ({acc_time*100:.2f}%)")
    print("\nBaseline Classification Report:")
    print(classification_report(y_test, y_pred_time))
    
    # Train combined model
    print("\n=== Training Combined (Time + FFT Features) ===")
    model_combined = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model_combined.fit(X_comb_train, y_train)
    y_pred_combined = model_combined.predict(X_comb_test)
    acc_combined = accuracy_score(y_test, y_pred_combined)
    
    print(f"Combined Accuracy: {acc_combined:.4f} ({acc_combined*100:.2f}%)")
    print("\nCombined Classification Report:")
    print(classification_report(y_test, y_pred_combined))
    
    # Calculate improvement
    improvement = (acc_combined - acc_time) / acc_time * 100
    absolute_improvement = (acc_combined - acc_time) * 100
    
    print(f"\n=== Improvement ===")
    print(f"Absolute: {absolute_improvement:+.2f} percentage points")
    print(f"Relative: {improvement:+.2f}%")
    
    # Feature importance for combined model
    feature_importance = pd.DataFrame({
        'feature': X_combined.columns,
        'importance': model_combined.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Most Important Features (Combined Model) ===")
    print(feature_importance.head(10).to_string(index=False))
    
    results = {
        'baseline_accuracy': float(acc_time),
        'combined_accuracy': float(acc_combined),
        'absolute_improvement_pct': float(absolute_improvement),
        'relative_improvement_pct': float(improvement),
        'feature_importance': feature_importance.to_dict('records'),
        'test_samples': int(len(y_test)),
        'regime_distribution': y.value_counts().to_dict(),
    }
    
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2.1: FFT Features for Regime Detection")
    p.add_argument('--symbol', default='BTC/USDT', help='Symbol to analyze')
    p.add_argument('--hours', type=int, default=3000, help='Hours of historical data')
    p.add_argument('--fft-window', type=int, default=168, help='FFT window size (hours)')
    p.add_argument('--output-dir', default=None, help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("EXPERIMENT 2.1: Regime Detection with FFT Features")
    print("=" * 70)
    print(f"Symbol: {args.symbol} | Hours: {args.hours} | FFT Window: {args.fft_window}h")
    
    results_dir = Path(args.output_dir) if args.output_dir else \
                  (Path(__file__).parent.parent / "results" / "05_regime_features")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        start = time.time()
        
        # Load data
        df = load_historical_data(args.symbol, hours=args.hours)
        
        # Initialize
        regime_detector = RegimeDetector()
        fft_extractor = FFTRegimeFeatureExtractor(window_size=args.fft_window)
        
        # Generate regime labels
        print("\n=== Generating Regime Labels ===")
        regimes = generate_regime_labels(df, regime_detector)
        
        # Extract features
        print("\n=== Extracting Features ===")
        print("Time-domain features...")
        X_time = extract_all_features(df, fft_extractor, include_fft=False)
        
        print("Time + FFT features...")
        X_combined = extract_all_features(df, fft_extractor, include_fft=True)
        
        # Train and evaluate
        print("\n=== Training Models ===")
        results = train_and_evaluate_models(X_time, X_combined, regimes)
        
        # Save results
        results['experiment'] = '2.1_regime_fft_features'
        results['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        results['symbol'] = args.symbol
        results['hours'] = args.hours
        results['fft_window'] = args.fft_window
        
        # Success criteria
        results['success_criteria'] = {
            'target_improvement_pct': 5.0,
            'achieved_improvement_pct': results['absolute_improvement_pct'],
            'success': bool(results['absolute_improvement_pct'] >= 5.0)
        }
        
        results_path = results_dir / f'experiment_2.1_{args.symbol.replace("/","_")}_{results["timestamp"]}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        elapsed = time.time() - start
        print(f"\n[TIME] Completed in {elapsed:.2f}s")
        
        # Summary
        print("\n" + "=" * 70)
        print("EXPERIMENT 2.1 SUMMARY")
        print("=" * 70)
        if results['success_criteria']['success']:
            print(f"[SUCCESS] {results['absolute_improvement_pct']:+.2f}pp improvement (target: 5pp)")
            print("\nKey Findings:")
            print(f"  - Baseline accuracy: {results['baseline_accuracy']*100:.2f}%")
            print(f"  - With FFT features: {results['combined_accuracy']*100:.2f}%")
            print("\nTop FFT Features:")
            fft_features = [f for f in results['feature_importance'][:10] 
                          if any(x in f['feature'] for x in ['freq', 'spectral', 'power', 'entropy'])]
            for feat in fft_features[:5]:
                print(f"  - {feat['feature']}: {feat['importance']:.4f}")
        else:
            print(f"[PARTIAL] {results['absolute_improvement_pct']:+.2f}pp improvement (target: 5pp)")
            print("FFT features provide some benefit but below target threshold")
        
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
