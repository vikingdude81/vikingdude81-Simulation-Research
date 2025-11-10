"""
Validate Regime Detector Against Known Market Events

Tests the calibrated regime detector against known crypto market events
to ensure it correctly classifies:
- Crisis periods (COVID crash, Terra/Luna collapse)
- Trending periods (2020-2021 bull run, 2023 recovery)
- Ranging/Volatile periods (2022 bear market, recent consolidation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from regime_detector import RegimeDetector
from datetime import datetime

def test_market_events():
    """Test regime detector on known market events"""
    
    print("🎯 VALIDATING REGIME DETECTOR ON KNOWN EVENTS")
    print("=" * 70)
    
    # Load data
    data_path = Path("DATA/yf_btc_1d.csv")
    print(f"\n📊 Loading BTC data from {data_path}...")
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    print(f"✅ Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Initialize detector with calibrated thresholds
    detector = RegimeDetector()
    
    # Define known market events
    events = [
        {
            'name': 'COVID Crash',
            'period': 'Feb-Mar 2020',
            'start': '2020-02-15',
            'end': '2020-03-31',
            'expected': ['crisis', 'volatile'],
            'description': 'Extreme volatility, BTC dropped from $10k to $5k in days'
        },
        {
            'name': 'Post-COVID Recovery',
            'period': 'Apr-Jul 2020',
            'start': '2020-04-01',
            'end': '2020-07-31',
            'expected': ['ranging', 'trending'],
            'description': 'Consolidation and gradual recovery'
        },
        {
            'name': 'Bull Run Start',
            'period': 'Oct-Dec 2020',
            'start': '2020-10-01',
            'end': '2020-12-31',
            'expected': ['trending'],
            'description': 'Strong uptrend, BTC $10k → $29k'
        },
        {
            'name': 'Bull Run Peak',
            'period': 'Jan-Apr 2021',
            'start': '2021-01-01',
            'end': '2021-04-30',
            'expected': ['trending', 'volatile'],
            'description': 'Parabolic rise to $64k ATH'
        },
        {
            'name': 'May 2021 Crash',
            'period': 'May 2021',
            'start': '2021-05-01',
            'end': '2021-05-31',
            'expected': ['crisis', 'volatile'],
            'description': 'China mining ban, $64k → $30k crash'
        },
        {
            'name': 'Summer 2021 Consolidation',
            'period': 'Jun-Sep 2021',
            'start': '2021-06-01',
            'end': '2021-09-30',
            'expected': ['ranging', 'volatile'],
            'description': 'Trading range $30k-$45k'
        },
        {
            'name': 'Q4 2021 Rally',
            'period': 'Oct-Nov 2021',
            'start': '2021-10-01',
            'end': '2021-11-30',
            'expected': ['trending'],
            'description': 'Second ATH attempt, $40k → $69k'
        },
        {
            'name': 'Bear Market Start',
            'period': 'Dec 2021 - Mar 2022',
            'start': '2021-12-01',
            'end': '2022-03-31',
            'expected': ['volatile', 'trending'],
            'description': 'Downtrend begins, $69k → $40k'
        },
        {
            'name': 'Terra/Luna Collapse',
            'period': 'May 2022',
            'start': '2022-05-01',
            'end': '2022-05-31',
            'expected': ['crisis', 'volatile'],
            'description': 'UST depeg, contagion, $40k → $26k'
        },
        {
            'name': 'Bear Market Bottom',
            'period': 'Jun-Nov 2022',
            'start': '2022-06-01',
            'end': '2022-11-30',
            'expected': ['volatile', 'ranging'],
            'description': 'FTX collapse, capitulation, $15k bottom'
        },
        {
            'name': '2023 Recovery',
            'period': 'Jan-Dec 2023',
            'start': '2023-01-01',
            'end': '2023-12-31',
            'expected': ['trending', 'ranging'],
            'description': 'ETF hopes, gradual recovery $16k → $44k'
        },
        {
            'name': '2024 ETF Rally',
            'period': 'Jan-Mar 2024',
            'start': '2024-01-01',
            'end': '2024-03-31',
            'expected': ['trending', 'volatile'],
            'description': 'Spot ETF approval, strong rally'
        },
        {
            'name': 'Recent Period',
            'period': 'Apr 2024 - Present',
            'start': '2024-04-01',
            'end': '2025-10-24',
            'expected': ['ranging', 'volatile', 'trending'],
            'description': 'Mixed conditions, consolidation and moves'
        }
    ]
    
    print(f"\n📅 Testing {len(events)} market events...\n")
    
    results = []
    
    for event in events:
        # Get data for this period
        period_data = df[(df.index >= event['start']) & (df.index <= event['end'])]
        
        if len(period_data) == 0:
            print(f"⚠️  {event['name']}: No data available")
            continue
        
        # Detect regime for this period
        regime = detector.detect_regime(period_data)
        confidence = detector.get_regime_confidence()
        
        # Check if detected regime matches expectations
        is_correct = regime in event['expected']
        status = "✅" if is_correct else "⚠️ "
        
        # Calculate price statistics
        price_start = period_data['close'].iloc[0]
        price_end = period_data['close'].iloc[-1]
        price_change = ((price_end / price_start) - 1) * 100
        price_min = period_data['close'].min()
        price_max = period_data['close'].max()
        price_range = ((price_max - price_min) / price_min) * 100
        
        result = {
            'event': event['name'],
            'period': event['period'],
            'expected': event['expected'],
            'detected': regime,
            'correct': is_correct,
            'confidence': confidence.get(regime, 0.0),
            'price_change': price_change,
            'price_range': price_range,
            'price_start': price_start,
            'price_end': price_end,
            'days': len(period_data)
        }
        results.append(result)
        
        print(f"{status} {event['name']} ({event['period']})")
        print(f"   Expected: {', '.join(event['expected'])}")
        print(f"   Detected: {regime} (confidence: {confidence.get(regime, 0.0):.2f})")
        print(f"   Price: ${price_start:,.0f} → ${price_end:,.0f} ({price_change:+.1f}%)")
        print(f"   Range: {price_range:.1f}% | Days: {len(period_data)}")
        print(f"   Note: {event['description']}")
        print()
    
    # Summary statistics
    print("=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    total_events = len(results)
    correct_detections = sum(1 for r in results if r['correct'])
    accuracy = (correct_detections / total_events * 100) if total_events > 0 else 0
    
    print(f"\nTotal events tested: {total_events}")
    print(f"Correct detections: {correct_detections}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Regime distribution
    detected_regimes = {}
    for r in results:
        regime = r['detected']
        detected_regimes[regime] = detected_regimes.get(regime, 0) + 1
    
    print(f"\n📈 Detected regime distribution:")
    for regime, count in sorted(detected_regimes.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_events * 100)
        print(f"   {regime:12s}: {count:2d} events ({pct:5.1f}%)")
    
    # Misclassifications
    misclassified = [r for r in results if not r['correct']]
    if misclassified:
        print(f"\n⚠️  Misclassified events ({len(misclassified)}):")
        for r in misclassified:
            print(f"   {r['event']}: expected {r['expected']}, got {r['detected']}")
    
    # Save results
    output_path = Path("outputs/regime_validation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'detector_config': {
                'vix_threshold_high': detector.vix_threshold_high,
                'vix_threshold_extreme': detector.vix_threshold_extreme,
                'adx_trending': detector.adx_trending,
                'adx_ranging': detector.adx_ranging
            },
            'summary': {
                'total_events': total_events,
                'correct_detections': correct_detections,
                'accuracy': accuracy
            },
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE!")
    print("=" * 70)
    
    if accuracy >= 70:
        print("\n🎉 Excellent! Detector is performing well on historical events.")
        print("   Ready to use for trading system!")
    elif accuracy >= 50:
        print("\n👍 Good! Detector is reasonably accurate.")
        print("   May need fine-tuning for some edge cases.")
    else:
        print("\n⚠️  Detector needs further calibration.")
        print("   Consider adjusting thresholds or logic.")
    
    return results

if __name__ == "__main__":
    results = test_market_events()
