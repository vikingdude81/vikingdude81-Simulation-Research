"""
Multi-Asset Future Price Predictions
Generate 24-hour forecasts for BTC, ETH, and SOL
Limits predictions to avoid recursive error compounding
"""

import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PREDICTION HORIZON (hours)
# 12 = Today (most accurate, 0.44% RMSE)
# 24 = Today + Tomorrow (good accuracy)
# 48 = 2 Days (moderate accuracy)
# 72 = 3 Days (higher uncertainty)
PREDICTION_HOURS = 24  # Recommended: 24 hours to avoid compounding errors

print("\n" + "="*80)
print(f"🔮 MULTI-ASSET {PREDICTION_HOURS}-HOUR PREDICTIONS - BTC, ETH, SOL")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Horizon: {PREDICTION_HOURS} hours ({PREDICTION_HOURS//24} days)")
print("="*80)

# Asset configurations
ASSETS = {
    'BTC': {
        'name': 'Bitcoin',
        'data_file': 'DATA/yf_btc_1h.csv',
        'model_dir': 'MODEL_STORAGE',
        'run_id': 'run_20251026_201759',  # Latest BTC run
        'emoji': '🟠'
    },
    'ETH': {
        'name': 'Ethereum',
        'data_file': 'DATA/yf_eth_1h.csv',
        'model_dir': 'MODEL_STORAGE',
        'run_id': 'run_20251026_204237',  # Latest ETH run
        'emoji': '🔵'
    },
    'SOL': {
        'name': 'Solana',
        'data_file': 'DATA/yf_sol_1h.csv',
        'model_dir': 'MODEL_STORAGE',
        'run_id': 'run_20251026_210518',  # Latest SOL run
        'emoji': '🟣'
    }
}

all_predictions = {}

# Import required modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from enhanced_features import add_all_enhanced_features

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

for asset_code, asset_info in ASSETS.items():
    print("\n" + "="*80)
    print(f"{asset_info['emoji']} {asset_info['name']} ({asset_code}) PREDICTIONS")
    print("="*80)
    
    # Check if model exists
    model_storage = Path(asset_info['model_dir']) / 'training_runs'
    if not model_storage.exists():
        print(f"   ⚠️  No models found for {asset_code} - skipping")
        continue
    
    # Use specified run_id
    latest_run = model_storage / asset_info['run_id']
    if not latest_run.exists():
        print(f"   ⚠️  Training run {asset_info['run_id']} not found for {asset_code} - skipping")
        continue
    
    print(f"\n📂 Using model from: {latest_run.name}")
    
    # Load metadata
    try:
        with open(latest_run / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"   Features: {metadata['config']['n_features']}")
    except:
        print(f"   ⚠️  Could not load metadata for {asset_code}")
        continue
    
    # Load data
    try:
        df = pd.read_csv(asset_info['data_file'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        current_price = df['close'].iloc[-1]
        current_time = df['time'].iloc[-1]
        
        print(f"\n📊 Current Market State:")
        print(f"   Time: {current_time}")
        print(f"   Price: ${current_price:,.2f}")
        
        # Generate features
        print(f"\n🔧 Generating features...")
        df_featured = add_all_enhanced_features(df.copy())
        df_featured = df_featured.dropna()
        
        # Use all numeric features (model was trained with all features, not selected subset)
        # Exclude only the target and raw price columns
        selected_features = [col for col in df_featured.columns 
                            if col not in ['time', 'target', 'close', 'open', 'high', 'low', 'volume', 
                                          'returns', 'log_returns', 'price']]
        
        # Ensure we have exactly the number of features the model expects
        expected_features = metadata['config']['n_features']
        if len(selected_features) != expected_features:
            print(f"   ⚠️  Feature count mismatch: have {len(selected_features)}, model expects {expected_features}")
            # Take first N features to match model
            selected_features = selected_features[:expected_features]
        
        print(f"   Using {len(selected_features)} features (matches model)")
        
        # Load MultiTask model from saved_models directory
        run_id = latest_run.name
        timestamp = run_id.replace('run_', '')
        multitask_model_path = Path(asset_info['model_dir']) / 'saved_models' / f'{run_id}_multitask_{timestamp}.pth'
        
        if not multitask_model_path.exists():
            # Try finding in training_runs directory
            multitask_files = list(latest_run.glob('*multitask*.pth'))
            if multitask_files:
                multitask_model_path = multitask_files[0]
            else:
                print(f"   ❌ MultiTask model not found for {asset_code}")
                print(f"      Looked in: {multitask_model_path}")
                continue
        
        print(f"\n🤖 Loading MultiTask model...")
        from main import MultiTaskTransformer
        
        checkpoint = torch.load(multitask_model_path, map_location=device)
        
        model = MultiTaskTransformer(
            input_size=metadata['config']['n_features'],
            d_model=metadata['config'].get('transformer_dim', 256),
            nhead=metadata['config'].get('transformer_heads', 8),
            num_layers=metadata['config'].get('transformer_layers', 4),
            dropout=0.1
        ).to(device)
        
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"   ✅ Model loaded")
        
        # Generate predictions
        SEQUENCE_LENGTH = metadata['config'].get('lstm_sequence_length', 48)
        predictions = []
        current_data = df_featured.tail(SEQUENCE_LENGTH).copy()
        
        print(f"\n🔮 Generating {PREDICTION_HOURS}-hour predictions...")
        
        with torch.no_grad():
            for hour in range(PREDICTION_HOURS):
                # Prepare input
                X_seq = current_data[selected_features].values[-SEQUENCE_LENGTH:]
                X_mean = X_seq.mean(axis=0)
                X_std = X_seq.std(axis=0) + 1e-8
                X_normalized = (X_seq - X_mean) / X_std
                X_tensor = torch.FloatTensor(X_normalized).unsqueeze(0).to(device)
                
                # Predict (model returns: price, volatility, direction_logits)
                price_pred, volatility_pred, direction_logits = model(X_tensor)
                pred_value = price_pred.cpu().numpy()[0, 0]
                
                # Get direction from logits
                direction = direction_logits.argmax(dim=-1).cpu().numpy()[0]
                
                # Use volatility as uncertainty proxy
                uncertainty = volatility_pred.cpu().numpy()[0, 0]
                
                # Calculate predicted price
                last_price = current_data['close'].iloc[-1]
                predicted_price = last_price * (1 + pred_value)
                
                last_time = current_data['time'].iloc[-1]
                pred_time = last_time + timedelta(hours=1)
                
                predictions.append({
                    'time': pred_time,
                    'predicted_price': predicted_price,
                    'predicted_change_pct': pred_value * 100,
                    'direction': direction,
                    'volatility': volatility_pred.cpu().numpy()[0, 0],
                    'uncertainty': uncertainty,
                    'hour_ahead': hour + 1
                })
                
                # Update for next iteration
                next_row = current_data.iloc[-1:].copy()
                next_row['time'] = pred_time
                next_row['close'] = predicted_price
                next_row['open'] = last_price
                next_row['high'] = max(last_price, predicted_price) * 1.002
                next_row['low'] = min(last_price, predicted_price) * 0.998
                current_data = pd.concat([current_data.iloc[1:], next_row], ignore_index=True)
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate metrics
        period_start_price = predictions_df['predicted_price'].iloc[0]
        period_end_price = predictions_df['predicted_price'].iloc[-1]
        period_change = ((period_end_price - period_start_price) / period_start_price) * 100
        period_high = predictions_df['predicted_price'].max()
        period_low = predictions_df['predicted_price'].min()
        avg_uncertainty = predictions_df['uncertainty'].mean()
        
        # Direction analysis
        direction_counts = predictions_df['direction'].value_counts()
        up_pct = (direction_counts.get(2, 0) / PREDICTION_HOURS) * 100
        down_pct = (direction_counts.get(0, 0) / PREDICTION_HOURS) * 100
        
        if up_pct > down_pct:
            trend = "🚀 BULLISH"
        elif down_pct > up_pct:
            trend = "🔻 BEARISH"
        else:
            trend = "➡️ NEUTRAL"
        
        # Store results
        all_predictions[asset_code] = {
            'current_price': float(current_price),
            'predictions': predictions_df,
            'summary': {
                '1h': float(predictions_df['predicted_price'].iloc[0]),
                '12h': float(predictions_df['predicted_price'].iloc[min(11, len(predictions_df)-1)]),
                '24h': float(predictions_df['predicted_price'].iloc[min(23, len(predictions_df)-1)]),
                'final': float(period_end_price),
                'change_pct': float(period_change),
                'high': float(period_high),
                'low': float(period_low),
                'trend': trend,
                'confidence_pct': float((1 - avg_uncertainty) * 100)
            }
        }
        
        # Print summary
        print(f"\n✅ Generated {len(predictions_df)} predictions")
        print(f"\n📊 {asset_code} {PREDICTION_HOURS}-HOUR OUTLOOK:")
        print(f"   Current:    ${current_price:,.2f}")
        if PREDICTION_HOURS >= 12:
            print(f"   12H:        ${all_predictions[asset_code]['summary']['12h']:,.2f}")
        if PREDICTION_HOURS >= 24:
            print(f"   24H:        ${all_predictions[asset_code]['summary']['24h']:,.2f}")
        print(f"   Final:      ${all_predictions[asset_code]['summary']['final']:,.2f}")
        print(f"   Change:     {period_change:+.2f}%")
        print(f"   High:       ${period_high:,.2f}")
        print(f"   Low:        ${period_low:,.2f}")
        print(f"   Trend:      {trend}")
        print(f"   Confidence: {(1 - avg_uncertainty) * 100:.1f}%")
        
        # Save individual predictions
        output_file = f'{asset_code}_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        
    except Exception as e:
        print(f"   ❌ Error processing {asset_code}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Multi-Asset Summary
print("\n" + "="*80)
print(f"📊 MULTI-ASSET SUMMARY - NEXT {PREDICTION_HOURS} HOURS")
print("="*80)

if all_predictions:
    # Dynamic header based on prediction horizon
    if PREDICTION_HOURS >= 24:
        print(f"\n{'Asset':<6} {'Current':<14} {'12H Target':<14} {'24H Target':<14} {'Change':<10} {'Trend':<15}")
    else:
        print(f"\n{'Asset':<6} {'Current':<14} {'Final Target':<14} {'Change':<10} {'Trend':<15}")
    print("-" * 80)
    
    for asset_code in ['BTC', 'ETH', 'SOL']:
        if asset_code in all_predictions:
            data = all_predictions[asset_code]
            emoji = ASSETS[asset_code]['emoji']
            if PREDICTION_HOURS >= 24:
                print(f"{emoji} {asset_code:<4} ${data['current_price']:>12,.2f} "
                      f"${data['summary']['12h']:>12,.2f} "
                      f"${data['summary']['24h']:>12,.2f} "
                      f"{data['summary']['change_pct']:>+8.2f}% "
                      f"{data['summary']['trend']:<15}")
            else:
                print(f"{emoji} {asset_code:<4} ${data['current_price']:>12,.2f} "
                      f"${data['summary']['final']:>12,.2f} "
                      f"{data['summary']['change_pct']:>+8.2f}% "
                      f"{data['summary']['trend']:<15}")
    
    # Trading recommendations
    print("\n" + "="*80)
    print("🎯 TRADING RECOMMENDATIONS")
    print("="*80)
    bullish_count = sum(1 for d in all_predictions.values() if '🚀' in d['summary']['trend'])
    bearish_count = sum(1 for d in all_predictions.values() if '🔻' in d['summary']['trend'])
    
    if bullish_count >= 2:
        print("\n✅ MARKET BIAS: BULLISH")
        print("   Strategy: Look for LONG opportunities on dips")
        print("   Focus: Assets with strongest uptrend")
    elif bearish_count >= 2:
        print("\n⚠️  MARKET BIAS: BEARISH")
        print("   Strategy: Caution on longs, consider shorts")
        print("   Focus: Wait for reversal signals")
    else:
        print("\n➡️  MARKET BIAS: MIXED/NEUTRAL")
        print("   Strategy: Asset-specific approach")
        print("   Focus: Trade based on individual signals")
    
    # Save combined summary
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'assets': {code: {
            'current_price': data['current_price'],
            **data['summary']
        } for code, data in all_predictions.items()}
    }
    
    summary_file = f'multi_asset_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")

else:
    print("\n❌ No predictions generated")

print("\n" + "="*80)
print("✅ MULTI-ASSET PREDICTION COMPLETE!")
print("="*80)
print("\n⚠️  DISCLAIMER: ML predictions are not financial advice.")
print("    Use proper risk management and do your own research!")
print("="*80 + "\n")
