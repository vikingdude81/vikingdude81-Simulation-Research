"""
Generate Future Price Predictions
Load latest trained model and predict next 7 days (168 hours)
"""

import pandas as pd
import numpy as np
import torch
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("\n" + "="*80)
print("🔮 FUTURE PRICE PREDICTIONS - NEXT 7 DAYS")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Load the latest trained model
model_storage = Path('MODEL_STORAGE/training_runs')
runs = sorted([d for d in model_storage.iterdir() if d.is_dir()], 
              key=lambda x: x.stat().st_mtime, reverse=True)

if len(runs) == 0:
    print("❌ No trained models found! Run main.py first.")
    exit(1)

latest_run = runs[0]
print(f"\n📂 Using model from: {latest_run.name}")

# Load run metadata
with open(latest_run / 'run_data.json', 'r') as f:
    run_data = json.load(f)

print(f"   Models: {', '.join(run_data['models'])}")
print(f"   Test RMSE: {run_data['metrics']['test_rmse']:.6f} ({run_data['metrics']['test_rmse_pct']:.2f}%)")
print(f"   Features: {run_data['config']['n_features']}")

# Load latest BTC data
df = pd.read_csv('DATA/yf_btc_1h.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

current_price = df['close'].iloc[-1]
current_time = df['time'].iloc[-1]

print(f"\n📊 Current Market State:")
print(f"   Time: {current_time}")
print(f"   Price: ${current_price:,.2f}")

# Prepare features (same as training)
from enhanced_features import add_all_enhanced_features

print("\n🔧 Generating features...")
df_featured = add_all_enhanced_features(df.copy())
df_featured = df_featured.dropna()

# Load selected features if available
selected_features_path = Path('MODEL_STORAGE/feature_data/selected_features_with_interactions.txt')
if selected_features_path.exists():
    with open(selected_features_path, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    print(f"   Using {len(selected_features)} selected features")
else:
    # Use all features except target and time
    selected_features = [col for col in df_featured.columns 
                        if col not in ['time', 'target', 'close', 'open', 'high', 'low', 'volume']]
    print(f"   Using all {len(selected_features)} features")

# Get the last window of data for recursive prediction
SEQUENCE_LENGTH = run_data['config'].get('lstm_sequence_length', 48)
last_window = df_featured.tail(SEQUENCE_LENGTH).copy()

print(f"\n🔮 Generating predictions for next 168 hours (7 days)...")
print("   Method: Recursive multi-step forecasting")

# Initialize predictions list
predictions = []
current_data = last_window.copy()

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# Try to load MultiTask model (best performer)
multitask_model_path = latest_run / 'multitask_model.pt'
if multitask_model_path.exists():
    print("\n   Loading MultiTask model...")
    
    # Import model architecture
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import MultiTaskTransformer
    
    # Load model state
    checkpoint = torch.load(multitask_model_path, map_location=device)
    
    n_features = run_data['config']['n_features']
    model = MultiTaskTransformer(
        n_features=n_features,
        d_model=run_data['config'].get('transformer_dim', 256),
        nhead=run_data['config'].get('transformer_heads', 8),
        num_layers=run_data['config'].get('transformer_layers', 4),
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint)
    model.eval()
    print("   ✅ MultiTask model loaded")
    
    # Generate predictions recursively
    with torch.no_grad():
        for hour in range(168):  # 7 days = 168 hours
            # Prepare input sequence
            X_seq = current_data[selected_features].values[-SEQUENCE_LENGTH:]
            
            # Normalize (using last window stats)
            X_mean = X_seq.mean(axis=0)
            X_std = X_seq.std(axis=0) + 1e-8
            X_normalized = (X_seq - X_mean) / X_std
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_normalized).unsqueeze(0).to(device)
            
            # Predict
            price_pred, direction_pred, volatility_pred, uncertainty = model(X_tensor)
            
            # Get prediction (denormalize)
            pred_value = price_pred.cpu().numpy()[0, 0]
            
            # Calculate predicted price
            last_price = current_data['_close'].iloc[-1]
            predicted_price = last_price * (1 + pred_value)
            
            # Predicted time
            last_time = current_data['time'].iloc[-1]
            pred_time = last_time + timedelta(hours=1)
            
            # Store prediction
            predictions.append({
                'time': pred_time,
                'predicted_price': predicted_price,
                'predicted_change_pct': pred_value * 100,
                'direction': direction_pred.cpu().argmax(dim=1).item(),  # 0=down, 1=stable, 2=up
                'volatility': volatility_pred.cpu().numpy()[0, 0],
                'uncertainty': uncertainty.cpu().numpy()[0, 0],
                'hour_ahead': hour + 1
            })
            
            # Create next row by copying last row and updating
            next_row = current_data.iloc[-1:].copy()
            next_row['time'] = pred_time
            next_row['_close'] = predicted_price
            next_row['_open'] = last_price
            next_row['_high'] = max(last_price, predicted_price) * 1.002
            next_row['_low'] = min(last_price, predicted_price) * 0.998
            
            # Update rolling features (simplified - in production would recalculate all)
            # For now, just append the row
            current_data = pd.concat([current_data.iloc[1:], next_row], ignore_index=True)
            
else:
    print("❌ MultiTask model not found. Please train the model first with main.py")
    exit(1)

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions)

print(f"\n✅ Generated {len(predictions_df)} hourly predictions")

# Create daily summary
print("\n" + "="*80)
print("📅 DAILY PRICE FORECAST")
print("="*80)

predictions_df['date'] = pd.to_datetime(predictions_df['time']).dt.date
predictions_df['day_name'] = pd.to_datetime(predictions_df['time']).dt.day_name()

daily_summary = predictions_df.groupby('date').agg({
    'predicted_price': ['first', 'max', 'min', 'last'],
    'day_name': 'first',
    'uncertainty': 'mean'
}).reset_index()

print(f"\n{'Date':<12} {'Day':<10} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Confidence':<12}")
print("-" * 92)

for idx, row in daily_summary.iterrows():
    date = row['date']
    day = row[('day_name', 'first')]
    open_p = row[('predicted_price', 'first')]
    high_p = row[('predicted_price', 'max')]
    low_p = row[('predicted_price', 'min')]
    close_p = row[('predicted_price', 'last')]
    uncertainty = row[('uncertainty', 'mean')]
    confidence = (1 - uncertainty) * 100
    
    print(f"{date} {day:<10} ${open_p:>10,.2f} ${high_p:>10,.2f} ${low_p:>10,.2f} ${close_p:>10,.2f}  {confidence:>6.1f}%")

# Weekly outlook
week_start_price = predictions_df['predicted_price'].iloc[0]
week_end_price = predictions_df['predicted_price'].iloc[-1]
week_change = ((week_end_price - week_start_price) / week_start_price) * 100
week_high = predictions_df['predicted_price'].max()
week_low = predictions_df['predicted_price'].min()
avg_uncertainty = predictions_df['uncertainty'].mean()

print("\n" + "="*80)
print("📊 WEEKLY OUTLOOK SUMMARY")
print("="*80)
print(f"Current Price:      ${current_price:,.2f}")
print(f"24H Prediction:     ${predictions_df['predicted_price'].iloc[23]:,.2f}")
print(f"48H Prediction:     ${predictions_df['predicted_price'].iloc[47]:,.2f}")
print(f"7-Day Prediction:   ${week_end_price:,.2f}")
print(f"\nExpected Change:    {week_change:+.2f}%")
print(f"Predicted High:     ${week_high:,.2f}")
print(f"Predicted Low:      ${week_low:,.2f}")
print(f"Price Range:        ${week_high - week_low:,.2f} ({((week_high - week_low) / week_low * 100):.2f}%)")
print(f"\nAvg Confidence:     {(1 - avg_uncertainty) * 100:.1f}%")

# Trend direction
direction_counts = predictions_df['direction'].value_counts()
if 2 in direction_counts and direction_counts.get(2, 0) > direction_counts.get(0, 0):
    trend = "🚀 BULLISH"
elif 0 in direction_counts and direction_counts.get(0, 0) > direction_counts.get(2, 0):
    trend = "🔻 BEARISH"
else:
    trend = "➡️ NEUTRAL"

print(f"Predicted Trend:    {trend}")

# Direction breakdown
total = len(predictions_df)
up_pct = (direction_counts.get(2, 0) / total) * 100
stable_pct = (direction_counts.get(1, 0) / total) * 100
down_pct = (direction_counts.get(0, 0) / total) * 100

print(f"\nDirection Breakdown:")
print(f"   UP:     {up_pct:>5.1f}% ({direction_counts.get(2, 0)} hours)")
print(f"   STABLE: {stable_pct:>5.1f}% ({direction_counts.get(1, 0)} hours)")
print(f"   DOWN:   {down_pct:>5.1f}% ({direction_counts.get(0, 0)} hours)")

# Price targets
print(f"\n🎯 KEY PRICE LEVELS:")
if week_change > 0:
    print(f"   Resistance: ${week_high:,.2f}")
    print(f"   Target:     ${week_end_price:,.2f}")
    print(f"   Support:    ${current_price:,.2f}")
else:
    print(f"   Resistance: ${current_price:,.2f}")
    print(f"   Target:     ${week_end_price:,.2f}")
    print(f"   Support:    ${week_low:,.2f}")

# Save predictions
output_file = f'future_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
predictions_df.to_csv(output_file, index=False)
print(f"\n💾 Predictions saved to: {output_file}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'current_price': float(current_price),
    'predictions': {
        '24h': float(predictions_df['predicted_price'].iloc[23]),
        '48h': float(predictions_df['predicted_price'].iloc[47]),
        '7d': float(week_end_price)
    },
    'week_change_pct': float(week_change),
    'predicted_high': float(week_high),
    'predicted_low': float(week_low),
    'trend': trend,
    'confidence_pct': float((1 - avg_uncertainty) * 100),
    'model_run': latest_run.name
}

summary_file = f'prediction_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"💾 Summary saved to: {summary_file}")

print("\n" + "="*80)
print("✅ PREDICTION COMPLETE!")
print("="*80)
print("\n⚠️  DISCLAIMER: These are ML model predictions, not financial advice.")
print("    Actual prices may vary significantly from predictions.")
print("    Always use proper risk management and do your own research!")
print("="*80 + "\n")
