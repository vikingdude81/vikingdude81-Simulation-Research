"""
System Audit - Verify all directories and model files are properly linked
"""
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def audit_system():
    """Comprehensive system audit"""
    
    print("="*80)
    print("🔍 SYSTEM AUDIT - PRICE DETECTION ML PIPELINE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    issues = []
    warnings = []
    
    # 1. Check data files
    print("📁 STEP 1: DATA FILES")
    print("-" * 80)
    
    data_dir = Path('DATA')
    if not data_dir.exists():
        issues.append("❌ DATA directory not found")
    else:
        print(f"✅ DATA directory exists")
        
        expected_files = ['yf_btc_1h.csv', 'yf_eth_1h.csv', 'yf_sol_1h.csv']
        for file in expected_files:
            filepath = data_dir / file
            if filepath.exists():
                df = pd.read_csv(filepath)
                print(f"   ✅ {file}: {len(df)} rows")
                print(f"      Latest: {df['time'].iloc[-1] if 'time' in df.columns else 'N/A'}")
            else:
                issues.append(f"❌ Missing {file}")
    
    print()
    
    # 2. Check MODEL_STORAGE structure
    print("📦 STEP 2: MODEL STORAGE STRUCTURE")
    print("-" * 80)
    
    model_storage = Path('MODEL_STORAGE')
    if not model_storage.exists():
        issues.append("❌ MODEL_STORAGE directory not found")
    else:
        print(f"✅ MODEL_STORAGE exists")
        
        expected_subdirs = ['training_runs', 'saved_models', 'predictions', 
                          'metrics', 'external_data', 'feature_data']
        
        for subdir in expected_subdirs:
            subdir_path = model_storage / subdir
            if subdir_path.exists():
                if subdir == 'training_runs':
                    runs = list(subdir_path.iterdir())
                    print(f"   ✅ {subdir}: {len(runs)} runs")
                else:
                    print(f"   ✅ {subdir}")
            else:
                warnings.append(f"⚠️  Missing {subdir}")
    
    print()
    
    # 3. Check training runs in detail
    print("🎯 STEP 3: TRAINING RUNS ANALYSIS")
    print("-" * 80)
    
    training_runs = model_storage / 'training_runs'
    if training_runs.exists():
        runs = sorted([d for d in training_runs.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"Total runs found: {len(runs)}")
        print()
        
        # Get last 5 runs
        recent_runs = runs[:5]
        
        run_details = []
        
        for run in recent_runs:
            metadata_path = run / 'metadata.json'
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check for required model files
                lstm_model = list(run.glob('*_lstm_*.pth'))
                transformer_model = list(run.glob('*_transformer_*.pth'))
                multitask_model = list(run.glob('*_multitask_*.pth'))
                
                detail = {
                    'run_id': run.name,
                    'timestamp': run.name.replace('run_', ''),
                    'features': metadata.get('config', {}).get('n_features', '?'),
                    'train_samples': metadata.get('config', {}).get('n_train_samples', '?'),
                    'test_samples': metadata.get('config', {}).get('n_test_samples', '?'),
                    'device': metadata.get('config', {}).get('device', '?'),
                    'has_lstm': '✅' if lstm_model else '❌',
                    'has_transformer': '✅' if transformer_model else '❌',
                    'has_multitask': '✅' if multitask_model else '❌',
                    'metadata': '✅'
                }
                
                run_details.append(detail)
                
            else:
                run_details.append({
                    'run_id': run.name,
                    'metadata': '❌',
                    'has_lstm': '?',
                    'has_transformer': '?',
                    'has_multitask': '?'
                })
                warnings.append(f"⚠️  {run.name} missing metadata.json")
        
        # Print table
        if run_details:
            print(f"{'Run ID':<25} {'Features':<10} {'Train':<8} {'Test':<8} {'LSTM':<6} {'Trans':<6} {'Multi':<6} {'Meta':<6}")
            print("-" * 80)
            for detail in run_details:
                print(f"{detail['run_id']:<25} "
                      f"{str(detail.get('features', '?')):<10} "
                      f"{str(detail.get('train_samples', '?')):<8} "
                      f"{str(detail.get('test_samples', '?')):<8} "
                      f"{detail.get('has_lstm', '?'):<6} "
                      f"{detail.get('has_transformer', '?'):<6} "
                      f"{detail.get('has_multitask', '?'):<6} "
                      f"{detail.get('metadata', '?'):<6}")
    
    print()
    
    # 4. Identify which runs are for which asset
    print("🏷️  STEP 4: ASSET IDENTIFICATION")
    print("-" * 80)
    
    # Based on training samples, identify assets
    asset_mapping = {}
    
    if training_runs.exists():
        for run in runs[:10]:  # Check last 10 runs
            metadata_path = run / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                train_samples = metadata.get('config', {}).get('n_train_samples', 0)
                
                # Try to identify asset by sample count (they differ slightly)
                if train_samples == 6990:
                    asset = 'BTC'
                elif train_samples == 6987:
                    asset = 'ETH'
                elif train_samples == 6988:
                    asset = 'SOL'
                else:
                    asset = 'UNKNOWN'
                
                if asset not in asset_mapping:
                    asset_mapping[asset] = []
                
                asset_mapping[asset].append({
                    'run_id': run.name,
                    'samples': train_samples,
                    'timestamp': run.stat().st_mtime
                })
    
    # Sort by timestamp and show latest for each asset
    for asset in ['BTC', 'ETH', 'SOL']:
        if asset in asset_mapping:
            latest = sorted(asset_mapping[asset], key=lambda x: x['timestamp'], reverse=True)[0]
            print(f"   {asset}: {latest['run_id']} (train samples: {latest['samples']})")
        else:
            issues.append(f"❌ No training run found for {asset}")
    
    if 'UNKNOWN' in asset_mapping:
        print(f"\n   ⚠️  {len(asset_mapping['UNKNOWN'])} runs with unknown asset:")
        for run in asset_mapping['UNKNOWN'][:3]:
            print(f"      - {run['run_id']} (samples: {run['samples']})")
    
    print()
    
    # 5. Check selected features file
    print("🎨 STEP 5: FEATURE SELECTION FILES")
    print("-" * 80)
    
    feature_files = [
        'MODEL_STORAGE/feature_data/selected_features_with_interactions.txt',
        'MODEL_STORAGE/feature_data/feature_importance.csv'
    ]
    
    for file in feature_files:
        filepath = Path(file)
        if filepath.exists():
            if file.endswith('.txt'):
                with open(filepath, 'r') as f:
                    features = [line.strip() for line in f if line.strip()]
                print(f"   ✅ {filepath.name}: {len(features)} features")
            else:
                print(f"   ✅ {filepath.name}")
        else:
            warnings.append(f"⚠️  Missing {filepath.name}")
    
    print()
    
    # 6. Summary and recommendations
    print("="*80)
    print("📊 AUDIT SUMMARY")
    print("="*80)
    
    if not issues and not warnings:
        print("✅ All systems operational - ready for predictions!")
    else:
        if issues:
            print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"   {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"   {warning}")
    
    print()
    
    # 7. Recommended asset-to-run mapping
    if asset_mapping:
        print("💡 RECOMMENDED ASSET-RUN MAPPING FOR PREDICTIONS:")
        print("-" * 80)
        
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in asset_mapping:
                latest = sorted(asset_mapping[asset], key=lambda x: x['timestamp'], reverse=True)[0]
                print(f"   {asset}: {latest['run_id']}")
        
        print()
        
        # Generate code snippet
        print("📝 CODE SNIPPET FOR predict_all_assets.py:")
        print("-" * 80)
        print("ASSETS = {")
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in asset_mapping:
                latest = sorted(asset_mapping[asset], key=lambda x: x['timestamp'], reverse=True)[0]
                asset_names = {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana'}
                emojis = {'BTC': '🟠', 'ETH': '🔵', 'SOL': '🟣'}
                print(f"    '{asset}': {{")
                print(f"        'name': '{asset_names[asset]}',")
                print(f"        'data_file': 'DATA/yf_{asset.lower()}_1h.csv',")
                print(f"        'model_dir': 'MODEL_STORAGE',")
                print(f"        'run_id': '{latest['run_id']}',")
                print(f"        'emoji': '{emojis[asset]}'")
                print(f"    }},")
        print("}")
    
    print()
    print("="*80)
    print("✅ AUDIT COMPLETE")
    print("="*80)
    
    return asset_mapping, issues, warnings

if __name__ == '__main__':
    audit_system()
