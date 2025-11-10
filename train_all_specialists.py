"""
Train All Trading Specialists

Trains regime-specific trading specialists using genetic algorithm.
This creates the BASELINE specialists (no GA Conductor yet).

Each specialist is optimized for its specific market regime:
- Volatile: Quick profits, tight stops, momentum-following
- Trending: Let winners run, strong trends, larger positions  
- Ranging: Mean reversion, moderate everything
- Crisis: Minimal risk, scalping, very tight stops

Author: GA Conductor Research Team
Date: November 5, 2025
"""

import pandas as pd
import numpy as np
from specialist_trainer import SpecialistTrainer
from trading_specialist import TradingSpecialist
import json
from datetime import datetime

def generate_predictions(df: pd.DataFrame, regime_type: str) -> np.ndarray:
    """
    Generate trading predictions based on regime type
    
    For now using simple momentum/mean-reversion signals.
    In production, would use ML model predictions.
    """
    df = df.copy()
    
    # Calculate indicators
    df['returns'] = df['close'].pct_change()
    df['sma_short'] = df['close'].rolling(5).mean()
    df['sma_long'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    if regime_type == 'volatile' or regime_type == 'trending':
        # Momentum-based predictions
        # Use crossover + momentum
        crossover = (df['sma_short'] - df['sma_long']) / df['close']
        momentum = df['returns'].rolling(5).mean()
        predictions = (crossover * 5 + momentum * 10).fillna(0).values
        
    elif regime_type == 'ranging':
        # Mean-reversion predictions
        # Use RSI + price deviation from mean
        rsi_signal = (50 - df['rsi']) / 50  # Buy when oversold, sell when overbought
        price_dev = (df['close'] - df['sma_long']) / df['sma_long']
        predictions = (rsi_signal * 3 - price_dev * 5).fillna(0).values
        
    else:  # crisis
        # Very conservative - only strong momentum
        strong_momentum = df['returns'].rolling(3).mean()
        volatility = df['returns'].rolling(10).std()
        # Only trade when momentum is strong AND volatility is decreasing
        predictions = (strong_momentum * 10 * (1 / (1 + volatility))).fillna(0).values
    
    return predictions


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def train_specialist(regime_type: str,
                     population_size: int = 200,
                     generations: int = 300,
                     mutation_rate: float = 0.1,
                     crossover_rate: float = 0.7):
    """
    Train a single specialist for given regime
    """
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING {regime_type.upper()} MARKET SPECIALIST")
    print(f"{'='*80}\n")
    
    # Load regime-specific data
    data_path = f'DATA/yf_btc_1d_{regime_type}.csv'
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} days of {regime_type} market data")
    except FileNotFoundError:
        print(f"❌ ERROR: {data_path} not found!")
        print(f"   Run label_historical_regimes.py first.")
        return None
    
    # Add required indicators
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    if 'atr' not in df.columns:
        high_low = df['high'] - df['low']
        df['atr'] = high_low.rolling(14).mean()
    
    # Generate predictions
    print(f"📊 Generating {regime_type}-specific trading signals...")
    predictions = generate_predictions(df, regime_type)
    
    # Create trainer
    trainer = SpecialistTrainer(
        regime_type=regime_type,
        training_data=df,
        predictions=predictions,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )
    
    # Train
    print(f"🚀 Starting genetic evolution...")
    best_genome, best_fitness, history = trainer.train(verbose=True)
    
    # Evaluate final specialist
    print(f"\n{'='*80}")
    print(f"📈 EVALUATING BEST {regime_type.upper()} SPECIALIST")
    print(f"{'='*80}\n")
    
    specialist = TradingSpecialist(best_genome, regime_type)
    specialist.evaluate_fitness(df, predictions)
    metrics = specialist.get_metrics()
    
    print(f"Performance Metrics:")
    print(f"  Fitness Score:    {metrics.fitness:.2f}")
    print(f"  Total Return:     {metrics.total_return*100:+.2f}%")
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"  Win Rate:         {metrics.win_rate*100:.1f}%")
    print(f"  Max Drawdown:     {metrics.max_drawdown*100:.2f}%")
    print(f"  Num Trades:       {metrics.num_trades}")
    print(f"  Avg Trade Return: {metrics.avg_trade_return*100:+.2f}%")
    print(f"  Profit Factor:    {metrics.profit_factor:.2f}")
    
    print(f"\nBest Genome:")
    gene_names = ['stop_loss', 'take_profit', 'position_size', 'entry_threshold',
                  'exit_threshold', 'max_hold_time', 'volatility_scaling', 'momentum_weight']
    for name, value in zip(gene_names, best_genome):
        print(f"  {name:20s}: {value:.4f}")
    
    # Save results
    results = trainer.save_results()
    trainer.plot_training()
    
    print(f"\n✅ {regime_type.upper()} specialist training complete!\n")
    
    return {
        'regime_type': regime_type,
        'genome': best_genome.tolist(),
        'fitness': best_fitness,
        'metrics': metrics.to_dict(),
        'training_history': history
    }


def train_all_specialists():
    """Train all 4 specialists"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING ALL TRADING SPECIALISTS - BASELINE GA")
    print("="*80)
    print("\nThis will train specialists for each market regime:")
    print("  • Volatile Market Specialist")
    print("  • Trending Market Specialist")
    print("  • Ranging Market Specialist")
    print("  • Crisis Market Specialist")
    print("\nUsing standard GA (no conductor yet - this is the baseline)")
    print("="*80 + "\n")
    
    results = {}
    
    # Train each specialist
    regimes = ['volatile', 'trending', 'ranging']  # Skip crisis for now (only 114 days)
    
    for regime in regimes:
        try:
            result = train_specialist(
                regime_type=regime,
                population_size=200,   # Larger population for better search
                generations=300,       # More generations for convergence
                mutation_rate=0.1,     # Standard mutation rate
                crossover_rate=0.7     # Standard crossover rate
            )
            
            if result is not None:
                results[regime] = result
                
        except Exception as e:
            print(f"\n❌ ERROR training {regime} specialist: {str(e)}\n")
            continue
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'outputs/all_specialists_baseline_{timestamp}.json'
    
    summary = {
        'training_method': 'standard_ga_baseline',
        'timestamp': datetime.now().isoformat(),
        'specialists': results,
        'summary': {
            regime: {
                'fitness': results[regime]['fitness'],
                'total_return': results[regime]['metrics']['total_return'],
                'sharpe_ratio': results[regime]['metrics']['sharpe_ratio'],
                'win_rate': results[regime]['metrics']['win_rate'],
                'num_trades': results[regime]['metrics']['num_trades']
            }
            for regime in results.keys()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL SPECIALISTS TRAINED!")
    print(f"{'='*80}\n")
    
    print("Summary:")
    for regime, data in results.items():
        print(f"\n{regime.upper():12s}:")
        print(f"  Fitness:      {data['fitness']:8.2f}")
        print(f"  Return:       {data['metrics']['total_return']*100:+7.2f}%")
        print(f"  Sharpe:       {data['metrics']['sharpe_ratio']:8.2f}")
        print(f"  Win Rate:     {data['metrics']['win_rate']*100:7.1f}%")
        print(f"  Trades:       {data['metrics']['num_trades']:8d}")
    
    print(f"\n✅ Saved combined results to: {output_path}")
    print(f"\n{'='*80}\n")
    
    return results


if __name__ == '__main__':
    # Train all specialists
    results = train_all_specialists()
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Review specialist performance in outputs/")
    print("  2. Build GA Conductor for enhanced training")
    print("  3. Compare baseline vs conductor performance")
    print("  4. Train crisis specialist (needs more data or different approach)")
    print()
