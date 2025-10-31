"""
🔥 EVOLUTION ON REAL CRYPTO DATA
Use actual BTC/ETH/SOL prices to evolve trading strategies

This connects your GA system to your real data pipeline!
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path to import from main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fetch_data import get_crypto_data
    HAS_DATA_FETCHER = True
except ImportError:
    print("⚠️  Could not import fetch_data.py - will use CSV files if available")
    HAS_DATA_FETCHER = False

from strategy_evolution import EvolutionSimulation
from visualize_evolution import EvolutionVisualizer

print("="*80)
print("🔥 EVOLUTION ON REAL CRYPTO DATA")
print("="*80)

def load_crypto_prices(symbol='BTC', periods=500):
    """
    Load real crypto prices from your data pipeline
    
    Args:
        symbol: BTC, ETH, or SOL
        periods: How many price points to use
    
    Returns:
        List of prices
    """
    print(f"\n📊 Loading {symbol} data...")
    
    # Try method 1: Use your fetch_data.py
    if HAS_DATA_FETCHER:
        try:
            df = get_crypto_data(symbol, periods=periods)
            if df is not None and not df.empty:
                prices = df['close'].values.tolist()
                print(f"   ✓ Loaded {len(prices)} prices from fetch_data.py")
                print(f"   ✓ Date range: {df.index[0]} to {df.index[-1]}")
                print(f"   ✓ Price range: ${prices[0]:.2f} → ${prices[-1]:.2f}")
                return prices
        except Exception as e:
            print(f"   ⚠️  fetch_data.py failed: {e}")
    
    # Try method 2: Load from CSV files
    csv_files = [
        f'../crypto_data_{symbol.lower()}.csv',
        f'../{symbol}_data.csv',
        f'../data/{symbol.lower()}_prices.csv',
    ]
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'close' in df.columns:
                prices = df['close'].values[-periods:].tolist()
                print(f"   ✓ Loaded {len(prices)} prices from {csv_file}")
                return prices
        except:
            continue
    
    print("   ⚠️  Could not load real data, using synthetic data as fallback")
    from strategy_evolution import generate_synthetic_market
    return generate_synthetic_market(periods=periods, trend=0.001, volatility=0.02)


def run_evolution_on_real_data(symbol='BTC', periods=500, generations=30):
    """
    Run full evolution on real crypto data
    """
    print("\n" + "="*80)
    print(f"🧬 EVOLVING STRATEGIES ON REAL {symbol} DATA")
    print("="*80)
    
    # Load data
    prices = load_crypto_prices(symbol, periods)
    
    # Calculate market stats
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    avg_return = np.mean(returns) * 100
    volatility = np.std(returns) * 100
    total_return = ((prices[-1] / prices[0]) - 1) * 100
    
    print(f"\n📈 Market Statistics:")
    print(f"   Periods: {len(prices)}")
    print(f"   Price: ${prices[0]:.2f} → ${prices[-1]:.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Avg Daily Return: {avg_return:+.4f}%")
    print(f"   Volatility: {volatility:.4f}%")
    
    # Create simulation
    print(f"\n⚙️  Creating evolution simulation...")
    sim = EvolutionSimulation(
        population_size=50,
        elite_size=5,
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    print(f"   Population: 50 agents")
    print(f"   Generations: {generations}")
    print(f"   Elite: 5 (top performers preserved)")
    print(f"   Mutation rate: 15%")
    print(f"   Crossover rate: 70%")
    
    # Run evolution
    print(f"\n🚀 Starting evolution...")
    print("="*80)
    
    sim.run(prices, generations=generations, verbose=True)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    viz = EvolutionVisualizer()
    viz.create_dashboard(sim, prices)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evolution_results_{symbol}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EVOLUTION RESULTS - {symbol}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MARKET DATA:\n")
        f.write(f"  Symbol: {symbol}\n")
        f.write(f"  Periods: {len(prices)}\n")
        f.write(f"  Price Range: ${prices[0]:.2f} → ${prices[-1]:.2f}\n")
        f.write(f"  Market Return: {total_return:+.2f}%\n")
        f.write(f"  Avg Daily Return: {avg_return:+.4f}%\n")
        f.write(f"  Volatility: {volatility:.4f}%\n\n")
        
        f.write("EVOLUTION CONFIGURATION:\n")
        f.write(f"  Population: {sim.population_size}\n")
        f.write(f"  Generations: {sim.generation}\n")
        f.write(f"  Elite Size: {sim.elite_size}\n")
        f.write(f"  Mutation Rate: {sim.mutation_rate}\n")
        f.write(f"  Crossover Rate: {sim.crossover_rate}\n\n")
        
        f.write("EVOLUTION RESULTS:\n")
        f.write(f"  Initial Best Fitness: ${sim.best_fitness_history[0]:.2f}\n")
        f.write(f"  Final Best Fitness: ${sim.best_fitness_history[-1]:.2f}\n")
        improvement = ((sim.best_fitness_history[-1] / sim.best_fitness_history[0]) - 1) * 100
        f.write(f"  Improvement: {improvement:+.2f}%\n\n")
        
        f.write("TOP 5 EVOLVED STRATEGIES:\n")
        sorted_pop = sorted(sim.population, key=lambda a: getattr(a, 'fitness', 0), reverse=True)
        for i, agent in enumerate(sorted_pop[:5], 1):
            fitness = getattr(agent, 'fitness', 0)
            f.write(f"\n#{i} Strategy: {agent.chromosome}\n")
            f.write(f"   Fitness: ${fitness:.2f}\n")
            f.write(f"   Trades: {len(agent.trades)}\n")
            f.write(f"   Wins/Losses: {agent.wins}/{agent.losses}\n")
            if agent.wins + agent.losses > 0:
                win_rate = agent.wins / (agent.wins + agent.losses) * 100
                f.write(f"   Win Rate: {win_rate:.1f}%\n")
            f.write(f"   Total P&L: ${agent.total_pnl:+.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("="*80 + "\n\n")
        
        best = sorted_pop[0]
        f.write("Best Strategy Behavior:\n")
        from trading_agent import CONDITION_NAMES
        for i, action in enumerate(best.chromosome.genes):
            condition = CONDITION_NAMES[i]
            f.write(f"  {condition}: {action}\n")
        
        f.write("\nStrategy Type: ")
        buy_count = best.chromosome.genes.count('B')
        sell_count = best.chromosome.genes.count('S')
        hold_count = best.chromosome.genes.count('H')
        
        if buy_count > sell_count and buy_count > hold_count:
            f.write("AGGRESSIVE BUYER (bullish bias)\n")
        elif sell_count > buy_count and sell_count > hold_count:
            f.write("AGGRESSIVE SELLER (bearish bias)\n")
        elif hold_count >= 3:
            f.write("CONSERVATIVE (wait-and-see approach)\n")
        else:
            f.write("BALANCED (mixed strategy)\n")
        
        f.write(f"\nVs Buy-and-Hold:\n")
        buy_hold_return = ((prices[-1] / prices[0]) - 1) * 10000  # $10,000 initial
        f.write(f"  Buy-and-Hold Final: ${10000 + buy_hold_return:.2f}\n")
        f.write(f"  Evolved Strategy Final: ${fitness:.2f}\n")
        if fitness > 10000 + buy_hold_return:
            f.write(f"  ✓ Strategy BEATS buy-and-hold by ${fitness - (10000 + buy_hold_return):.2f}\n")
        else:
            f.write(f"  ✗ Strategy underperforms buy-and-hold\n")
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return sim, prices


def compare_multiple_assets():
    """
    Run evolution on BTC, ETH, and SOL simultaneously
    """
    print("\n" + "="*80)
    print("🌐 MULTI-ASSET EVOLUTION COMPARISON")
    print("="*80)
    
    assets = ['BTC', 'ETH', 'SOL']
    results = {}
    
    for asset in assets:
        print(f"\n{'='*80}")
        print(f"Processing {asset}...")
        print(f"{'='*80}")
        
        try:
            sim, prices = run_evolution_on_real_data(asset, periods=500, generations=20)
            results[asset] = {
                'sim': sim,
                'prices': prices,
                'best_fitness': sim.best_fitness_history[-1],
                'improvement': ((sim.best_fitness_history[-1] / sim.best_fitness_history[0]) - 1) * 100
            }
        except Exception as e:
            print(f"❌ Error processing {asset}: {e}")
            continue
    
    # Summary comparison
    print("\n" + "="*80)
    print("📊 MULTI-ASSET COMPARISON SUMMARY")
    print("="*80)
    
    for asset, data in results.items():
        print(f"\n{asset}:")
        print(f"  Best Fitness: ${data['best_fitness']:.2f}")
        print(f"  Improvement: {data['improvement']:+.2f}%")
        
        best_agent = max(data['sim'].population, key=lambda a: getattr(a, 'fitness', 0))
        print(f"  Best Strategy: {best_agent.chromosome}")
        print(f"  Trades: {len(best_agent.trades)} | W/L: {best_agent.wins}/{best_agent.losses}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    print("\n🎯 Choose mode:")
    print("1. Single asset (BTC)")
    print("2. Single asset (ETH)")
    print("3. Single asset (SOL)")
    print("4. Compare all three (BTC, ETH, SOL)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        run_evolution_on_real_data('BTC', periods=500, generations=30)
    elif choice == '2':
        run_evolution_on_real_data('ETH', periods=500, generations=30)
    elif choice == '3':
        run_evolution_on_real_data('SOL', periods=500, generations=30)
    elif choice == '4':
        compare_multiple_assets()
    else:
        print("Invalid choice, defaulting to BTC")
        run_evolution_on_real_data('BTC', periods=500, generations=30)
    
    print("\n" + "="*80)
    print("✅ EVOLUTION COMPLETE!")
    print("="*80)
    print("\nFiles created:")
    print("  - evolution_dashboard.png (visualization)")
    print("  - evolution_results_*.txt (detailed results)")
    print("\n🎯 Next: Compare with your ML models!")
    print("="*80)
