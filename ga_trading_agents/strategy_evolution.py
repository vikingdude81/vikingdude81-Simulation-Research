"""
🧬 Strategy Evolution - GA + ABM Combined
Evolves trading strategies through natural selection

Process:
1. Create population of trading agents (random strategies)
2. Run them through market simulation (ABM)
3. Evaluate fitness (profit/loss)
4. Select winners, breed new generation (GA)
5. Repeat until convergence

This is Holland's "Hidden Order" in action:
→ Agents compete for resources (profit)
→ Successful strategies reproduce
→ Population adapts to market
→ Winning strategies EMERGE
"""
import random
from typing import List, Tuple
from trading_agent import (
    TradingAgent, TradingChromosome, MarketState,
    detect_market_condition, CONDITION_NAMES
)

class EvolutionSimulation:
    """Runs GA evolution on trading strategies"""
    
    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[TradingAgent] = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self, initial_cash: float = 10000):
        """Create random initial population"""
        self.population = [
            TradingAgent(i, chromosome=TradingChromosome(), initial_cash=initial_cash)
            for i in range(self.population_size)
        ]
        print(f"✓ Created {self.population_size} agents with random strategies")
    
    def run_market_simulation(self, prices: List[float], verbose: bool = False):
        """
        Run all agents through market data
        
        Args:
            prices: Historical price data
            verbose: Print progress
        """
        if verbose:
            print(f"  📊 Simulating {len(prices)} periods...")
        
        # Reset all agents
        for agent in self.population:
            agent.cash = 10000
            agent.position = 0
            agent.trades = []
            agent.total_pnl = 0
            agent.wins = 0
            agent.losses = 0
        
        # Run through market
        for i, price in enumerate(prices[1:], 1):
            condition = detect_market_condition(prices[:i+1])
            market = MarketState(price, condition, i)
            
            # Each agent decides and acts
            for agent in self.population:
                action = agent.decide_action(market)
                agent.execute_trade(action, market, position_size=0.5)
        
        # Calculate fitness (final portfolio value)
        final_price = prices[-1]
        for agent in self.population:
            agent.fitness = agent.calculate_fitness(final_price)
        
        if verbose:
            print(f"  ✓ Simulation complete")
    
    def evaluate_fitness(self) -> Tuple[float, float, float]:
        """Get fitness statistics"""
        fitnesses = [agent.fitness for agent in self.population]
        return max(fitnesses), sum(fitnesses) / len(fitnesses), min(fitnesses)
    
    def select_parents(self) -> List[TradingAgent]:
        """
        Tournament selection
        
        Better fitness = higher chance to reproduce
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        
        # Keep elite (best performers automatically advance)
        elite = sorted_pop[:self.elite_size]
        
        # Tournament selection for rest
        parents = elite.copy()
        
        while len(parents) < self.population_size:
            # Random tournament
            tournament = random.sample(self.population, 3)
            winner = max(tournament, key=lambda a: a.fitness)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: TradingAgent, parent2: TradingAgent) -> Tuple[TradingAgent, TradingAgent]:
        """Breed two parents"""
        child1_chrome, child2_chrome = parent1.chromosome.crossover(parent2.chromosome)
        
        child1 = TradingAgent(parent1.id, chromosome=child1_chrome, initial_cash=10000)
        child2 = TradingAgent(parent2.id, chromosome=child2_chrome, initial_cash=10000)
        
        return child1, child2
    
    def mutate(self, agent: TradingAgent):
        """Random mutation"""
        agent.chromosome.mutate(self.mutation_rate)
    
    def evolve_generation(self, prices: List[float], verbose: bool = True):
        """
        Run one complete generation
        
        1. Simulate market
        2. Evaluate fitness
        3. Select parents
        4. Create next generation
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation {self.generation}")
            print(f"{'='*60}")
        
        # 1. Simulate
        self.run_market_simulation(prices, verbose=verbose)
        
        # 2. Evaluate
        best, avg, worst = self.evaluate_fitness()
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)
        
        if verbose:
            best_agent = max(self.population, key=lambda a: a.fitness)
            print(f"\n  🏆 Best:    ${best:.2f}")
            print(f"  📊 Average: ${avg:.2f}")
            print(f"  💸 Worst:   ${worst:.2f}")
            print(f"  🧬 Best Strategy: {best_agent.chromosome}")
            print(f"     Trades: {len(best_agent.trades)} | W/L: {best_agent.wins}/{best_agent.losses}")
        
        # 3. Selection
        parents = self.select_parents()
        
        # 4. Create next generation
        next_generation = []
        
        # Keep elite unchanged
        sorted_pop = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        elite = sorted_pop[:self.elite_size]
        for agent in elite:
            # Clone with same chromosome
            clone = TradingAgent(agent.id, chromosome=TradingChromosome(agent.chromosome.genes))
            next_generation.append(clone)
        
        # Breed rest
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate children
                self.mutate(child1)
                self.mutate(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            else:
                # Just clone parent with mutation
                clone = TradingAgent(
                    parent1.id,
                    chromosome=TradingChromosome(parent1.chromosome.genes)
                )
                self.mutate(clone)
                next_generation.append(clone)
        
        self.population = next_generation[:self.population_size]
        self.generation += 1
    
    def run(self, prices: List[float], generations: int = 20, verbose: bool = True):
        """Run full evolution"""
        print("\n" + "="*80)
        print("🧬 STRATEGY EVOLUTION - GA + ABM")
        print("="*80)
        print(f"\n⚙️  Configuration:")
        print(f"   Population: {self.population_size} agents")
        print(f"   Generations: {generations}")
        print(f"   Elite size: {self.elite_size}")
        print(f"   Mutation rate: {self.mutation_rate}")
        print(f"   Crossover rate: {self.crossover_rate}")
        print(f"\n💹 Market data: {len(prices)} periods")
        print(f"   Price: ${prices[0]:.2f} → ${prices[-1]:.2f}")
        print(f"   Return: {((prices[-1]/prices[0])-1)*100:.2f}%")
        
        # Initialize
        if not self.population:
            self.initialize_population()
        
        # Evolve
        for gen in range(generations):
            self.evolve_generation(prices, verbose=verbose)
        
        # Final results
        self.print_summary()
    
    def print_summary(self):
        """Print evolution results"""
        print("\n" + "="*80)
        print("📊 EVOLUTION COMPLETE - RESULTS")
        print("="*80)
        
        # Best performers
        sorted_pop = sorted(self.population, key=lambda a: getattr(a, 'fitness', 0), reverse=True)
        
        print(f"\n🏆 TOP 5 EVOLVED STRATEGIES:")
        for i, agent in enumerate(sorted_pop[:5], 1):
            fitness = getattr(agent, 'fitness', 0)
            print(f"\n#{i} | {agent.chromosome}")
            print(f"    Fitness: ${fitness:.2f}")
            print(f"    Trades: {len(agent.trades)} | W/L: {agent.wins}/{agent.losses}")
            
            if agent.wins + agent.losses > 0:
                win_rate = agent.wins / (agent.wins + agent.losses) * 100
                print(f"    Win Rate: {win_rate:.1f}%")
        
        # Convergence
        print(f"\n📈 Evolution Progress:")
        print(f"   Generation 0:  Best = ${self.best_fitness_history[0]:.2f}")
        print(f"   Generation {self.generation-1}: Best = ${self.best_fitness_history[-1]:.2f}")
        improvement = ((self.best_fitness_history[-1] / self.best_fitness_history[0]) - 1) * 100
        print(f"   Improvement: {improvement:+.2f}%")
        
        print("\n💡 EMERGENCE:")
        print("   → Started with RANDOM strategies")
        print("   → Evolution discovered PATTERNS")
        print("   → Best strategies EMERGED through selection")
        print("   → Population ADAPTED to market conditions")
        
        print("\n" + "="*80)


def generate_synthetic_market(periods: int = 200, trend: float = 0.0005, volatility: float = 0.02) -> List[float]:
    """Generate realistic price data"""
    prices = [100.0]
    
    for _ in range(periods - 1):
        # Random walk with drift and volatility
        change = random.gauss(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Prevent negative prices
    
    return prices


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 LAUNCHING STRATEGY EVOLUTION")
    print("=" * 80)
    
    # Generate market data
    print("\n📊 Generating synthetic market data...")
    prices = generate_synthetic_market(periods=100, trend=0.001, volatility=0.02)
    print(f"   ✓ Created {len(prices)} periods")
    print(f"   ✓ Trend: +0.1% per period (moderate uptrend)")
    print(f"   ✓ Volatility: 2% (realistic)")
    
    # Create evolution simulation
    sim = EvolutionSimulation(
        population_size=30,
        elite_size=3,
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    # Run evolution
    sim.run(prices, generations=15, verbose=True)
    
    print("\n🎯 Next steps:")
    print("   → Test on real crypto data (BTC/ETH)")
    print("   → Add more sophisticated fitness functions")
    print("   → Implement risk-adjusted returns")
    print("   → Add portfolio constraints")
    print("   → Visualize strategy evolution")
    print("=" * 80)
