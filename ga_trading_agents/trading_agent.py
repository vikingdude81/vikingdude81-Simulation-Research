"""
📈 Trading Agent with GA Chromosome
Combines Agent-Based Model with Genetic Algorithm

Each agent has a "chromosome" (strategy) that evolves:
- Chromosome: "B-S-H-B-S" = Buy, Sell, Hold decisions for different market conditions
- Fitness: Total profit/loss from trades
- Evolution: Successful strategies reproduce, unsuccessful ones die out
"""
import random
from dataclasses import dataclass
from typing import List, Tuple

# === MARKET CONDITIONS ===
CONDITION_TRENDING_UP = 0
CONDITION_TRENDING_DOWN = 1
CONDITION_VOLATILE = 2
CONDITION_STABLE = 3
CONDITION_BREAKOUT = 4

CONDITION_NAMES = {
    0: "TRENDING_UP",
    1: "TRENDING_DOWN",
    2: "VOLATILE",
    3: "STABLE",
    4: "BREAKOUT"
}

# === ACTIONS ===
ACTION_BUY = 'B'
ACTION_SELL = 'S'
ACTION_HOLD = 'H'

ALL_ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]


@dataclass
class MarketState:
    """Current market information"""
    price: float
    condition: int  # One of CONDITION_* constants
    timestamp: int


class TradingChromosome:
    """
    GA Chromosome = Trading Strategy
    
    Format: List of actions for each market condition
    Example: ['B', 'S', 'H', 'B', 'H']
      → TRENDING_UP: Buy
      → TRENDING_DOWN: Sell
      → VOLATILE: Hold
      → STABLE: Buy
      → BREAKOUT: Hold
    """
    def __init__(self, genes: List[str] = None):
        if genes is None:
            # Random initialization
            self.genes = [random.choice(ALL_ACTIONS) for _ in range(5)]
        else:
            self.genes = genes.copy()
    
    def get_action(self, condition: int) -> str:
        """Get action for market condition"""
        return self.genes[condition]
    
    def mutate(self, mutation_rate: float = 0.1):
        """Randomly change genes"""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = random.choice(ALL_ACTIONS)
    
    def crossover(self, other: 'TradingChromosome') -> Tuple['TradingChromosome', 'TradingChromosome']:
        """Combine with another chromosome"""
        point = random.randint(1, len(self.genes) - 1)
        
        child1_genes = self.genes[:point] + other.genes[point:]
        child2_genes = other.genes[:point] + self.genes[point:]
        
        return TradingChromosome(child1_genes), TradingChromosome(child2_genes)
    
    def __repr__(self):
        return '-'.join(self.genes)


class TradingAgent:
    """
    Agent with GA chromosome strategy
    
    Tracks:
    - Strategy (chromosome)
    - Portfolio (cash + position)
    - Performance (fitness)
    """
    def __init__(self, agent_id: int, chromosome: TradingChromosome = None, initial_cash: float = 10000):
        self.id = agent_id
        self.chromosome = chromosome or TradingChromosome()
        
        # Portfolio
        self.cash = initial_cash
        self.position = 0  # Number of units held
        self.entry_price = 0  # Price we bought at
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        self.wins = 0
        self.losses = 0
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Current total value"""
        return self.cash + (self.position * current_price)
    
    def decide_action(self, market: MarketState) -> str:
        """Use chromosome to decide what to do"""
        return self.chromosome.get_action(market.condition)
    
    def execute_trade(self, action: str, market: MarketState, position_size: float = 1.0):
        """
        Execute trading action
        
        Args:
            action: 'B', 'S', or 'H'
            market: Current market state
            position_size: Fraction of capital to use (0.0 to 1.0)
        """
        if action == ACTION_BUY and self.position == 0:
            # Buy with available cash
            units_to_buy = (self.cash * position_size) / market.price
            cost = units_to_buy * market.price
            
            if cost <= self.cash:
                self.position = units_to_buy
                self.cash -= cost
                self.entry_price = market.price
                
                self.trades.append({
                    'timestamp': market.timestamp,
                    'action': 'BUY',
                    'price': market.price,
                    'units': units_to_buy,
                    'cost': cost
                })
        
        elif action == ACTION_SELL and self.position > 0:
            # Sell all position
            revenue = self.position * market.price
            pnl = revenue - (self.position * self.entry_price)
            
            self.cash += revenue
            self.position = 0
            self.total_pnl += pnl
            
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            self.trades.append({
                'timestamp': market.timestamp,
                'action': 'SELL',
                'price': market.price,
                'units': self.position,
                'revenue': revenue,
                'pnl': pnl
            })
        
        # ACTION_HOLD: Do nothing
    
    def calculate_fitness(self, current_price: float) -> float:
        """
        Fitness = Portfolio performance
        
        Could use:
        - Total return %
        - Sharpe ratio
        - Win rate
        - Risk-adjusted return
        
        For simplicity: Total portfolio value
        """
        portfolio_value = self.get_portfolio_value(current_price)
        return portfolio_value
    
    def clone(self) -> 'TradingAgent':
        """Create a copy with same chromosome"""
        new_agent = TradingAgent(
            agent_id=self.id,
            chromosome=TradingChromosome(self.chromosome.genes),
            initial_cash=self.cash
        )
        return new_agent
    
    def __repr__(self):
        return f"Agent#{self.id}[{self.chromosome}] Cash:{self.cash:.2f} Pos:{self.position:.4f} PnL:{self.total_pnl:.2f}"


# === HELPER FUNCTIONS ===

def detect_market_condition(prices: List[float], lookback: int = 20) -> int:
    """
    Analyze recent prices to determine market condition
    
    Args:
        prices: Recent price history
        lookback: How many periods to analyze
    
    Returns:
        CONDITION_* constant
    """
    if len(prices) < lookback:
        return CONDITION_STABLE
    
    recent = prices[-lookback:]
    
    # Calculate metrics
    returns = [(recent[i] - recent[i-1]) / recent[i-1] for i in range(1, len(recent))]
    avg_return = sum(returns) / len(returns)
    volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
    
    trend = (recent[-1] - recent[0]) / recent[0]
    
    # Classify
    if volatility > 0.03:  # High volatility
        return CONDITION_VOLATILE
    elif trend > 0.05:  # Strong uptrend
        return CONDITION_TRENDING_UP
    elif trend < -0.05:  # Strong downtrend
        return CONDITION_TRENDING_DOWN
    elif abs(recent[-1] - recent[-2]) / recent[-2] > 0.02:  # Sudden move
        return CONDITION_BREAKOUT
    else:
        return CONDITION_STABLE


# === DEMO ===
if __name__ == "__main__":
    print("=" * 80)
    print("📈 TRADING AGENT - GA CHROMOSOME")
    print("=" * 80)
    
    # Create some agents with different strategies
    print("\n🤖 Creating trading agents...")
    
    agent1 = TradingAgent(1)
    agent2 = TradingAgent(2)
    agent3 = TradingAgent(3)
    
    print(f"\nAgent 1 Strategy: {agent1.chromosome}")
    print(f"Agent 2 Strategy: {agent2.chromosome}")
    print(f"Agent 3 Strategy: {agent3.chromosome}")
    
    # Simulate some market conditions
    print("\n📊 Simulating market conditions...")
    
    prices = [100]
    for i in range(50):
        # Random walk
        change = random.gauss(0, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    print(f"\n💹 Price movement: ${prices[0]:.2f} → ${prices[-1]:.2f}")
    print(f"   Return: {((prices[-1] / prices[0]) - 1) * 100:.2f}%")
    
    # Run agents through market
    print("\n🎮 Running trading simulation...")
    
    agents = [agent1, agent2, agent3]
    
    for i, price in enumerate(prices[1:], 1):
        condition = detect_market_condition(prices[:i+1])
        market = MarketState(price, condition, i)
        
        for agent in agents:
            action = agent.decide_action(market)
            agent.execute_trade(action, market, position_size=0.5)
    
    # Show results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    
    for agent in agents:
        fitness = agent.calculate_fitness(prices[-1])
        trades_made = len(agent.trades)
        win_rate = agent.wins / max(agent.wins + agent.losses, 1) * 100
        
        print(f"\n{agent}")
        print(f"   Fitness: ${fitness:.2f}")
        print(f"   Trades: {trades_made}")
        print(f"   W/L: {agent.wins}/{agent.losses} ({win_rate:.1f}% wins)")
    
    # Demonstrate evolution
    print("\n" + "=" * 80)
    print("🧬 EVOLUTION OPERATORS")
    print("=" * 80)
    
    print(f"\nParent 1: {agent1.chromosome}")
    print(f"Parent 2: {agent2.chromosome}")
    
    child1, child2 = agent1.chromosome.crossover(agent2.chromosome)
    
    print(f"\nAfter CROSSOVER:")
    print(f"Child 1: {child1}")
    print(f"Child 2: {child2}")
    
    child1.mutate(mutation_rate=0.3)
    
    print(f"\nAfter MUTATION (child 1):")
    print(f"Child 1: {child1}")
    
    print("\n🎯 Next: Run full GA evolution to find best strategies!")
    print("=" * 80)
