"""
🐜 Simple Agent-Based Model - Wealth Distribution
Based on "Hidden Order" by John Holland (Echo Model, Fig 4.2)

This demonstrates how simple agent rules create emergent complexity.
Agents move randomly and exchange resources, leading to wealth inequality.

Concepts demonstrated:
- Agent-based modeling
- Resource exchange
- Emergent behavior
- Wealth distribution (inequality emerges naturally!)
"""
import random
from mesa import Agent, Model
from mesa.space import MultiGrid

print("=" * 80)
print("🐜 AGENT-BASED MODEL - WEALTH DISTRIBUTION")
print("=" * 80)

class MoneyAgent(Agent):
    """
    An agent with fixed initial wealth.
    Simple rules:
    1. Move randomly
    2. Give money to neighbors if present
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1  # Everyone starts equal

    def move(self):
        """Move to a random adjacent empty cell."""
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,  # All 8 surrounding cells
            include_center=False
        )
        
        # Find empty cells
        valid_steps = [step for step in possible_steps 
                      if self.model.grid.is_cell_empty(step)]
        
        if valid_steps:
            new_position = self.random.choice(valid_steps)
            self.model.grid.move_agent(self, new_position)

    def give_money(self):
        """
        Resource exchange: Give money to a random neighbor.
        This simple rule creates complex wealth distribution!
        """
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        
        if len(cellmates) > 1:  # If there are neighbors
            other_agent = self.random.choice(cellmates)
            if other_agent != self:
                # Transfer 1 unit of wealth
                other_agent.wealth += 1
                self.wealth -= 1

    def step(self):
        """Agent's turn in the simulation."""
        self.move()
        if self.wealth > 0:
            self.give_money()


class MoneyModel(Model):
    """
    The 'world' containing agents and grid.
    This is like the Echo environment from the book.
    """
    def __init__(self, N, width, height):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, torus=False)
        
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            
            # Place agent randomly on grid
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        
        print(f"\n🌍 Created world: {width}x{height} grid")
        print(f"👥 Population: {N} agents")
        print(f"💰 Initial wealth per agent: 1 unit")
        print(f"💵 Total wealth in system: {N} units")

    def step(self):
        """Advance the model by one step."""
        # Mesa 3+ automatically manages agents
        for agent in self.agents:
            agent.step()


# === RUN SIMULATION ===

print("\n" + "=" * 80)
print("🚀 STARTING SIMULATION")
print("=" * 80)

# Create model: 50 agents on 10x10 grid
model = MoneyModel(50, 10, 10)

# Run for 100 time steps
STEPS = 100
print(f"\n⏱️  Running {STEPS} time steps...")

for i in range(STEPS):
    model.step()
    
    # Show progress
    if (i + 1) % 20 == 0:
        all_wealth = [agent.wealth for agent in model.agents]
        avg_wealth = sum(all_wealth) / len(all_wealth)
        max_wealth = max(all_wealth)
        min_wealth = min(all_wealth)
        print(f"   Step {i+1:3d} | Avg: {avg_wealth:.2f} | Range: {min_wealth}-{max_wealth}")

# === ANALYSIS ===

print("\n" + "=" * 80)
print("📊 SIMULATION COMPLETE - RESULTS")
print("=" * 80)

all_wealth = [agent.wealth for agent in model.agents]
all_wealth_sorted = sorted(all_wealth, reverse=True)

print(f"\n💰 Wealth Statistics:")
print(f"   Total wealth: {sum(all_wealth)} units (same as start)")
print(f"   Average: {sum(all_wealth) / len(all_wealth):.2f} units")
print(f"   Maximum: {max(all_wealth)} units")
print(f"   Minimum: {min(all_wealth)} units")
print(f"   Median: {sorted(all_wealth)[len(all_wealth)//2]} units")

# Count wealth distribution
zero_wealth = all_wealth.count(0)
rich_agents = len([w for w in all_wealth if w >= 3])

print(f"\n📈 Distribution:")
print(f"   Agents with 0 wealth: {zero_wealth} ({zero_wealth/len(all_wealth)*100:.1f}%)")
print(f"   Agents with 3+ wealth: {rich_agents} ({rich_agents/len(all_wealth)*100:.1f}%)")

# Show top and bottom agents
print(f"\n🏆 Top 10 Wealthiest Agents:")
print(f"   {all_wealth_sorted[:10]}")

print(f"\n💸 Bottom 10 Agents:")
print(f"   {all_wealth_sorted[-10:]}")

# Gini coefficient (wealth inequality measure)
def gini_coefficient(wealth_list):
    """Calculate Gini coefficient: 0 = perfect equality, 1 = total inequality"""
    sorted_wealth = sorted(wealth_list)
    n = len(sorted_wealth)
    cumsum = 0
    for i, w in enumerate(sorted_wealth):
        cumsum += (2 * (i + 1) - n - 1) * w
    return cumsum / (n * sum(sorted_wealth))

gini = gini_coefficient(all_wealth)
print(f"\n📊 Gini Coefficient: {gini:.3f}")
if gini < 0.3:
    print("   → Low inequality (relatively equal)")
elif gini < 0.5:
    print("   → Moderate inequality")
else:
    print("   → High inequality (wealth concentrated)")

print("\n" + "=" * 80)
print("💡 KEY INSIGHT - EMERGENCE!")
print("=" * 80)
print("""
Everyone started with equal wealth (1 unit).
Simple rule: Move randomly, give money to neighbors.

Result: INEQUALITY EMERGES!
- Some agents became rich (3+ units)
- Some agents became poor (0 units)
- Wealth concentrated in few agents

This wasn't programmed - it EMERGED from simple interactions!
This is Holland's "Hidden Order" in action.

In trading simulation:
→ Simple trading rules will create complex market dynamics
→ Some strategies will dominate
→ Market patterns will emerge
→ Evolution will discover these patterns
""")

print("\n🎯 Ready to add this to trading? Agents will compete with")
print("   evolving strategies instead of random moves!")
print("=" * 80)
