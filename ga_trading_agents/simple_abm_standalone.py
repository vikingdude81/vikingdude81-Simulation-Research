"""
🐜 Simple Agent-Based Model - Wealth Distribution (Standalone)
Based on "Hidden Order" by John Holland (Echo Model)

This demonstrates emergence: Simple rules create complex wealth inequality.
No external libraries needed!
"""
import random

print("=" * 80)
print("🐜 AGENT-BASED MODEL - WEALTH DISTRIBUTION")
print("=" * 80)

class MoneyAgent:
    """Agent with wealth that moves and exchanges resources"""
    def __init__(self, unique_id, world):
        self.id = unique_id
        self.world = world
        self.wealth = 1  # Everyone starts equal
        self.pos = None
    
    def move(self):
        """Move to random adjacent cell"""
        if self.pos is None:
            return
        
        x, y = self.pos
        # Get all 8 neighbors (Moore neighborhood)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                # Check bounds
                if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                    neighbors.append((nx, ny))
        
        # Find empty neighbors
        empty = [pos for pos in neighbors if self.world.is_empty(pos)]
        
        if empty:
            new_pos = random.choice(empty)
            self.world.move_agent(self, new_pos)
    
    def give_money(self):
        """Give money to a random neighbor if have wealth"""
        if self.wealth <= 0 or self.pos is None:
            return
        
        # Get all neighbors (Moore neighborhood)
        x, y = self.pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                    agents_there = self.world.get_agents_at((nx, ny))
                    neighbors.extend(agents_there)
        
        # Also include agents in same cell
        cellmates = self.world.get_agents_at(self.pos)
        neighbors.extend([a for a in cellmates if a.id != self.id])
        
        if neighbors:
            recipient = random.choice(neighbors)
            recipient.wealth += 1
            self.wealth -= 1
    
    def step(self):
        """Agent's turn"""
        self.move()
        self.give_money()


class World:
    """The simulation world (grid)"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = {}  # {(x, y): [agents]}
        self.agents = []
    
    def add_agent(self, agent, pos):
        """Place agent at position"""
        agent.pos = pos
        if pos not in self.grid:
            self.grid[pos] = []
        self.grid[pos].append(agent)
        self.agents.append(agent)
    
    def move_agent(self, agent, new_pos):
        """Move agent to new position"""
        # Remove from old position
        if agent.pos in self.grid:
            self.grid[agent.pos].remove(agent)
            if not self.grid[agent.pos]:
                del self.grid[agent.pos]
        
        # Add to new position
        agent.pos = new_pos
        if new_pos not in self.grid:
            self.grid[new_pos] = []
        self.grid[new_pos].append(agent)
    
    def is_empty(self, pos):
        """Check if cell is empty"""
        return pos not in self.grid or len(self.grid[pos]) == 0
    
    def get_agents_at(self, pos):
        """Get all agents at position"""
        return self.grid.get(pos, [])
    
    def step(self):
        """Run one time step"""
        # Randomize order each step
        agents = self.agents.copy()
        random.shuffle(agents)
        for agent in agents:
            agent.step()


# === CREATE WORLD ===
WIDTH, HEIGHT = 10, 10
NUM_AGENTS = 50

world = World(WIDTH, HEIGHT)

print(f"\n🌍 Created world: {WIDTH}x{HEIGHT} grid")
print(f"👥 Population: {NUM_AGENTS} agents")
print(f"💰 Initial wealth per agent: 1 unit")
print(f"💵 Total wealth in system: {NUM_AGENTS} units")

# Place agents randomly
for i in range(NUM_AGENTS):
    agent = MoneyAgent(i, world)
    x = random.randint(0, WIDTH - 1)
    y = random.randint(0, HEIGHT - 1)
    world.add_agent(agent, (x, y))

print("\n" + "=" * 80)
print("🚀 RUNNING SIMULATION")
print("=" * 80)

# Run simulation
STEPS = 100
print(f"\n⏱️  Running {STEPS} time steps...")

for step in range(STEPS):
    world.step()
    
    if (step + 1) % 20 == 0:
        all_wealth = [a.wealth for a in world.agents]
        avg = sum(all_wealth) / len(all_wealth)
        print(f"   Step {step+1:3d} | Avg: {avg:.2f} | Range: {min(all_wealth)}-{max(all_wealth)}")

# === ANALYSIS ===
print("\n" + "=" * 80)
print("📊 SIMULATION COMPLETE - RESULTS")
print("=" * 80)

all_wealth = [a.wealth for a in world.agents]
all_wealth_sorted = sorted(all_wealth, reverse=True)

print(f"\n💰 Wealth Statistics:")
print(f"   Total wealth: {sum(all_wealth)} units (conservation check)")
print(f"   Average: {sum(all_wealth) / len(all_wealth):.2f} units")
print(f"   Maximum: {max(all_wealth)} units")
print(f"   Minimum: {min(all_wealth)} units")
print(f"   Median: {sorted(all_wealth)[len(all_wealth)//2]} units")

zero_wealth = all_wealth.count(0)
rich_agents = len([w for w in all_wealth if w >= 3])

print(f"\n📈 Distribution:")
print(f"   Agents with 0 wealth: {zero_wealth} ({zero_wealth/len(all_wealth)*100:.1f}%)")
print(f"   Agents with 3+ wealth: {rich_agents} ({rich_agents/len(all_wealth)*100:.1f}%)")

print(f"\n🏆 Top 10 Wealthiest: {all_wealth_sorted[:10]}")
print(f"💸 Bottom 10: {all_wealth_sorted[-10:]}")

# Gini coefficient
def gini(wealth):
    sorted_w = sorted(wealth)
    n = len(sorted_w)
    cumsum = 0
    for i, w in enumerate(sorted_w):
        cumsum += (2 * (i + 1) - n - 1) * w
    return cumsum / (n * sum(sorted_w))

gini_coef = gini(all_wealth)
print(f"\n📊 Gini Coefficient: {gini_coef:.3f}")
if gini_coef < 0.3:
    print("   → Low inequality")
elif gini_coef < 0.5:
    print("   → Moderate inequality")
else:
    print("   → High inequality")

print("\n" + "=" * 80)
print("💡 EMERGENCE - HIDDEN ORDER")
print("=" * 80)
print("""
Started: Everyone equal (1 unit each)
Rule: Move randomly + give to neighbors

Result: INEQUALITY EMERGED!
✓ Some agents rich ({}+ units)
✓ Some agents poor (0 units)  
✓ Wealth concentrated
✓ NOT programmed - it EMERGED!

This is Holland's "Hidden Order":
→ Simple local rules
→ Complex global behavior
→ Emergent structures
→ Self-organization

Next: Add GA to evolve STRATEGIES!
→ Agents compete with different trading rules
→ Evolution discovers winning strategies
→ Market dynamics emerge
""".format(max(all_wealth)))

print("🎯 Ready to combine GA + ABM for trading!")
print("=" * 80)
