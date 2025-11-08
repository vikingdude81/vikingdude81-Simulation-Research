"""
💰 WEALTH INEQUALITY & ELITE EMERGENCE TRACKER
==============================================

Tracks wealth distribution, elite emergence, and "super citizens" across simulations.

Key Metrics:
1. **Wealth Distribution**: Top 1%, Top 10%, Bottom 50%
2. **Elite Emergence**: Who becomes wealthy? Cooperators or defectors?
3. **Wealth Concentration**: Gini coefficient, wealth share ratios
4. **Super Citizens**: Track individuals who accumulate extreme wealth
5. **Wealth Mobility**: Do agents move between classes?
6. **Dynasty Formation**: Do wealthy lineages persist across generations?

Research Questions:
- Do certain government types create oligarchies?
- Do cooperators or defectors become wealthy?
- Does genetic diversity correlate with wealth inequality?
- Can "super citizens" emerge and dominate?
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class WealthSnapshot:
    """Snapshot of wealth distribution at a single generation."""
    generation: int
    
    # Overall distribution
    total_wealth: float
    mean_wealth: float
    median_wealth: float
    std_wealth: float
    
    # Inequality metrics
    gini_coefficient: float
    top_1_percent_share: float  # % of total wealth held by top 1%
    top_10_percent_share: float  # % of total wealth held by top 10%
    bottom_50_percent_share: float  # % of total wealth held by bottom 50%
    
    # Elite metrics
    richest_wealth: float
    richest_strategy: int  # 0=Defector, 1=Cooperator
    richest_age: int
    num_super_citizens: int  # Agents with wealth > 10x median
    
    # Wealth by strategy
    cooperator_mean_wealth: float
    defector_mean_wealth: float
    cooperator_count: int
    defector_count: int
    
    # Wealth mobility (if tracking over time)
    wealth_mobility_index: float = 0.0  # % of agents who changed class


@dataclass
class SuperCitizen:
    """Tracks an individual "super citizen" who accumulates extreme wealth."""
    agent_id: int
    birth_generation: int
    death_generation: Optional[int]
    peak_wealth: float
    peak_generation: int
    strategy: int  # 0=Defector, 1=Cooperator
    final_wealth: float
    lifespan: int
    wealth_history: List[float]
    
    def is_alive(self, current_gen: int) -> bool:
        """Check if super citizen is still alive."""
        return self.death_generation is None or self.death_generation > current_gen


class WealthInequalityTracker:
    """
    Comprehensive wealth inequality tracking system.
    
    Tracks:
    - Wealth distribution over time
    - Elite emergence patterns
    - Super citizen lifespans
    - Wealth mobility between classes
    """
    
    def __init__(self):
        self.snapshots: List[WealthSnapshot] = []
        self.super_citizens: Dict[int, SuperCitizen] = {}  # agent_id -> SuperCitizen
        self.wealth_classes_history: List[Dict[int, str]] = []  # Track class membership
        
        # Thresholds
        self.SUPER_CITIZEN_THRESHOLD = 10  # 10x median wealth
        self.ELITE_THRESHOLD = 5  # 5x median wealth
        
    def capture_snapshot(self, agents: List, generation: int) -> WealthSnapshot:
        """
        Capture comprehensive wealth snapshot at current generation.
        
        Args:
            agents: List of agents with wealth, strategy, age attributes
            generation: Current generation number
            
        Returns:
            WealthSnapshot with all metrics
        """
        if len(agents) == 0:
            return self._empty_snapshot(generation)
        
        # Extract wealth data
        wealths = np.array([a.wealth for a in agents if a.wealth > 0])
        if len(wealths) == 0:
            return self._empty_snapshot(generation)
        
        wealths_sorted = np.sort(wealths)
        total_wealth = np.sum(wealths)
        mean_wealth = np.mean(wealths)
        median_wealth = np.median(wealths)
        
        # Calculate Gini coefficient
        gini = self._calculate_gini(wealths_sorted)
        
        # Calculate wealth concentration by percentile
        n = len(wealths)
        top_1_idx = max(0, int(n * 0.99))
        top_10_idx = max(0, int(n * 0.90))
        bottom_50_idx = int(n * 0.50)
        
        top_1_wealth = np.sum(wealths_sorted[top_1_idx:])
        top_10_wealth = np.sum(wealths_sorted[top_10_idx:])
        bottom_50_wealth = np.sum(wealths_sorted[:bottom_50_idx])
        
        top_1_share = (top_1_wealth / total_wealth * 100) if total_wealth > 0 else 0
        top_10_share = (top_10_wealth / total_wealth * 100) if total_wealth > 0 else 0
        bottom_50_share = (bottom_50_wealth / total_wealth * 100) if total_wealth > 0 else 0
        
        # Find richest agent
        richest_idx = np.argmax([a.wealth for a in agents])
        richest_agent = agents[richest_idx]
        richest_wealth = richest_agent.wealth
        richest_strategy = richest_agent.get_strategy()
        richest_age = getattr(richest_agent, 'age', 0)
        
        # Count super citizens (wealth > 10x median)
        num_super = np.sum(wealths > median_wealth * self.SUPER_CITIZEN_THRESHOLD)
        
        # Wealth by strategy
        cooperators = [a for a in agents if a.get_strategy() == 1]
        defectors = [a for a in agents if a.get_strategy() == 0]
        
        cooperator_mean = np.mean([a.wealth for a in cooperators]) if cooperators else 0
        defector_mean = np.mean([a.wealth for a in defectors]) if defectors else 0
        
        # Calculate wealth mobility (if we have previous snapshot)
        mobility = self._calculate_wealth_mobility(agents, generation)
        
        snapshot = WealthSnapshot(
            generation=generation,
            total_wealth=total_wealth,
            mean_wealth=mean_wealth,
            median_wealth=median_wealth,
            std_wealth=np.std(wealths),
            gini_coefficient=gini,
            top_1_percent_share=top_1_share,
            top_10_percent_share=top_10_share,
            bottom_50_percent_share=bottom_50_share,
            richest_wealth=richest_wealth,
            richest_strategy=richest_strategy,
            richest_age=richest_age,
            num_super_citizens=num_super,
            cooperator_mean_wealth=cooperator_mean,
            defector_mean_wealth=defector_mean,
            cooperator_count=len(cooperators),
            defector_count=len(defectors),
            wealth_mobility_index=mobility
        )
        
        self.snapshots.append(snapshot)
        
        # Update super citizen tracking
        self._update_super_citizens(agents, generation, median_wealth)
        
        return snapshot
    
    def _calculate_gini(self, wealths_sorted: np.ndarray) -> float:
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
        if len(wealths_sorted) <= 1:
            return 0.0
        
        n = len(wealths_sorted)
        cumsum = np.cumsum(wealths_sorted)
        total = cumsum[-1]
        
        if total == 0:
            return 0.0
        
        # Gini = (2 * sum(i * wealth_i)) / (n * sum(wealth)) - (n + 1) / n
        gini = (2 * np.sum((np.arange(n) + 1) * wealths_sorted)) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, gini))
    
    def _calculate_wealth_mobility(self, agents: List, generation: int) -> float:
        """
        Calculate wealth mobility: % of agents who changed wealth class.
        
        Classes: Poor (bottom 33%), Middle (33-67%), Rich (top 33%)
        """
        if len(self.wealth_classes_history) == 0:
            # First generation - no mobility yet
            current_classes = self._classify_agents(agents)
            self.wealth_classes_history.append(current_classes)
            return 0.0
        
        # Get previous and current classifications
        prev_classes = self.wealth_classes_history[-1]
        current_classes = self._classify_agents(agents)
        self.wealth_classes_history.append(current_classes)
        
        # Find agents that exist in both timepoints
        common_agents = set(prev_classes.keys()) & set(current_classes.keys())
        
        if len(common_agents) == 0:
            return 0.0
        
        # Count how many changed class
        changed = sum(1 for agent_id in common_agents 
                     if prev_classes[agent_id] != current_classes[agent_id])
        
        return (changed / len(common_agents)) * 100
    
    def _classify_agents(self, agents: List) -> Dict[int, str]:
        """Classify agents into wealth classes (Poor/Middle/Rich)."""
        wealths = [(id(a), a.wealth) for a in agents]
        wealths_sorted = sorted(wealths, key=lambda x: x[1])
        
        n = len(wealths_sorted)
        bottom_33 = int(n * 0.33)
        top_33 = int(n * 0.67)
        
        classes = {}
        for i, (agent_id, wealth) in enumerate(wealths_sorted):
            if i < bottom_33:
                classes[agent_id] = "Poor"
            elif i < top_33:
                classes[agent_id] = "Middle"
            else:
                classes[agent_id] = "Rich"
        
        return classes
    
    def _update_super_citizens(self, agents: List, generation: int, median_wealth: float):
        """Track super citizens (agents with extreme wealth)."""
        threshold = median_wealth * self.SUPER_CITIZEN_THRESHOLD
        
        for agent in agents:
            agent_id = id(agent)
            
            if agent.wealth >= threshold:
                # Agent is a super citizen
                if agent_id not in self.super_citizens:
                    # New super citizen emerged!
                    self.super_citizens[agent_id] = SuperCitizen(
                        agent_id=agent_id,
                        birth_generation=generation,
                        death_generation=None,
                        peak_wealth=agent.wealth,
                        peak_generation=generation,
                        strategy=agent.get_strategy(),
                        final_wealth=agent.wealth,
                        lifespan=0,
                        wealth_history=[agent.wealth]
                    )
                else:
                    # Existing super citizen - update
                    sc = self.super_citizens[agent_id]
                    sc.wealth_history.append(agent.wealth)
                    sc.final_wealth = agent.wealth
                    sc.lifespan = generation - sc.birth_generation
                    
                    if agent.wealth > sc.peak_wealth:
                        sc.peak_wealth = agent.wealth
                        sc.peak_generation = generation
        
        # Mark super citizens who died (no longer in agent list)
        current_agent_ids = {id(a) for a in agents}
        for agent_id, sc in self.super_citizens.items():
            if sc.death_generation is None and agent_id not in current_agent_ids:
                sc.death_generation = generation
    
    def _empty_snapshot(self, generation: int) -> WealthSnapshot:
        """Return empty snapshot for extinct populations."""
        return WealthSnapshot(
            generation=generation,
            total_wealth=0, mean_wealth=0, median_wealth=0, std_wealth=0,
            gini_coefficient=0, top_1_percent_share=0, top_10_percent_share=0,
            bottom_50_percent_share=0, richest_wealth=0, richest_strategy=0,
            richest_age=0, num_super_citizens=0, cooperator_mean_wealth=0,
            defector_mean_wealth=0, cooperator_count=0, defector_count=0
        )
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of wealth inequality over time."""
        if len(self.snapshots) == 0:
            return {"error": "No snapshots captured"}
        
        # Time-averaged metrics
        avg_gini = np.mean([s.gini_coefficient for s in self.snapshots])
        avg_top_1_share = np.mean([s.top_1_percent_share for s in self.snapshots])
        avg_top_10_share = np.mean([s.top_10_percent_share for s in self.snapshots])
        avg_mobility = np.mean([s.wealth_mobility_index for s in self.snapshots])
        
        # Final snapshot metrics
        final = self.snapshots[-1]
        
        # Super citizen analysis
        total_super_citizens = len(self.super_citizens)
        super_cooperators = sum(1 for sc in self.super_citizens.values() if sc.strategy == 1)
        super_defectors = total_super_citizens - super_cooperators
        
        avg_super_lifespan = np.mean([sc.lifespan for sc in self.super_citizens.values()]) if self.super_citizens else 0
        max_super_wealth = max([sc.peak_wealth for sc in self.super_citizens.values()]) if self.super_citizens else 0
        
        return {
            'time_period': {
                'generations': len(self.snapshots),
                'start_gen': self.snapshots[0].generation,
                'end_gen': self.snapshots[-1].generation
            },
            'average_inequality': {
                'gini_coefficient': avg_gini,
                'top_1_percent_share': avg_top_1_share,
                'top_10_percent_share': avg_top_10_share,
                'wealth_mobility': avg_mobility
            },
            'final_state': {
                'gini': final.gini_coefficient,
                'top_1_share': final.top_1_percent_share,
                'richest_wealth': final.richest_wealth,
                'richest_strategy': 'Cooperator' if final.richest_strategy == 1 else 'Defector',
                'num_super_citizens': final.num_super_citizens
            },
            'super_citizens': {
                'total_emerged': total_super_citizens,
                'cooperators': super_cooperators,
                'defectors': super_defectors,
                'avg_lifespan': avg_super_lifespan,
                'max_wealth_ever': max_super_wealth
            },
            'wealth_by_strategy': {
                'cooperator_mean': final.cooperator_mean_wealth,
                'defector_mean': final.defector_mean_wealth,
                'wealth_gap': final.defector_mean_wealth - final.cooperator_mean_wealth
            }
        }
    
    def plot_wealth_inequality(self, save_path: str = "wealth_inequality_analysis.png"):
        """Create comprehensive wealth inequality visualization."""
        if len(self.snapshots) < 2:
            print("Not enough snapshots for visualization")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('🏛️ Wealth Inequality & Elite Emergence Analysis', fontsize=16, fontweight='bold')
        
        generations = [s.generation for s in self.snapshots]
        
        # 1. Gini Coefficient over time
        ax = axes[0, 0]
        gini_values = [s.gini_coefficient for s in self.snapshots]
        ax.plot(generations, gini_values, linewidth=2, color='darkred')
        ax.fill_between(generations, 0, gini_values, alpha=0.3, color='darkred')
        ax.set_title('Gini Coefficient (Inequality)', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Gini (0=Equal, 1=Unequal)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.4, color='orange', linestyle='--', label='Moderate (0.4)')
        ax.legend()
        
        # 2. Wealth Concentration (Top 1%, 10%, Bottom 50%)
        ax = axes[0, 1]
        top_1 = [s.top_1_percent_share for s in self.snapshots]
        top_10 = [s.top_10_percent_share for s in self.snapshots]
        bottom_50 = [s.bottom_50_percent_share for s in self.snapshots]
        ax.plot(generations, top_1, label='Top 1%', linewidth=2, color='darkred')
        ax.plot(generations, top_10, label='Top 10%', linewidth=2, color='orange')
        ax.plot(generations, bottom_50, label='Bottom 50%', linewidth=2, color='blue')
        ax.set_title('Wealth Concentration', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('% of Total Wealth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Super Citizens Count
        ax = axes[0, 2]
        super_count = [s.num_super_citizens for s in self.snapshots]
        ax.plot(generations, super_count, linewidth=2, color='gold', marker='o', markersize=4)
        ax.fill_between(generations, 0, super_count, alpha=0.3, color='gold')
        ax.set_title('Super Citizens (>10x Median Wealth)', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # 4. Wealth by Strategy (Cooperator vs Defector)
        ax = axes[1, 0]
        coop_wealth = [s.cooperator_mean_wealth for s in self.snapshots]
        def_wealth = [s.defector_mean_wealth for s in self.snapshots]
        ax.plot(generations, coop_wealth, label='Cooperators', linewidth=2, color='green')
        ax.plot(generations, def_wealth, label='Defectors', linewidth=2, color='red')
        ax.set_title('Average Wealth by Strategy', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Wealth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Richest Agent Wealth
        ax = axes[1, 1]
        richest = [s.richest_wealth for s in self.snapshots]
        median = [s.median_wealth for s in self.snapshots]
        ax.plot(generations, richest, label='Richest', linewidth=2, color='purple')
        ax.plot(generations, median, label='Median', linewidth=2, color='gray', linestyle='--')
        ax.set_title('Richest vs Median Wealth', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Wealth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 6. Wealth Mobility
        ax = axes[1, 2]
        mobility = [s.wealth_mobility_index for s in self.snapshots]
        ax.plot(generations, mobility, linewidth=2, color='teal')
        ax.fill_between(generations, 0, mobility, alpha=0.3, color='teal')
        ax.set_title('Wealth Mobility (Class Changes)', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('% Agents Changed Class')
        ax.grid(True, alpha=0.3)
        
        # 7. Final Wealth Distribution (Histogram)
        ax = axes[2, 0]
        final_wealths = [a.wealth for a in self.snapshots[-1].cooperator_count + self.snapshots[-1].defector_count] if hasattr(self.snapshots[-1], 'agents') else []
        if len(final_wealths) > 0:
            ax.hist(final_wealths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title('Final Wealth Distribution', fontweight='bold')
        ax.set_xlabel('Wealth')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # 8. Super Citizen Lifespan Distribution
        ax = axes[2, 1]
        if len(self.super_citizens) > 0:
            lifespans = [sc.lifespan for sc in self.super_citizens.values()]
            ax.hist(lifespans, bins=20, color='gold', edgecolor='black', alpha=0.7)
            ax.set_title('Super Citizen Lifespans', fontweight='bold')
            ax.set_xlabel('Generations Alive')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Super Citizens', ha='center', va='center', fontsize=12)
            ax.set_title('Super Citizen Lifespans', fontweight='bold')
        
        # 9. Summary Statistics Table
        ax = axes[2, 2]
        ax.axis('off')
        summary = self.get_summary()
        
        summary_text = f"""
        📊 INEQUALITY SUMMARY
        
        Average Gini: {summary['average_inequality']['gini_coefficient']:.3f}
        Top 1% Share: {summary['average_inequality']['top_1_percent_share']:.1f}%
        Top 10% Share: {summary['average_inequality']['top_10_percent_share']:.1f}%
        
        💰 SUPER CITIZENS
        Total Emerged: {summary['super_citizens']['total_emerged']}
        Cooperators: {summary['super_citizens']['cooperators']}
        Defectors: {summary['super_citizens']['defectors']}
        Max Wealth: {summary['super_citizens']['max_wealth_ever']:.1f}
        
        🔄 WEALTH MOBILITY
        Avg Mobility: {summary['average_inequality']['wealth_mobility']:.1f}%
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Wealth inequality analysis saved to: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("💰 WEALTH INEQUALITY & ELITE EMERGENCE TRACKER")
    print("=" * 60)
    print("\nUsage:")
    print("  from wealth_inequality_tracker import WealthInequalityTracker")
    print("  tracker = WealthInequalityTracker()")
    print("  ")
    print("  # Each generation:")
    print("  snapshot = tracker.capture_snapshot(agents, generation)")
    print("  ")
    print("  # At end:")
    print("  summary = tracker.get_summary()")
    print("  tracker.plot_wealth_inequality('results.png')")
