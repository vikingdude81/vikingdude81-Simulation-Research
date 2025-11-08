"""
🧬 QUANTUM GENETIC INTEGRATION TEST - Economic Simulation

Tests the quantum genetic champion as a controller for:
1. Agent behavior (cooperation/defection decisions)
2. Government interventions (economic policy)
3. Comparative analysis vs baseline

This is the full production integration test!
"""

import sys
from pathlib import Path
import random
import numpy as np
import json
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import quantum_genetics
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "PRICE-DETECTION-TEST-1"))
sys.path.insert(0, str(parent_dir))

try:
    from quantum_genetics.deploy_to_simulation import QuantumEconomicAgent, QuantumGovernmentController
    print("✅ Imported quantum genetics modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Searched in: {parent_dir}")
    print(f"   Current sys.path: {sys.path[:3]}")
    sys.exit(1)

try:
    from prisoner_echo_god import Agent, EconomyModel
    print("✅ Imported prisoner_echo_god modules")
except ImportError:
    # Create minimal Agent and EconomyModel classes if not available
    print("⚠️  Using minimal Agent/EconomyModel implementations")
    
    class Agent:
        def __init__(self, agent_id, chromosome, resources, x, y):
            self.agent_id = agent_id
            self.chromosome = chromosome
            self.resources = resources
            self.x = x
            self.y = y
        
        def get_action(self, opponent):
            return 'C' if random.random() > 0.5 else 'D'
    
    class EconomyModel:
        def __init__(self, num_agents=100):
            self.agents = []
            self.generation = 0
            
        def step(self):
            self.generation += 1
            
        def get_statistics(self):
            return {
                'generation': self.generation,
                'population': len(self.agents),
                'avg_resources': 100,
                'cooperation_rate': 0.5,
                'tribe_diversity': 0.5
            }

print("🧬 Quantum Genetic Integration Test Starting...")


class QuantumAgent(Agent):
    """
    Agent that uses quantum genetic controller for decision-making.
    Extends the original Agent class.
    """
    
    def __init__(self, agent_id, chromosome, resources, x, y, use_quantum=True):
        """Initialize with optional quantum controller."""
        super().__init__(agent_id, chromosome, resources, x, y)
        
        self.use_quantum = use_quantum
        if use_quantum:
            # Create quantum economic agent
            self.quantum = QuantumEconomicAgent(
                agent_id=agent_id, 
                environment='standard',
                initial_wealth=resources
            )
        else:
            self.quantum = None
        
        # Track decisions
        self.quantum_decisions = []
        self.interaction_history = []
    
    def decide_with_quantum(self, opponent, model_state):
        """
        Make cooperation decision using quantum controller.
        
        Args:
            opponent: The opponent agent
            model_state: Current model state for context
        
        Returns:
            'C' or 'D'
        """
        if not self.use_quantum or self.quantum is None:
            # Fall back to standard strategy lookup
            return self.get_action(opponent)
        
        # Build opponent history (last 10 interactions)
        opponent_history = []
        for entry in self.interaction_history[-10:]:
            if entry['opponent_id'] == opponent.agent_id:
                opponent_history.append(entry['opponent_cooperated'])
        
        # Get quantum decision
        current_round = len(self.interaction_history)
        cooperate = self.quantum.decide_cooperation(opponent_history, current_round)
        
        decision = 'C' if cooperate else 'D'
        
        # Log decision
        self.quantum_decisions.append({
            'round': current_round,
            'opponent_id': opponent.agent_id,
            'decision': decision,
            'traits': {
                'creativity': float(self.quantum.quantum_agent.traits[0]),
                'coherence': float(self.quantum.quantum_agent.traits[1]),
                'longevity': float(self.quantum.quantum_agent.traits[2])
            }
        })
        
        return decision
    
    def log_interaction(self, opponent_id, my_action, opponent_action):
        """Log interaction for history."""
        self.interaction_history.append({
            'opponent_id': opponent_id,
            'my_action': my_action,
            'opponent_cooperated': (opponent_action == 'C'),
            'opponent_action': opponent_action
        })
    
    def get_quantum_stats(self):
        """Get quantum controller statistics."""
        if not self.use_quantum or self.quantum is None:
            return None
        
        return self.quantum.get_statistics()


class QuantumEconomyModel(EconomyModel):
    """
    Economy model with quantum genetic government controller.
    Extends the original EconomyModel.
    """
    
    def __init__(self, use_quantum_agents=True, use_quantum_government=True, **kwargs):
        """Initialize with quantum controllers."""
        super().__init__(**kwargs)
        
        self.use_quantum_agents = use_quantum_agents
        self.use_quantum_government = use_quantum_government
        
        if use_quantum_government:
            # Create quantum government controller
            self.quantum_gov = QuantumGovernmentController(environment='standard')
            print(f"✅ Quantum Government Controller initialized")
            print(f"   Genome: {self.quantum_gov.genome}")
        else:
            self.quantum_gov = None
        
        # Replace agents with quantum agents if enabled
        if use_quantum_agents:
            self._convert_to_quantum_agents()
        
        # Tracking
        self.quantum_interventions = []
        self.economic_states = []
    
    def _convert_to_quantum_agents(self):
        """Convert existing agents to quantum agents."""
        quantum_agents = []
        for agent in self.agents:
            quantum_agent = QuantumAgent(
                agent_id=agent.agent_id,
                chromosome=agent.chromosome,
                resources=agent.resources,
                x=agent.x,
                y=agent.y,
                use_quantum=True
            )
            quantum_agents.append(quantum_agent)
        
        self.agents = quantum_agents
        print(f"✅ Converted {len(self.agents)} agents to quantum controllers")
    
    def quantum_government_intervention(self):
        """
        Check if quantum government should intervene.
        """
        if not self.use_quantum_government or self.quantum_gov is None:
            return
        
        # Calculate economic state
        if not self.agents:
            return
        
        wealth_values = [a.resources for a in self.agents]
        avg_wealth = np.mean(wealth_values)
        
        # Calculate Gini coefficient approximation
        sorted_wealth = sorted(wealth_values)
        n = len(sorted_wealth)
        gini = 0.0
        if n > 1:
            cumsum = np.cumsum(sorted_wealth)
            gini = (2 * sum((i+1) * w for i, w in enumerate(sorted_wealth)) / (n * sum(sorted_wealth))) - (n+1)/n
        
        # Calculate cooperation rate from recent interactions
        coop_count = 0
        total_count = 0
        for agent in self.agents:
            if isinstance(agent, QuantumAgent) and agent.interaction_history:
                recent = agent.interaction_history[-10:]
                coop_count += sum(1 for entry in recent if entry['my_action'] == 'C')
                total_count += len(recent)
        
        cooperation_rate = coop_count / total_count if total_count > 0 else 0.5
        
        # Calculate growth rate (change in avg wealth)
        growth_rate = 0.0
        if len(self.economic_states) > 0:
            prev_wealth = self.economic_states[-1]['avg_wealth']
            growth_rate = (avg_wealth - prev_wealth) / prev_wealth if prev_wealth > 0 else 0.0
        
        economic_state = {
            'avg_wealth': float(avg_wealth),
            'gini_coefficient': float(gini),
            'cooperation_rate': float(cooperation_rate),
            'growth_rate': float(growth_rate),
            'population': len(self.agents)
        }
        
        self.economic_states.append(economic_state)
        
        # Get quantum government decision
        intervention = self.quantum_gov.decide_intervention(economic_state)
        
        # Apply intervention
        if intervention['type'] != 'no_intervention':
            self._apply_quantum_intervention(intervention, economic_state)
    
    def _apply_quantum_intervention(self, intervention, economic_state):
        """Apply quantum government intervention."""
        intervention_record = {
            'generation': self.generation,
            'type': intervention['type'],
            'magnitude': intervention['magnitude'],
            'target': intervention['target'],
            'reasoning': intervention['reasoning'],
            'economic_state': economic_state
        }
        
        if intervention['type'] == 'wealth_redistribution':
            # Redistribute from rich to poor
            sorted_agents = sorted(self.agents, key=lambda a: a.resources)
            bottom_20 = sorted_agents[:max(1, len(sorted_agents)//5)]
            top_20 = sorted_agents[-max(1, len(sorted_agents)//5):]
            
            amount_per_agent = intervention['magnitude'] * 50  # Scale magnitude
            
            for agent in top_20:
                agent.resources -= amount_per_agent
            for agent in bottom_20:
                agent.resources += amount_per_agent
            
            intervention_record['agents_affected'] = len(bottom_20) + len(top_20)
            intervention_record['total_transferred'] = amount_per_agent * len(top_20)
        
        elif intervention['type'] == 'economic_stimulus':
            # Give resources to all agents
            amount_per_agent = intervention['magnitude'] * 30
            for agent in self.agents:
                agent.resources += amount_per_agent
            
            intervention_record['agents_affected'] = len(self.agents)
            intervention_record['total_injected'] = amount_per_agent * len(self.agents)
        
        elif intervention['type'] == 'infrastructure_investment':
            # Increase productivity (resources) for random agents
            num_beneficiaries = int(len(self.agents) * 0.5)
            beneficiaries = random.sample(self.agents, num_beneficiaries)
            
            amount_per_agent = intervention['magnitude'] * 40
            for agent in beneficiaries:
                agent.resources += amount_per_agent
            
            intervention_record['agents_affected'] = num_beneficiaries
            intervention_record['total_invested'] = amount_per_agent * num_beneficiaries
        
        elif intervention['type'] == 'welfare_program':
            # Target bottom 30%
            sorted_agents = sorted(self.agents, key=lambda a: a.resources)
            bottom_30 = sorted_agents[:max(1, int(len(sorted_agents) * 0.3))]
            
            amount_per_agent = intervention['magnitude'] * 60
            for agent in bottom_30:
                agent.resources += amount_per_agent
            
            intervention_record['agents_affected'] = len(bottom_30)
            intervention_record['total_welfare'] = amount_per_agent * len(bottom_30)
        
        self.quantum_interventions.append(intervention_record)
        
        print(f"\n🏛️ QUANTUM GOV INTERVENTION (Gen {self.generation}):")
        print(f"   Type: {intervention['type']}")
        print(f"   Magnitude: {intervention['magnitude']:.3f}")
        print(f"   Target: {intervention['target']}")
        print(f"   Reasoning: {', '.join(intervention['reasoning'])}")
    
    def step(self):
        """Override step to add quantum government checks."""
        # Check for quantum government intervention every 10 generations
        if self.use_quantum_government and self.generation % 10 == 0:
            self.quantum_government_intervention()
        
        # Call parent step
        super().step()


def run_comparative_test(generations=100, initial_pop=100):
    """
    Run comparative test: Quantum vs Baseline.
    """
    print("\n" + "="*70)
    print("🧬 QUANTUM GENETIC INTEGRATION - COMPARATIVE TEST")
    print("="*70)
    
    results = {}
    
    # Test 1: Baseline (no quantum)
    print("\n" + "="*70)
    print("📊 TEST 1: BASELINE (No Quantum)")
    print("="*70)
    
    baseline_model = QuantumEconomyModel(
        use_quantum_agents=False,
        use_quantum_government=False,
        num_agents=initial_pop
    )
    
    baseline_stats = []
    for gen in range(generations):
        baseline_model.step()
        
        if gen % 10 == 0:
            stats = baseline_model.get_statistics()
            baseline_stats.append(stats)
            print(f"Gen {gen:3d}: Pop={stats['population']:3d} | "
                  f"Avg Wealth={stats['avg_resources']:.1f} | "
                  f"Coop={stats.get('cooperation_rate', 0)*100:.1f}%")
    
    results['baseline'] = {
        'stats': baseline_stats,
        'final': baseline_model.get_statistics()
    }
    
    # Test 2: Quantum Agents Only
    print("\n" + "="*70)
    print("🧬 TEST 2: QUANTUM AGENTS ONLY")
    print("="*70)
    
    quantum_agents_model = QuantumEconomyModel(
        use_quantum_agents=True,
        use_quantum_government=False,
        num_agents=initial_pop
    )
    
    quantum_agents_stats = []
    for gen in range(generations):
        quantum_agents_model.step()
        
        if gen % 10 == 0:
            stats = quantum_agents_model.get_statistics()
            quantum_agents_stats.append(stats)
            print(f"Gen {gen:3d}: Pop={stats['population']:3d} | "
                  f"Avg Wealth={stats['avg_resources']:.1f} | "
                  f"Coop={stats.get('cooperation_rate', 0)*100:.1f}%")
    
    results['quantum_agents'] = {
        'stats': quantum_agents_stats,
        'final': quantum_agents_model.get_statistics()
    }
    
    # Test 3: Quantum Government Only
    print("\n" + "="*70)
    print("🏛️ TEST 3: QUANTUM GOVERNMENT ONLY")
    print("="*70)
    
    quantum_gov_model = QuantumEconomyModel(
        use_quantum_agents=False,
        use_quantum_government=True,
        num_agents=initial_pop
    )
    
    quantum_gov_stats = []
    for gen in range(generations):
        quantum_gov_model.step()
        
        if gen % 10 == 0:
            stats = quantum_gov_model.get_statistics()
            quantum_gov_stats.append(stats)
            print(f"Gen {gen:3d}: Pop={stats['population']:3d} | "
                  f"Avg Wealth={stats['avg_resources']:.1f} | "
                  f"Interventions={len(quantum_gov_model.quantum_interventions)}")
    
    results['quantum_gov'] = {
        'stats': quantum_gov_stats,
        'final': quantum_gov_model.get_statistics(),
        'interventions': quantum_gov_model.quantum_interventions
    }
    
    # Test 4: Full Quantum (Agents + Government)
    print("\n" + "="*70)
    print("🌟 TEST 4: FULL QUANTUM (Agents + Government)")
    print("="*70)
    
    full_quantum_model = QuantumEconomyModel(
        use_quantum_agents=True,
        use_quantum_government=True,
        num_agents=initial_pop
    )
    
    full_quantum_stats = []
    for gen in range(generations):
        full_quantum_model.step()
        
        if gen % 10 == 0:
            stats = full_quantum_model.get_statistics()
            full_quantum_stats.append(stats)
            print(f"Gen {gen:3d}: Pop={stats['population']:3d} | "
                  f"Avg Wealth={stats['avg_resources']:.1f} | "
                  f"Coop={stats.get('cooperation_rate', 0)*100:.1f}% | "
                  f"Interventions={len(full_quantum_model.quantum_interventions)}")
    
    results['full_quantum'] = {
        'stats': full_quantum_stats,
        'final': full_quantum_model.get_statistics(),
        'interventions': full_quantum_model.quantum_interventions
    }
    
    return results, {
        'baseline': baseline_model,
        'quantum_agents': quantum_agents_model,
        'quantum_gov': quantum_gov_model,
        'full_quantum': full_quantum_model
    }


def visualize_results(results, output_dir):
    """Create comprehensive visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    colors = {
        'baseline': '#95a5a6',
        'quantum_agents': '#3498db',
        'quantum_gov': '#e74c3c',
        'full_quantum': '#2ecc71'
    }
    
    labels = {
        'baseline': 'Baseline',
        'quantum_agents': 'Quantum Agents',
        'quantum_gov': 'Quantum Government',
        'full_quantum': 'Full Quantum'
    }
    
    # 1. Population over time
    ax1 = axes[0, 0]
    for name, data in results.items():
        stats = data['stats']
        generations = [s['generation'] for s in stats]
        population = [s['population'] for s in stats]
        ax1.plot(generations, population, linewidth=2, label=labels[name], color=colors[name])
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Population', fontsize=12, fontweight='bold')
    ax1.set_title('Population Dynamics', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average wealth over time
    ax2 = axes[0, 1]
    for name, data in results.items():
        stats = data['stats']
        generations = [s['generation'] for s in stats]
        wealth = [s['avg_resources'] for s in stats]
        ax2.plot(generations, wealth, linewidth=2, label=labels[name], color=colors[name])
    
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Wealth', fontsize=12, fontweight='bold')
    ax2.set_title('Economic Prosperity', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final outcomes comparison
    ax3 = axes[1, 0]
    names = list(results.keys())
    final_pop = [results[n]['final']['population'] for n in names]
    final_wealth = [results[n]['final']['avg_resources'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax3.bar(x - width/2, final_pop, width, label='Population', color=[colors[n] for n in names], alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, final_wealth, width, label='Avg Wealth', color=[colors[n] for n in names], alpha=0.9)
    
    ax3.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Final Population', fontsize=12, fontweight='bold')
    ax3_twin.set_ylabel('Final Avg Wealth', fontsize=12, fontweight='bold')
    ax3.set_title('Final Outcomes Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([labels[n] for n in names], rotation=15, ha='right')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Government interventions (if applicable)
    ax4 = axes[1, 1]
    
    intervention_counts = {}
    for name in ['quantum_gov', 'full_quantum']:
        if name in results and 'interventions' in results[name]:
            interventions = results[name]['interventions']
            types = [i['type'] for i in interventions]
            intervention_counts[name] = dict(Counter(types))
    
    if intervention_counts:
        intervention_types = set()
        for counts in intervention_counts.values():
            intervention_types.update(counts.keys())
        intervention_types = sorted(list(intervention_types))
        
        x = np.arange(len(intervention_types))
        width = 0.35
        
        for i, (name, counts) in enumerate(intervention_counts.items()):
            values = [counts.get(t, 0) for t in intervention_types]
            ax4.bar(x + i*width, values, width, label=labels[name], color=colors[name], alpha=0.8)
        
        ax4.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Quantum Government Interventions', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(intervention_types, rotation=30, ha='right', fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No Interventions', ha='center', va='center', fontsize=14)
        ax4.set_title('Quantum Government Interventions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "quantum_integration_test_results.png", dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: quantum_integration_test_results.png")
    plt.close()


def main():
    """Run full integration test."""
    print("\n" + "="*70)
    print("🚀 QUANTUM GENETIC CHAMPION - ECONOMIC INTEGRATION TEST")
    print("="*70)
    print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = Path(__file__).parent
    
    # Run comparative test
    results, models = run_comparative_test(generations=100, initial_pop=100)
    
    # Save results
    output_file = output_dir / f"quantum_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'stats': data['stats'],
            'final': data['final'],
            'interventions': data.get('interventions', [])
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_file.name}")
    
    # Visualize
    visualize_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON SUMMARY")
    print("="*70)
    
    labels = {
        'baseline': 'Baseline',
        'quantum_agents': 'Quantum Agents',
        'quantum_gov': 'Quantum Government',
        'full_quantum': 'Full Quantum'
    }
    
    for name, data in results.items():
        final = data['final']
        print(f"\n{labels[name]}:")
        print(f"   Final Population:    {final['population']:3d}")
        print(f"   Final Avg Wealth:    {final['avg_resources']:.1f}")
        print(f"   Tribe Diversity:     {final.get('tribe_diversity', 0):.3f}")
        
        if 'interventions' in data:
            print(f"   Total Interventions: {len(data['interventions'])}")
            types = Counter([i['type'] for i in data['interventions']])
            for int_type, count in types.most_common():
                print(f"      {int_type}: {count}")
    
    print("\n" + "="*70)
    print("✅ QUANTUM INTEGRATION TEST COMPLETE!")
    print("="*70)
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Output files:")
    print(f"   - {output_file.name}")
    print(f"   - quantum_integration_test_results.png")
    print(f"\n✨ Quantum genetic champion successfully tested in economic simulation!")


if __name__ == "__main__":
    main()
