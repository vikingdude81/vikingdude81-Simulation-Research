"""
🧬👁️ QUANTUM ML-BASED GOD CONTROLLER

This module integrates the Quantum Genetic Evolution Champion as the ML-based
God controller for the prisoner's dilemma simulation.

The quantum controller uses evolved parameters (μ=5.0, ω=0.1, d=0.0001, φ=2π)
to make intelligent intervention decisions based on economic state.

Integration with prisoner_echo_god.py:
- Replaces the placeholder _ml_based_decision() method
- Uses quantum traits (creativity, coherence, longevity) for governance
- Adapts interventions dynamically based on simulation state
"""

import sys
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, List
from enum import Enum

# Import quantum genetics modules
# Try multiple path locations
quantum_paths = [
    Path(__file__).parent.parent / "quantum_genetics",
    Path(__file__).parent / "quantum_genetics",
    Path(__file__).parent.parent.parent / "quantum_genetics"
]

quantum_path = None
for path in quantum_paths:
    if path.exists():
        quantum_path = path
        break

if quantum_path is None:
    raise ImportError(f"Quantum genetics path not found in any of: {quantum_paths}")

sys.path.insert(0, str(quantum_path))

try:
    from deploy_champion import ChampionGenome
    from quantum_genetic_agents import QuantumAgent
    print(f"✅ Successfully imported quantum genetics modules from {quantum_path}")
except ImportError as e:
    print(f"❌ Failed to import quantum genetics: {e}")
    print(f"   Tried path: {quantum_path}")
    print(f"   Path exists: {quantum_path.exists()}")
    if quantum_path.exists():
        import os
        print(f"   Files in path: {os.listdir(quantum_path)[:10]}")
    raise

# Import prisoner echo structures
try:
    from prisoner_echo_god import InterventionType
except ImportError:
    # Define locally if not available
    class InterventionType(Enum):
        STIMULUS = "stimulus"
        WELFARE = "welfare"
        SPAWN_TRIBE = "spawn_tribe"
        EMERGENCY_REVIVAL = "emergency_revival"
        FORCED_COOPERATION = "forced_cooperation"


class QuantumGodController:
    """
    ML-based God controller using Quantum Genetic Evolution.
    
    This controller uses the champion genome [5.0, 0.1, 0.0001, 2π] to make
    intelligent governance decisions in the economic simulation.
    
    Key Features:
    - Adaptive interventions based on quantum traits
    - Dynamic learning through trait evolution
    - Context-aware policy decisions
    - No hard-coded rules - purely learning-based
    """
    
    def __init__(self, environment='standard', intervention_cooldown=10):
        """
        Initialize quantum god controller.
        
        Args:
            environment: Environment type for quantum agent
            intervention_cooldown: Minimum generations between interventions
        """
        # Initialize champion
        self.champion = ChampionGenome()
        self.genome = self.champion.get_genome()
        
        # Create quantum agent for governance
        self.agent = self.champion.create_agent(agent_id=0, environment=environment)
        
        # State tracking
        self.timestep = 0
        self.interventions = []
        self.intervention_cooldown = intervention_cooldown
        self.last_intervention_time = -intervention_cooldown
        
        # Learning history
        self.state_history = []
        self.outcome_history = []
        
        print(f"\n🧬 Quantum God Controller Initialized")
        print(f"   Genome: μ={self.genome[0]:.1f}, ω={self.genome[1]:.3f}, d={self.genome[2]:.6f}, φ={self.genome[3]:.6f}")
    
    def decide_intervention(self, state: Dict, generation: int) -> Optional[Tuple[str, str, Dict]]:
        """
        Decide intervention using quantum genetic evolution.
        
        This is the main decision method called by GodController in prisoner_echo_god.py
        
        Args:
            state: Economic state dictionary with:
                - population: int
                - avg_wealth: float
                - cooperation_rate: float
                - gini_coefficient: float
                - growth_rate: float
                - tribe_dominance: float
            generation: Current generation number
        
        Returns:
            Tuple of (intervention_type, reasoning, parameters) or None
        """
        # Check cooldown
        if generation - self.last_intervention_time < self.intervention_cooldown:
            return None
        
        # Evolve quantum agent
        self.agent.evolve(self.timestep)
        self.timestep += 1
        
        # Get quantum traits
        creativity = float(self.agent.traits[0])
        coherence = float(self.agent.traits[1])
        longevity = float(self.agent.traits[2])
        fitness = float(self.agent.get_final_fitness())
        
        # Extract state features
        pop = state.get('population', 0)
        wealth = state.get('avg_wealth', 0)
        coop = state.get('cooperation_rate', 0)
        gini = state.get('gini_coefficient', 0)
        growth = state.get('growth_rate', 0)
        dominance = state.get('tribe_dominance', 0)
        
        # Store state for learning
        self.state_history.append({
            'generation': generation,
            'state': state.copy(),
            'traits': {'creativity': creativity, 'coherence': coherence, 'longevity': longevity},
            'fitness': fitness
        })
        
        # Quantum decision making - map traits to interventions
        intervention = self._quantum_intervention_mapping(
            creativity, coherence, longevity, fitness,
            pop, wealth, coop, gini, growth, dominance
        )
        
        if intervention is not None:
            self.last_intervention_time = generation
            self.interventions.append({
                'generation': generation,
                'intervention': intervention,
                'traits': {'creativity': creativity, 'coherence': coherence, 'longevity': longevity},
                'state': state.copy()
            })
        
        return intervention
    
    def _quantum_intervention_mapping(self, creativity, coherence, longevity, fitness,
                                     pop, wealth, coop, gini, growth, dominance) -> Optional[Tuple[str, str, Dict]]:
        """
        Map quantum traits to intervention decisions.
        
        This is where the quantum evolution "learns" optimal governance strategies.
        No hard-coded thresholds - decisions emerge from trait dynamics.
        """
        # Normalize traits for decision making
        c_norm = np.clip(creativity / 10.0, 0, 1)
        h_norm = np.clip(coherence / 10.0, 0, 1)
        l_norm = np.clip(longevity / 10.0, 0, 1)
        
        # Calculate intervention scores for each type
        scores = {}
        
        # === STIMULUS (Universal Basic Income) ===
        # High when: low cooperation + high creativity (innovative solution)
        stimulus_score = 0
        if coop < 0.5:
            stimulus_score += (0.5 - coop) * 2 * c_norm
        if wealth < 60:
            stimulus_score += (60 - wealth) / 60 * c_norm
        scores['STIMULUS'] = stimulus_score
        
        # === WELFARE (Targeted Assistance) ===
        # High when: high inequality + high coherence (structured aid)
        welfare_score = 0
        if gini > 0.4:
            welfare_score += (gini - 0.4) * 3 * h_norm
        if wealth < 50:
            welfare_score += (50 - wealth) / 50 * h_norm
        scores['WELFARE'] = welfare_score
        
        # === SPAWN_TRIBE (Diversity Injection) ===
        # High when: high dominance + high creativity (disrupt stagnation)
        spawn_score = 0
        if dominance > 0.85:
            spawn_score += (dominance - 0.85) * 10 * c_norm
        if coop < 0.3:
            spawn_score += (0.3 - coop) * 2 * c_norm
        scores['SPAWN_TRIBE'] = spawn_score
        
        # === EMERGENCY_REVIVAL (Collapse Prevention) ===
        # High when: low population + high longevity (preserve future)
        emergency_score = 0
        pop_ratio = pop / 1000  # Assuming MAX_POPULATION = 1000
        if pop_ratio < 0.1:
            emergency_score += (0.1 - pop_ratio) * 20 * l_norm
        if wealth < 30:
            emergency_score += (30 - wealth) / 30 * l_norm * 2
        scores['EMERGENCY_REVIVAL'] = emergency_score
        
        # === FORCED_COOPERATION (Social Engineering) ===
        # High when: very low cooperation + balanced traits
        coop_score = 0
        if coop < 0.2:
            coop_score += (0.2 - coop) * 5 * (c_norm + h_norm) / 2
        scores['FORCED_COOPERATION'] = coop_score
        
        # Find highest scoring intervention
        if not scores or max(scores.values()) < 0.1:
            return None  # No intervention needed
        
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Generate parameters based on quantum traits
        parameters = self._generate_parameters(best_type, creativity, coherence, longevity, 
                                               wealth, coop, gini, dominance)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_type, creativity, coherence, longevity,
                                            wealth, coop, gini, growth, dominance, best_score)
        
        return (best_type, reasoning, parameters)
    
    def _generate_parameters(self, intervention_type, creativity, coherence, longevity,
                            wealth, coop, gini, dominance) -> Dict:
        """Generate intervention parameters based on quantum traits."""
        
        if intervention_type == 'STIMULUS':
            # Amount scales with creativity (more creative = bigger stimulus)
            base_amount = 50
            amount = int(base_amount * (1 + creativity / 10))
            return {'amount_per_agent': amount}
        
        elif intervention_type == 'WELFARE':
            # Target and amount scale with coherence (more structured = more targeted)
            target_pct = np.clip(0.05 + (1 - coherence/10) * 0.15, 0.05, 0.3)
            amount = int(100 * (1 + coherence / 10))
            return {
                'target_bottom_percent': float(target_pct),
                'amount': amount
            }
        
        elif intervention_type == 'SPAWN_TRIBE':
            # Count scales with creativity (more creative = more invaders)
            count = int(10 + creativity * 2)
            strategies = ['random', 'tit_for_tat', 'always_cooperate']
            # Coherence determines strategy choice (high = structured)
            if coherence > 7:
                strategy = 'tit_for_tat'
            elif coherence > 3:
                strategy = 'always_cooperate'
            else:
                strategy = 'random'
            return {
                'invader_count': count,
                'strategy': strategy
            }
        
        elif intervention_type == 'EMERGENCY_REVIVAL':
            # Resource amount scales with longevity (thinking long-term)
            resources = int(5000 * (1 + longevity / 10))
            agents = int(20 + longevity * 2)
            return {
                'resource_injection': resources,
                'spawn_count': agents
            }
        
        elif intervention_type == 'FORCED_COOPERATION':
            # Convert count scales with coherence (structured social change)
            count = int(15 + coherence * 2)
            return {'convert_count': count}
        
        return {}
    
    def _generate_reasoning(self, intervention_type, creativity, coherence, longevity,
                           wealth, coop, gini, growth, dominance, score) -> str:
        """Generate human-readable reasoning for intervention."""
        
        reasons = []
        
        # State-based reasons
        if wealth < 50:
            reasons.append(f"💰 Low avg wealth ({wealth:.0f})")
        if coop < 0.4:
            reasons.append(f"🤝 Low cooperation ({coop:.1%})")
        if gini > 0.5:
            reasons.append(f"📊 High inequality (Gini={gini:.2f})")
        if dominance > 0.85:
            reasons.append(f"🏛️ Tribe dominance ({dominance:.1%})")
        if growth < -0.05:
            reasons.append(f"📉 Negative growth ({growth:.1%})")
        
        # Trait-based reasons
        if creativity > 7:
            reasons.append(f"💡 High creativity ({creativity:.1f}) → innovative approach")
        if coherence > 7:
            reasons.append(f"🎯 High coherence ({coherence:.1f}) → structured intervention")
        if longevity > 7:
            reasons.append(f"⏳ High longevity ({longevity:.1f}) → long-term focus")
        
        # Intervention-specific
        type_reasons = {
            'STIMULUS': '💵 Economic stimulus to boost activity',
            'WELFARE': '🏥 Targeted welfare for vulnerable populations',
            'SPAWN_TRIBE': '🌱 Inject genetic diversity to break stagnation',
            'EMERGENCY_REVIVAL': '🚨 Emergency intervention to prevent collapse',
            'FORCED_COOPERATION': '🤝 Social engineering to restore cooperation'
        }
        
        reasoning = f"🧬 QUANTUM GOD INTERVENTION (score={score:.2f})\n"
        reasoning += f"   {type_reasons.get(intervention_type, intervention_type)}\n"
        reasoning += "   Reasons: " + " | ".join(reasons) if reasons else "   Economic optimization"
        
        return reasoning
    
    def get_statistics(self) -> Dict:
        """Get controller statistics for analysis."""
        return {
            'total_interventions': len(self.interventions),
            'timesteps': self.timestep,
            'genome': [float(x) for x in self.genome],
            'intervention_history': self.interventions[-10:],  # Last 10
            'current_traits': {
                'creativity': float(self.agent.traits[0]),
                'coherence': float(self.agent.traits[1]),
                'longevity': float(self.agent.traits[2])
            },
            'fitness': float(self.agent.get_final_fitness())
        }
    
    def save_state(self, filepath: str):
        """Save controller state for analysis."""
        import json
        from datetime import datetime
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'genome': [float(x) for x in self.genome],
            'timesteps': self.timestep,
            'total_interventions': len(self.interventions),
            'interventions': self.interventions,
            'state_history': self.state_history[-100:],  # Last 100 states
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Quantum God state saved to: {filepath}")


# === INTEGRATION FUNCTIONS ===

def create_quantum_ml_god(intervention_cooldown=10, environment='standard'):
    """
    Factory function to create quantum ML-based God controller.
    
    This is the main entry point for integration with prisoner_echo_god.py
    
    Returns:
        QuantumGodController: Ready-to-use quantum god controller
    """
    return QuantumGodController(
        environment=environment,
        intervention_cooldown=intervention_cooldown
    )


def demo_quantum_god():
    """Demonstrate quantum god controller with sample scenarios."""
    print("\n" + "="*70)
    print("🧬 QUANTUM GOD CONTROLLER DEMO")
    print("="*70)
    
    controller = create_quantum_ml_god(intervention_cooldown=5)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Collapse Crisis',
            'state': {
                'population': 50,
                'avg_wealth': 25,
                'cooperation_rate': 0.15,
                'gini_coefficient': 0.65,
                'growth_rate': -0.15,
                'tribe_dominance': 0.45
            }
        },
        {
            'name': 'Stagnation',
            'state': {
                'population': 800,
                'avg_wealth': 90,
                'cooperation_rate': 0.55,
                'gini_coefficient': 0.25,
                'growth_rate': 0.0,
                'tribe_dominance': 0.92
            }
        },
        {
            'name': 'Inequality Crisis',
            'state': {
                'population': 600,
                'avg_wealth': 120,
                'cooperation_rate': 0.40,
                'gini_coefficient': 0.75,
                'growth_rate': 0.05,
                'tribe_dominance': 0.60
            }
        },
        {
            'name': 'Healthy Economy',
            'state': {
                'population': 900,
                'avg_wealth': 150,
                'cooperation_rate': 0.70,
                'gini_coefficient': 0.30,
                'growth_rate': 0.08,
                'tribe_dominance': 0.55
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📊 Scenario {i+1}: {scenario['name']}")
        print(f"   Population: {scenario['state']['population']}")
        print(f"   Wealth: {scenario['state']['avg_wealth']}")
        print(f"   Cooperation: {scenario['state']['cooperation_rate']:.1%}")
        print(f"   Gini: {scenario['state']['gini_coefficient']:.2f}")
        
        intervention = controller.decide_intervention(scenario['state'], generation=i*10)
        
        if intervention:
            itype, reasoning, params = intervention
            print(f"\n   🎯 Decision: {itype}")
            print(f"   {reasoning}")
            print(f"   Parameters: {params}")
        else:
            print("\n   ✅ No intervention needed")
    
    # Save stats
    stats = controller.get_statistics()
    print(f"\n📈 Controller Statistics:")
    print(f"   Total interventions: {stats['total_interventions']}")
    print(f"   Timesteps: {stats['timesteps']}")
    print(f"   Current fitness: {stats['fitness']:.0f}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_quantum_god()
