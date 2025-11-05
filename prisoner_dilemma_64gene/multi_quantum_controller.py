"""
Multi-Quantum Controller Framework
====================================
Orchestrates multiple specialized Quantum ML genomes for different phases/conditions.

This framework is designed to be reusable for:
- Prisoner's Dilemma simulations
- Stock trading strategies
- Crypto trading strategies
- Any multi-phase decision-making system

Architecture:
- Multiple specialist genomes trained for different conditions
- Meta-controller that selects which genome to use
- Performance tracking and adaptive learning
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math


@dataclass
class QuantumGenome:
    """Represents a single Quantum ML genome with metadata."""
    name: str
    genome: List[float]
    trained_for: str  # Description of what this genome was optimized for
    optimal_phase: str  # e.g., "early" (0-60), "mid" (60-120), "late" (120+)
    performance_history: List[float] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
    
    def to_dict(self) -> dict:
        return asdict(self)


class MetaController:
    """
    Meta-controller that orchestrates multiple Quantum ML genomes.
    
    Selection Strategies:
    - phase_based: Select based on simulation generation/phase
    - performance_based: Select based on recent performance
    - adaptive: Learn which genome works best in current conditions
    - round_robin: Cycle through genomes for exploration
    """
    
    def __init__(self, genomes: List[QuantumGenome], strategy: str = "phase_based"):
        self.genomes = genomes
        self.strategy = strategy
        self.selection_history: List[Tuple[int, str, float]] = []  # (gen, genome_name, score)
        self.current_genome_index = 0
        
    def select_genome(self, generation: int, max_generations: int, 
                     current_metrics: Optional[Dict] = None) -> QuantumGenome:
        """
        Select which genome to use based on current conditions.
        
        Args:
            generation: Current generation/timestep
            max_generations: Total generations in simulation
            current_metrics: Optional dict with metrics like cooperation_rate, avg_wealth, etc.
        
        Returns:
            Selected QuantumGenome
        """
        if self.strategy == "phase_based":
            return self._select_by_phase(generation, max_generations)
        elif self.strategy == "performance_based":
            return self._select_by_performance()
        elif self.strategy == "adaptive":
            return self._select_adaptive(generation, max_generations, current_metrics)
        elif self.strategy == "round_robin":
            return self._select_round_robin()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _select_by_phase(self, generation: int, max_generations: int) -> QuantumGenome:
        """Select genome based on simulation phase."""
        progress = generation / max_generations
        
        if progress < 0.33:  # Early phase (0-33%)
            candidates = [g for g in self.genomes if g.optimal_phase == "early"]
        elif progress < 0.67:  # Mid phase (33-67%)
            candidates = [g for g in self.genomes if g.optimal_phase == "mid"]
        else:  # Late phase (67-100%)
            candidates = [g for g in self.genomes if g.optimal_phase == "late"]
        
        # If no genome for this phase, fall back to the first one
        return candidates[0] if candidates else self.genomes[0]
    
    def _select_by_performance(self) -> QuantumGenome:
        """Select genome with best recent performance."""
        if not any(g.performance_history for g in self.genomes):
            return self.genomes[0]
        
        # Calculate average of last 3 performances
        avg_performances = []
        for genome in self.genomes:
            if genome.performance_history:
                recent = genome.performance_history[-3:]
                avg_performances.append(np.mean(recent))
            else:
                avg_performances.append(0)
        
        best_idx = np.argmax(avg_performances)
        return self.genomes[best_idx]
    
    def _select_adaptive(self, generation: int, max_generations: int, 
                        current_metrics: Optional[Dict]) -> QuantumGenome:
        """
        Adaptive selection using contextual bandits approach.
        Combines phase-based prior with performance feedback.
        """
        # Start with phase-based selection
        phase_genome = self._select_by_phase(generation, max_generations)
        
        # If we have enough performance data, consider switching
        if len(self.selection_history) > 10:
            # Calculate success rate for each genome in similar phases
            phase_progress = generation / max_generations
            genome_scores = {g.name: [] for g in self.genomes}
            
            for hist_gen, hist_name, hist_score in self.selection_history[-20:]:
                hist_progress = hist_gen / max_generations
                # Only consider similar phases (within 0.2 progress)
                if abs(hist_progress - phase_progress) < 0.2:
                    genome_scores[hist_name].append(hist_score)
            
            # If another genome has significantly better performance, switch
            for genome in self.genomes:
                if genome.name != phase_genome.name and genome_scores[genome.name]:
                    avg_score = np.mean(genome_scores[genome.name])
                    phase_avg = np.mean(genome_scores.get(phase_genome.name, [0]))
                    if avg_score > phase_avg * 1.1:  # 10% better
                        return genome
        
        return phase_genome
    
    def _select_round_robin(self) -> QuantumGenome:
        """Cycle through genomes sequentially."""
        genome = self.genomes[self.current_genome_index]
        self.current_genome_index = (self.current_genome_index + 1) % len(self.genomes)
        return genome
    
    def update_performance(self, genome_name: str, generation: int, score: float):
        """Update performance tracking for a genome."""
        self.selection_history.append((generation, genome_name, score))
        for genome in self.genomes:
            if genome.name == genome_name:
                genome.performance_history.append(score)
                break
    
    def get_statistics(self) -> Dict:
        """Get statistics about genome usage and performance."""
        stats = {
            "total_selections": len(self.selection_history),
            "genome_usage": {},
            "genome_performance": {}
        }
        
        for genome in self.genomes:
            name = genome.name
            usage_count = sum(1 for _, n, _ in self.selection_history if n == name)
            stats["genome_usage"][name] = usage_count
            
            if genome.performance_history:
                stats["genome_performance"][name] = {
                    "mean": float(np.mean(genome.performance_history)),
                    "std": float(np.std(genome.performance_history)),
                    "min": float(np.min(genome.performance_history)),
                    "max": float(np.max(genome.performance_history)),
                    "count": len(genome.performance_history)
                }
        
        return stats
    
    def save_state(self, filepath: str):
        """Save controller state to JSON."""
        state = {
            "genomes": [g.to_dict() for g in self.genomes],
            "strategy": self.strategy,
            "selection_history": self.selection_history,
            "statistics": self.get_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Multi-Quantum controller state saved to {filepath}")


class MultiQuantumGodController:
    """
    God controller that uses multiple Quantum ML genomes.
    Integrates with the prisoner's dilemma simulation.
    """
    
    def __init__(self, meta_controller: MetaController, max_generations: int):
        self.meta_controller = meta_controller
        self.max_generations = max_generations
        self.current_genome: Optional[QuantumGenome] = None
        self.intervention_count = 0
        self.last_switch_generation = 0
        
    def should_intervene(self, population, generation: int) -> Tuple[bool, str, float]:
        """
        Decide if God should intervene based on current genome's parameters.
        
        Returns:
            (should_intervene, action_type, action_amount)
        """
        # Select appropriate genome for current phase
        current_metrics = self._calculate_metrics(population)
        self.current_genome = self.meta_controller.select_genome(
            generation, self.max_generations, current_metrics
        )
        
        # Track if we switched genomes
        if generation > self.last_switch_generation:
            self.last_switch_generation = generation
        
        # Extract genome parameters (assuming 8-parameter format)
        genome = self.current_genome.genome
        intervention_threshold = genome[0]
        intervention_probability = genome[1]
        welfare_amount = genome[2]
        stimulus_amount = genome[3]
        welfare_targeting = genome[4]
        stimulus_targeting = genome[5]
        intervention_timing = genome[6]
        intervention_cooldown = genome[7]
        
        # Calculate intervention score
        cooperation_rate = current_metrics.get('cooperation_rate', 0.5)
        avg_wealth = current_metrics.get('avg_wealth', 100)
        population_size = current_metrics.get('population_size', 100)
        
        # Normalized score (0-10 scale)
        intervention_score = (
            (1 - cooperation_rate) * 5 +  # Low cooperation triggers intervention
            (1 - min(avg_wealth / 200, 1)) * 3 +  # Low wealth triggers intervention
            (1 - min(population_size / 150, 1)) * 2  # Small population triggers intervention
        )
        
        # Check cooldown (simplified - would need to track last intervention)
        should_intervene = (
            intervention_score > intervention_threshold and
            np.random.random() < intervention_probability
        )
        
        if should_intervene:
            self.intervention_count += 1
            # Choose action type based on what's most needed
            if cooperation_rate < 0.5:
                return True, "stimulus", stimulus_amount
            elif avg_wealth < 100:
                return True, "welfare", welfare_amount
            else:
                return True, "stimulus", stimulus_amount
        
        return False, "none", 0.0
    
    def _calculate_metrics(self, population) -> Dict:
        """Calculate current population metrics."""
        if not hasattr(population, 'agents') or not population.agents:
            return {
                'cooperation_rate': 0.5,
                'avg_wealth': 100,
                'population_size': 100
            }
        
        agents = population.agents
        total_agents = len(agents)
        
        if total_agents == 0:
            return {
                'cooperation_rate': 0.5,
                'avg_wealth': 100,
                'population_size': 0
            }
        
        # Calculate cooperation rate
        cooperators = sum(1 for a in agents if hasattr(a, 'strategy') and a.strategy == 'C')
        cooperation_rate = cooperators / total_agents if total_agents > 0 else 0.5
        
        # Calculate average wealth
        avg_wealth = np.mean([a.wealth for a in agents if hasattr(a, 'wealth')])
        
        return {
            'cooperation_rate': cooperation_rate,
            'avg_wealth': avg_wealth,
            'population_size': total_agents
        }
    
    def get_report(self) -> Dict:
        """Generate performance report."""
        return {
            "total_interventions": self.intervention_count,
            "current_genome": self.current_genome.name if self.current_genome else None,
            "meta_statistics": self.meta_controller.get_statistics()
        }


def create_default_ensemble() -> List[QuantumGenome]:
    """
    Create default ensemble of Quantum ML genomes for prisoner's dilemma.
    
    Returns:
        List of QuantumGenome objects representing different specialists
    """
    genomes = []
    
    # Early-game specialist (0-60 generations)
    # Our champion 50-gen genome: minimal intervention, magic 2π stimulus
    genomes.append(QuantumGenome(
        name="EarlyGame_Specialist",
        genome=[5.0, 0.1, 0.0001, 6.283185307179586, 0.6, 0.3, 0.7, 10.0],
        trained_for="50-generation scenarios, early optimization",
        optimal_phase="early"
    ))
    
    # Mid-game specialist (60-120 generations)
    # Balanced approach with moderate intervention
    genomes.append(QuantumGenome(
        name="MidGame_Balanced",
        genome=[2.5, 0.15, 0.01, 10.0, 0.7, 0.5, 0.6, 12.0],
        trained_for="Mid-phase stability and growth",
        optimal_phase="mid"
    ))
    
    # Late-game specialist (120+ generations)
    # More aggressive to maintain cooperation in mature populations
    genomes.append(QuantumGenome(
        name="LateGame_Stabilizer",
        genome=[1.5, 0.2, 0.1, 15.0, 0.8, 0.6, 0.5, 15.0],
        trained_for="Long-term cooperation maintenance",
        optimal_phase="late"
    ))
    
    # Crisis manager (can be used at any phase when metrics are bad)
    genomes.append(QuantumGenome(
        name="Crisis_Manager",
        genome=[1.0, 0.3, 1.0, 20.0, 0.9, 0.7, 0.4, 8.0],
        trained_for="Emergency intervention for failing populations",
        optimal_phase="early"  # Default to early but can be selected adaptively
    ))
    
    return genomes


def load_custom_genomes(filepath: str) -> List[QuantumGenome]:
    """Load custom genomes from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    genomes = []
    for g in data['genomes']:
        genome = QuantumGenome(
            name=g['name'],
            genome=g['genome'],
            trained_for=g['trained_for'],
            optimal_phase=g['optimal_phase'],
            performance_history=g.get('performance_history', [])
        )
        genomes.append(genome)
    
    return genomes


# Example usage for future trading applications
def create_trading_ensemble() -> List[QuantumGenome]:
    """
    Example ensemble for trading applications (stocks/crypto).
    This is a template - would need to be trained for actual trading.
    """
    genomes = []
    
    # Bull market specialist
    genomes.append(QuantumGenome(
        name="Bull_Market_Specialist",
        genome=[0.5, 0.8, 100, 500, 0.7, 0.6, 0.9, 5],
        trained_for="Bull market conditions, high volume",
        optimal_phase="early"
    ))
    
    # Bear market specialist
    genomes.append(QuantumGenome(
        name="Bear_Market_Specialist",
        genome=[2.0, 0.3, 50, 200, 0.5, 0.4, 0.3, 10],
        trained_for="Bear market conditions, capital preservation",
        optimal_phase="mid"
    ))
    
    # Sideways market specialist
    genomes.append(QuantumGenome(
        name="Sideways_Market_Specialist",
        genome=[1.5, 0.5, 75, 300, 0.6, 0.5, 0.5, 8],
        trained_for="Ranging markets, mean reversion",
        optimal_phase="late"
    ))
    
    # Volatility specialist
    genomes.append(QuantumGenome(
        name="High_Volatility_Specialist",
        genome=[0.8, 0.6, 150, 400, 0.8, 0.7, 0.7, 6],
        trained_for="High volatility conditions",
        optimal_phase="early"
    ))
    
    return genomes


if __name__ == "__main__":
    # Demonstration
    print("Multi-Quantum Controller Framework")
    print("=" * 50)
    
    # Create ensemble
    ensemble = create_default_ensemble()
    print(f"\nCreated ensemble with {len(ensemble)} specialized genomes:")
    for g in ensemble:
        print(f"  - {g.name}: {g.trained_for}")
    
    # Create meta-controller
    meta = MetaController(ensemble, strategy="phase_based")
    print(f"\nMeta-controller initialized with strategy: {meta.strategy}")
    
    # Simulate genome selection across generations
    print("\nSimulating genome selection across 150 generations:")
    max_gens = 150
    for gen in [0, 30, 60, 90, 120, 150]:
        selected = meta.select_genome(gen, max_gens)
        print(f"  Gen {gen:3d}: {selected.name}")
    
    print("\n✅ Framework ready for testing!")
    print("   Use test_multi_quantum_ensemble.py to run experiments")
