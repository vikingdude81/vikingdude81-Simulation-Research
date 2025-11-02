"""
🧠👁️ ECHO MODEL WITH GOD-AI CONTROLLER

Revolutionary enhancement: AI oversight layer that monitors and intervenes
in the simulation to test governance policies, stabilize collapse, and explore
"digital twin" scenarios for policy design.

This implements three "God" modes:
1. RULE-BASED: Simple if/then logic (baseline)
2. ML-BASED: Learns optimal interventions from historical data
3. API-BASED: Uses external LLM (GPT-4/Claude) for governance decisions

Based on Holland's Echo model + Ultimate variant with external shocks.
Now with a meta-controller that can reshape the world itself.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import json
from datetime import datetime
import time
import os
from colorama import init, Fore, Style
from collections import Counter
from dataclasses import dataclass, asdict
from enum import Enum

init(autoreset=True)

# --- CONFIGURATION (inherited from Ultimate) ---
TAG_LENGTH = 8
STRATEGY_LENGTH = 64
CHROMOSOME_LENGTH = TAG_LENGTH + STRATEGY_LENGTH
MATCH_THRESHOLD = 2

GRID_WIDTH = 50
GRID_HEIGHT = 50
NEIGHBORHOOD_RADIUS = 1

INITIAL_RESOURCES = 100
REPRODUCTION_THRESHOLD = 200
DEATH_THRESHOLD = 0
METABOLISM_COST = 1
MAX_POPULATION = 1000

# External shock parameters (still active alongside God)
DROUGHT_CHANCE = 0.05
DROUGHT_PENALTY = 50
DISASTER_CHANCE = 0.02
DISASTER_RADIUS = 5
DISASTER_KILL_CHANCE = 0.5
PREDATOR_CHANCE = 0.03
PREDATOR_COUNT = 3
PREDATOR_KILL_CHANCE = 0.3

# --- GOD-AI CONFIGURATION ---
GOD_MODE = "RULE_BASED"  # Options: RULE_BASED, ML_BASED, API_BASED, DISABLED

# Intervention thresholds (for rule-based)
STAGNATION_THRESHOLD = 0.90  # If one tribe > 90%, spawn invaders
LOW_WEALTH_THRESHOLD = 50    # Avg wealth below this → stimulus
POVERTY_THRESHOLD = 0.10     # Bottom 10% get targeted welfare
COLLAPSE_THRESHOLD = 0.05    # If pop < 5% of max → emergency intervention
INEQUALITY_THRESHOLD = 10    # Gini coefficient analog (max/min wealth ratio)

# Intervention parameters
STIMULUS_AMOUNT = 50         # Universal Basic Income per agent
WELFARE_AMOUNT = 100         # Targeted welfare for poor
INVADER_COUNT = 15           # Number of new agents when spawning tribe
RESOURCE_INJECTION = 5000    # Total resources to inject during collapse

# God intervention frequency
GOD_INTERVENTION_COOLDOWN = 10  # Wait N generations between interventions

# --- GENETIC FUNCTIONS (same as Ultimate) ---

def create_random_chromosome() -> str:
    tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "".join(random.choice(['C', 'D']) for _ in range(STRATEGY_LENGTH))
    return tag + strategy

def create_tit_for_tat_with_tag(tag: str | None = None) -> str:
    if tag is None:
        tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "CD" * 32
    return tag + strategy

def create_always_cooperate_with_tag(tag: str | None = None) -> str:
    if tag is None:
        tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "C" * 64
    return tag + strategy

def crossover(parent1: str, parent2: str) -> str:
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

def mutate(chromosome: str, rate: float = 0.01) -> str:
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < rate:
            if i < TAG_LENGTH:
                chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
            else:
                chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
    return "".join(chrom_list)

def hamming_distance(tag1: str, tag2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(tag1, tag2))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- INTERVENTION TYPES ---

class InterventionType(Enum):
    """Types of God interventions."""
    NONE = "none"
    STIMULUS = "stimulus"              # Give everyone resources
    WELFARE = "welfare"                # Give poor agents resources
    SPAWN_TRIBE = "spawn_tribe"        # Introduce new genetic variant
    RESOURCE_INJECTION = "resource_injection"  # Add resources to system
    DISASTER_PREVENTION = "disaster_prevention"  # Block external shocks
    FORCED_COOPERATION = "forced_cooperation"   # Change defector genes
    EMERGENCY_REVIVAL = "emergency_revival"     # Save from extinction

@dataclass
class InterventionRecord:
    """Record of a single God intervention."""
    generation: int
    intervention_type: InterventionType
    reason: str
    parameters: Dict[str, Any]
    before_state: Dict[str, float]
    after_state: Dict[str, float]
    effectiveness: Optional[float] = None  # Calculated later
    
    def to_dict(self):
        d = asdict(self)
        d['intervention_type'] = self.intervention_type.value
        return d

# --- GOD-AI CONTROLLER ---

class GodController:
    """
    The "Meta-Agent" that monitors and intervenes in the simulation.
    
    This is the revolutionary part: a controller that sits above the agents
    and can reshape the world based on global observations.
    
    Three modes:
    1. RULE_BASED: Simple if/then logic (what we're implementing now)
    2. ML_BASED: Learns optimal interventions (future)
    3. API_BASED: Uses external LLM for decisions (future)
    """
    
    def __init__(self, mode: str = "RULE_BASED"):
        self.mode = mode
        self.intervention_history: List[InterventionRecord] = []
        self.last_intervention_generation = -GOD_INTERVENTION_COOLDOWN
        self.total_interventions = 0
        
        # Statistics tracking
        self.interventions_by_type = {itype: 0 for itype in InterventionType}
        
    def should_intervene(self, generation: int) -> bool:
        """Check cooldown period."""
        return generation - self.last_intervention_generation >= GOD_INTERVENTION_COOLDOWN
    
    def capture_state(self, population: 'GodEchoPopulation') -> Dict[str, float]:
        """Capture current world state for analysis."""
        if not population.agents:
            return {
                'population': 0,
                'avg_wealth': 0,
                'total_wealth': 0,
                'cooperation_rate': 0,
                'clustering': 0,
                'tribe_diversity': 0,
                'max_tribe_dominance': 0,
                'wealth_inequality': 0,
                'avg_age': 0
            }
        
        resources = [a.resources for a in population.agents]
        ages = [a.age for a in population.agents]
        
        # Tag diversity (number of unique tags)
        unique_tags = len(set(a.tag for a in population.agents))
        tribe_diversity = unique_tags / len(population.agents)
        
        # Dominant tribe percentage
        tag_counts = Counter(a.tag for a in population.agents)
        max_tribe_count = tag_counts.most_common(1)[0][1]
        max_tribe_dominance = max_tribe_count / len(population.agents)
        
        # Wealth inequality (simple: max/min ratio)
        min_wealth = min(resources)
        max_wealth = max(resources)
        wealth_inequality = max_wealth / max(min_wealth, 1)  # Avoid division by zero
        
        # Cooperation rate
        total_actions = sum(a.cooperations + a.defections for a in population.agents)
        total_coop = sum(a.cooperations for a in population.agents)
        coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
        
        return {
            'population': len(population.agents),
            'avg_wealth': np.mean(resources),
            'total_wealth': sum(resources),
            'cooperation_rate': coop_rate,
            'clustering': population.calculate_clustering(),
            'tribe_diversity': tribe_diversity,
            'max_tribe_dominance': max_tribe_dominance,
            'wealth_inequality': wealth_inequality,
            'avg_age': np.mean(ages)
        }
    
    def decide_intervention(self, population: 'GodEchoPopulation', generation: int) -> Optional[Tuple[InterventionType, str, Dict]]:
        """
        Core decision logic: What intervention (if any) should God make?
        
        Returns: (intervention_type, reason, parameters) or None
        """
        if not self.should_intervene(generation):
            return None
        
        state = self.capture_state(population)
        
        if self.mode == "RULE_BASED":
            return self._rule_based_decision(state, population)
        elif self.mode == "ML_BASED":
            return self._ml_based_decision(state, population)
        elif self.mode == "API_BASED":
            return self._api_based_decision(state, population)
        else:
            return None
    
    def _rule_based_decision(self, state: Dict, population: 'GodEchoPopulation') -> Optional[Tuple[InterventionType, str, Dict]]:
        """Simple if/then logic for interventions."""
        
        # PRIORITY 1: Emergency - Prevent extinction
        if state['population'] < MAX_POPULATION * COLLAPSE_THRESHOLD:
            return (
                InterventionType.EMERGENCY_REVIVAL,
                f"Emergency! Population critical ({state['population']} agents)",
                {'resource_boost': RESOURCE_INJECTION, 'spawn_count': 50}
            )
        
        # PRIORITY 2: Combat stagnation - One tribe dominating
        if state['max_tribe_dominance'] > STAGNATION_THRESHOLD:
            return (
                InterventionType.SPAWN_TRIBE,
                f"Stagnation detected! One tribe controls {state['max_tribe_dominance']*100:.1f}%",
                {'invader_count': INVADER_COUNT, 'strategy': 'random'}
            )
        
        # PRIORITY 3: Economic collapse - Low average wealth
        if state['avg_wealth'] < LOW_WEALTH_THRESHOLD:
            return (
                InterventionType.STIMULUS,
                f"Economic crisis! Avg wealth = {state['avg_wealth']:.1f}",
                {'amount_per_agent': STIMULUS_AMOUNT}
            )
        
        # PRIORITY 4: Extreme inequality
        if state['wealth_inequality'] > INEQUALITY_THRESHOLD:
            return (
                InterventionType.WELFARE,
                f"Extreme inequality! Wealth ratio = {state['wealth_inequality']:.1f}:1",
                {'target_bottom_percent': POVERTY_THRESHOLD, 'amount': WELFARE_AMOUNT}
            )
        
        # PRIORITY 5: Too much defection (optional - more interventionist)
        # Uncomment if you want God to enforce cooperation
        # if state['cooperation_rate'] < 0.3:
        #     return (
        #         InterventionType.FORCED_COOPERATION,
        #         f"Cooperation crisis! Only {state['cooperation_rate']*100:.1f}% cooperating",
        #         {'convert_count': 20}
        #     )
        
        return None
    
    def _ml_based_decision(self, state: Dict, population: 'GodEchoPopulation') -> Optional[Tuple[InterventionType, str, Dict]]:
        """Use ML model to predict best intervention (FUTURE)."""
        # TODO: Implement reinforcement learning agent
        # Input: state vector [population, avg_wealth, cooperation, etc.]
        # Output: [intervention_type, parameters]
        return None
    
    def _api_based_decision(self, state: Dict, population: 'GodEchoPopulation') -> Optional[Tuple[InterventionType, str, Dict]]:
        """Call external LLM API for governance decision (FUTURE)."""
        # TODO: Serialize state, call GPT-4/Claude, parse response
        return None
    
    def execute_intervention(self, population: 'GodEchoPopulation', 
                           intervention_type: InterventionType, 
                           parameters: Dict) -> str:
        """Execute the chosen intervention and return description."""
        
        if intervention_type == InterventionType.STIMULUS:
            return self._execute_stimulus(population, parameters)
        elif intervention_type == InterventionType.WELFARE:
            return self._execute_welfare(population, parameters)
        elif intervention_type == InterventionType.SPAWN_TRIBE:
            return self._execute_spawn_tribe(population, parameters)
        elif intervention_type == InterventionType.EMERGENCY_REVIVAL:
            return self._execute_emergency_revival(population, parameters)
        elif intervention_type == InterventionType.FORCED_COOPERATION:
            return self._execute_forced_cooperation(population, parameters)
        else:
            return "Unknown intervention type"
    
    def _execute_stimulus(self, population: 'GodEchoPopulation', params: Dict) -> str:
        """Universal Basic Income - give everyone resources."""
        amount = params['amount_per_agent']
        for agent in population.agents:
            agent.resources += amount
        
        total_injected = amount * len(population.agents)
        return f"💰 STIMULUS: Gave {amount} resources to all {len(population.agents)} agents (Total: {total_injected})"
    
    def _execute_welfare(self, population: 'GodEchoPopulation', params: Dict) -> str:
        """Targeted welfare for poorest agents."""
        target_percent = params['target_bottom_percent']
        amount = params['amount']
        
        # Sort by wealth
        sorted_agents = sorted(population.agents, key=lambda a: a.resources)
        poorest_count = max(1, int(len(sorted_agents) * target_percent))
        poorest = sorted_agents[:poorest_count]
        
        for agent in poorest:
            agent.resources += amount
        
        total_given = amount * poorest_count
        return f"🏥 WELFARE: Gave {amount} to poorest {poorest_count} agents (Total: {total_given})"
    
    def _execute_spawn_tribe(self, population: 'GodEchoPopulation', params: Dict) -> str:
        """Spawn new tribe with different genetics."""
        invader_count = params['invader_count']
        strategy = params.get('strategy', 'random')
        
        # Create new unique tag
        new_tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
        
        spawned = 0
        for _ in range(invader_count):
            # Find empty position
            position = population.find_random_empty_position()
            if position is None:
                break
            
            # Create invader with new tag
            if strategy == 'random':
                chromosome = new_tag + "".join(random.choice(['C', 'D']) for _ in range(STRATEGY_LENGTH))
            elif strategy == 'cooperate':
                chromosome = new_tag + ("C" * STRATEGY_LENGTH)
            else:  # tit-for-tat
                chromosome = create_tit_for_tat_with_tag(new_tag)
            
            agent = GodEchoAgent(population.next_id, chromosome, position)
            population.next_id += 1
            
            x, y = position
            population.grid[y][x] = agent
            population.agents.append(agent)
            spawned += 1
        
        return f"🌟 SPAWN TRIBE: Created {spawned} invaders with tag '{new_tag}' ({strategy} strategy)"
    
    def _execute_emergency_revival(self, population: 'GodEchoPopulation', params: Dict) -> str:
        """Emergency intervention to prevent extinction."""
        resource_boost = params['resource_boost']
        spawn_count = params['spawn_count']
        
        # Give existing agents massive resource boost
        for agent in population.agents:
            agent.resources += resource_boost / len(population.agents)
        
        # Spawn new diverse population
        spawned = 0
        for _ in range(spawn_count):
            position = population.find_random_empty_position()
            if position is None:
                break
            
            chromosome = create_random_chromosome()
            agent = GodEchoAgent(population.next_id, chromosome, position)
            agent.resources = INITIAL_RESOURCES * 2  # Give them advantage
            population.next_id += 1
            
            x, y = position
            population.grid[y][x] = agent
            population.agents.append(agent)
            spawned += 1
        
        return f"🚨 EMERGENCY REVIVAL: Boosted {len(population.agents)} agents + spawned {spawned} new agents"
    
    def _execute_forced_cooperation(self, population: 'GodEchoPopulation', params: Dict) -> str:
        """Force some defectors to become cooperators (interventionist!)."""
        convert_count = params['convert_count']
        
        # Find most defecting agents
        sorted_by_defection = sorted(population.agents, 
                                     key=lambda a: a.defections / max(a.cooperations + a.defections, 1),
                                     reverse=True)
        
        to_convert = sorted_by_defection[:convert_count]
        
        for agent in to_convert:
            # Replace their strategy with Tit-for-Tat
            agent.chromosome = agent.tag + ("CD" * 32)
            agent.strategy_genes = "CD" * 32
        
        return f"⚖️ FORCED COOPERATION: Converted {len(to_convert)} defectors to Tit-for-Tat"
    
    def record_intervention(self, generation: int, intervention_type: InterventionType,
                          reason: str, parameters: Dict,
                          before_state: Dict, after_state: Dict):
        """Record intervention for later analysis."""
        record = InterventionRecord(
            generation=generation,
            intervention_type=intervention_type,
            reason=reason,
            parameters=parameters,
            before_state=before_state,
            after_state=after_state
        )
        
        self.intervention_history.append(record)
        self.interventions_by_type[intervention_type] += 1
        self.total_interventions += 1
        self.last_intervention_generation = generation
    
    def get_summary_stats(self) -> Dict:
        """Get summary of God's intervention history."""
        if not self.intervention_history:
            return {'total_interventions': 0}
        
        return {
            'total_interventions': self.total_interventions,
            'interventions_by_type': {k.value: v for k, v in self.interventions_by_type.items() if v > 0},
            'avg_effectiveness': np.mean([r.effectiveness for r in self.intervention_history if r.effectiveness is not None]),
            'mode': self.mode
        }

# --- AGENT CLASS (same as Ultimate but renamed) ---

class GodEchoAgent:
    """Agent in the God-controlled Echo simulation."""
    
    MOVE_MAP = {
        ('C', 'C'): 0, ('C', 'D'): 1,
        ('D', 'C'): 2, ('D', 'D'): 3
    }
    
    RESOURCE_PAYOFFS = {
        ('C', 'C'): (20, 20),
        ('D', 'C'): (30, -10),
        ('C', 'D'): (-10, 30),
        ('D', 'D'): (5, 5),
    }
    
    LONER_SCORE = 5
    
    def __init__(self, agent_id: int, chromosome: str, position: Tuple[int, int]):
        self.id = agent_id
        self.chromosome = chromosome
        self.tag = chromosome[:TAG_LENGTH]
        self.strategy_genes = chromosome[TAG_LENGTH:]
        self.position = position
        self.resources = INITIAL_RESOURCES
        self.age = 0
        self.cooperations = 0
        self.defections = 0
        self.survived_droughts = 0
        self.survived_disasters = 0
        self.survived_predators = 0
        self.god_interventions_received = 0  # Track God's blessings
        
    def can_match(self, other: 'GodEchoAgent') -> bool:
        return hamming_distance(self.tag, other.tag) <= MATCH_THRESHOLD
    
    def get_move(self, history: List[str]) -> str:
        if not history:
            return self.strategy_genes[0]
        
        opponent_history = tuple(history[-min(len(history), 6):])
        if len(opponent_history) < 6:
            opponent_history = ('C',) * (6 - len(opponent_history)) + opponent_history
        
        index = int(sum(
            self.MOVE_MAP.get((opponent_history[i], opponent_history[i+1]), 0) * (4 ** (2-i))
            for i in range(0, 6, 2)
        ))
        
        return self.strategy_genes[min(index, STRATEGY_LENGTH - 1)]
    
    def interact(self, other: 'GodEchoAgent', rounds: int = 5) -> Tuple[int, int]:
        my_history = []
        other_history = []
        
        my_total = 0
        other_total = 0
        
        for _ in range(rounds):
            my_move = self.get_move(other_history)
            other_move = other.get_move(my_history)
            
            my_gain, other_gain = self.RESOURCE_PAYOFFS[(my_move, other_move)]
            my_total += my_gain
            other_total += other_gain
            
            my_history.append(my_move)
            other_history.append(other_move)
            
            if my_move == 'C':
                self.cooperations += 1
            else:
                self.defections += 1
            
            if other_move == 'C':
                other.cooperations += 1
            else:
                other.defections += 1
        
        return my_total, other_total
    
    def metabolize(self):
        self.resources -= METABOLISM_COST
        self.age += 1
    
    def is_alive(self) -> bool:
        return self.resources > DEATH_THRESHOLD
    
    def can_reproduce(self) -> bool:
        return self.resources >= REPRODUCTION_THRESHOLD

# --- EXTERNAL SHOCKS (unchanged from Ultimate) ---

class ExternalShock:
    """External shocks still exist alongside God interventions."""
    
    @staticmethod
    def apply_drought(agents: List[GodEchoAgent]) -> int:
        affected = 0
        for agent in agents:
            agent.resources -= DROUGHT_PENALTY
            agent.survived_droughts += 1
            affected += 1
        return affected
    
    @staticmethod
    def apply_disaster(agents: List[GodEchoAgent], grid: np.ndarray, 
                       center: Tuple[int, int]) -> Tuple[int, int]:
        cx, cy = center
        killed = 0
        affected = 0
        
        for agent in agents[:]:
            ax, ay = agent.position
            distance = max(abs(ax - cx), abs(ay - cy))
            
            if distance <= DISASTER_RADIUS:
                affected += 1
                if random.random() < DISASTER_KILL_CHANCE:
                    agent.resources = -1000
                    killed += 1
                else:
                    agent.survived_disasters += 1
        
        return killed, affected
    
    @staticmethod
    def apply_predators(agents: List[GodEchoAgent], grid: np.ndarray, 
                       predator_positions: List[Tuple[int, int]]) -> int:
        killed = 0
        
        for pred_x, pred_y in predator_positions:
            for agent in agents[:]:
                ax, ay = agent.position
                distance = max(abs(ax - pred_x), abs(ay - pred_y))
                
                if distance <= 1:
                    if random.random() < PREDATOR_KILL_CHANCE:
                        agent.resources = -1000
                        killed += 1
                    else:
                        agent.survived_predators += 1
        
        return killed

# --- GOD-CONTROLLED POPULATION ---

class GodEchoPopulation:
    """Echo population with God-AI oversight."""
    
    def __init__(self, initial_size: int = 100, god_mode: str = "RULE_BASED"):
        self.agents: List[GodEchoAgent] = []
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), None, dtype=object)
        self.rounds_per_interaction = 5
        
        # Initialize God Controller
        self.god = GodController(mode=god_mode)
        
        # Initialize population (same as Ultimate)
        positions_used = set()
        
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            
            while True:
                x = random.randint(0, GRID_WIDTH - 1)
                y = random.randint(0, GRID_HEIGHT - 1)
                if (x, y) not in positions_used:
                    break
            
            agent = GodEchoAgent(i, chromosome, (x, y))
            self.agents.append(agent)
            self.grid[y][x] = agent
            positions_used.add((x, y))
        
        # Add TFT cluster
        tft_tag = "11110000"
        for i in range(10):
            while True:
                x = random.randint(0, 10)
                y = random.randint(0, 10)
                if (x, y) not in positions_used:
                    break
            
            tft_agent = GodEchoAgent(len(self.agents), 
                                    create_tit_for_tat_with_tag(tft_tag),
                                    (x, y))
            self.agents.append(tft_agent)
            self.grid[y][x] = tft_agent
            positions_used.add((x, y))
        
        self.generation = 0
        self.next_id = len(self.agents)
        self.start_time = time.time()
        
        self.history = {
            'population': [],
            'resources': [],
            'cooperation': [],
            'births': [],
            'deaths': [],
            'clustering': [],
            'shocks': [],
            'god_interventions': []  # NEW: Track God's actions
        }
        
        self.shock_log = []
    
    def find_random_empty_position(self) -> Optional[Tuple[int, int]]:
        """Find any empty position on grid."""
        empty_positions = []
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] is None:
                    empty_positions.append((x, y))
        
        if empty_positions:
            return random.choice(empty_positions)
        return None
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[GodEchoAgent]:
        x, y = position
        neighbors = []
        
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                
                neighbor = self.grid[ny][nx]
                if neighbor is not None and neighbor.is_alive():
                    neighbors.append(neighbor)
        
        return neighbors
    
    def run_interactions(self):
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if not agent.is_alive():
                continue
            
            neighbors = self.get_neighbors(agent.position)
            
            if not neighbors:
                agent.resources += GodEchoAgent.LONER_SCORE
                continue
            
            matches = [n for n in neighbors if agent.can_match(n)]
            
            if matches:
                partner = random.choice(matches)
                agent_gain, partner_gain = agent.interact(partner, self.rounds_per_interaction)
                agent.resources += agent_gain
                partner.resources += partner_gain
            else:
                agent.resources += GodEchoAgent.LONER_SCORE
    
    def apply_external_shocks(self) -> Optional[str]:
        """Apply random external shocks (unchanged from Ultimate)."""
        shocks = []
        
        if random.random() < DROUGHT_CHANCE:
            affected = ExternalShock.apply_drought(self.agents)
            shock_msg = f"🌵 DROUGHT! All {affected} agents lose {DROUGHT_PENALTY} resources"
            shocks.append(shock_msg)
            self.shock_log.append(('drought', self.generation, affected))
        
        if random.random() < DISASTER_CHANCE:
            center = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            killed, affected = ExternalShock.apply_disaster(self.agents, self.grid, center)
            shock_msg = f"💥 DISASTER at ({center[0]},{center[1]})! {killed} killed, {affected} affected"
            shocks.append(shock_msg)
            self.shock_log.append(('disaster', self.generation, killed, affected, center))
        
        if random.random() < PREDATOR_CHANCE:
            predator_positions = [
                (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
                for _ in range(PREDATOR_COUNT)
            ]
            killed = ExternalShock.apply_predators(self.agents, self.grid, predator_positions)
            shock_msg = f"🦖 PREDATORS! {PREDATOR_COUNT} predators kill {killed} agents"
            shocks.append(shock_msg)
            self.shock_log.append(('predators', self.generation, killed, predator_positions))
        
        return " | ".join(shocks) if shocks else None
    
    def god_intervention_step(self) -> Optional[str]:
        """
        🧠 THE KEY NEW FUNCTION: God monitors and decides intervention.
        
        This is where the "Meta-Agent" makes decisions.
        """
        if self.god.mode == "DISABLED":
            return None
        
        # Capture state before intervention
        before_state = self.god.capture_state(self)
        
        # God decides if intervention is needed
        decision = self.god.decide_intervention(self, self.generation)
        
        if decision is None:
            return None
        
        intervention_type, reason, parameters = decision
        
        # Execute the intervention
        intervention_msg = self.god.execute_intervention(self, intervention_type, parameters)
        
        # Capture state after intervention
        after_state = self.god.capture_state(self)
        
        # Record the intervention
        self.god.record_intervention(
            self.generation,
            intervention_type,
            reason,
            parameters,
            before_state,
            after_state
        )
        
        return f"🧠 GOD: {intervention_msg} | Reason: {reason}"
    
    def metabolize_all(self):
        for agent in self.agents:
            agent.metabolize()
    
    def remove_dead(self) -> int:
        initial_count = len(self.agents)
        dead_agents = [a for a in self.agents if not a.is_alive()]
        
        for agent in dead_agents:
            x, y = agent.position
            self.grid[y][x] = None
        
        self.agents = [a for a in self.agents if a.is_alive()]
        return initial_count - len(self.agents)
    
    def find_empty_neighbor(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = position
        empty_spots = []
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                
                if self.grid[ny][nx] is None:
                    empty_spots.append((nx, ny))
        
        return random.choice(empty_spots) if empty_spots else None
    
    def handle_reproduction(self) -> int:
        if len(self.agents) >= MAX_POPULATION:
            return 0
        
        parents = [a for a in self.agents if a.can_reproduce()]
        if not parents:
            return 0
        
        new_agents = []
        
        for _ in range(min(len(parents), MAX_POPULATION - len(self.agents))):
            parent = random.choice(parents)
            
            position = self.find_empty_neighbor(parent.position)
            if position is None:
                continue
            
            parent.resources -= REPRODUCTION_THRESHOLD // 2
            
            child_chromosome = mutate(parent.chromosome)
            child = GodEchoAgent(self.next_id, child_chromosome, position)
            self.next_id += 1
            
            x, y = position
            self.grid[y][x] = child
            new_agents.append(child)
        
        self.agents.extend(new_agents)
        return len(new_agents)
    
    def calculate_clustering(self) -> float:
        if len(self.agents) < 2:
            return 0.0
        
        same_tag_neighbors = 0
        total_neighbors = 0
        
        for agent in self.agents:
            neighbors = self.get_neighbors(agent.position)
            if not neighbors:
                continue
            
            for neighbor in neighbors:
                total_neighbors += 1
                if agent.tag == neighbor.tag:
                    same_tag_neighbors += 1
        
        if total_neighbors == 0:
            return 0.0
        
        return same_tag_neighbors / total_neighbors
    
    def step(self):
        """
        Modified step function with God intervention.
        
        Order of operations:
        1. Normal interactions
        2. External shocks (droughts, disasters, predators)
        3. 🧠 GOD INTERVENTION (NEW!)
        4. Metabolism
        5. Remove dead
        6. Reproduction
        """
        self.run_interactions()
        
        # External shocks
        shock_msg = self.apply_external_shocks()
        
        # 🧠 GOD INTERVENTION - The new piece!
        god_msg = self.god_intervention_step()
        
        self.metabolize_all()
        deaths = self.remove_dead()
        births = self.handle_reproduction()
        
        # Record history
        if self.agents:
            total_res = sum(a.resources for a in self.agents)
            total_actions = sum(a.cooperations + a.defections for a in self.agents)
            total_coop = sum(a.cooperations for a in self.agents)
            coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
            clustering = self.calculate_clustering()
            
            self.history['population'].append(len(self.agents))
            self.history['resources'].append(total_res)
            self.history['cooperation'].append(coop_rate)
            self.history['births'].append(births)
            self.history['deaths'].append(deaths)
            self.history['clustering'].append(clustering)
            self.history['shocks'].append(shock_msg)
            self.history['god_interventions'].append(god_msg)
        
        self.generation += 1
        return deaths, births, shock_msg, god_msg
    
    def print_live_dashboard(self, shock_msg: Optional[str] = None, god_msg: Optional[str] = None):
        """Enhanced dashboard showing God interventions."""
        clear_screen()
        import sys
        sys.stdout.flush()
        
        elapsed = time.time() - self.start_time
        
        # Header with God mode indicator
        print(f"{Fore.CYAN}{'='*100}")
        print(f"{Fore.MAGENTA}🧠👁️  ECHO MODEL WITH GOD-AI CONTROLLER (Mode: {self.god.mode}) 👁️🧠{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
        
        # Alert banners
        if shock_msg:
            print(f"{Fore.RED}⚠️  {shock_msg} ⚠️{Style.RESET_ALL}")
        
        if god_msg:
            print(f"{Fore.MAGENTA}{god_msg}{Style.RESET_ALL}")
        
        if shock_msg or god_msg:
            print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
        
        # Timing
        print(f"{Fore.WHITE}Generation: {Fore.YELLOW}{self.generation}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Elapsed: {Fore.GREEN}{elapsed:.1f}s{Style.RESET_ALL} | ", end="")
        speed = self.generation / elapsed if elapsed > 0 else 0
        print(f"{Fore.WHITE}Speed: {Fore.GREEN}{speed:.2f} gen/s{Style.RESET_ALL}\n")
        
        if not self.agents:
            print(f"{Fore.RED}💀 EXTINCTION{Style.RESET_ALL}")
            return
        
        # Population stats
        resources = [a.resources for a in self.agents]
        ages = [a.age for a in self.agents]
        
        print(f"{Fore.CYAN}{'─'*100}")
        print(f"{Fore.YELLOW}POPULATION STATS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Size: {Fore.GREEN}{len(self.agents)}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Avg Age: {Fore.CYAN}{np.mean(ages):.1f}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Clustering: {Fore.MAGENTA}{self.history['clustering'][-1]*100:.1f}%{Style.RESET_ALL}\n")
        
        # Resource distribution
        print(f"{Fore.WHITE}Resources: Avg={Fore.GREEN}{np.mean(resources):.1f}{Style.RESET_ALL}, ", end="")
        print(f"Min={Fore.RED}{min(resources):.0f}{Style.RESET_ALL}, ", end="")
        print(f"Max={Fore.GREEN}{max(resources):.0f}{Style.RESET_ALL}, ", end="")
        print(f"Total={Fore.CYAN}{sum(resources):.0f}{Style.RESET_ALL}\n")
        
        # Cooperation
        total_actions = sum(a.cooperations + a.defections for a in self.agents)
        total_coop = sum(a.cooperations for a in self.agents)
        coop_rate = (total_coop / total_actions * 100) if total_actions > 0 else 0
        
        print(f"{Fore.WHITE}Cooperation: {Fore.GREEN}{coop_rate:.1f}%{Style.RESET_ALL} ", end="")
        coop_bar = int(coop_rate / 5)
        print(f"[{Fore.GREEN}{'█' * coop_bar}{Fore.WHITE}{'░' * (20 - coop_bar)}{Style.RESET_ALL}]\n")
        
        # GOD STATS
        god_stats = self.god.get_summary_stats()
        print(f"{Fore.CYAN}{'─'*100}")
        print(f"{Fore.MAGENTA}🧠 GOD-AI INTERVENTIONS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Total: {Fore.YELLOW}{god_stats.get('total_interventions', 0)}{Style.RESET_ALL} | ", end="")
        
        if 'interventions_by_type' in god_stats:
            for itype, count in god_stats['interventions_by_type'].items():
                print(f"{Fore.WHITE}{itype}: {Fore.CYAN}{count}{Style.RESET_ALL} | ", end="")
        print()
        
        # Simplified grid (smaller sample)
        print(f"\n{Fore.CYAN}{'─'*100}")
        print(f"{Fore.YELLOW}SPATIAL GRID (30×30 sample){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
        
        start_y = (GRID_HEIGHT - 30) // 2
        start_x = (GRID_WIDTH - 30) // 2
        
        for y in range(start_y, start_y + 30):
            row = ""
            for x in range(start_x, start_x + 30):
                agent = self.grid[y][x]
                if agent is None:
                    row += Fore.WHITE + "·"
                elif agent.resources > 5000:
                    row += Fore.GREEN + "█"
                elif agent.resources > 1000:
                    row += Fore.YELLOW + "▓"
                else:
                    row += Fore.RED + "░"
            print(row + Style.RESET_ALL)
        
        print(f"\n{Fore.WHITE}Legend: {Fore.GREEN}█{Style.RESET_ALL}=Rich(>5K) ", end="")
        print(f"{Fore.YELLOW}▓{Style.RESET_ALL}=Medium(>1K) ", end="")
        print(f"{Fore.RED}░{Style.RESET_ALL}=Poor(<1K) ", end="")
        print(f"{Fore.WHITE}·{Style.RESET_ALL}=Empty\n")
    
    def save_results(self, filename: str = None):
        """Save simulation results including God intervention log."""
        if filename is None:
            filename = f"god_echo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            'config': {
                'initial_size': len(self.agents),
                'grid_size': f"{GRID_WIDTH}×{GRID_HEIGHT}",
                'god_mode': self.god.mode,
                'generations': self.generation
            },
            'final_stats': {
                'population': len(self.agents),
                'avg_wealth': np.mean([a.resources for a in self.agents]) if self.agents else 0,
                'cooperation_rate': self.history['cooperation'][-1] if self.history['cooperation'] else 0,
                'clustering': self.history['clustering'][-1] if self.history['clustering'] else 0
            },
            'history': {k: v for k, v in self.history.items() if k != 'god_interventions'},
            'god_interventions': [r.to_dict() for r in self.god.intervention_history],
            'god_summary': self.god.get_summary_stats()
        }
        
        filepath = os.path.join('outputs', 'god_ai', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")

# --- MAIN SIMULATION ---

def run_god_echo_simulation(
    generations: int = 500,
    initial_size: int = 100,
    god_mode: str = "RULE_BASED",
    update_frequency: int = 5
):
    """
    Run Echo simulation with God-AI controller.
    
    Args:
        generations: Number of generations to run
        initial_size: Starting population
        god_mode: "RULE_BASED", "ML_BASED", "API_BASED", or "DISABLED"
        update_frequency: Update dashboard every N generations
    """
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.MAGENTA}🧠👁️  INITIALIZING GOD-CONTROLLED ECHO SIMULATION 👁️🧠{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}God Mode: {Fore.YELLOW}{god_mode}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Generations: {Fore.YELLOW}{generations}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Initial Population: {Fore.YELLOW}{initial_size}{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}Starting simulation...{Style.RESET_ALL}\n")
    
    time.sleep(2)
    
    population = GodEchoPopulation(initial_size=initial_size, god_mode=god_mode)
    
    for gen in range(generations):
        deaths, births, shock_msg, god_msg = population.step()
        
        # Update dashboard periodically or when events occur
        if gen % update_frequency == 0 or shock_msg or god_msg:
            population.print_live_dashboard(shock_msg, god_msg)
        
        # Check extinction
        if not population.agents:
            population.print_live_dashboard()
            print(f"\n{Fore.RED}💀 EXTINCTION at generation {gen}{Style.RESET_ALL}")
            break
        
        time.sleep(0.01)  # Small delay for readability
    
    # Final dashboard
    population.print_live_dashboard()
    
    # Print summary
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.YELLOW}🏁 SIMULATION COMPLETE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    
    if population.agents:
        print(f"{Fore.GREEN}✅ Population survived!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Final Population: {Fore.CYAN}{len(population.agents)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Final Cooperation: {Fore.CYAN}{population.history['cooperation'][-1]*100:.1f}%{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Final Clustering: {Fore.CYAN}{population.history['clustering'][-1]*100:.1f}%{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}💀 Population went extinct{Style.RESET_ALL}")
    
    # God summary
    god_stats = population.god.get_summary_stats()
    print(f"\n{Fore.MAGENTA}🧠 God-AI Summary:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Total Interventions: {Fore.YELLOW}{god_stats.get('total_interventions', 0)}{Style.RESET_ALL}")
    
    if 'interventions_by_type' in god_stats:
        print(f"{Fore.WHITE}Breakdown:{Style.RESET_ALL}")
        for itype, count in god_stats['interventions_by_type'].items():
            print(f"  {Fore.CYAN}{itype}{Style.RESET_ALL}: {count}")
    
    # Save results
    population.save_results()
    
    return population

if __name__ == "__main__":
    # Run with Rule-Based God
    population = run_god_echo_simulation(
        generations=500,
        initial_size=100,
        god_mode="RULE_BASED",
        update_frequency=10
    )
