"""
🏛️ ENHANCED GOVERNMENT STYLES - Granular Political-Economic Systems
=====================================================================

Extended government framework with:
1. COMMUNIST - State ownership, wealth equality enforcement, collective redistribution
2. FASCIST - Corporatist authoritarianism, state-directed economy, nationalist purges
3. Granular parameters that vary by government type

New Government Styles (in addition to existing 5):
- COMMUNIST: Aggressive wealth leveling, state control of production
- FASCIST: Nationalist enforcement + economic dirigisme
- SOCIAL_DEMOCRACY: Nordic model (high tax, strong welfare)
- LIBERTARIAN: Minimal state (lower than laissez-faire, no regulations)
- THEOCRACY: Morality-based (cooperation = virtue, defection = sin)
- OLIGARCHY: Rule by wealthy elite (inverse welfare state)
- TECHNOCRACY: Data-driven, algorithmic governance

Granular Parameters by Type:
- Tax rates: 0% (Libertarian) → 90% (Communist)
- Stimulus amounts: Vary by ideology
- Enforcement severity: None (Libertarian) → Total (Fascist/Communist)
- Redistribution: None (Oligarchy) → Total (Communist)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


class EnhancedGovernmentStyle(Enum):
    """Extended government types with ideological variants."""
    # Original 5
    LAISSEZ_FAIRE = "laissez_faire"
    WELFARE_STATE = "welfare_state"
    AUTHORITARIAN = "authoritarian"
    CENTRAL_BANKER = "central_banker"
    MIXED_ECONOMY = "mixed_economy"
    
    # New ideological types
    COMMUNIST = "communist"
    FASCIST = "fascist"
    SOCIAL_DEMOCRACY = "social_democracy"
    LIBERTARIAN = "libertarian"
    THEOCRACY = "theocracy"
    OLIGARCHY = "oligarchy"
    TECHNOCRACY = "technocracy"


@dataclass
class GovernmentParameters:
    """
    Granular parameters that vary by government type.
    
    Philosophy: Different ideologies have different policy tools and priorities.
    """
    # Taxation (0.0 to 1.0)
    wealth_tax_rate: float = 0.3  # % taken from rich
    income_tax_rate: float = 0.2  # % of generation wealth gain
    
    # Redistribution
    universal_basic_income: float = 0  # Per-agent stimulus
    targeted_welfare: float = 10  # Amount for poor agents
    poverty_line: float = 10  # Below this = poor
    wealth_threshold: float = 50  # Above this = rich
    
    # Enforcement
    defector_enforcement: str = "none"  # "none", "tax", "jail", "execute"
    enforcement_severity: float = 0.5  # 0.0 (lenient) to 1.0 (harsh)
    enforcement_frequency: int = 10  # Every N generations
    
    # Economic management
    stimulus_threshold: float = 15  # Avg wealth below this triggers stimulus
    stimulus_amount: float = 10  # Per-agent stimulus
    inflation_control: bool = False  # Adjust stimulus for population?
    
    # Ideological quirks
    cooperator_bonus: float = 0  # Extra reward for cooperators
    defector_penalty: float = 0  # Extra punishment for defectors
    equality_enforcement: bool = False  # Force wealth equality?
    nationalist_purge: bool = False  # Remove "outsiders"?
    meritocratic_bonus: bool = False  # Reward high performers?


@dataclass
class PolicyAction:
    """Records a government policy action."""
    generation: int
    style: str
    action_type: str
    affected_agents: int
    wealth_transferred: float
    reason: str
    before_metrics: Dict
    after_metrics: Dict


class EnhancedGovernmentController:
    """
    Implements granular government styles with ideological parameters.
    
    Key Innovation: Each government type has DIFFERENT policy tools and priorities.
    """
    
    def __init__(self, style: EnhancedGovernmentStyle):
        self.style = style
        self.policy_history: List[PolicyAction] = []
        self.generation = 0
        
        # Load parameters based on government type
        self.params = self._get_parameters_for_style(style)
        
    def _get_parameters_for_style(self, style: EnhancedGovernmentStyle) -> GovernmentParameters:
        """
        Return granular parameters for each government type.
        
        This is where the magic happens - each type gets unique settings.
        """
        if style == EnhancedGovernmentStyle.LIBERTARIAN:
            return GovernmentParameters(
                wealth_tax_rate=0.0,  # No taxation
                income_tax_rate=0.0,
                universal_basic_income=0,
                targeted_welfare=0,
                defector_enforcement="none",
                enforcement_severity=0.0,
                stimulus_threshold=0,  # Never stimulate
                stimulus_amount=0,
                cooperator_bonus=0,
                defector_penalty=0,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=False
            )
        
        elif style == EnhancedGovernmentStyle.LAISSEZ_FAIRE:
            return GovernmentParameters(
                wealth_tax_rate=0.0,
                income_tax_rate=0.05,  # Minimal tax
                universal_basic_income=0,
                targeted_welfare=0,
                defector_enforcement="none",
                enforcement_severity=0.0,
                stimulus_threshold=5,  # Only extreme crisis
                stimulus_amount=5,  # Small stimulus
                cooperator_bonus=0,
                defector_penalty=0,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=False
            )
        
        elif style == EnhancedGovernmentStyle.WELFARE_STATE:
            return GovernmentParameters(
                wealth_tax_rate=0.3,  # 30% wealth tax
                income_tax_rate=0.25,
                universal_basic_income=0,
                targeted_welfare=15,  # Generous welfare
                poverty_line=15,
                wealth_threshold=50,
                defector_enforcement="none",
                enforcement_severity=0.0,
                stimulus_threshold=15,
                stimulus_amount=10,
                cooperator_bonus=0,
                defector_penalty=0,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=False
            )
        
        elif style == EnhancedGovernmentStyle.SOCIAL_DEMOCRACY:
            return GovernmentParameters(
                wealth_tax_rate=0.5,  # 50% wealth tax (Nordic model)
                income_tax_rate=0.4,
                universal_basic_income=8,  # Universal basic income
                targeted_welfare=20,  # Strong safety net
                poverty_line=20,
                wealth_threshold=40,
                defector_enforcement="tax",  # Tax defectors
                enforcement_severity=0.3,  # Moderate
                enforcement_frequency=5,
                stimulus_threshold=20,
                stimulus_amount=15,
                cooperator_bonus=5,  # Bonus for cooperation
                defector_penalty=0,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=True
            )
        
        elif style == EnhancedGovernmentStyle.COMMUNIST:
            return GovernmentParameters(
                wealth_tax_rate=0.9,  # 90% wealth tax (near-total redistribution)
                income_tax_rate=0.8,
                universal_basic_income=15,  # Strong UBI
                targeted_welfare=25,  # High welfare
                poverty_line=30,  # High poverty threshold
                wealth_threshold=25,  # Low wealth threshold (equality)
                defector_enforcement="jail",  # "Re-educate" defectors
                enforcement_severity=0.8,  # Severe
                enforcement_frequency=3,  # Frequent
                stimulus_threshold=25,
                stimulus_amount=20,
                cooperator_bonus=10,  # Reward collective behavior
                defector_penalty=15,  # Punish individualism
                equality_enforcement=True,  # Force wealth equality
                nationalist_purge=False,
                meritocratic_bonus=False  # Anti-meritocratic
            )
        
        elif style == EnhancedGovernmentStyle.AUTHORITARIAN:
            return GovernmentParameters(
                wealth_tax_rate=0.2,  # Moderate tax
                income_tax_rate=0.15,
                universal_basic_income=0,
                targeted_welfare=5,  # Minimal welfare
                defector_enforcement="jail",  # Harsh enforcement
                enforcement_severity=0.9,  # Very severe
                enforcement_frequency=5,
                stimulus_threshold=10,
                stimulus_amount=8,
                cooperator_bonus=0,
                defector_penalty=20,  # Harsh punishment
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=False
            )
        
        elif style == EnhancedGovernmentStyle.FASCIST:
            return GovernmentParameters(
                wealth_tax_rate=0.4,  # Corporatist (state + industry)
                income_tax_rate=0.3,
                universal_basic_income=0,
                targeted_welfare=12,  # Welfare for "insiders" only
                poverty_line=10,
                wealth_threshold=60,
                defector_enforcement="execute",  # Extreme enforcement
                enforcement_severity=1.0,  # Maximum severity
                enforcement_frequency=2,  # Very frequent
                stimulus_threshold=18,
                stimulus_amount=12,
                cooperator_bonus=8,  # Reward "patriots"
                defector_penalty=30,  # Punish "traitors"
                equality_enforcement=False,
                nationalist_purge=True,  # Purge outsiders
                meritocratic_bonus=True  # Reward strength
            )
        
        elif style == EnhancedGovernmentStyle.OLIGARCHY:
            return GovernmentParameters(
                wealth_tax_rate=0.05,  # Minimal tax on rich
                income_tax_rate=0.1,
                universal_basic_income=0,
                targeted_welfare=2,  # Almost no welfare
                poverty_line=5,
                wealth_threshold=100,  # Very high threshold
                defector_enforcement="none",
                enforcement_severity=0.0,
                stimulus_threshold=8,  # Only help rich
                stimulus_amount=5,
                cooperator_bonus=0,
                defector_penalty=0,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=True  # Wealth = merit
            )
        
        elif style == EnhancedGovernmentStyle.THEOCRACY:
            return GovernmentParameters(
                wealth_tax_rate=0.25,  # Moderate tax (tithe)
                income_tax_rate=0.2,
                universal_basic_income=5,  # Charity
                targeted_welfare=15,  # Help the faithful poor
                poverty_line=12,
                wealth_threshold=45,
                defector_enforcement="jail",  # Punish sinners
                enforcement_severity=0.7,  # Harsh (morality-based)
                enforcement_frequency=7,
                stimulus_threshold=15,
                stimulus_amount=10,
                cooperator_bonus=12,  # Reward virtue
                defector_penalty=18,  # Punish sin
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=False
            )
        
        elif style == EnhancedGovernmentStyle.CENTRAL_BANKER:
            return GovernmentParameters(
                wealth_tax_rate=0.15,
                income_tax_rate=0.15,
                universal_basic_income=0,
                targeted_welfare=8,
                defector_enforcement="none",
                enforcement_severity=0.0,
                stimulus_threshold=15,  # Counter-cyclical
                stimulus_amount=10,
                inflation_control=True,  # Adjust for population
                cooperator_bonus=0,
                defector_penalty=0,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=False
            )
        
        elif style == EnhancedGovernmentStyle.TECHNOCRACY:
            return GovernmentParameters(
                wealth_tax_rate=0.35,  # Data-driven optimal tax
                income_tax_rate=0.28,
                universal_basic_income=10,  # Algorithmic UBI
                targeted_welfare=18,
                poverty_line=18,
                wealth_threshold=45,
                defector_enforcement="tax",  # Rational enforcement
                enforcement_severity=0.5,  # Calibrated
                enforcement_frequency=10,
                stimulus_threshold=17,
                stimulus_amount=12,
                cooperator_bonus=7,  # Incentivize efficiency
                defector_penalty=10,  # Disincentivize inefficiency
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=True
            )
        
        elif style == EnhancedGovernmentStyle.MIXED_ECONOMY:
            return GovernmentParameters(
                wealth_tax_rate=0.25,
                income_tax_rate=0.2,
                universal_basic_income=5,
                targeted_welfare=12,
                poverty_line=12,
                wealth_threshold=50,
                defector_enforcement="tax",
                enforcement_severity=0.4,
                enforcement_frequency=8,
                stimulus_threshold=15,
                stimulus_amount=10,
                cooperator_bonus=3,
                defector_penalty=5,
                equality_enforcement=False,
                nationalist_purge=False,
                meritocratic_bonus=True
            )
        
        else:
            # Default to mixed economy
            return self._get_parameters_for_style(EnhancedGovernmentStyle.MIXED_ECONOMY)
    
    def apply_policy(self, agents: List, grid_size: Tuple[int, int]) -> Optional[PolicyAction]:
        """
        Apply government policy based on the selected style and parameters.
        
        Args:
            agents: List of agents in the simulation
            grid_size: (width, height) of the grid
            
        Returns:
            PolicyAction if an intervention occurred, None otherwise
        """
        self.generation += 1
        
        if len(agents) == 0:
            return None
        
        # Capture before-state metrics
        before_metrics = self._capture_metrics(agents)
        
        # Route to appropriate policy function based on style
        if self.style == EnhancedGovernmentStyle.LIBERTARIAN:
            return None  # Pure non-intervention
        
        elif self.style == EnhancedGovernmentStyle.LAISSEZ_FAIRE:
            return self._laissez_faire(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.WELFARE_STATE:
            return self._welfare_state(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.SOCIAL_DEMOCRACY:
            return self._social_democracy(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.COMMUNIST:
            return self._communist(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.AUTHORITARIAN:
            return self._authoritarian(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.FASCIST:
            return self._fascist(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.OLIGARCHY:
            return self._oligarchy(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.THEOCRACY:
            return self._theocracy(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.CENTRAL_BANKER:
            return self._central_banker(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.TECHNOCRACY:
            return self._technocracy(agents, before_metrics)
        
        elif self.style == EnhancedGovernmentStyle.MIXED_ECONOMY:
            return self._mixed_economy(agents, before_metrics)
        
        return None
    
    def _laissez_faire(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """Minimal intervention - only extreme crisis response."""
        # Only act in severe crisis
        if before_metrics['avg_wealth'] < self.params.stimulus_threshold:
            for agent in agents:
                agent.wealth += self.params.stimulus_amount
            
            after_metrics = self._capture_metrics(agents)
            return PolicyAction(
                generation=self.generation,
                style=self.style.value,
                action_type="CRISIS_STIMULUS",
                affected_agents=len(agents),
                wealth_transferred=self.params.stimulus_amount * len(agents),
                reason=f"Extreme crisis: avg wealth {before_metrics['avg_wealth']:.1f}",
                before_metrics=before_metrics,
                after_metrics=after_metrics
            )
        return None
    
    def _welfare_state(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """Standard redistributive welfare state."""
        rich_agents = [a for a in agents if a.wealth > self.params.wealth_threshold]
        poor_agents = [a for a in agents if a.wealth < self.params.poverty_line]
        
        if len(rich_agents) == 0 or len(poor_agents) == 0:
            return None
        
        # Tax rich
        total_tax = 0
        for agent in rich_agents:
            tax_amount = (agent.wealth - self.params.wealth_threshold) * self.params.wealth_tax_rate
            agent.wealth -= tax_amount
            total_tax += tax_amount
        
        # Redistribute to poor
        per_person = total_tax / len(poor_agents)
        for agent in poor_agents:
            agent.wealth += per_person
        
        after_metrics = self._capture_metrics(agents)
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="WEALTH_REDISTRIBUTION",
            affected_agents=len(rich_agents) + len(poor_agents),
            wealth_transferred=total_tax,
            reason=f"Taxed {len(rich_agents)} rich, helped {len(poor_agents)} poor",
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _social_democracy(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Nordic model: High taxes + strong welfare + UBI + cooperation incentives.
        
        Combines redistribution with positive incentives for cooperation.
        """
        # Universal Basic Income (always)
        for agent in agents:
            agent.wealth += self.params.universal_basic_income
        
        # Progressive taxation + redistribution
        rich_agents = [a for a in agents if a.wealth > self.params.wealth_threshold]
        poor_agents = [a for a in agents if a.wealth < self.params.poverty_line]
        
        total_tax = 0
        if len(rich_agents) > 0:
            for agent in rich_agents:
                tax_amount = (agent.wealth - self.params.wealth_threshold) * self.params.wealth_tax_rate
                agent.wealth -= tax_amount
                total_tax += tax_amount
            
            if len(poor_agents) > 0:
                per_person = total_tax / len(poor_agents)
                for agent in poor_agents:
                    agent.wealth += per_person
        
        # Bonus for cooperators (positive reinforcement)
        cooperators = [a for a in agents if a.get_strategy() == 1]
        for agent in cooperators:
            agent.wealth += self.params.cooperator_bonus
        
        after_metrics = self._capture_metrics(agents)
        total_transferred = (self.params.universal_basic_income * len(agents)) + total_tax + (self.params.cooperator_bonus * len(cooperators))
        
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="SOCIAL_DEMOCRATIC_PACKAGE",
            affected_agents=len(agents),
            wealth_transferred=total_transferred,
            reason=f"UBI for all, taxed {len(rich_agents)} rich, bonus for {len(cooperators)} cooperators",
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _communist(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Communist system: Aggressive wealth equality + collective ownership + ideological enforcement.
        
        Key features:
        - Near-total wealth redistribution (90% tax)
        - Wealth equality enforcement (cap maximum wealth)
        - Strong UBI (collective provision)
        - Harsh punishment of "individualist" defectors
        - Reward for "collective" cooperators
        """
        total_wealth = sum(a.wealth for a in agents)
        
        # 1. WEALTH EQUALITY ENFORCEMENT (core communist principle)
        if self.params.equality_enforcement:
            # Calculate equal share
            equal_share = total_wealth / len(agents)
            
            # Redistribute to achieve equality
            for agent in agents:
                agent.wealth = equal_share
            
            equality_action = True
        else:
            equality_action = False
        
        # 2. UNIVERSAL BASIC INCOME (from the collective)
        for agent in agents:
            agent.wealth += self.params.universal_basic_income
        
        # 3. REWARD COOPERATORS (collective behavior = virtue)
        cooperators = [a for a in agents if a.get_strategy() == 1]
        for agent in cooperators:
            agent.wealth += self.params.cooperator_bonus
        
        # 4. PUNISH DEFECTORS (individualism = vice)
        if self.generation % self.params.enforcement_frequency == 0:
            defectors = [a for a in agents if a.get_strategy() == 0]
            
            if self.params.defector_enforcement == "jail":
                # "Re-education" (remove poorest defectors)
                defectors_sorted = sorted(defectors, key=lambda a: a.wealth)
                num_to_remove = int(len(defectors) * self.params.enforcement_severity)
                
                for agent in defectors_sorted[:num_to_remove]:
                    agent.wealth = -9999  # Mark for removal
            
            elif self.params.defector_enforcement == "tax":
                # Heavy taxation of defectors
                for agent in defectors:
                    penalty = agent.wealth * self.params.defector_penalty / 100
                    agent.wealth -= penalty
        
        after_metrics = self._capture_metrics([a for a in agents if a.wealth > -9999])
        
        action_desc = []
        if equality_action:
            action_desc.append("wealth equality enforced")
        action_desc.append(f"UBI {self.params.universal_basic_income}")
        action_desc.append(f"{len(cooperators)} cooperators rewarded")
        
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="COMMUNIST_REDISTRIBUTION",
            affected_agents=len(agents),
            wealth_transferred=total_wealth if equality_action else self.params.universal_basic_income * len(agents),
            reason=", ".join(action_desc),
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _authoritarian(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """Harsh enforcement of cooperation through punishment."""
        if self.generation % self.params.enforcement_frequency != 0:
            return None
        
        defectors = [a for a in agents if a.get_strategy() == 0]
        if len(defectors) == 0:
            return None
        
        # Jail defectors
        for agent in defectors:
            agent.wealth = -9999  # Mark for removal
        
        after_metrics = self._capture_metrics([a for a in agents if a.wealth > -9999])
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="AUTHORITARIAN_PURGE",
            affected_agents=len(defectors),
            wealth_transferred=0,
            reason=f"Jailed {len(defectors)} defectors",
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _fascist(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Fascist system: Corporatist state + extreme nationalism + total enforcement.
        
        Key features:
        - State-directed economy (corporatism)
        - Extreme enforcement (execute defectors)
        - Nationalist purges (genetic purity)
        - Reward "patriots" (cooperators), harshly punish "traitors" (defectors)
        - Meritocratic bonus (strength = virtue)
        """
        actions_taken = []
        total_affected = 0
        total_wealth = 0
        
        # 1. CORPORATIST REDISTRIBUTION (state + industry cooperation)
        rich_agents = [a for a in agents if a.wealth > self.params.wealth_threshold]
        poor_cooperators = [a for a in agents if a.wealth < self.params.poverty_line and a.get_strategy() == 1]
        
        if len(rich_agents) > 0 and len(poor_cooperators) > 0:
            total_tax = 0
            for agent in rich_agents:
                tax = (agent.wealth - self.params.wealth_threshold) * self.params.wealth_tax_rate
                agent.wealth -= tax
                total_tax += tax
            
            per_person = total_tax / len(poor_cooperators)
            for agent in poor_cooperators:
                agent.wealth += per_person
            
            actions_taken.append(f"corporatist welfare for {len(poor_cooperators)} insiders")
            total_affected += len(rich_agents) + len(poor_cooperators)
            total_wealth += total_tax
        
        # 2. EXTREME ENFORCEMENT (every few generations)
        if self.generation % self.params.enforcement_frequency == 0:
            defectors = [a for a in agents if a.get_strategy() == 0]
            
            if len(defectors) > 0:
                # Execute defectors (mark all for removal)
                for agent in defectors:
                    agent.wealth = -9999
                
                actions_taken.append(f"executed {len(defectors)} traitors")
                total_affected += len(defectors)
        
        # 3. REWARD PATRIOTS (cooperators)
        cooperators = [a for a in agents if a.get_strategy() == 1]
        for agent in cooperators:
            agent.wealth += self.params.cooperator_bonus
        
        if len(cooperators) > 0:
            actions_taken.append(f"rewarded {len(cooperators)} patriots")
            total_wealth += self.params.cooperator_bonus * len(cooperators)
        
        # 4. NATIONALIST PURGE (remove "outsiders" - poorest X%)
        if self.params.nationalist_purge and self.generation % (self.params.enforcement_frequency * 2) == 0:
            # Define "outsiders" as poorest 10%
            agents_sorted = sorted([a for a in agents if a.wealth > -9999], key=lambda a: a.wealth)
            num_to_purge = max(1, int(len(agents_sorted) * 0.1))
            
            for agent in agents_sorted[:num_to_purge]:
                agent.wealth = -9999
            
            actions_taken.append(f"purged {num_to_purge} outsiders")
            total_affected += num_to_purge
        
        if len(actions_taken) == 0:
            return None
        
        after_metrics = self._capture_metrics([a for a in agents if a.wealth > -9999])
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="FASCIST_TOTAL_CONTROL",
            affected_agents=total_affected,
            wealth_transferred=total_wealth,
            reason="; ".join(actions_taken),
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _oligarchy(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Rule by wealthy elite: Inverse welfare state.
        
        Features:
        - Minimal tax on rich
        - Almost no welfare for poor
        - Stimulus goes to wealthy (bailouts)
        - Meritocratic (wealth = merit)
        """
        # Only intervene in crisis - but help the rich, not poor
        if before_metrics['avg_wealth'] < self.params.stimulus_threshold:
            # Bailout for top 20% wealthiest
            agents_sorted = sorted(agents, key=lambda a: a.wealth, reverse=True)
            num_rich = max(1, int(len(agents) * 0.2))
            rich_agents = agents_sorted[:num_rich]
            
            for agent in rich_agents:
                agent.wealth += self.params.stimulus_amount
            
            after_metrics = self._capture_metrics(agents)
            return PolicyAction(
                generation=self.generation,
                style=self.style.value,
                action_type="OLIGARCH_BAILOUT",
                affected_agents=len(rich_agents),
                wealth_transferred=self.params.stimulus_amount * len(rich_agents),
                reason=f"Bailout for {len(rich_agents)} wealthy elites",
                before_metrics=before_metrics,
                after_metrics=after_metrics
            )
        
        return None
    
    def _theocracy(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Religious governance: Cooperation = virtue, defection = sin.
        
        Features:
        - Moderate redistribution (charity)
        - Strong moral enforcement (punish "sinners")
        - Reward "virtuous" cooperators
        - Harsh but less frequent than fascist
        """
        actions_taken = []
        total_affected = 0
        total_wealth = 0
        
        # 1. CHARITY (help faithful poor)
        poor_agents = [a for a in agents if a.wealth < self.params.poverty_line]
        if len(poor_agents) > 0:
            for agent in poor_agents:
                agent.wealth += self.params.targeted_welfare
            
            actions_taken.append(f"charity for {len(poor_agents)} faithful poor")
            total_affected += len(poor_agents)
            total_wealth += self.params.targeted_welfare * len(poor_agents)
        
        # 2. REWARD VIRTUE (cooperators)
        cooperators = [a for a in agents if a.get_strategy() == 1]
        for agent in cooperators:
            agent.wealth += self.params.cooperator_bonus
        
        if len(cooperators) > 0:
            actions_taken.append(f"blessed {len(cooperators)} virtuous")
            total_wealth += self.params.cooperator_bonus * len(cooperators)
        
        # 3. PUNISH SIN (defectors)
        if self.generation % self.params.enforcement_frequency == 0:
            defectors = [a for a in agents if a.get_strategy() == 0]
            
            if len(defectors) > 0:
                # Harsh punishment (but don't execute - that's fascist)
                for agent in defectors:
                    penalty = agent.wealth * (self.params.defector_penalty / 100)
                    agent.wealth = max(1, agent.wealth - penalty)  # Leave them alive but poor
                
                actions_taken.append(f"punished {len(defectors)} sinners")
                total_affected += len(defectors)
        
        if len(actions_taken) == 0:
            return None
        
        after_metrics = self._capture_metrics(agents)
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="THEOCRATIC_MORAL_ORDER",
            affected_agents=total_affected,
            wealth_transferred=total_wealth,
            reason="; ".join(actions_taken),
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _central_banker(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """Macroeconomic management with granular parameters."""
        if before_metrics['avg_wealth'] < self.params.stimulus_threshold:
            stimulus = self.params.stimulus_amount
            
            # Adjust for population if inflation control enabled
            if self.params.inflation_control:
                stimulus = stimulus * (100 / len(agents))  # Scale down for large populations
            
            for agent in agents:
                agent.wealth += stimulus
            
            after_metrics = self._capture_metrics(agents)
            return PolicyAction(
                generation=self.generation,
                style=self.style.value,
                action_type="MONETARY_STIMULUS",
                affected_agents=len(agents),
                wealth_transferred=stimulus * len(agents),
                reason=f"Recession detected: avg wealth {before_metrics['avg_wealth']:.1f}",
                before_metrics=before_metrics,
                after_metrics=after_metrics
            )
        
        return None
    
    def _technocracy(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Data-driven algorithmic governance.
        
        Features:
        - Optimal tax rates (balanced)
        - Algorithmic UBI
        - Rational enforcement (proportional)
        - Meritocratic rewards
        """
        actions_taken = []
        total_affected = 0
        total_wealth = 0
        
        # 1. ALGORITHMIC UBI
        for agent in agents:
            agent.wealth += self.params.universal_basic_income
        
        actions_taken.append(f"UBI for all {len(agents)}")
        total_wealth += self.params.universal_basic_income * len(agents)
        
        # 2. OPTIMAL REDISTRIBUTION (only if high inequality)
        if before_metrics['wealth_inequality'] > 5:
            rich_agents = [a for a in agents if a.wealth > self.params.wealth_threshold]
            poor_agents = [a for a in agents if a.wealth < self.params.poverty_line]
            
            if len(rich_agents) > 0 and len(poor_agents) > 0:
                total_tax = 0
                for agent in rich_agents:
                    tax = (agent.wealth - self.params.wealth_threshold) * self.params.wealth_tax_rate
                    agent.wealth -= tax
                    total_tax += tax
                
                per_person = total_tax / len(poor_agents)
                for agent in poor_agents:
                    agent.wealth += per_person
                
                actions_taken.append(f"optimal redistribution: {len(rich_agents)}→{len(poor_agents)}")
                total_wealth += total_tax
        
        # 3. RATIONAL ENFORCEMENT (proportional to defection rate)
        if self.generation % self.params.enforcement_frequency == 0:
            defectors = [a for a in agents if a.get_strategy() == 0]
            
            if before_metrics['cooperation_rate'] < 0.7 and len(defectors) > 0:
                # Tax defectors (not execution - that's inefficient)
                for agent in defectors:
                    penalty = agent.wealth * (self.params.defector_penalty / 100)
                    agent.wealth -= penalty
                
                actions_taken.append(f"taxed {len(defectors)} inefficient defectors")
                total_affected += len(defectors)
        
        # 4. MERITOCRATIC BONUS (reward high performers = cooperators)
        if self.params.meritocratic_bonus:
            cooperators = [a for a in agents if a.get_strategy() == 1]
            for agent in cooperators:
                agent.wealth += self.params.cooperator_bonus
            
            if len(cooperators) > 0:
                actions_taken.append(f"efficiency bonus for {len(cooperators)}")
                total_wealth += self.params.cooperator_bonus * len(cooperators)
        
        after_metrics = self._capture_metrics(agents)
        return PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="TECHNOCRATIC_OPTIMIZATION",
            affected_agents=len(agents),
            wealth_transferred=total_wealth,
            reason="; ".join(actions_taken),
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _mixed_economy(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """Adaptive policy with granular parameters."""
        # Emergency: Stimulus if population very low
        if len(agents) < 50:
            return self._central_banker(agents, before_metrics)
        
        # High inequality: Wealth redistribution
        if before_metrics['wealth_inequality'] > 10:
            return self._welfare_state(agents, before_metrics)
        
        # High defection: Enforcement
        if before_metrics['cooperation_rate'] < 0.5:
            return self._authoritarian(agents, before_metrics)
        
        return None
    
    def _capture_metrics(self, agents: List) -> Dict:
        """Capture current state metrics."""
        if len(agents) == 0:
            return {
                'population': 0,
                'avg_wealth': 0,
                'total_wealth': 0,
                'cooperation_rate': 0,
                'wealth_inequality': 0
            }
        
        wealths = [a.wealth for a in agents if a.wealth > 0]
        cooperators = sum(1 for a in agents if a.get_strategy() == 1)
        
        return {
            'population': len(agents),
            'avg_wealth': np.mean(wealths) if wealths else 0,
            'total_wealth': sum(wealths),
            'cooperation_rate': cooperators / len(agents) if len(agents) > 0 else 0,
            'wealth_inequality': (max(wealths) / min(wealths)) if (wealths and min(wealths) > 0) else 0
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics for all policy actions."""
        if not self.policy_history:
            return {
                'total_actions': 0,
                'style': self.style.value,
                'parameters': vars(self.params)
            }
        
        action_types = {}
        total_wealth_transferred = 0
        
        for action in self.policy_history:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
            total_wealth_transferred += action.wealth_transferred
        
        return {
            'style': self.style.value,
            'parameters': vars(self.params),
            'total_actions': len(self.policy_history),
            'action_breakdown': action_types,
            'total_wealth_transferred': total_wealth_transferred,
            'first_action_generation': self.policy_history[0].generation if self.policy_history else None,
            'last_action_generation': self.policy_history[-1].generation if self.policy_history else None
        }


if __name__ == "__main__":
    print("🏛️ ENHANCED GOVERNMENT STYLES MODULE")
    print("=" * 70)
    print("\n📋 Available Government Types:")
    print("\nOriginal 5:")
    print("  1. LAISSEZ_FAIRE - Minimal intervention")
    print("  2. WELFARE_STATE - Redistributive policies")
    print("  3. AUTHORITARIAN - Enforcement-based")
    print("  4. CENTRAL_BANKER - Macroeconomic management")
    print("  5. MIXED_ECONOMY - Adaptive policy")
    print("\n✨ New Ideological Types:")
    print("  6. COMMUNIST - State ownership, wealth equality (90% tax)")
    print("  7. FASCIST - Corporatist authoritarianism, nationalist purges")
    print("  8. SOCIAL_DEMOCRACY - Nordic model (50% tax, UBI, cooperation bonus)")
    print("  9. LIBERTARIAN - Pure non-intervention (0% tax)")
    print(" 10. THEOCRACY - Morality-based (cooperation=virtue, defection=sin)")
    print(" 11. OLIGARCHY - Rule by wealthy elite (inverse welfare)")
    print(" 12. TECHNOCRACY - Data-driven algorithmic governance")
    
    print("\n📊 Granular Parameters by Type:")
    print("  • Tax rates: 0% (Libertarian) → 90% (Communist)")
    print("  • Enforcement: None (Libertarian) → Total (Fascist)")
    print("  • Redistribution: Inverse (Oligarchy) → Total (Communist)")
    print("  • UBI: 0 (Libertarian) → 15 (Communist)")
    
    print("\n🔬 To use:")
    print("  from enhanced_government_styles import EnhancedGovernmentController, EnhancedGovernmentStyle")
    print("  gov = EnhancedGovernmentController(EnhancedGovernmentStyle.COMMUNIST)")
    print("  action = gov.apply_policy(agents, (50, 50))")
    print("  summary = gov.get_summary()")
