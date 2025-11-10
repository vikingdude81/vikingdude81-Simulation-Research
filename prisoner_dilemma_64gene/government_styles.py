"""
🏛️ GOVERNMENT STYLES - Top-Down Economic Policy Framework
===========================================================

This module implements different government/economic policy styles as
rule-based interventions in the Echo simulation. Each style tests a
different hypothesis about top-down control vs. bottom-up emergence.

Government Styles:
1. LAISSEZ_FAIRE: No intervention (pure agent interactions)
2. WELFARE_STATE: Redistributive policies (safety net + wealth tax)
3. AUTHORITARIAN: Enforcement-based (punish defectors)
4. CENTRAL_BANKER: Macroeconomic management (stimulus during recession)
5. MIXED_ECONOMY: Adaptive policy (switches based on conditions)

Research Questions:
- Does welfare protect cooperators or subsidize defectors?
- Does authoritarian enforcement create stable cooperation?
- Does monetary stimulus help cooperation or just cause inflation?
- Which style produces the most wealth/cooperation/diversity?
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


class GovernmentStyle(Enum):
    """Different economic governance approaches."""
    LAISSEZ_FAIRE = "laissez_faire"
    WELFARE_STATE = "welfare_state"
    AUTHORITARIAN = "authoritarian"
    CENTRAL_BANKER = "central_banker"
    MIXED_ECONOMY = "mixed_economy"


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


class GovernmentController:
    """
    Implements different government/economic policy styles.
    
    This is a more sophisticated version of the God-AI controller,
    with explicit economic policy frameworks instead of generic interventions.
    """
    
    def __init__(self, style: GovernmentStyle):
        self.style = style
        self.policy_history: List[PolicyAction] = []
        self.generation = 0
        
        # Policy parameters (tunable)
        self.WEALTH_TAX_THRESHOLD = 50  # Tax agents with wealth > this
        self.POVERTY_LINE = 10  # Safety net for agents below this
        self.RECESSION_THRESHOLD = 15  # Avg wealth below this triggers stimulus
        self.STIMULUS_AMOUNT = 10  # Universal basic income during recession
        self.WEALTH_TRANSFER_RATE = 0.3  # Take 30% from rich, give to poor
        self.JAIL_DEFECTORS = True  # Authoritarian: remove defectors?
        
    def apply_policy(self, agents: List, grid_size: Tuple[int, int]) -> Optional[PolicyAction]:
        """
        Apply government policy based on the selected style.
        
        Args:
            agents: List of agents in the simulation
            grid_size: (width, height) of the grid
            
        Returns:
            PolicyAction if an intervention occurred, None otherwise
        """
        self.generation += 1
        
        # Capture before-state metrics
        before_metrics = self._capture_metrics(agents)
        
        # Route to appropriate policy function
        if self.style == GovernmentStyle.LAISSEZ_FAIRE:
            return self._laissez_faire(agents)
        elif self.style == GovernmentStyle.WELFARE_STATE:
            return self._welfare_state(agents, before_metrics)
        elif self.style == GovernmentStyle.AUTHORITARIAN:
            return self._authoritarian(agents, before_metrics)
        elif self.style == GovernmentStyle.CENTRAL_BANKER:
            return self._central_banker(agents, before_metrics)
        elif self.style == GovernmentStyle.MIXED_ECONOMY:
            return self._mixed_economy(agents, before_metrics)
        else:
            return None
    
    def _laissez_faire(self, agents: List) -> Optional[PolicyAction]:
        """Pure free market - no intervention."""
        return None  # Do nothing
    
    def _welfare_state(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Redistributive policy: Wealth tax + safety net.
        
        Theory Test:
        - Does this protect cooperators and help them thrive?
        - Or does it prop up defectors who would otherwise die?
        """
        if len(agents) == 0:
            return None
        
        # Find rich agents (above wealth tax threshold)
        rich_agents = [a for a in agents if a.wealth > self.WEALTH_TAX_THRESHOLD]
        
        # Find poor agents (below poverty line)
        poor_agents = [a for a in agents if a.wealth < self.POVERTY_LINE]
        
        if len(rich_agents) == 0 or len(poor_agents) == 0:
            return None  # No redistribution needed
        
        # Wealth tax: Take from rich
        total_tax_revenue = 0
        for agent in rich_agents:
            tax_amount = (agent.wealth - self.WEALTH_TAX_THRESHOLD) * self.WEALTH_TRANSFER_RATE
            agent.wealth -= tax_amount
            total_tax_revenue += tax_amount
        
        # Safety net: Give to poor
        per_person_benefit = total_tax_revenue / len(poor_agents)
        for agent in poor_agents:
            agent.wealth += per_person_benefit
        
        # Capture after-state metrics
        after_metrics = self._capture_metrics(agents)
        
        # Record policy action
        action = PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="WEALTH_REDISTRIBUTION",
            affected_agents=len(rich_agents) + len(poor_agents),
            wealth_transferred=total_tax_revenue,
            reason=f"Taxed {len(rich_agents)} rich agents, distributed to {len(poor_agents)} poor agents",
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
        
        self.policy_history.append(action)
        return action
    
    def _authoritarian(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Enforcement-based: Punish defectors.
        
        Theory Test:
        - Does top-down enforcement create stable cooperation?
        - Or does it collapse when enforcement stops?
        
        Note: This actually REMOVES defectors from the agent list.
        The caller must handle this (remove agents marked for deletion).
        """
        if len(agents) == 0:
            return None
        
        # Find defectors (agents with strategy == 0)
        defectors = [a for a in agents if a.get_strategy() == 0]
        
        if len(defectors) == 0:
            return None  # No enforcement needed
        
        # "Jail" defectors (mark for removal)
        for agent in defectors:
            agent.wealth = -9999  # Mark for deletion (caller will remove)
        
        # Capture after-state metrics (before actual removal)
        after_metrics = self._capture_metrics([a for a in agents if a.wealth > -9999])
        
        # Record policy action
        action = PolicyAction(
            generation=self.generation,
            style=self.style.value,
            action_type="DEFECTOR_PUNISHMENT",
            affected_agents=len(defectors),
            wealth_transferred=0,
            reason=f"Jailed {len(defectors)} defectors ({len(defectors)/len(agents)*100:.1f}% of population)",
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
        
        self.policy_history.append(action)
        return action
    
    def _central_banker(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Macroeconomic management: Stimulus during recession.
        
        Theory Test:
        - Does stimulus help new cooperative tribes form?
        - Or does it just create inflation with no real change?
        """
        if len(agents) == 0:
            return None
        
        # Check for recession (low average wealth)
        avg_wealth = before_metrics['avg_wealth']
        
        if avg_wealth < self.RECESSION_THRESHOLD:
            # Quantitative easing: Give all agents stimulus
            for agent in agents:
                agent.wealth += self.STIMULUS_AMOUNT
            
            # Capture after-state metrics
            after_metrics = self._capture_metrics(agents)
            
            # Record policy action
            action = PolicyAction(
                generation=self.generation,
                style=self.style.value,
                action_type="STIMULUS_PACKAGE",
                affected_agents=len(agents),
                wealth_transferred=self.STIMULUS_AMOUNT * len(agents),
                reason=f"Recession detected (avg wealth {avg_wealth:.1f} < {self.RECESSION_THRESHOLD})",
                before_metrics=before_metrics,
                after_metrics=after_metrics
            )
            
            self.policy_history.append(action)
            return action
        
        return None  # No stimulus needed
    
    def _mixed_economy(self, agents: List, before_metrics: Dict) -> Optional[PolicyAction]:
        """
        Adaptive policy: Switches between styles based on conditions.
        
        Priority:
        1. Emergency (low population) -> Stimulus
        2. High inequality -> Wealth redistribution
        3. High defection -> Authoritarian enforcement
        """
        if len(agents) == 0:
            return None
        
        # Emergency: Stimulus if population very low
        if len(agents) < 50:
            return self._central_banker(agents, before_metrics)
        
        # High inequality: Wealth redistribution
        if before_metrics['wealth_inequality'] > 10:
            return self._welfare_state(agents, before_metrics)
        
        # High defection: Enforcement
        if before_metrics['cooperation_rate'] < 0.5:
            return self._authoritarian(agents, before_metrics)
        
        return None  # No intervention needed
    
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
                'style': self.style.value
            }
        
        action_types = {}
        total_wealth_transferred = 0
        
        for action in self.policy_history:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
            total_wealth_transferred += action.wealth_transferred
        
        return {
            'style': self.style.value,
            'total_actions': len(self.policy_history),
            'action_breakdown': action_types,
            'total_wealth_transferred': total_wealth_transferred,
            'first_action_generation': self.policy_history[0].generation if self.policy_history else None,
            'last_action_generation': self.policy_history[-1].generation if self.policy_history else None
        }


def compare_government_styles(
    generations: int = 500,
    initial_size: int = 100,
    trials: int = 5
) -> Dict:
    """
    Run comparative experiment across all government styles.
    
    This will run the simulation multiple times with each style and
    compare outcomes.
    
    Args:
        generations: Number of generations per trial
        initial_size: Starting population size
        trials: Number of trials per style
        
    Returns:
        Dict with comparative results
    """
    # This function would integrate with prisoner_echo_god.py
    # to run multiple trials with different government styles
    pass  # Implementation depends on integration with main simulation


if __name__ == "__main__":
    print("🏛️ GOVERNMENT STYLES MODULE")
    print("=" * 60)
    print("\nAvailable Government Styles:")
    for style in GovernmentStyle:
        print(f"  • {style.value}")
    print("\nTo use: Import GovernmentController and GovernmentStyle")
    print("Example:")
    print("  from government_styles import GovernmentController, GovernmentStyle")
    print("  gov = GovernmentController(GovernmentStyle.WELFARE_STATE)")
    print("  action = gov.apply_policy(agents, (50, 50))")
