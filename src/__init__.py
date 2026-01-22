"""
Complex Systems Simulation Package

A comprehensive framework for simulating complex systems including:
- Government policy simulations
- Agent-based modeling
- Economic systems
- Social networks
- Multi-scale dynamics
"""

__version__ = "1.0.0"
__author__ = "Simulation Research Team"

from src.simulations.government_simulation import GovernmentSimulation, Policy, PolicyType
from src.models.agent_based_models import BaseAgent, EconomicAgent, SocialAgent, AdaptiveAgent
from src.analysis.simulation_analyzer import SimulationAnalyzer

__all__ = [
    'GovernmentSimulation',
    'Policy',
    'PolicyType',
    'BaseAgent',
    'EconomicAgent',
    'SocialAgent',
    'AdaptiveAgent',
    'SimulationAnalyzer'
]
