"""
⚙️ Quantum Research Platform - Configuration
Centralized configuration and constants for the quantum research platform
"""

import os

VISUALIZATION_DIR = "visualizations"
GENOME_DIR = "."

DEFAULT_EVOLUTION_PARAMS = {
    "num_populations": 15,
    "population_size": 30,
    "num_generations": 300,
    "mutation_rate": 0.1,
    "crossover_rate": 0.7,
}

QUANTUM_PARAMS = {
    "hbar": 1.0,
    "mass": 1.0,
    "omega": 1.0,
}

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.environ.get('PORT', 5000)),
    "debug": False,
}

ENVIRONMENT_TYPES = [
    "standard",
    "harsh",
    "gentle",
    "chaotic",
    "oscillating"
]

PLOTTING_PARAMS = {
    "dpi": 150,
    "figsize": (10, 7),
    "style": "seaborn-v0_8-darkgrid",
}

os.makedirs(VISUALIZATION_DIR, exist_ok=True)
