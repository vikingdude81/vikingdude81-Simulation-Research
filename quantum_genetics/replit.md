# Quantum-Genetic Hybrid Agent Evolution System

## Overview
This project implements a sophisticated quantum-genetic hybrid evolution system with parallel ensemble training and a deployment server for evolved genomes. The system combines quantum mechanics principles with genetic algorithms to evolve adaptive agents.

**Business Vision:** To provide a robust platform for evolving highly adaptive agents using quantum-genetic principles.
**Market Potential:** Applications in AI, scientific research, and complex system optimization.
**Project Ambitions:** To push the boundaries of evolutionary computation by integrating quantum mechanics, offering a powerful tool for agent development and deployment.

## User Preferences
I prefer simple language and direct communication. I want iterative development with clear explanations for each step. Ask before making major architectural changes or introducing new dependencies. Do not make changes to the `visualizations/` folder.

## System Architecture

### Core Components
1.  **`quantum_genetic_agents.py`**: The main evolution engine, responsible for running quantum-genetic hybrid evolution with parallel ensemble populations, multi-environment testing, and genome export.
2.  **`genome_deployment_server.py`**: A Flask-based REST API for deploying and monitoring evolved genomes, providing a dashboard UI and real-time performance monitoring.
3.  **`schrodinger_cat.py`**: A visualization suite utilizing PyVista for 3D quantum state visualization and GIF export.

### Workflows
*   **"Generate Genomes" Workflow**: Executes `quantum_genetic_agents.py` to run the evolution process, producing JSON genome files in `data/genomes/production/` directory (`best_individual_genome.json`, `averaged_ensemble_genome.json`).
*   **"Deploy Genome Server" Workflow**: Starts the `genome_deployment_server.py` on port 5000, enabling access to the web dashboard and API endpoints for genome deployment and monitoring.

### Project Structure
The project is organized with a clean folder structure focused on the quantum evolution workflows:

**Root Directory:**
- `quantum_genetic_agents.py` - Main evolution engine
- `genome_deployment_server.py` - Deployment server
- `schrodinger_cat.py` - Visualization suite

**Data Organization:**
- `data/genomes/production/` - Active genome files used by the deployment server
  - `best_individual_genome.json` - Best evolved genome from latest run
  - `averaged_ensemble_genome.json` - Averaged genome across all populations
- `data/genomes/archive/` - Historical genome files from previous experiments

**Supporting Directories:**
- `templates/` - HTML templates for the web dashboard
- `visualizations/` - Generated visualization images and animations (do not modify manually)
- `archive/` - Experimental scripts and research utilities
- `attached_assets/` - User-uploaded files and paste history

All genome generation and deployment operations use the `data/genomes/production/` directory as the canonical location for active genomes.

### Technical Implementations & Features
*   **Evolution Parameters**: 15 parallel populations, 30 agents per population, 300 generations, with multi-environment testing (standard, harsh, gentle, chaotic, oscillating).
*   **Genome Structure**: Each genome is defined by `mutation_rate`, `oscillation_freq`, `decoherence_rate`, and `phase_offset`.
*   **Deployment Server**: Flask-based API with endpoints for deploying best, averaged, or custom genomes, listing available genomes, undeploying, and retrieving status/history.
*   **Dashboard UI**: Provides a user interface for viewing available genomes, previewing properties, deploying multiple genomes, real-time performance monitoring with dynamic color assignment, and exporting configurations.
*   **Security Hardening**: Includes XSS vulnerability elimination by replacing `innerHTML` with safe DOM manipulation and path traversal protection for file operations.
*   **Performance & Efficiency**: Optimized by organizing visualization files, disabling live visualization by default, and capping metrics storage to prevent memory leaks.
*   **Robustness**: Implemented comprehensive error handling in the dashboard and server-side to prevent crashes and provide user-friendly alerts.
*   **UI/UX Decisions**: Dynamic color assignment for deployed genomes, a genome properties modal for pre-deployment viewing, and an auto-refreshing genome list.

## External Dependencies
*   **Python 3.x**
*   **flask**: Web server for the deployment API.
*   **numpy**: Numerical computing.
*   **matplotlib**: Plotting.
*   **scikit-learn**: Machine learning predictions.
*   **scipy**: Scientific computing.
*   **seaborn**: Statistical visualization.
*   **pyvista**: 3D visualization.
*   **imageio**: Image processing.
*   **termcolor**: Colored terminal output.
*   **tqdm**: Progress bars.