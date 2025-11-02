# Quantum Research Platform

## Overview
A comprehensive quantum research platform that combines quantum mechanics simulations, evolutionary computation, and advanced analysis tools. This platform provides an integrated environment for exploring quantum phenomena, evolving quantum agents using genetic algorithms, and deploying optimized genomes.

**Project Vision:** To provide a unified platform for quantum research, combining theoretical simulations with practical evolutionary optimization.

**Key Features:**
- 14 interactive quantum physics simulations
- 9 comprehensive analysis and research tools
- 5 evolutionary computation engines
- Web-based genome deployment and monitoring system

## User Preferences
I prefer simple language and direct communication. I want iterative development with clear explanations for each step. Ask before making major architectural changes or introducing new dependencies. Do not make changes to the `visualizations/` folder.

## Project Structure

The project is now organized into four main directories:

### 📊 quantum_examples/
Interactive quantum physics simulations and visualizations:
- **Quantum States**: Bloch sphere, hydrogen atom, 3D/4D oscillators, Schrödinger's cat
- **Quantum Dynamics**: Rabi oscillations, wavepacket evolution, decoherence
- **Quantum Phenomena**: Double-slit experiment, quantum tunneling, entanglement
- **Quantum Computing**: Quantum gate operations
- See `quantum_examples/README.md` for detailed descriptions

### 📈 analysis_tools/
Research and analysis tools for studying quantum systems:
- **Evolution Analysis**: Evolution dynamics, genome comparison, deep analysis
- **Genome Testing**: App tester, quantum genome tester, extreme frontier testing
- **Research Tools**: Comprehensive analysis, quantum data analysis, research utilities
- See `analysis_tools/README.md` for usage instructions

### 🧬 evolution_engine/
Genetic algorithm implementations for evolving quantum agents:
- **quantum_genetic_agents.py** - Main evolution engine with 15 parallel populations
- **multi_objective_evolution.py** - Pareto frontier optimization
- **phase_focused_evolution.py** - Phase parameter specialization
- **quantum_evolution_agents.py** - Quantum-inspired evolutionary strategies
- **quantum_ml.py** - Machine learning genome prediction
- See `evolution_engine/README.md` for configuration and usage

### 🚀 server/
Web servers for genome deployment and monitoring:
- **genome_deployment_server.py** - Flask API and dashboard (port 5000)
- **co_evolution_server.py** - Co-evolutionary experiments
- **templates/** - HTML templates for web UI
- See `server/README.md` for API documentation

### 📄 Root Files
- **config.py** - Centralized configuration and constants
- **main.py** - Legacy quantum harmonic oscillator visualization
- **replit.md** - This documentation file

## Workflows

The platform is organized around 4 focused workflows for quantum research:

### 1. Quantum Examples
**Command:** `python quantum_examples/schrodinger_cat.py`
**Type:** Console/Visualization
**Purpose:** Demonstrate quantum phenomena through interactive simulations

**What it does:**
- Runs Schrödinger's cat quantum superposition visualization
- Creates 3D PyVista visualizations of cat states
- Generates comparison plots and orbital animations
- Can be swapped with any other quantum example script

**Other examples you can run:**
- `bloch_sphere.py` - Qubit state visualization
- `double_slit.py` - Interference patterns
- `hydrogen_atom.py` - Atomic orbitals
- `rabi_oscillations.py` - Two-level system dynamics
- `quantum_tunneling.py` - Tunneling phenomena
- And 8 more quantum simulations!

### 2. Analysis & Research
**Command:** `python analysis_tools/quantum_data_analysis.py`
**Type:** Console/Data Analysis
**Purpose:** Analyze quantum datasets and research results

**What it does:**
- Hydrogen spectroscopy analysis
- Double-slit interference pattern analysis
- Qubit state tomography
- Generates analysis plots and statistical reports

**Other analysis tools:**
- `compare_all_genomes.py` - Benchmark genome performance
- `deep_genome_analysis.py` - In-depth genome inspection
- `comprehensive_analysis.py` - Full system analysis
- `extreme_frontier_test.py` - Test extreme conditions

### 3. Evolution Engine
**Command:** `python evolution_engine/quantum_genetic_agents.py`
**Type:** Console/Long-running
**Purpose:** Evolve quantum agents using genetic algorithms

**What it does:**
- Runs 15 parallel populations with 30 agents each
- Evolves over 300 generations (~10 minutes)
- Tests in 5 different environments
- Exports `best_individual_genome.json` and `averaged_ensemble_genome.json`
- Generates evolution visualization plots

**Other evolution experiments:**
- `multi_objective_evolution.py` - Pareto frontier optimization
- `phase_focused_evolution.py` - Phase parameter specialization
- `quantum_ml.py` - ML-guided genome prediction

### 4. Deployment Server
**Command:** `python server/genome_deployment_server.py`
**Type:** Web Server (Port 5000)
**Purpose:** Web-based dashboard for deploying and monitoring evolved genomes

**API Endpoints:**
- `GET /` - Dashboard UI
- `POST /api/deploy` - Deploy best individual genome
- `POST /api/deploy_averaged` - Deploy averaged ensemble genome
- `POST /api/deploy_file/<filename>` - Deploy any genome file
- `POST /api/undeploy/<genome_id>` - Remove deployed genome
- `GET /api/list_genomes` - List available genome files
- `GET /api/start` - Start evolution simulation
- `GET /api/stop` - Stop simulation
- `GET /api/status` - Get current status
- `GET /api/history/<genome_id>` - Get performance history
- `GET /api/export_config` - Export deployment config

## System Architecture

### Quantum Examples
Each quantum example is a self-contained simulation demonstrating a specific quantum phenomenon. Examples use matplotlib for 2D/3D visualization and PyVista for advanced 3D rendering. All visualizations are saved to the `visualizations/` directory.

### Evolution Engine
The evolution system uses genetic algorithms with:
- **15 parallel populations** for diversity
- **30 agents per population** for sufficient sampling
- **300 generations** (configurable) for convergence
- **Multi-environment testing**: standard, harsh, gentle, chaotic, oscillating
- **Genome structure**: mutation_rate, oscillation_freq, decoherence_rate, phase_offset

### Analysis Tools
Analysis tools provide comprehensive insights into:
- Evolutionary dynamics and convergence patterns
- Genome performance benchmarking
- Fitness landscape visualization
- Statistical analysis of quantum datasets
- Extreme condition testing

### Deployment System
The deployment server provides:
- Real-time performance monitoring
- Multiple genome deployment (best, averaged, custom)
- Historical performance tracking
- Dynamic color-coded visualization
- Export capabilities for deployment configurations

## Quick Start Guide

### 1. Explore Quantum Examples
Use the **Quantum Examples** workflow, or run directly:
```bash
python quantum_examples/schrodinger_cat.py
python quantum_examples/double_slit.py
python quantum_examples/hydrogen_atom.py
# ... any quantum example
```

### 2. Run Evolution
Use the **Evolution Engine** workflow, or run directly:
```bash
python evolution_engine/quantum_genetic_agents.py
# Select experiment type when prompted
# Generates: best_individual_genome.json and averaged_ensemble_genome.json
```

### 3. Analyze Results
Use the **Analysis & Research** workflow, or run directly:
```bash
python analysis_tools/quantum_data_analysis.py
python analysis_tools/compare_all_genomes.py
python analysis_tools/deep_genome_analysis.py
# ... any analysis tool
```

### 4. Deploy & Monitor Genomes
Use the **Deployment Server** workflow, or run directly:
```bash
python server/genome_deployment_server.py
# Access web dashboard at http://0.0.0.0:5000
```

## Technical Implementation

### Dependencies
- **Python 3.x** - Core language
- **flask** - Web server framework
- **numpy** - Numerical computing
- **matplotlib** - 2D/3D plotting
- **scikit-learn** - Machine learning
- **scipy** - Scientific computing
- **seaborn** - Statistical visualization
- **pyvista** - Advanced 3D visualization
- **imageio** - Image/GIF processing
- **termcolor** - Colored terminal output
- **tqdm** - Progress bars

### Configuration
Edit `config.py` to adjust:
- Evolution parameters (populations, generations, mutation rates)
- Quantum constants (ℏ, mass, omega)
- Server settings (host, port)
- Plotting defaults (DPI, figure size, style)
- Environment types for testing

### Security & Performance
- **XSS Protection**: Safe DOM manipulation (no innerHTML)
- **Path Traversal Protection**: File operation sanitization
- **Memory Management**: Capped metrics storage (500 max)
- **Error Handling**: Comprehensive try-catch blocks
- **Optimized Rendering**: Headless mode for PyVista

## Recent Changes (Nov 2, 2025)

### Project Reorganization
- Restructured entire codebase into organized directories
- Created `quantum_examples/`, `analysis_tools/`, `evolution_engine/`, `server/`
- Added README files to each directory
- Created centralized `config.py` for configuration

### New Features
- **Workflow-focused approach** - Direct access to quantum examples and analysis through 4 workflows
- Organized directory structure for all tools and simulations
- Fixed import paths for new directory layout
- Streamlined access to all 28 quantum tools

### Workflow Updates
- Created 4 focused workflows for direct access:
  - **Quantum Examples**: Run quantum simulations (Schrödinger's cat)
  - **Analysis & Research**: Run quantum data analysis
  - **Evolution Engine**: Run genetic algorithm evolution
  - **Deployment Server**: Web dashboard on port 5000
- Fixed import paths for new directory structure
- Updated all server files for compatibility

### Documentation
- Complete rewrite of replit.md for new structure
- Added comprehensive README files to all directories
- Created quick start guide
- Documented all API endpoints and workflows

## Future Enhancements

Potential areas for expansion:
- Additional quantum phenomena simulations (quantum error correction, quantum annealing)
- Enhanced ML models for genome prediction
- Multi-objective optimization improvements
- Real-time collaborative evolution experiments
- Advanced visualization techniques (VR/AR quantum states)
- Integration with quantum computing frameworks (Qiskit, Cirq)

## Support & Documentation

For detailed information on specific components:
- Quantum Examples: See `quantum_examples/README.md`
- Analysis Tools: See `analysis_tools/README.md`
- Evolution Engine: See `evolution_engine/README.md`
- Server Management: See `server/README.md`
- Configuration: Edit `config.py`
- Scaling Guide: See `SCALING_GUIDE.md`
