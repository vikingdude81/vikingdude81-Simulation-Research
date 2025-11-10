# Analysis Tools

This directory contains research and analysis tools for studying quantum systems and evolutionary dynamics.

## Available Tools

### Evolution Analysis
- **analyze_evolution_dynamics.py** - Study evolutionary trends and convergence
- **compare_all_genomes.py** - Benchmark and compare genome performance
- **deep_genome_analysis.py** - In-depth genome inspection and visualization

### Genome Testing
- **genome_app_tester.py** - Test genome applications in various scenarios
- **quantum_genome_tester.py** - Test quantum genome fitness and stability
- **extreme_frontier_test.py** - Test genomes under extreme conditions

### Research Tools
- **comprehensive_analysis.py** - Full system analysis with multiple metrics
- **quantum_data_analysis.py** - Analyze quantum datasets and statistics
- **quantum_research.py** - General quantum research utilities

## Usage

Use the **Analysis & Research** workflow to run the featured analysis (quantum data analysis), or run any tool directly:
```bash
python analysis_tools/compare_all_genomes.py
python analysis_tools/deep_genome_analysis.py
python analysis_tools/comprehensive_analysis.py
# ... or any other analysis tool
```

## Prerequisites

Some analysis tools require evolved genomes to be present. Run the evolution engine first:
```bash
python evolution_engine/quantum_genetic_agents.py
```

## Output

Analysis results are typically saved as:
- PNG images in the root directory
- Text reports in the root directory
- Console output with colored statistics
