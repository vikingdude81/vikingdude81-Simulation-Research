# Server

This directory contains web servers for deploying and monitoring evolved genomes.

## Available Servers

### Genome Deployment Server
**genome_deployment_server.py**

Flask-based REST API and web dashboard for genome deployment and monitoring.

**Features:**
- Web dashboard UI for genome management
- Deploy best individual, averaged ensemble, or custom genomes
- Real-time performance monitoring
- Historical performance tracking
- Export deployment configurations

**API Endpoints:**
- `GET /` - Dashboard UI
- `POST /api/deploy` - Deploy best individual genome
- `POST /api/deploy_averaged` - Deploy averaged ensemble genome
- `POST /api/deploy_file/<filename>` - Deploy any genome file
- `POST /api/undeploy/<genome_id>` - Remove deployed genome
- `GET /api/list_genomes` - List all available genome files
- `GET /api/start` - Start evolution simulation
- `GET /api/stop` - Stop simulation
- `GET /api/status` - Get current status
- `GET /api/history/<genome_id>` - Get performance history
- `GET /api/export_config` - Export deployment config

**Usage:**
```bash
python server/genome_deployment_server.py
```
Access at: http://0.0.0.0:5000

### Co-Evolution Server
**co_evolution_server.py**

Server for co-evolutionary experiments with multiple interacting populations.

**Features:**
- Multiple populations evolving together
- Competition and cooperation dynamics
- Advanced evolutionary strategies

**Usage:**
```bash
python server/co_evolution_server.py
```

## Prerequisites

The servers require evolved genomes to function properly:
```bash
python evolution_engine/quantum_genetic_agents.py
```

This will generate:
- `best_individual_genome.json`
- `averaged_ensemble_genome.json`

## Templates

The `templates/` subdirectory contains HTML templates for the web dashboard.

## Configuration

Server configuration is in `config.py`:
- Host: 0.0.0.0 (accessible from all interfaces)
- Port: 5000 (or from PORT environment variable)
- Debug: False (production mode)
