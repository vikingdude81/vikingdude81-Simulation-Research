
"""
🚀 Quantum Genome Deployment Server
Real-time monitoring dashboard for deployed evolved genomes
"""

from flask import Flask, render_template, jsonify, request
from quantum_genetic_agents import QuantumAgent, load_genome
import numpy as np
import json
from datetime import datetime
import threading
import time
from pathlib import Path

app = Flask(__name__)

# Genome directory configuration
GENOME_DIR = Path('data/genomes/production')

# Global state
deployed_agents = {}
performance_history = []
is_running = False
agents_lock = threading.Lock()  # Protect deployed_agents from race conditions

class DeployedGenome:
    """Production deployment wrapper for evolved genomes"""
    MAX_METRICS = 500  # Prevent memory leak by capping metrics history
    
    def __init__(self, genome_id, genome, environment='production'):
        self.genome_id = genome_id
        self.genome = genome
        self.environment = environment
        self.agent = QuantumAgent(genome_id, genome, environment)
        self.timestep = 0
        self.metrics = {
            'energy': [],
            'coherence': [],
            'fitness': [],
            'uptime': 0,
            'stability_score': 0
        }
        
    def tick(self):
        """Execute one timestep"""
        self.timestep += 1
        self.agent.evolve(self.timestep)
        
        # Record metrics with size limit to prevent memory leak
        traits = self.agent.traits
        self.metrics['energy'].append(float(traits[0]))
        self.metrics['coherence'].append(float(traits[1]))
        self.metrics['fitness'].append(float(traits[3]))
        
        # Trim arrays to prevent unbounded growth
        for key in ['energy', 'coherence', 'fitness']:
            if len(self.metrics[key]) > self.MAX_METRICS:
                self.metrics[key].pop(0)
        
        self.metrics['uptime'] = self.timestep
        
        # Calculate stability (lower variance = more stable)
        if len(self.metrics['fitness']) > 10:
            recent_fitness = self.metrics['fitness'][-10:]
            self.metrics['stability_score'] = 1.0 / (1.0 + np.std(recent_fitness))
        
        return self.get_current_state()
    
    def get_current_state(self):
        """Get current state for API"""
        return {
            'genome_id': self.genome_id,
            'timestep': self.timestep,
            'energy': float(self.agent.traits[0]),
            'coherence': float(self.agent.traits[1]),
            'phase': float(self.agent.traits[2]),
            'fitness': float(self.agent.traits[3]),
            'stability': self.metrics['stability_score'],
            'uptime': self.metrics['uptime']
        }

def background_runner():
    """Background thread to run deployed genomes"""
    global is_running, deployed_agents, performance_history
    
    while is_running:
        timestamp = datetime.now().isoformat()
        current_states = {}
        
        # Use lock to prevent "dictionary changed size during iteration" error
        with agents_lock:
            # Create a copy of items to safely iterate
            agents_snapshot = list(deployed_agents.items())
        
        # Process agents outside the lock to minimize lock time
        for genome_id, deployed in agents_snapshot:
            state = deployed.tick()
            current_states[genome_id] = state
        
        # Record snapshot
        performance_history.append({
            'timestamp': timestamp,
            'states': current_states
        })
        
        # Keep only last 500 snapshots to prevent memory issues
        if len(performance_history) > 500:
            performance_history.pop(0)
        
        time.sleep(0.1)  # 10 ticks per second

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard (alternative route)"""
    return render_template('dashboard.html')

@app.route('/frontier')
def frontier():
    """Extreme frontier results viewer"""
    from flask import send_from_directory
    return send_from_directory('.', 'view_frontier.html')

@app.route('/comparison')
def comparison():
    """Co-evolution comparison dashboard"""
    from flask import send_from_directory
    return send_from_directory('.', 'deploy_comparison.html')

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization images"""
    from flask import send_from_directory
    import os
    # Security: prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 400
    return send_from_directory('visualizations', filename)

@app.route('/api/deploy', methods=['POST'])
def deploy_genome():
    """Deploy a genome from file"""
    global deployed_agents
    
    try:
        # Load best individual genome
        genome, metadata = load_genome('best_individual_genome.json')
        genome_id = 'best_individual'
        
        with agents_lock:
            deployed_agents[genome_id] = DeployedGenome(genome_id, genome)
        
        return jsonify({
            'status': 'success',
            'genome_id': genome_id,
            'metadata': metadata
        })
    except FileNotFoundError:
        return jsonify({
            'status': 'error', 
            'message': 'Genome file not found. Run quantum_genetic_agents.py first to generate genomes!'
        }), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/deploy_averaged', methods=['POST'])
def deploy_averaged():
    """Deploy averaged ensemble genome"""
    global deployed_agents
    
    try:
        genome, metadata = load_genome('averaged_ensemble_genome.json')
        genome_id = 'averaged_ensemble'
        
        with agents_lock:
            deployed_agents[genome_id] = DeployedGenome(genome_id, genome)
        
        return jsonify({
            'status': 'success',
            'genome_id': genome_id,
            'metadata': metadata
        })
    except FileNotFoundError:
        return jsonify({
            'status': 'error',
            'message': 'Genome file not found. Run quantum_genetic_agents.py first to generate genomes!'
        }), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/start')
def start_system():
    """Start the deployment system"""
    global is_running
    
    if not is_running:
        is_running = True
        thread = threading.Thread(target=background_runner, daemon=True)
        thread.start()
        return jsonify({'status': 'running'})
    
    return jsonify({'status': 'already_running'})

@app.route('/api/stop')
def stop_system():
    """Stop the deployment system"""
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})

@app.route('/api/status')
def get_status():
    """Get current system status"""
    states = {}
    
    with agents_lock:
        for genome_id, deployed in deployed_agents.items():
            states[genome_id] = deployed.get_current_state()
        deployed_count = len(deployed_agents)
    
    return jsonify({
        'running': is_running,
        'deployed_count': deployed_count,
        'agents': states,
        'history_length': len(performance_history)
    })

@app.route('/api/history/<genome_id>')
def get_history(genome_id):
    """Get performance history for a genome"""
    with agents_lock:
        if genome_id not in deployed_agents:
            return jsonify({'error': 'Genome not found'}), 404
        
        deployed = deployed_agents[genome_id]
        
        # Return last 100 datapoints
        history_slice = {
            'energy': deployed.metrics['energy'][-100:],
            'coherence': deployed.metrics['coherence'][-100:],
            'fitness': deployed.metrics['fitness'][-100:],
            'timesteps': list(range(max(0, len(deployed.metrics['fitness']) - 100), 
                                   len(deployed.metrics['fitness'])))
        }
    
    return jsonify(history_slice)

@app.route('/api/list_genomes')
def list_genomes():
    """List all available genome JSON files"""
    import os
    import glob
    
    # List files in genome directory
    GENOME_DIR.mkdir(parents=True, exist_ok=True)
    genome_pattern = str(GENOME_DIR / '*.json')
    genome_files = glob.glob(genome_pattern)
    genome_info = []
    
    for filepath in sorted(genome_files):
        try:
            # Security: verify file is in genome directory
            resolved_path = Path(filepath).resolve()
            if not str(resolved_path).startswith(str(GENOME_DIR.resolve())):
                continue  # Skip files outside genome directory
            
            # Get just the filename for display
            filename = os.path.basename(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Combine genome data with metadata for properties viewer
                metadata = data.get('metadata', {})
                metadata['genome'] = data.get('genome', {})
                
                genome_info.append({
                    'filename': filename,  # Only return basename
                    'exported': data.get('export_timestamp', 'Unknown'),
                    'metadata': metadata,
                    'size': os.path.getsize(filepath)
                })
        except Exception as e:
            genome_info.append({
                'filename': os.path.basename(filepath),
                'error': str(e)
            })
    
    return jsonify({
        'available_genomes': genome_info,
        'count': len(genome_info)
    })

@app.route('/api/deploy_file/<filename>', methods=['POST'])
def deploy_file(filename):
    """Deploy a specific genome file"""
    global deployed_agents
    
    try:
        import os
        
        # Security: only allow JSON files with _genome or co_evolved suffix
        if not (filename.endswith('_genome.json') or filename.endswith('.json')):
            return jsonify({
                'status': 'error',
                'message': 'Invalid filename. Must be a JSON genome file'
            }), 400
        
        # Security: prevent path traversal attacks
        # Only allow files in the genome directory (no paths, no ..)
        if '/' in filename or '\\' in filename or '..' in filename:
            return jsonify({
                'status': 'error',
                'message': 'Invalid filename. Path separators not allowed.'
            }), 400
        
        # Resolve to absolute path and verify it's in the genome directory
        genome_path = (GENOME_DIR / filename).resolve()
        
        # Ensure the resolved path is within the genome directory
        if not str(genome_path).startswith(str(GENOME_DIR.resolve())):
            return jsonify({
                'status': 'error',
                'message': 'Invalid filename. Path traversal not allowed.'
            }), 400
        
        # Verify file exists before attempting to load
        if not genome_path.exists():
            return jsonify({
                'status': 'error', 
                'message': f'Genome file {filename} not found!'
            }), 404
        
        # Load the genome using the sanitized filename
        genome, metadata = load_genome(str(genome_path))
        
        # Create genome_id from filename (basename only)
        genome_id = filename.replace('.json', '')
        
        with agents_lock:
            deployed_agents[genome_id] = DeployedGenome(genome_id, genome)
        
        return jsonify({
            'status': 'success',
            'genome_id': genome_id,
            'filename': filename,
            'metadata': metadata
        })
    except FileNotFoundError:
        return jsonify({
            'status': 'error', 
            'message': f'Genome file {filename} not found!'
        }), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/undeploy/<genome_id>', methods=['POST'])
def undeploy_genome(genome_id):
    """Remove a deployed genome"""
    global deployed_agents
    
    with agents_lock:
        if genome_id in deployed_agents:
            del deployed_agents[genome_id]
            return jsonify({
                'status': 'success',
                'message': f'Undeployed {genome_id}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Genome not found'
            }), 404

@app.route('/api/deploy_production_gen2117', methods=['POST'])
def deploy_production_gen2117():
    """Deploy Gen 2117 genome for production"""
    global deployed_agents
    
    try:
        genome, metadata = load_genome('co_evolved_best_gen_2117.json')
        genome_id = 'production_gen2117'
        
        with agents_lock:
            deployed_agents[genome_id] = DeployedGenome(genome_id, genome)
        
        return jsonify({
            'status': 'success',
            'genome_id': genome_id,
            'metadata': metadata
        })
    except FileNotFoundError:
        return jsonify({
            'status': 'error',
            'message': 'Gen 2117 genome file not found!'
        }), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/deploy_custom', methods=['POST'])
def deploy_custom():
    """Deploy a custom genome from user input"""
    global deployed_agents
    
    try:
        data = request.json
        genome = data.get('genome', [])
        
        if len(genome) != 4:
            return jsonify({
                'status': 'error',
                'message': 'Genome must have exactly 4 parameters'
            }), 400
        
        # Validate ranges
        if not (0.01 <= genome[0] <= 1.5 and 
                0.1 <= genome[1] <= 2.0 and 
                0.01 <= genome[2] <= 0.1 and 
                0.05 <= genome[3] <= 0.5):
            return jsonify({
                'status': 'error',
                'message': 'Genome parameters out of valid range'
            }), 400
        
        genome_id = f'custom_{int(time.time())}'
        filename = f'{genome_id}_genome.json'
        
        # Export custom genome to file
        metadata = {
            'type': 'custom',
            'created': datetime.now().isoformat(),
            'fitness': 0.0,  # Will be calculated after deployment
            'parameters': {
                'mutation_rate': genome[0],
                'oscillation_freq': genome[1],
                'decoherence_rate': genome[2],
                'phase_offset': genome[3]
            }
        }
        
        export_data = {
            'genome': {
                'mutation_rate': float(genome[0]),
                'oscillation_freq': float(genome[1]),
                'decoherence_rate': float(genome[2]),
                'phase_offset': float(genome[3])
            },
            'export_timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        # Save to genome directory
        GENOME_DIR.mkdir(parents=True, exist_ok=True)
        filepath = GENOME_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Deploy the genome
        with agents_lock:
            deployed_agents[genome_id] = DeployedGenome(genome_id, genome)
        
        return jsonify({
            'status': 'success',
            'genome_id': genome_id,
            'filename': filename,
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/export_config')
def export_config():
    """Export current deployment configuration"""
    config = {
        'deployed_genomes': {},
        'export_timestamp': datetime.now().isoformat()
    }
    
    with agents_lock:
        for genome_id, deployed in deployed_agents.items():
            config['deployed_genomes'][genome_id] = {
                'genome': deployed.genome,
                'environment': deployed.environment,
                'current_fitness': float(deployed.agent.traits[3]),
                'uptime': deployed.metrics['uptime']
            }
    
    return jsonify(config)

if __name__ == '__main__':
    import sys
    import os
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "=" * 80)
    print("🚀 QUANTUM GENOME DEPLOYMENT SERVER")
    print("=" * 80)
    print(f"\n📊 Server starting on http://0.0.0.0:{port}")
    print("\n💡 API Endpoints:")
    print("   GET  /              - Dashboard UI")
    print("   POST /api/deploy    - Deploy best individual genome")
    print("   POST /api/deploy_averaged - Deploy averaged ensemble genome")
    print("   POST /api/deploy_file/<filename> - Deploy any genome file")
    print("   POST /api/undeploy/<genome_id> - Remove deployed genome")
    print("   GET  /api/list_genomes - List all available genome files")
    print("   GET  /api/start     - Start evolution simulation")
    print("   GET  /api/stop      - Stop simulation")
    print("   GET  /api/status    - Get current status")
    print("   GET  /api/history/<genome_id> - Get performance history")
    print("   GET  /api/export_config - Export deployment config")
    print("\n💾 Looking for genome files in: data/genomes/production/")
    print("   - best_individual_genome.json")
    print("   - averaged_ensemble_genome.json")
    
    # Check if genome files exist
    genome_dir = Path('data/genomes/production')
    if not (genome_dir / 'best_individual_genome.json').exists():
        print("\n⚠️  WARNING: best_individual_genome.json not found!")
        print("   Run 'python quantum_genetic_agents.py' first to generate genomes.")
    if not (genome_dir / 'averaged_ensemble_genome.json').exists():
        print("⚠️  WARNING: averaged_ensemble_genome.json not found!")
        print("   Run 'python quantum_genetic_agents.py' first to generate genomes.")
    
    print("\n" + "=" * 80 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ ERROR: Port {port} is already in use!")
            print("   Kill the existing process or set a different PORT environment variable.")
            print(f"   Example: PORT=5001 python {sys.argv[0]}")
            sys.exit(1)
        raise
