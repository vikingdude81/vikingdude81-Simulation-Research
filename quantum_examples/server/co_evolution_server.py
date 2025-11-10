
"""
🧬 Co-Evolution Dashboard
Multiple genomes evolve together, learning from successful interactions
"""

from flask import Flask, render_template, jsonify
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evolution_engine.quantum_genetic_agents import QuantumAgent, create_genome, crossover, mutate
import numpy as np
import json
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Global state
population = []
generation = 0
is_running = False
evolution_history = []

class CoEvolvingAgent:
    """Agent that learns from interactions with other agents"""
    def __init__(self, genome_id, genome):
        self.genome_id = genome_id
        self.genome = genome
        self.agent = QuantumAgent(genome_id, genome)
        self.timestep = 0
        self.fitness_history = []
        self.interaction_count = 0
        self.successful_interactions = 0
        
    def tick(self):
        """Execute one timestep"""
        self.timestep += 1
        self.agent.evolve(self.timestep)
        current_fitness = float(self.agent.traits[3])
        self.fitness_history.append(current_fitness)
        return current_fitness
    
    def interact_with(self, other_agent):
        """Interact with another agent - learn from high performers"""
        self.interaction_count += 1
        
        my_fitness = self.fitness_history[-1] if self.fitness_history else 0
        their_fitness = other_agent.fitness_history[-1] if other_agent.fitness_history else 0
        
        # If other agent is significantly better, learn from them
        if their_fitness > my_fitness * 1.5:
            # Crossover genomes
            new_genome = crossover(self.genome, other_agent.genome)
            new_genome = mutate(new_genome, mutation_rate=0.1)
            
            # Test new genome
            test_agent = QuantumAgent(999, new_genome)
            for t in range(1, min(50, self.timestep)):
                test_agent.evolve(t)
            test_fitness = test_agent.get_final_fitness()
            
            # Adopt if better
            if test_fitness > my_fitness:
                self.genome = new_genome
                self.agent = QuantumAgent(self.genome_id, new_genome)
                self.successful_interactions += 1
                return True
        return False

def background_evolution():
    """Background thread for co-evolution"""
    global is_running, population, generation, evolution_history
    
    while is_running:
        generation += 1
        
        # Each agent ticks
        for agent in population:
            agent.tick()
        
        # Random interactions
        if len(population) >= 2:
            for _ in range(len(population) // 2):
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                population[idx1].interact_with(population[idx2])
        
        # Record generation stats
        fitness_scores = [a.fitness_history[-1] for a in population if a.fitness_history]
        if fitness_scores:
            evolution_history.append({
                'generation': generation,
                'max_fitness': max(fitness_scores),
                'mean_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores),
                'timestamp': datetime.now().isoformat()
            })
        
        # Keep only last 500 generations
        if len(evolution_history) > 500:
            evolution_history.pop(0)
        
        time.sleep(0.2)

@app.route('/')
def index():
    """Co-evolution dashboard"""
    return render_template('co_evolution.html')

@app.route('/api/add_genome/<filename>')
def add_genome(filename):
    """Add a genome file to the population"""
    global population
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        genome = [
            data['genome']['mutation_rate'],
            data['genome']['oscillation_freq'],
            data['genome']['decoherence_rate'],
            data['genome']['phase_offset']
        ]
        
        genome_id = len(population)
        agent = CoEvolvingAgent(genome_id, genome)
        population.append(agent)
        
        return jsonify({
            'status': 'success',
            'genome_id': genome_id,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/start')
def start_evolution():
    """Start co-evolution"""
    global is_running
    
    if not is_running and len(population) >= 2:
        is_running = True
        thread = threading.Thread(target=background_evolution, daemon=True)
        thread.start()
        return jsonify({'status': 'running'})
    elif len(population) < 2:
        return jsonify({'status': 'error', 'message': 'Need at least 2 genomes'}), 400
    return jsonify({'status': 'already_running'})

@app.route('/api/stop')
def stop_evolution():
    """Stop co-evolution"""
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})

@app.route('/api/status')
def get_status():
    """Get current status"""
    agents_data = []
    for agent in population:
        agents_data.append({
            'genome_id': agent.genome_id,
            'fitness': agent.fitness_history[-1] if agent.fitness_history else 0,
            'timestep': agent.timestep,
            'interactions': agent.interaction_count,
            'successful_learns': agent.successful_interactions,
            'genome': agent.genome
        })
    
    return jsonify({
        'running': is_running,
        'generation': generation,
        'population_size': len(population),
        'agents': agents_data,
        'history': evolution_history[-100:]  # Last 100 generations
    })

@app.route('/api/export_best')
def export_best():
    """Export the best performing genome"""
    if not population:
        return jsonify({'error': 'No population'}), 404
    
    best_agent = max(population, key=lambda a: a.fitness_history[-1] if a.fitness_history else 0)
    
    export_data = {
        'genome': {
            'mutation_rate': float(best_agent.genome[0]),
            'oscillation_freq': float(best_agent.genome[1]),
            'decoherence_rate': float(best_agent.genome[2]),
            'phase_offset': float(best_agent.genome[3])
        },
        'export_timestamp': datetime.now().isoformat(),
        'metadata': {
            'type': 'co_evolved',
            'generation': generation,
            'fitness': float(best_agent.fitness_history[-1]),
            'interactions': best_agent.interaction_count,
            'successful_learns': best_agent.successful_interactions
        }
    }
    
    filename = f'co_evolved_best_gen_{generation}.json'
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return jsonify({'status': 'success', 'filename': filename})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    
    print("\n" + "=" * 80)
    print("🧬 CO-EVOLUTION DASHBOARD")
    print("=" * 80)
    print(f"\n📊 Server starting on http://0.0.0.0:{port}")
    print("\n💡 Features:")
    print("   • Multiple genomes evolve together")
    print("   • Agents learn from successful peers")
    print("   • Real-time fitness tracking")
    print("   • Export co-evolved champions")
    print("\n" + "=" * 80 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
