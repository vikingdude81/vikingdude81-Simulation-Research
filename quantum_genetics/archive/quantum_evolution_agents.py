
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, exp
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

class QuantumAgent:
    """Agent with quantum-inspired traits that evolve over time"""
    def __init__(self, agent_id, initial_traits):
        self.id = agent_id
        self.traits = np.array(initial_traits)  # [energy, coherence, phase, fitness]
        self.history = [self.traits.copy()]
        
    def evolve(self, timestep, mutation_rate=0.1):
        """Evolve agent traits with quantum-inspired dynamics"""
        # Quantum evolution: traits oscillate and interfere
        t = timestep * 0.1
        
        # Energy evolves with damped oscillation
        self.traits[0] = self.traits[0] * np.cos(t) + mutation_rate * np.random.randn()
        
        # Coherence decays over time (decoherence)
        self.traits[1] = self.traits[1] * np.exp(-0.05 * t) + mutation_rate * np.random.randn()
        
        # Phase rotates
        self.traits[2] = (self.traits[2] + 0.2 * t) % (2 * pi)
        
        # Fitness is emergent from other traits
        self.traits[3] = abs(self.traits[0]) * self.traits[1]
        
        self.history.append(self.traits.copy())
        
    def get_state_vector(self, timestep):
        """Get feature vector for ML prediction"""
        if timestep < len(self.history):
            return self.history[timestep]
        return self.traits

class EvolutionPredictor:
    """ML model to predict agent trait evolution"""
    def __init__(self, model_type='gradient_boosting'):
        self.scaler = StandardScaler()
        
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        else:  # neural network
            self.model = MLPRegressor(hidden_layer_sizes=(50, 30, 20), max_iter=500, random_state=42)
            
        self.model_type = model_type
        
    def prepare_training_data(self, agents, lookback=3, forecast_steps=5):
        """Create training dataset from agent histories"""
        X_train = []
        y_train = []
        
        for agent in agents:
            history = np.array(agent.history)
            
            # Create sequences for time-series prediction
            for i in range(lookback, len(history) - forecast_steps):
                # Input: past 'lookback' states flattened
                past_states = history[i-lookback:i].flatten()
                
                # Target: future state after 'forecast_steps'
                future_state = history[i + forecast_steps]
                
                X_train.append(past_states)
                y_train.append(future_state)
                
        return np.array(X_train), np.array(y_train)
    
    def train(self, agents, lookback=3, forecast_steps=5):
        """Train the evolution predictor"""
        X, y = self.prepare_training_data(agents, lookback, forecast_steps)
        
        if len(X) == 0:
            print("⚠ Warning: Not enough data to train")
            return
            
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict_evolution(self, agent, lookback=3, n_steps=10):
        """Predict future evolution of agent traits"""
        history = np.array(agent.history)
        
        if len(history) < lookback:
            return None
            
        predictions = []
        current_history = history.copy()
        
        for _ in range(n_steps):
            # Use last 'lookback' states
            past_states = current_history[-lookback:].flatten()
            past_scaled = self.scaler.transform([past_states])
            
            # Predict next state
            next_state = self.model.predict(past_scaled)[0]
            predictions.append(next_state)
            
            # Update history for next prediction
            current_history = np.vstack([current_history, next_state])
            
        return np.array(predictions)

def generate_agent_population(n_agents=20, n_timesteps=50):
    """Create a population of evolving agents"""
    print(f"🧬 Generating population of {n_agents} agents...")
    print(f"   Simulating {n_timesteps} timesteps of evolution\n")
    
    agents = []
    
    for i in range(n_agents):
        # Random initial traits: [energy, coherence, phase, fitness]
        initial_traits = [
            np.random.uniform(-2, 2),      # energy
            np.random.uniform(0.5, 1.0),   # coherence
            np.random.uniform(0, 2*pi),    # phase
            0.0                             # fitness (will be calculated)
        ]
        
        agent = QuantumAgent(i, initial_traits)
        
        # Evolve agent over time
        for t in range(1, n_timesteps):
            agent.evolve(t, mutation_rate=0.15)
            
        agents.append(agent)
        
    return agents

def visualize_agent_evolution(agents, predictor, agent_idx=0):
    """Visualize actual vs predicted evolution for an agent"""
    print(f"\n🔮 Predicting evolution for Agent #{agent_idx}...")
    
    agent = agents[agent_idx]
    lookback = 3
    forecast_horizon = 15
    
    # Split history: train on first 70%, predict remaining
    split_point = int(len(agent.history) * 0.7)
    
    # Create a test agent with partial history
    test_agent = QuantumAgent(agent.id, agent.history[0])
    test_agent.history = agent.history[:split_point]
    
    # Predict future evolution
    predictions = predictor.predict_evolution(test_agent, lookback=lookback, n_steps=forecast_horizon)
    
    if predictions is None:
        print("⚠ Not enough history for prediction")
        return
        
    # Actual future evolution
    actual_future = np.array(agent.history[split_point:split_point+forecast_horizon])
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    trait_names = ['Energy', 'Coherence', 'Phase', 'Fitness']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (trait_name, color) in enumerate(zip(trait_names, colors)):
        ax = axes[i // 2, i % 2]
        
        # Historical data
        history_vals = [state[i] for state in agent.history[:split_point]]
        ax.plot(range(len(history_vals)), history_vals, 'o-', 
               color=color, lw=2, markersize=4, label='Historical', alpha=0.7)
        
        # Actual future
        future_times = range(split_point, split_point + len(actual_future))
        actual_vals = actual_future[:, i]
        ax.plot(future_times, actual_vals, 's-', 
               color='black', lw=2, markersize=5, label='Actual future')
        
        # Predicted future
        pred_vals = predictions[:, i]
        ax.plot(future_times, pred_vals, '^--', 
               color='red', lw=2, markersize=5, label='ML Prediction', alpha=0.7)
        
        # Error bars
        errors = np.abs(actual_vals - pred_vals)
        ax.fill_between(future_times, 
                        pred_vals - errors, 
                        pred_vals + errors,
                        alpha=0.2, color='red')
        
        ax.axvline(split_point, color='gray', linestyle='--', alpha=0.5, label='Prediction start')
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel(trait_name, fontsize=10)
        ax.set_title(f'{trait_name} Evolution - Agent #{agent_idx}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
    plt.suptitle(f'Agent Evolution Prediction with {predictor.model_type.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'agent_evolution_prediction_{agent_idx}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: agent_evolution_prediction_{agent_idx}.png")
    plt.close()
    
    # Calculate prediction accuracy
    mse_per_trait = np.mean((actual_future - predictions)**2, axis=0)
    print(f"\n📊 Prediction Error (MSE):")
    for trait_name, mse in zip(trait_names, mse_per_trait):
        print(f"   {trait_name}: {mse:.6f}")

def analyze_population_dynamics(agents):
    """Analyze emergent population-level dynamics"""
    print("\n📊 Analyzing population dynamics...")
    
    n_timesteps = len(agents[0].history)
    n_agents = len(agents)
    
    # Extract population statistics over time
    mean_traits = np.zeros((n_timesteps, 4))
    std_traits = np.zeros((n_timesteps, 4))
    
    for t in range(n_timesteps):
        traits_at_t = np.array([agent.history[t] for agent in agents])
        mean_traits[t] = np.mean(traits_at_t, axis=0)
        std_traits[t] = np.std(traits_at_t, axis=0)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    trait_names = ['Energy', 'Coherence', 'Phase', 'Fitness']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (trait_name, color) in enumerate(zip(trait_names, colors)):
        ax = axes[i // 2, i % 2]
        
        # Mean trajectory
        ax.plot(mean_traits[:, i], lw=3, color=color, label='Population mean')
        
        # Standard deviation envelope
        ax.fill_between(range(n_timesteps),
                        mean_traits[:, i] - std_traits[:, i],
                        mean_traits[:, i] + std_traits[:, i],
                        alpha=0.3, color=color, label='±1 std dev')
        
        # Individual agents (sample)
        for agent_idx in range(min(5, n_agents)):
            agent_trait = [agent.history[t][i] for t in range(n_timesteps)]
            ax.plot(agent_trait, alpha=0.3, lw=1, color='gray')
        
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel(trait_name, fontsize=10)
        ax.set_title(f'Population {trait_name} Dynamics', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Population Dynamics - {n_agents} Quantum Agents', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('population_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: population_dynamics.png")
    plt.close()

def compare_ml_models(agents):
    """Compare different ML models for evolution prediction"""
    print("\n🤖 Comparing ML models for evolution prediction...")
    
    models = {
        'Gradient Boosting': EvolutionPredictor('gradient_boosting'),
        'Random Forest': EvolutionPredictor('random_forest'),
        'Neural Network': EvolutionPredictor('neural_network')
    }
    
    # Train all models
    for name, model in models.items():
        print(f"   Training {name}...")
        model.train(agents, lookback=3, forecast_steps=5)
    
    # Test on a held-out agent
    test_agent_idx = len(agents) - 1
    test_agent = agents[test_agent_idx]
    
    # Predict with each model
    results = {}
    for name, model in models.items():
        test_agent_partial = QuantumAgent(test_agent.id, test_agent.history[0])
        test_agent_partial.history = test_agent.history[:30]
        
        predictions = model.predict_evolution(test_agent_partial, lookback=3, n_steps=10)
        if predictions is not None:
            results[name] = predictions
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    trait_names = ['Energy', 'Coherence', 'Phase', 'Fitness']
    colors = {'Gradient Boosting': 'red', 'Random Forest': 'green', 'Neural Network': 'blue'}
    
    for i, trait_name in enumerate(trait_names):
        ax = axes[i // 2, i % 2]
        
        # Actual values
        actual = np.array([test_agent.history[t][i] for t in range(30, 40)])
        ax.plot(range(30, 40), actual, 'ko-', lw=3, markersize=7, label='Actual', zorder=10)
        
        # Model predictions
        for name, predictions in results.items():
            pred_vals = predictions[:, i]
            ax.plot(range(30, 40), pred_vals, 'o--', lw=2, 
                   color=colors[name], label=name, alpha=0.7)
        
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel(trait_name, fontsize=10)
        ax.set_title(f'{trait_name} - Model Comparison', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('ML Model Comparison for Agent Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ml_model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ml_model_comparison.png")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("  QUANTUM-INSPIRED EVOLUTION PREDICTION FOR AGENT SYSTEMS")
    print("=" * 70)
    print("\n🧬 Simulating evolving agent population with ML prediction\n")
    
    # Generate population
    n_agents = 25
    n_timesteps = 50
    agents = generate_agent_population(n_agents, n_timesteps)
    
    print(f"✓ Created {n_agents} agents with {n_timesteps} timesteps of evolution\n")
    
    # Train predictor
    print("🎯 Training evolution predictor...")
    predictor = EvolutionPredictor('gradient_boosting')
    predictor.train(agents, lookback=3, forecast_steps=5)
    print("✓ Training complete!\n")
    
    # Visualize individual agent prediction
    visualize_agent_evolution(agents, predictor, agent_idx=0)
    visualize_agent_evolution(agents, predictor, agent_idx=5)
    
    # Analyze population
    analyze_population_dynamics(agents)
    
    # Compare models
    compare_ml_models(agents)
    
    print("\n" + "=" * 70)
    print("✨ EVOLUTION PREDICTION COMPLETE!")
    print("=" * 70)
    print("\n💡 What you can do with this system:")
    print("   • Predict how your agents will evolve before running simulations")
    print("   • Identify agents with optimal trait trajectories")
    print("   • Forecast population-level emergent behaviors")
    print("   • Test different evolution rules and compare outcomes")
    print("   • Integrate with your existing agent systems")
    print("\n🚀 Next steps:")
    print("   • Replace QuantumAgent with your own agent class")
    print("   • Add your specific traits (speed, intelligence, cooperation, etc.)")
    print("   • Use predictions to optimize selection/breeding strategies")
    print("   • Scale to larger populations (100s or 1000s of agents)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
