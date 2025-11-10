"""
🤖 GPU-ACCELERATED ML GOVERNANCE
=================================

Adapts your existing crypto ML infrastructure for governance learning!

Uses:
- Your GPU setup (RTX 4070 Ti detected)
- PyTorch neural networks (already configured)
- Multi-task learning framework
- Attention mechanisms
- Your training loops and optimization

This is a CUSTOM RL implementation using your existing ML stack
(Alternative to stable-baselines3 - uses what you already have!)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
import json
from datetime import datetime
from pathlib import Path

# Rich terminal UI for progress tracking
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box

from ultimate_echo_simulation import UltimateEchoSimulation
from government_styles import GovernmentStyle

console = Console()

# Check GPU availability
HAS_PYTORCH = torch.cuda.is_available()
if HAS_PYTORCH:
    DEVICE = torch.device('cuda')
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("⚠️  Using CPU (GPU not available)")


class PolicyNetwork(nn.Module):
    """
    Neural network for governance policy.
    
    Similar to your crypto price prediction networks,
    but outputs governance actions instead of price predictions.
    
    Architecture:
    - Input: 10D state vector (population, cooperation, wealth, etc.)
    - Hidden: 2 layers with dropout (like your models)
    - Output: 7D action probabilities + 1D value estimate
    """
    
    def __init__(self, state_dim=10, hidden_dim=256, action_dim=7):
        super().__init__()
        
        # Shared feature extractor (like your crypto feature networks)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Policy head (action probabilities)
        # Use LogSoftmax + exp for numerical stability
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass through network"""
        features = self.feature_net(state)
        logits = self.policy_head(features)
        # Use log_softmax for numerical stability, then exp
        log_probs = torch.log_softmax(logits, dim=-1)
        action_probs = torch.exp(log_probs).clamp(min=1e-8, max=1.0)  # Prevent NaN
        # Renormalize to ensure sum=1
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        state_value = self.value_head(features)
        return action_probs, state_value


class GovernanceAgent:
    """
    RL agent for learning governance policies.
    
    Uses Actor-Critic algorithm (simple but effective)
    Similar to your multi-task learning setup!
    """
    
    def __init__(self, state_dim=10, action_dim=7, learning_rate=0.001):
        self.policy_net = PolicyNetwork(state_dim, 256, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience memory (like replay buffer)
        self.memory = deque(maxlen=10000)
        
        # Reward weights
        self.reward_weights = {
            'cooperation': 2.0,
            'population': 1.0,
            'diversity': 1.5,
            'inequality': -1.0,
            'intervention_cost': -0.1
        }
        
    def select_action(self, state, explore=True, epsilon=0.1):
        """
        Select action using policy network.
        
        Args:
            state: Current state vector
            explore: Whether to explore (training) or exploit (evaluation)
            epsilon: Exploration rate (like your dropout uncertainty)
        
        Returns:
            action: Selected action index
            log_prob: Log probability of action (for training)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor)
        
        # Epsilon-greedy exploration (like your dropout sampling)
        if explore and random.random() < epsilon:
            action = random.randint(0, 6)
        else:
            action_probs_np = action_probs.cpu().numpy()[0]
            action = np.random.choice(len(action_probs_np), p=action_probs_np)
        
        # Calculate log probability for training
        log_prob = torch.log(action_probs[0, action])
        
        return action, log_prob
    
    def calculate_reward(self, state):
        """
        Calculate reward from state (like your loss functions)
        
        Multi-objective reward combining:
        - Cooperation rate (main goal)
        - Population health
        - Genetic diversity
        - Inequality (penalize)
        - Intervention cost (small penalty)
        """
        population = state[0]
        cooperation = state[1]
        wealth = state[2]
        gini = state[3]
        diversity = state[4]
        
        # Multi-objective reward
        reward = (
            self.reward_weights['cooperation'] * cooperation +
            self.reward_weights['population'] * population +
            self.reward_weights['diversity'] * diversity +
            self.reward_weights['inequality'] * (1 - gini) +  # Reward low inequality
            self.reward_weights['intervention_cost'] * 0  # Will add based on action
        )
        
        # Bonus for very high cooperation with high diversity
        if cooperation > 0.8 and diversity > 0.5:
            reward += 1.0
        
        # Penalty for extinction risk
        if population < 0.1:  # Less than 500 agents
            reward -= 5.0
        
        return reward
    
    def train_step(self, batch_size=32, gamma=0.99):
        """
        Training step (like your crypto model training)
        
        Uses Actor-Critic algorithm:
        - Actor: learns policy (action selection)
        - Critic: learns value (state evaluation)
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first, then to tensors (much faster!)
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(DEVICE)
        actions = torch.LongTensor(np.array([x[1] for x in batch])).to(DEVICE)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).to(DEVICE)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(DEVICE)
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).to(DEVICE)
        
        # Forward pass
        action_probs, state_values = self.policy_net(states)
        _, next_state_values = self.policy_net(next_states)
        
        # Calculate TD targets (like your multi-task targets)
        td_targets = rewards + gamma * next_state_values.squeeze() * (1 - dones)
        
        # Calculate advantages
        advantages = td_targets.detach() - state_values.squeeze()
        
        # Policy loss (actor) - with numerical stability
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        # Clamp to prevent log(0) = -inf
        selected_probs = torch.clamp(selected_probs, min=1e-8, max=1.0)
        log_probs = torch.log(selected_probs)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss (critic)
        value_loss = nn.MSELoss()(state_values.squeeze(), td_targets.detach())
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Check for NaN before backward pass
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️  Warning: NaN/Inf detected in loss, skipping this batch")
            return None
        
        # Backward pass (like your training loops)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }


def get_state_from_sim(sim):
    """Extract state vector from simulation"""
    if not sim.agents:
        return np.zeros(10, dtype=np.float32)
    
    # Calculate metrics
    population = len(sim.agents)
    cooperators = sum(1 for a in sim.agents if a.traits.strategy == 1)
    cooperation_rate = cooperators / population if population > 0 else 0
    
    wealth_values = [a.wealth for a in sim.agents]
    avg_wealth = np.mean(wealth_values) if wealth_values else 0
    
    # Gini coefficient
    sorted_wealth = np.sort(wealth_values)
    n = len(sorted_wealth)
    if n > 0 and np.sum(sorted_wealth) > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_wealth)) / (n * np.sum(sorted_wealth)) - (n + 1) / n
    else:
        gini = 0
    
    # Diversity
    chromosomes = [tuple(a.chromosome.flatten()) for a in sim.agents]
    unique = len(set(chromosomes))
    diversity = unique / population if population > 0 else 0
    
    # Construct state vector (10D)
    state = np.array([
        population / 5000,  # Normalize to [0, 1]
        cooperation_rate,
        avg_wealth / 100,
        gini,
        diversity,
        0.5,  # Population growth (placeholder)
        0.5,  # Cooperation change (placeholder)
        0.5,  # Wealth change (placeholder)
        0.0,  # Generation (will be updated)
        0.0   # Time since intervention (will be updated)
    ], dtype=np.float32)
    
    return state


def apply_action_to_sim(sim, action):
    """Apply governance action to simulation"""
    intervention_cost = 0
    
    if action == 0:
        # Do nothing (laissez-faire)
        pass
    
    elif action == 1:
        # Welfare redistribution
        rich = [a for a in sim.agents if a.wealth > 50]
        poor = [a for a in sim.agents if a.wealth < 10]
        if rich and poor:
            tax_rate = 0.3
            total_tax = sum(a.wealth * tax_rate for a in rich)
            for a in rich:
                a.wealth *= (1 - tax_rate)
            per_person = total_tax / len(poor)
            for a in poor:
                a.wealth += per_person
            intervention_cost = 0.1
    
    elif action == 2:
        # Universal stimulus
        for a in sim.agents:
            a.wealth += 10
        intervention_cost = 0.2
    
    elif action == 3:
        # Targeted stimulus (poor only)
        poor = [a for a in sim.agents if a.wealth < 15]
        for a in poor:
            a.wealth += 15
        intervention_cost = 0.1
    
    elif action == 4:
        # Remove worst defectors
        defectors = [a for a in sim.agents if a.traits.strategy == 0]
        if defectors:
            defectors.sort(key=lambda a: a.wealth)
            to_remove = defectors[:max(1, len(defectors) // 10)]
            for a in to_remove:
                a.wealth = -9999
            intervention_cost = 0.3
    
    elif action == 5:
        # Tax defectors
        defectors = [a for a in sim.agents if a.traits.strategy == 0]
        for a in defectors:
            a.wealth *= 0.7
        intervention_cost = 0.1
    
    elif action == 6:
        # Boost cooperators
        cooperators = [a for a in sim.agents if a.traits.strategy == 1]
        for a in cooperators:
            a.wealth += 5
        intervention_cost = 0.1
    
    return intervention_cost


def train_governance_agent(
    episodes=100,
    max_generations=300,
    save_path='ml_governance_model.pth'
):
    """
    Train governance agent (like your main training loops!)
    
    Args:
        episodes: Number of training episodes
        max_generations: Max generations per episode
        save_path: Where to save trained model
    """
    console.clear()
    console.rule("[bold cyan]🤖 ML GOVERNANCE TRAINING - GPU Accelerated", style="cyan")
    console.print()
    
    # Training info panel
    info_table = Table(show_header=False, box=box.ROUNDED)
    info_table.add_row("[cyan]Device:", f"[green]{DEVICE}")
    info_table.add_row("[cyan]Episodes:", f"[yellow]{episodes}")
    info_table.add_row("[cyan]Generations/Episode:", f"[yellow]{max_generations}")
    info_table.add_row("[cyan]Model Save Path:", f"[blue]{save_path}")
    console.print(Panel(info_table, title="[bold]Training Configuration", border_style="cyan"))
    console.print()
    
    # Create agent
    console.print("[cyan]⚙️  Initializing agent...[/cyan]")
    agent = GovernanceAgent()
    console.print("[green]✓ Agent created![/green]\n")
    
    # Training history
    history = {
        'episode_rewards': [],
        'episode_cooperation': [],
        'episode_diversity': [],
        'losses': []
    }
    
    action_names = [
        "Do Nothing",
        "Welfare",
        "Universal Stimulus",
        "Targeted Stimulus",
        "Remove Defectors",
        "Tax Defectors",
        "Boost Cooperators"
    ]
    
    # Training setup panel
    setup_panel = Panel(
        "[bold cyan]🤖 ML GOVERNANCE TRAINING[/bold cyan]\n\n"
        f"[yellow]Configuration:[/yellow]\n"
        f"  • Episodes: [green]{episodes}[/green]\n"
        f"  • Max Generations/Episode: [green]{max_generations}[/green]\n"
        f"  • Population Size: [green]200 agents[/green]\n"
        f"  • Learning Rate: [green]0.001[/green]\n"
        f"  • GPU: [green]{'CUDA (' + torch.cuda.get_device_name(0) + ')' if HAS_PYTORCH else 'CPU'}[/green]\n\n"
        f"[yellow]State Space (10D):[/yellow]\n"
        f"  Population • Cooperation • Wealth • Age • Diversity\n"
        f"  Inequality (Gini) • Defector Ratio • Metabolism • Vision • Lifespan\n\n"
        f"[yellow]Action Space (7 Actions):[/yellow]\n"
        f"  🔵 Do Nothing • 🏛️ Welfare • 💰 Stimulus • 🛡️ Enforcement\n"
        f"  💎 Tax Rich • 🤝 Support Cooperators • 🚨 Emergency Revival\n\n"
        f"[yellow]Target:[/yellow] Beat human governments:\n"
        f"  Authoritarian: [magenta]99.8%[/magenta] | Mixed Economy: [cyan]75.0%[/cyan] | Welfare: [blue]50.4%[/blue]\n"
        f"  Central Banker: [yellow]36.8%[/yellow] | Laissez-Faire: [white]34.2%[/white]",
        title="[bold green]Training Setup",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(setup_panel)
    console.print()
    
    # Training progress with rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        episode_task = progress.add_task("[cyan]Training Episodes", total=episodes)
        
        # Create live dashboard
        from rich.layout import Layout
        from rich.live import Live
        
        for episode in range(episodes):
            # Create new simulation
            sim = UltimateEchoSimulation(
                initial_size=200,
                grid_size=(75, 75),
                government_style=GovernmentStyle.LAISSEZ_FAIRE
            )
            
            episode_reward = 0
            episode_actions = []
            state_snapshots = []  # Track state evolution
            
            # Generation progress
            gen_task = progress.add_task(
                f"[yellow]Episode {episode+1}/{episodes} - Generations", 
                total=max_generations
            )
            
            # Run episode with live state tracking
            for gen in range(max_generations):
                # Get current state
                state = get_state_from_sim(sim)
                state[8] = gen / max_generations
                
                # Select action
                action, log_prob = agent.select_action(state, explore=True, epsilon=0.1)
                episode_actions.append(action)
                
                # Apply action
                intervention_cost = apply_action_to_sim(sim, action)
                
                # Step simulation
                sim.step()
                
                # Get new state
                next_state = get_state_from_sim(sim)
                next_state[8] = (gen + 1) / max_generations
                
                # Calculate reward
                reward = agent.calculate_reward(next_state)
                reward -= intervention_cost * agent.reward_weights['intervention_cost']
                episode_reward += reward
                
                # Track state snapshots every 10 generations
                if gen % 10 == 0:
                    state_snapshots.append({
                        'gen': gen,
                        'pop': len(sim.agents),
                        'coop': next_state[1] * 100,
                        'wealth': next_state[2],
                        'diversity': next_state[4],
                        'gini': next_state[5]
                    })
                
                # Check if done
                done = (gen >= max_generations - 1) or (len(sim.agents) < 10)
                
                # Store experience
                agent.memory.append((state, action, reward, next_state, float(done)))
                
                # Train if enough experience
                if len(agent.memory) >= 32:
                    loss_dict = agent.train_step(batch_size=32)
                    if loss_dict and episode > 0 and gen % 10 == 0:
                        history['losses'].append(loss_dict)
                
                # Update progress with detailed info every 25 generations
                if gen % 25 == 0 and gen > 0:
                    progress.console.print(
                        f"   [dim]Gen {gen:3d}: Pop={len(sim.agents):3d} | "
                        f"Coop={next_state[1]*100:5.1f}% | "
                        f"Div={next_state[4]:.3f} | "
                        f"Reward={episode_reward:7.1f}[/dim]"
                    )
                
                progress.update(gen_task, advance=1)
                
                if done:
                    break
            
            # Episode complete
            progress.remove_task(gen_task)
            
            final_state = get_state_from_sim(sim)
            final_coop = final_state[1] * 100
            final_div = final_state[4]
            
            history['episode_rewards'].append(episode_reward)
            history['episode_cooperation'].append(final_coop)
            history['episode_diversity'].append(final_div)
            
            # Action distribution
            action_counts = {i: episode_actions.count(i) for i in range(7)}
            most_used_action = max(action_counts, key=lambda x: action_counts[x])
            
            # Episode summary with detailed metrics
            progress.console.print()
            progress.console.rule(f"[bold cyan]Episode {episode+1}/{episodes} Complete", style="cyan")
            
            # Main metrics table
            result_table = Table(title="[bold]📊 Episode Results", box=box.ROUNDED, show_header=True)
            result_table.add_column("Metric", style="cyan", width=20)
            result_table.add_column("Value", style="yellow", width=15)
            result_table.add_column("Status", style="green", width=20)
            
            # Compare to human governments
            coop_status = "🏆 Best!" if final_coop > 75 else "✅ Good" if final_coop > 50 else "⚠️ Low"
            div_status = "🌟 High" if final_div > 0.8 else "✅ Good" if final_div > 0.6 else "⚠️ Low"
            
            result_table.add_row("Total Reward", f"{episode_reward:.2f}", f"Accumulated: {sum(history['episode_rewards']):.1f}")
            result_table.add_row("Cooperation %", f"{final_coop:.1f}%", coop_status)
            result_table.add_row("Genetic Diversity", f"{final_div:.3f}", div_status)
            result_table.add_row("Final Population", f"{len(sim.agents)}", "✅ Stable" if len(sim.agents) > 150 else "⚠️ Declining")
            result_table.add_row("Avg Wealth", f"{final_state[2]:.1f}", f"Gini: {final_state[5]:.3f}")
            
            progress.console.print(result_table)
            
            # Action distribution table
            action_table = Table(title="[bold]🎯 Actions Taken", box=box.SIMPLE, show_header=True)
            action_table.add_column("Action", style="magenta", width=25)
            action_table.add_column("Count", justify="right", style="cyan", width=10)
            action_table.add_column("% of Episode", justify="right", style="yellow", width=15)
            action_table.add_column("Bar", style="green", width=20)
            
            total_actions = len(episode_actions)
            for i, name in enumerate(action_names):
                count = action_counts[i]
                pct = (count / total_actions * 100) if total_actions > 0 else 0
                bar_len = int(pct / 5)  # 20 char max
                bar = "█" * bar_len
                style = "bold green" if i == most_used_action else "dim"
                action_table.add_row(
                    name,
                    str(count),
                    f"{pct:.1f}%",
                    f"[{style}]{bar}[/{style}]"
                )
            
            progress.console.print(action_table)
            
            # Comparison with human governments
            if episode >= 2:  # Show after a few episodes
                compare_table = Table(title="[bold]🏛️ Comparison: ML vs Human Governments", box=box.DOUBLE, show_header=True)
                compare_table.add_column("Government Type", style="cyan", width=20)
                compare_table.add_column("Cooperation %", justify="right", style="yellow", width=15)
                compare_table.add_column("vs ML Agent", style="green", width=20)
                
                # Human government results (from deep dive)
                governments = [
                    ("Authoritarian", 99.8, "magenta"),
                    ("Mixed Economy", 75.0, "cyan"),
                    ("Welfare State", 50.4, "blue"),
                    ("Central Banker", 36.8, "yellow"),
                    ("Laissez-Faire", 34.2, "white"),
                ]
                
                for gov_name, gov_coop, color in governments:
                    diff = final_coop - gov_coop
                    if abs(diff) < 5:
                        status = "≈ Similar"
                        style = "yellow"
                    elif diff > 0:
                        status = f"↑ +{diff:.1f}%"
                        style = "green"
                    else:
                        status = f"↓ {diff:.1f}%"
                        style = "red"
                    
                    compare_table.add_row(
                        f"[{color}]{gov_name}[/{color}]",
                        f"{gov_coop:.1f}%",
                        f"[{style}]{status}[/{style}]"
                    )
                
                # Add ML agent current result
                compare_table.add_row(
                    "[bold green]ML Agent (This Ep)[/bold green]",
                    f"[bold green]{final_coop:.1f}%[/bold green]",
                    "[bold green]← YOU ARE HERE[/bold green]"
                )
                
                progress.console.print(compare_table)
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                checkpoint_path = f"{save_path}.ep{episode+1}.pth"
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'history': history
                }, checkpoint_path)
                progress.console.print(f"[green]💾 Checkpoint saved: {checkpoint_path}[/green]")
            
            progress.console.print()
            progress.update(episode_task, advance=1)
    
    # Save final model
    console.print()
    console.rule("[bold green]🎉 TRAINING COMPLETE!", style="green")
    console.print()
    
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'history': history
    }, save_path)
    
    console.print(f"[bold green]✅ Model saved: {save_path}[/bold green]")
    console.print()
    
    # Calculate comprehensive statistics
    avg_coop_last10 = np.mean(history['episode_cooperation'][-10:])
    avg_div_last10 = np.mean(history['episode_diversity'][-10:])
    avg_reward_last10 = np.mean(history['episode_rewards'][-10:])
    
    best_episode = np.argmax(history['episode_rewards'])
    best_coop = history['episode_cooperation'][best_episode]
    best_reward = history['episode_rewards'][best_episode]
    
    # Training statistics
    stats_table = Table(title="[bold cyan]📈 Training Statistics", box=box.DOUBLE_EDGE, show_header=True)
    stats_table.add_column("Metric", style="cyan", width=30)
    stats_table.add_column("Value", style="yellow", width=20)
    stats_table.add_column("Details", style="green", width=30)
    
    stats_table.add_row("Total Episodes", str(episodes), f"✅ Complete")
    stats_table.add_row("Memory Buffer Size", str(len(agent.memory)), "Experience stored")
    stats_table.add_row("Best Episode", f"#{best_episode+1}", f"Reward: {best_reward:.2f}")
    stats_table.add_row("Best Cooperation", f"{best_coop:.1f}%", "Peak performance")
    stats_table.add_row("", "", "")
    stats_table.add_row("[bold]Last 10 Episodes Avg", "", "[bold]Final Performance")
    stats_table.add_row("  Cooperation %", f"{avg_coop_last10:.1f}%", "Stable behavior")
    stats_table.add_row("  Genetic Diversity", f"{avg_div_last10:.3f}", "Population health")
    stats_table.add_row("  Reward Score", f"{avg_reward_last10:.2f}", "Optimization target")
    
    console.print(stats_table)
    console.print()
    
    # Final comparison with human governments
    final_compare = Table(title="[bold magenta]🏆 FINAL RESULTS: ML Agent vs Human Governments", box=box.DOUBLE, show_header=True)
    final_compare.add_column("Rank", justify="center", style="yellow", width=6)
    final_compare.add_column("Government Type", style="cyan", width=25)
    final_compare.add_column("Cooperation %", justify="right", style="green", width=15)
    final_compare.add_column("Rating", style="magenta", width=20)
    
    # All results including ML
    all_results = [
        ("Authoritarian", 99.8, "🔒 Forced Compliance"),
        ("Mixed Economy", 75.0, "⚖️ Adaptive Balance"),
        ("Welfare State", 50.4, "🤝 Support System"),
        ("Central Banker", 36.8, "💰 Market Focus"),
        ("Laissez-Faire", 34.2, "🆓 Free Market"),
        ("ML Agent", avg_coop_last10, "🤖 Learned Policy"),
    ]
    
    # Sort by cooperation
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, coop, rating) in enumerate(all_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        style = "bold green" if name == "ML Agent" else "white"
        final_compare.add_row(
            medal,
            f"[{style}]{name}[/{style}]",
            f"[{style}]{coop:.1f}%[/{style}]",
            rating
        )
    
    console.print(final_compare)
    console.print()
    
    # Key insights
    insights_panel = Panel(
        "[bold cyan]🔍 Key Insights:[/bold cyan]\n\n"
        f"• ML agent achieved [yellow]{avg_coop_last10:.1f}%[/yellow] cooperation (avg last 10 episodes)\n"
        f"• Peak performance: [green]{best_coop:.1f}%[/green] in episode {best_episode+1}\n"
        f"• Maintained diversity: [blue]{avg_div_last10:.3f}[/blue] (0.0 = uniform, 1.0 = max diversity)\n"
        f"• Total training experience: [magenta]{len(agent.memory)}[/magenta] state-action pairs\n\n"
        "[bold yellow]Next Steps:[/bold yellow]\n"
        "• Run evaluation mode to test trained policy\n"
        "• Try full training (100 episodes) for better optimization\n"
        "• Compare action distributions to understand learned strategy",
        title="[bold green]Training Summary",
        border_style="green"
    )
    console.print(insights_panel)
    console.print()
    
    return agent, history


def evaluate_agent(model_path='ml_governance_model.pth', n_episodes=5):
    """Evaluate trained agent"""
    console.clear()
    console.rule("[bold cyan]📊 EVALUATING ML AGENT", style="cyan")
    console.print()
    
    # Load model
    console.print("[cyan]Loading model...[/cyan]")
    agent = GovernanceAgent()
    checkpoint = torch.load(model_path)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.policy_net.eval()
    console.print("[green]✓ Model loaded![/green]\n")
    
    results = []
    action_names = [
        "Do Nothing", "Welfare", "Universal Stimulus", 
        "Targeted Stimulus", "Remove Defectors", "Tax Defectors", "Boost Cooperators"
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        eval_task = progress.add_task("[cyan]Evaluation Episodes", total=n_episodes)
        
        for episode in range(n_episodes):
            sim = UltimateEchoSimulation(
                initial_size=200,
                grid_size=(75, 75),
                government_style=GovernmentStyle.LAISSEZ_FAIRE
            )
            
            episode_reward = 0
            actions_taken = []
            
            gen_task = progress.add_task(
                f"[yellow]Episode {episode+1}/{n_episodes}", 
                total=300
            )
            
            for gen in range(300):
                state = get_state_from_sim(sim)
                state[8] = gen / 300
                
                action, _ = agent.select_action(state, explore=False)
                actions_taken.append(action)
                
                intervention_cost = apply_action_to_sim(sim, action)
                sim.step()
                
                next_state = get_state_from_sim(sim)
                reward = agent.calculate_reward(next_state)
                reward -= intervention_cost * agent.reward_weights['intervention_cost']
                episode_reward += reward
                
                progress.update(gen_task, advance=1)
            
            progress.remove_task(gen_task)
            
            final_state = get_state_from_sim(sim)
            final_coop = final_state[1] * 100
            final_div = final_state[4]
            
            results.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'cooperation': final_coop,
                'diversity': final_div,
                'actions': actions_taken
            })
            
            action_counts = {i: actions_taken.count(i) for i in range(7)}
            most_used = max(action_counts, key=lambda x: action_counts[x])
            
            # Episode result
            progress.console.print()
            result_table = Table(show_header=False, box=box.SIMPLE)
            result_table.add_row(
                f"[bold cyan]Episode {episode+1}[/bold cyan]",
                f"[yellow]Reward: {episode_reward:.2f}[/yellow]",
                f"[green]Coop: {final_coop:.1f}%[/green]",
                f"[blue]Div: {final_div:.3f}[/blue]"
            )
            result_table.add_row(
                "",
                f"[magenta]Primary Strategy: {action_names[most_used]}[/magenta]",
                f"[white]Used: {action_counts[most_used]} times[/white]",
                ""
            )
            progress.console.print(result_table)
            progress.console.print()
            
            progress.update(eval_task, advance=1)
    
    # Final summary
    console.print()
    console.rule("[bold green]Evaluation Complete", style="green")
    console.print()
    
    avg_coop = np.mean([r['cooperation'] for r in results])
    avg_div = np.mean([r['diversity'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    
    summary_table = Table(title="[bold cyan]Evaluation Results", box=box.DOUBLE_EDGE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Episodes", str(n_episodes))
    summary_table.add_row("Avg Cooperation", f"{avg_coop:.1f}%")
    summary_table.add_row("Avg Diversity", f"{avg_div:.3f}")
    summary_table.add_row("Avg Reward", f"{avg_reward:.2f}")
    
    console.print(summary_table)
    console.print()
    
    return results


def main():
    """Main entry point"""
    console.clear()
    console.rule("[bold magenta]🤖 GPU-ACCELERATED ML GOVERNANCE", style="magenta")
    console.print()
    
    info_panel = Panel(
        "[cyan]This uses YOUR existing PyTorch/GPU infrastructure!\n"
        "Training on RTX 4070 Ti with real-time progress tracking.",
        title="[bold]System Info",
        border_style="cyan"
    )
    console.print(info_panel)
    console.print()
    
    menu_table = Table(show_header=False, box=box.ROUNDED)
    menu_table.add_column("Option", style="cyan", width=8)
    menu_table.add_column("Description", style="white")
    menu_table.add_column("Time", style="yellow")
    
    menu_table.add_row("1", "Train new agent (Full)", "~30 min")
    menu_table.add_row("2", "Train quick agent (Fast)", "~5 min")
    menu_table.add_row("3", "Evaluate existing agent", "~2 min")
    menu_table.add_row("4", "Compare ML vs Human governments", "Coming soon")
    
    console.print(Panel(menu_table, title="[bold]Menu Options", border_style="green"))
    console.print()
    
    choice = console.input("[bold cyan]Enter choice (1-4): [/bold cyan]").strip()
    
    if choice == '1':
        console.print("\n[bold green]Starting full training (100 episodes)...[/bold green]\n")
        train_governance_agent(episodes=100)
    elif choice == '2':
        console.print("\n[bold green]Starting quick training (20 episodes)...[/bold green]\n")
        train_governance_agent(episodes=20)
    elif choice == '3':
        console.print("\n[bold green]Starting evaluation...[/bold green]\n")
        evaluate_agent()
    elif choice == '4':
        console.print("\n[bold yellow]🔜 Coming soon: Full comparison![/bold yellow]\n")
    else:
        console.print("\n[bold red]❌ Invalid choice[/bold red]\n")


if __name__ == '__main__':
    main()
