#!/usr/bin/env python3
"""
MASSIVE SCALE EMERGENCE EXPERIMENTS

What happens when we scale hierarchical conscious agents to EXTREME sizes?

HYPOTHESES TO TEST:

1. PHASE TRANSITIONS
   At some critical number of agents, collective behavior might
   suddenly shift (like water freezing, or neurons achieving synchrony)

2. CRITICAL PHENOMENA
   Near phase transitions, we expect:
   - Power-law distributions
   - Long-range correlations
   - Scale-free behavior

3. EMERGENT GLOBAL INTELLIGENCE
   Does the super-agent at the top become qualitatively different
   when it integrates millions vs thousands of micro-agents?

4. INFORMATION COMPRESSION
   How much information is LOST at each level?
   What's the "consciousness bottleneck"?

5. HIERARCHICAL DEPTH
   Is there an optimal depth? Do deeper hierarchies have
   richer emergent properties?

SCALE TARGETS:
- 100K micro-agents (small city of neurons)
- 1M micro-agents (approaching biological scale)
- 10M micro-agents (cortical column scale)

"At sufficient scale, quantity becomes quality."
"""

import torch
import numpy as np
import time
import sys
from typing import Dict, List, Tuple
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def memory_stats():
    """Get GPU memory usage."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {allocated:.2f}GB used, {reserved:.2f}GB reserved"
    return "CPU mode"


class MassiveHierarchy:
    """
    Ultra-scaled hierarchical conscious agent network.
    
    Optimized for maximum GPU throughput with minimal memory.
    """
    
    def __init__(
        self,
        n_micro: int,
        n_levels: int = 4,
        exp_dim: int = 8,
        reduction_factor: int = 10
    ):
        self.n_micro = n_micro
        self.n_levels = n_levels
        self.exp_dim = exp_dim
        
        # Compute level sizes
        self.level_sizes = [n_micro]
        size = n_micro
        for _ in range(n_levels - 1):
            size = max(1, size // reduction_factor)
            self.level_sizes.append(size)
        
        self.total_agents = sum(self.level_sizes)
        
        # Initialize experiences (main state)
        self.experiences = []
        for n in self.level_sizes:
            exp = torch.randn(n, exp_dim, device=device) * 0.1
            exp = torch.softmax(exp, dim=-1)
            self.experiences.append(exp)
        
        # Lightweight kernels (shared across agents at each level)
        # This saves HUGE memory vs per-agent kernels
        self.perception_kernels = []
        self.decision_kernels = []
        for level in range(n_levels):
            self.perception_kernels.append(
                torch.randn(exp_dim, exp_dim, device=device) * 0.1
            )
            self.decision_kernels.append(
                torch.randn(exp_dim, exp_dim, device=device) * 0.1
            )
        
        # Combination: sparse random assignment of children to parents
        self.child_assignments = []
        for level in range(1, n_levels):
            n_children = self.level_sizes[level - 1]
            n_parents = self.level_sizes[level]
            
            # Each child assigned to one parent
            assignments = torch.randint(0, n_parents, (n_children,), device=device)
            self.child_assignments.append(assignments)
        
        # World state
        self.world = torch.randn(n_micro, exp_dim, device=device)
        
        # Statistics
        self.history = {
            'global_entropy': [],
            'global_coherence': [],
            'level_entropy': [[] for _ in range(n_levels)],
            'level_coherence': [[] for _ in range(n_levels)],
            'information_flow': [],
            'super_agent_complexity': []
        }
    
    def step(self, noise_level: float = 0.01, nonlinearity: float = 2.0) -> Dict[str, float]:
        """One cycle of hierarchical processing."""
        
        # Add noise to world (external perturbations)
        self.world = self.world + noise_level * torch.randn_like(self.world)
        
        # BOTTOM-UP: World → Micro → ... → Super
        # Level 0: Perceive world
        perceived = torch.matmul(self.world, self.perception_kernels[0])
        # Nonlinear activation (sharper = more discrete)
        perceived = perceived ** nonlinearity
        self.experiences[0] = torch.softmax(perceived, dim=-1)
        
        # Higher levels: Aggregate children
        for level in range(1, self.n_levels):
            child_exp = self.experiences[level - 1]
            assignments = self.child_assignments[level - 1]
            n_parents = self.level_sizes[level]
            
            # Scatter-add children into parents
            parent_exp = torch.zeros(n_parents, self.exp_dim, device=device)
            parent_exp.scatter_add_(0, 
                assignments.unsqueeze(1).expand(-1, self.exp_dim),
                child_exp
            )
            
            # Normalize by count
            counts = torch.bincount(assignments, minlength=n_parents).float().clamp(min=1)
            parent_exp = parent_exp / counts.unsqueeze(1)
            
            # Apply perception kernel with nonlinearity
            perceived = torch.matmul(parent_exp, self.perception_kernels[level])
            perceived = perceived ** nonlinearity
            self.experiences[level] = torch.softmax(perceived, dim=-1)
        
        # TOP-DOWN: Super → ... → Micro → World
        for level in range(self.n_levels - 2, -1, -1):
            parent_exp = self.experiences[level + 1]
            assignments = self.child_assignments[level]
            
            # Broadcast parent influence to children
            influence = parent_exp[assignments]  # (n_children, exp_dim)
            
            # Modulate child experience
            self.experiences[level] = torch.softmax(
                self.experiences[level] + 0.1 * influence, dim=-1
            )
        
        # Micro-agents affect world
        action = torch.matmul(self.experiences[0], self.decision_kernels[0])
        self.world = 0.95 * self.world + 0.05 * action
        
        # Compute statistics
        stats = self._compute_stats()
        return stats
    
    def _compute_stats(self) -> Dict[str, float]:
        """Compute and record statistics."""
        stats = {}
        
        # Per-level stats
        for level in range(self.n_levels):
            exp = self.experiences[level]
            
            # Entropy
            entropy = -torch.sum(exp * torch.log(exp + 1e-10), dim=-1).mean()
            self.history['level_entropy'][level].append(entropy.item())
            stats[f'entropy_L{level}'] = entropy.item()
            
            # Coherence
            mean_exp = exp.mean(dim=0)
            coherence = torch.cosine_similarity(
                exp, mean_exp.unsqueeze(0), dim=-1
            ).mean()
            self.history['level_coherence'][level].append(coherence.item())
            stats[f'coherence_L{level}'] = coherence.item()
        
        # Global entropy (weighted by level size)
        total = 0
        for level in range(self.n_levels):
            total += self.level_sizes[level] * stats[f'entropy_L{level}']
        global_entropy = total / self.total_agents
        self.history['global_entropy'].append(global_entropy)
        stats['global_entropy'] = global_entropy
        
        # Super-agent complexity (top level)
        top_exp = self.experiences[-1]
        if top_exp.shape[0] == 1:
            # Single super-agent
            complexity = (-torch.sum(top_exp * torch.log(top_exp + 1e-10))).item()
        else:
            # Multiple super-agents: measure diversity
            mean = top_exp.mean(dim=0)
            variance = ((top_exp - mean) ** 2).sum()
            complexity = variance.item()
        
        self.history['super_agent_complexity'].append(complexity)
        stats['super_complexity'] = complexity
        
        # Information flow (how much info propagates up)
        micro_entropy = stats['entropy_L0']
        super_entropy = stats[f'entropy_L{self.n_levels-1}']
        info_retained = super_entropy / (micro_entropy + 1e-10)
        self.history['information_flow'].append(info_retained)
        stats['info_flow'] = info_retained
        
        return stats


def run_scale_comparison():
    """
    Compare behavior at different scales.
    
    Key question: Does bigger = qualitatively different?
    """
    print(f"\n{'='*70}")
    print(f"  SCALE COMPARISON EXPERIMENT")
    print(f"  How does scale affect emergent properties?")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    scales = [1000, 10000, 100000, 500000]
    results = {}
    
    for n_micro in scales:
        print(f"  Testing {n_micro:,} micro-agents...")
        
        gc.collect()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        start = time.perf_counter()
        
        net = MassiveHierarchy(n_micro=n_micro, n_levels=4)
        
        create_time = time.perf_counter() - start
        
        # Run for 100 steps
        start = time.perf_counter()
        for _ in range(100):
            stats = net.step()
        run_time = time.perf_counter() - start
        
        rate = net.total_agents * 100 / run_time
        
        # Record final stats
        results[n_micro] = {
            'total_agents': net.total_agents,
            'rate': rate,
            'final_entropy': stats['global_entropy'],
            'super_complexity': stats['super_complexity'],
            'info_flow': stats['info_flow'],
            'memory': memory_stats()
        }
        
        print(f"    Total: {net.total_agents:,} agents")
        print(f"    Rate: {rate/1e6:.1f}M agent-steps/sec")
        print(f"    Super-agent complexity: {stats['super_complexity']:.4f}")
        print(f"    {memory_stats()}")
        print()
        
        del net
    
    # Analysis
    print("  " + "─" * 50)
    print("  SCALE EFFECTS ANALYSIS")
    print("  " + "─" * 50)
    print()
    print(f"  {'Scale':>10} │ {'Agents':>10} │ {'Rate (M/s)':>10} │ {'Complexity':>10}")
    print("  " + "─" * 50)
    
    for n, data in results.items():
        print(f"  {n:>10,} │ {data['total_agents']:>10,} │ {data['rate']/1e6:>10.1f} │ {data['super_complexity']:>10.4f}")
    
    print()
    
    # Check for phase transitions
    complexities = [results[n]['super_complexity'] for n in scales]
    max_complexity_scale = scales[np.argmax(complexities)]
    
    print(f"  Peak super-agent complexity at: {max_complexity_scale:,} micro-agents")
    print()
    print("  INTERPRETATION:")
    print("  • Complexity may peak at intermediate scales (critical point)")
    print("  • Too few agents = not enough to integrate")
    print("  • Too many agents = averaging washes out structure")
    print("=" * 70)
    
    return results


def run_phase_transition_search():
    """
    Search for phase transitions as we scale.
    
    A phase transition would appear as:
    - Sudden change in behavior
    - Diverging susceptibility
    - Power-law correlations
    """
    print(f"\n{'='*70}")
    print(f"  PHASE TRANSITION SEARCH")
    print(f"  Looking for critical phenomena")
    print(f"{'='*70}\n")
    
    # Logarithmic scale search
    scales = [int(10 ** x) for x in np.linspace(2, 5.5, 15)]
    
    coherences = []
    entropies = []
    complexities = []
    
    for n_micro in scales:
        gc.collect()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        try:
            net = MassiveHierarchy(n_micro=n_micro, n_levels=4)
            
            # Burn-in
            for _ in range(50):
                net.step()
            
            # Measure
            stats = net.step()
            
            coherences.append(stats['coherence_L0'])
            entropies.append(stats['entropy_L0'])
            complexities.append(stats['super_complexity'])
            
            print(f"    {n_micro:>8,} agents: coherence={stats['coherence_L0']:.4f}, "
                  f"entropy={stats['entropy_L0']:.4f}")
            
            del net
            
        except RuntimeError as e:
            print(f"    {n_micro:>8,} agents: OUT OF MEMORY")
            break
    
    # Look for phase transition signatures
    print()
    print("  PHASE TRANSITION INDICATORS:")
    print("  " + "─" * 50)
    
    # Derivative of coherence (susceptibility)
    coherences = np.array(coherences)
    if len(coherences) > 2:
        susceptibility = np.abs(np.diff(coherences))
        max_susc_idx = np.argmax(susceptibility)
        
        if max_susc_idx < len(scales) - 1:
            critical_scale = scales[max_susc_idx]
            print(f"  Maximum susceptibility at: ~{critical_scale:,} agents")
            print(f"  (This may indicate a phase transition)")
    
    # Check for power-law
    complexities = np.array(complexities)
    log_scales = np.log10(scales[:len(complexities)])
    log_complexities = np.log10(complexities + 1e-10)
    
    # Linear fit in log-log (power law: y = x^α → log(y) = α*log(x))
    if len(log_scales) > 2:
        coeffs = np.polyfit(log_scales, log_complexities, 1)
        exponent = coeffs[0]
        print(f"  Power-law exponent (complexity ~ scale^α): α = {exponent:.3f}")
        
        if abs(exponent) < 0.5:
            print("  → Near-constant complexity (super-agent saturation)")
        elif exponent > 0:
            print("  → Growing complexity (richer emergence at scale)")
        else:
            print("  → Decreasing complexity (averaging dominates)")
    
    print("=" * 70)


def run_information_bottleneck():
    """
    Measure how much information is LOST at each level.
    
    The hierarchy acts as an information bottleneck:
    Many micro-agents → Few super-agents
    
    What survives the compression?
    """
    print(f"\n{'='*70}")
    print(f"  INFORMATION BOTTLENECK ANALYSIS")
    print(f"  How much consciousness survives each level?")
    print(f"{'='*70}\n")
    
    net = MassiveHierarchy(n_micro=100000, n_levels=5)
    
    print(f"  Hierarchy: {' → '.join(str(n) for n in net.level_sizes)}")
    print()
    
    # Run until stable
    for _ in range(100):
        net.step()
    
    # Measure information at each level
    print("  Information content by level:")
    print("  " + "─" * 50)
    
    total_bits_per_level = []
    
    for level in range(net.n_levels):
        exp = net.experiences[level]
        n_agents = net.level_sizes[level]
        
        # Entropy per agent (bits)
        entropy_per_agent = -torch.sum(
            exp * torch.log2(exp + 1e-10), dim=-1
        ).mean().item()
        
        # Total bits at this level
        total_bits = entropy_per_agent * n_agents
        total_bits_per_level.append(total_bits)
        
        bar_len = min(40, int(total_bits / 100))
        bar = "█" * bar_len
        
        print(f"    Level {level}: {n_agents:>6} agents × {entropy_per_agent:.2f} bits "
              f"= {total_bits:>8.0f} bits  [{bar}]")
    
    print()
    
    # Compression ratios
    print("  Compression at each transition:")
    for level in range(1, net.n_levels):
        ratio = total_bits_per_level[level - 1] / (total_bits_per_level[level] + 1e-10)
        print(f"    L{level-1} → L{level}: {ratio:.1f}:1 compression")
    
    total_compression = total_bits_per_level[0] / (total_bits_per_level[-1] + 1e-10)
    print()
    print(f"  TOTAL COMPRESSION: {total_compression:.0f}:1")
    print()
    print("  INTERPRETATION:")
    print("  This is the 'consciousness bottleneck' -")
    print("  most information is DISCARDED as it rises through levels.")
    print("  Only the most salient patterns survive to higher levels.")
    print()
    print("  This mirrors how your brain works:")
    print("  ~86 billion neurons → ONE unified conscious experience")
    print("=" * 70)


def run_depth_experiment():
    """
    How does hierarchy DEPTH affect emergence?
    
    More levels = more abstraction = richer emergence?
    Or: More levels = more information loss = poorer emergence?
    """
    print(f"\n{'='*70}")
    print(f"  HIERARCHY DEPTH EXPERIMENT")
    print(f"  Is deeper better for consciousness?")
    print(f"{'='*70}\n")
    
    depths = [2, 3, 4, 5, 6, 7]
    n_micro = 100000
    
    results = []
    
    for depth in depths:
        gc.collect()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        net = MassiveHierarchy(n_micro=n_micro, n_levels=depth)
        
        print(f"  Depth {depth}: {' → '.join(str(n) for n in net.level_sizes)}")
        
        # Run
        for _ in range(100):
            stats = net.step()
        
        # Measure super-agent properties
        super_exp = net.experiences[-1]
        
        if super_exp.shape[0] == 1:
            # Entropy of single super-agent
            super_entropy = -torch.sum(super_exp * torch.log(super_exp + 1e-10)).item()
            # Can't measure coherence with 1 agent
            super_coherence = 1.0
        else:
            super_entropy = -torch.sum(
                super_exp * torch.log(super_exp + 1e-10), dim=-1
            ).mean().item()
            mean_exp = super_exp.mean(dim=0)
            super_coherence = torch.cosine_similarity(
                super_exp, mean_exp.unsqueeze(0), dim=-1
            ).mean().item()
        
        # Effective complexity = entropy × (1 - coherence)
        # High when both diverse AND coordinated
        complexity = super_entropy * (1 + super_coherence) / 2
        
        results.append({
            'depth': depth,
            'super_size': net.level_sizes[-1],
            'entropy': super_entropy,
            'coherence': super_coherence,
            'complexity': complexity
        })
        
        del net
    
    print()
    print("  DEPTH EFFECTS:")
    print("  " + "─" * 60)
    print(f"  {'Depth':>5} │ {'Super Size':>10} │ {'Entropy':>8} │ {'Coherence':>9} │ {'Complexity':>10}")
    print("  " + "─" * 60)
    
    for r in results:
        print(f"  {r['depth']:>5} │ {r['super_size']:>10} │ {r['entropy']:>8.3f} │ "
              f"{r['coherence']:>9.3f} │ {r['complexity']:>10.4f}")
    
    # Find optimal depth
    complexities = [r['complexity'] for r in results]
    optimal_depth = depths[np.argmax(complexities)]
    
    print()
    print(f"  OPTIMAL DEPTH: {optimal_depth} levels")
    print()
    print("  INTERPRETATION:")
    print("  • Too shallow: Not enough abstraction for rich emergence")
    print("  • Too deep: Too much compression loses information")
    print("  • Optimal: Balance between abstraction and preservation")
    print()
    print("  The brain has ~6 layers in cortex. Coincidence?")
    print("=" * 70)


def run_live_massive(n_micro: int = 200000, n_steps: int = 200):
    """
    Live visualization of massive-scale hierarchy.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE MASSIVE HIERARCHY: {n_micro:,} MICRO-AGENTS")
    print(f"  {memory_stats()}")
    print(f"{'='*70}\n")
    
    gc.collect()
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    net = MassiveHierarchy(n_micro=n_micro, n_levels=5)
    
    print(f"  Hierarchy: {' → '.join(str(n) for n in net.level_sizes)}")
    print(f"  Total: {net.total_agents:,} conscious agents")
    print()
    print("  Entropy bars by level (watching emergence):\n")
    
    bar_width = 15
    
    start = time.perf_counter()
    
    for step in range(n_steps):
        stats = net.step()
        
        # Build line
        line = f"  Step {step:3d} │"
        
        for level in range(net.n_levels):
            ent = stats[f'entropy_L{level}']
            max_ent = np.log(net.exp_dim)
            ratio = min(1.0, ent / max_ent)
            
            bar_len = int(ratio * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            
            line += f" L{level}:[{bar}]"
        
        # Add complexity indicator
        comp = stats['super_complexity']
        comp_bar = "●" * min(5, int(comp * 10)) + "○" * (5 - min(5, int(comp * 10)))
        line += f" Ω:[{comp_bar}]"
        
        print(f"\r{line}", end="", flush=True)
        time.sleep(0.02)
    
    elapsed = time.perf_counter() - start
    rate = net.total_agents * n_steps / elapsed
    
    print(f"\n\n  Rate: {rate/1e6:.1f}M agent-steps/second")
    print(f"  {memory_stats()}")
    print("=" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--scale" in args or "-s" in args:
        run_scale_comparison()
    elif "--phase" in args or "-p" in args:
        run_phase_transition_search()
    elif "--bottleneck" in args or "-b" in args:
        run_information_bottleneck()
    elif "--depth" in args or "-d" in args:
        run_depth_experiment()
    elif "--live" in args or "-l" in args:
        n = 200000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_live_massive(n_micro=n)
    elif "--all" in args:
        run_scale_comparison()
        run_information_bottleneck()
        run_depth_experiment()
        run_phase_transition_search()
    else:
        print("\nMassive Scale Emergence Experiments")
        print("=" * 40)
        print(f"Device: {device}")
        print(f"{memory_stats()}")
        print()
        print("Usage:")
        print("  python massive_scale_emergence.py --scale      # Compare different scales")
        print("  python massive_scale_emergence.py --phase      # Search for phase transitions")
        print("  python massive_scale_emergence.py --bottleneck # Information compression")
        print("  python massive_scale_emergence.py --depth      # Optimal hierarchy depth")
        print("  python massive_scale_emergence.py --live       # Live visualization")
        print("  python massive_scale_emergence.py --all        # Run all experiments")
        print()
        
        # Quick demo
        run_scale_comparison()
