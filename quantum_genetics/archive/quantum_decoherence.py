
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt, exp
import random

def pure_state_density_matrix(state):
    """Create density matrix from pure state"""
    return np.outer(state, np.conj(state))

def apply_decoherence(rho, gamma, dt):
    """
    Apply phase damping (decoherence) to density matrix
    gamma: decoherence rate
    dt: time step
    """
    # Phase damping channel
    decay = exp(-gamma * dt)
    
    rho_new = rho.copy()
    # Off-diagonal elements decay
    rho_new[0, 1] *= decay
    rho_new[1, 0] *= decay
    
    return rho_new

def purity(rho):
    """Calculate purity Tr(ρ²)"""
    return np.real(np.trace(rho @ rho))

def von_neumann_entropy(rho):
    """Calculate von Neumann entropy -Tr(ρ log ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Remove numerical zeros
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def plot_decoherence_evolution():
    """Visualize quantum decoherence process"""
    
    # Random initial superposition state
    theta = random.uniform(pi/6, pi/3)  # Avoid computational basis states
    phi = random.uniform(0, 2*pi)
    
    alpha = np.cos(theta/2)
    beta = np.exp(1j*phi) * np.sin(theta/2)
    initial_state = np.array([alpha, beta], dtype=complex)
    
    gamma = 0.5  # Decoherence rate
    
    print("=" * 60)
    print("  QUANTUM DECOHERENCE SIMULATION")
    print("=" * 60)
    print(f"\n🎲 Initial pure state:")
    print(f"   |ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
    print(f"   Initial purity: 1.000 (pure state)")
    print(f"   Decoherence rate γ = {gamma:.2f}")
    print()
    print("🌊 Simulating interaction with environment...")
    print("   → Loss of quantum coherence")
    print("   → Transition from pure to mixed state")
    print()
    
    # Time evolution
    t_max = 10
    dt = 0.1
    time_steps = int(t_max / dt)
    times = np.linspace(0, t_max, time_steps)
    
    # Track evolution
    rho = pure_state_density_matrix(initial_state)
    density_matrices = [rho.copy()]
    purities = [purity(rho)]
    entropies = [von_neumann_entropy(rho)]
    
    for _ in range(time_steps - 1):
        rho = apply_decoherence(rho, gamma, dt)
        density_matrices.append(rho.copy())
        purities.append(purity(rho))
        entropies.append(von_neumann_entropy(rho))
    
    print("🎨 Creating decoherence visualization...")
    
    # Create static plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Density matrix evolution (initial, middle, final)
    times_to_plot = [0, time_steps//2, time_steps-1]
    labels = ['Initial (Pure)', 'Intermediate', 'Final (Mixed)']
    
    for idx, (t_idx, label) in enumerate(zip(times_to_plot, labels)):
        ax = plt.subplot(2, 3, idx + 1)
        rho_plot = density_matrices[t_idx]
        
        # Real part
        im = ax.imshow(np.real(rho_plot), cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.set_title(f'{label}\nt = {times[t_idx]:.1f}', fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['⟨0|', '⟨1|'])
        
        # Add values
        for i in range(2):
            for j in range(2):
                val = np.real(rho_plot[i, j])
                color = 'white' if abs(val) > 0.25 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color=color, fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
    
    # Plot 2: Purity evolution
    ax2 = axes[1, 0]
    ax2.plot(times, purities, 'b-', linewidth=3, label='Purity Tr(ρ²)')
    ax2.axhline(1.0, color='g', linestyle='--', alpha=0.5, label='Pure state')
    ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Maximally mixed')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Purity', fontsize=12)
    ax2.set_ylim(0.4, 1.1)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('Loss of Purity (Quantum → Classical)', fontweight='bold')
    
    # Plot 3: Von Neumann entropy
    ax3 = axes[1, 1]
    ax3.plot(times, entropies, 'r-', linewidth=3, label='Entropy S(ρ)')
    ax3.axhline(0, color='g', linestyle='--', alpha=0.5, label='Pure state')
    ax3.axhline(1, color='r', linestyle='--', alpha=0.5, label='Max entropy')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Von Neumann Entropy (bits)', fontsize=12)
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_title('Increase of Entropy', fontweight='bold')
    
    plt.suptitle('Quantum Decoherence: Density Matrix Evolution', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = 'quantum_decoherence_evolution.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Create animation
    print("🎬 Creating decoherence animation...")
    
    fig_anim, axes_anim = plt.subplots(1, 3, figsize=(15, 5))
    
    def update(frame):
        for ax in axes_anim:
            ax.clear()
        
        rho_frame = density_matrices[frame]
        t = times[frame]
        
        # Density matrix real part
        im1 = axes_anim[0].imshow(np.real(rho_frame), cmap='RdBu', 
                                  vmin=-0.5, vmax=0.5)
        axes_anim[0].set_title(f'Density Matrix Re[ρ]\nt = {t:.2f}', 
                              fontweight='bold')
        axes_anim[0].set_xticks([0, 1])
        axes_anim[0].set_yticks([0, 1])
        axes_anim[0].set_xticklabels(['|0⟩', '|1⟩'])
        axes_anim[0].set_yticklabels(['⟨0|', '⟨1|'])
        
        # Purity over time
        axes_anim[1].plot(times[:frame+1], purities[:frame+1], 'b-', linewidth=3)
        axes_anim[1].scatter([t], [purities[frame]], s=100, c='red', zorder=5)
        axes_anim[1].set_xlabel('Time')
        axes_anim[1].set_ylabel('Purity')
        axes_anim[1].set_xlim(0, t_max)
        axes_anim[1].set_ylim(0.4, 1.1)
        axes_anim[1].grid(alpha=0.3)
        axes_anim[1].set_title(f'Purity = {purities[frame]:.3f}', fontweight='bold')
        axes_anim[1].axhline(1.0, color='g', linestyle='--', alpha=0.3)
        
        # Off-diagonal element magnitude
        coherence = np.abs(rho_frame[0, 1])
        axes_anim[2].plot(times[:frame+1], 
                         [np.abs(density_matrices[i][0, 1]) for i in range(frame+1)],
                         'g-', linewidth=3)
        axes_anim[2].scatter([t], [coherence], s=100, c='red', zorder=5)
        axes_anim[2].set_xlabel('Time')
        axes_anim[2].set_ylabel('|ρ₀₁| (Coherence)')
        axes_anim[2].set_xlim(0, t_max)
        axes_anim[2].set_ylim(0, 0.6)
        axes_anim[2].grid(alpha=0.3)
        axes_anim[2].set_title(f'Coherence = {coherence:.3f}', fontweight='bold')
        
        return axes_anim
    
    anim = FuncAnimation(fig_anim, update, frames=time_steps, 
                        interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('quantum_decoherence_animation.gif', writer=writer)
    print(f"✓ Saved: quantum_decoherence_animation.gif")
    plt.close()
    
    print(f"\n   Final purity: {purities[-1]:.3f}")
    print(f"   Final entropy: {entropies[-1]:.3f} bits")
    print(f"   Coherence lost: {100*(1-np.abs(density_matrices[-1][0,1])/np.abs(density_matrices[0][0,1])):.1f}%")
    
    print("\n" + "=" * 60)
    print("Quantum decoherence visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_decoherence_evolution()
