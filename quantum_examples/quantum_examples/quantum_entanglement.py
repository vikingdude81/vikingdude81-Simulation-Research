
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
import random

def bell_state(state_type):
    """
    Create Bell states (maximally entangled 2-qubit states)
    |Φ+⟩ = (|00⟩ + |11⟩)/√2
    |Φ-⟩ = (|00⟩ - |11⟩)/√2
    |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    """
    states = {
        'phi_plus': np.array([1, 0, 0, 1]) / sqrt(2),
        'phi_minus': np.array([1, 0, 0, -1]) / sqrt(2),
        'psi_plus': np.array([0, 1, 1, 0]) / sqrt(2),
        'psi_minus': np.array([0, 1, -1, 0]) / sqrt(2)
    }
    return states[state_type]

def density_matrix(state):
    """Compute density matrix ρ = |ψ⟩⟨ψ|"""
    return np.outer(state, np.conj(state))

def plot_entanglement():
    bell_types = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']
    bell_names = ['|Φ+⟩', '|Φ-⟩', '|Ψ+⟩', '|Ψ-⟩']
    
    chosen = random.choice(bell_types)
    chosen_name = bell_names[bell_types.index(chosen)]
    
    print("=" * 60)
    print("  QUANTUM ENTANGLEMENT - BELL STATES")
    print("=" * 60)
    print(f"\n🎲 Random Bell state: {chosen_name}")
    print(f"   Basis: |00⟩, |01⟩, |10⟩, |11⟩")
    print(f"   Property: Measuring one qubit instantly determines the other!")
    print()
    
    state = bell_state(chosen)
    rho = density_matrix(state)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: State vector
    ax1 = fig.add_subplot(131)
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    colors = ['red' if np.abs(amp) > 0.6 else 'blue' for amp in state]
    ax1.bar(basis_labels, np.abs(state)**2, color=colors, alpha=0.7)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(f'Bell State {chosen_name}\nProbability Distribution', fontweight='bold')
    ax1.set_ylim(0, 0.6)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Density matrix (real part)
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(np.real(rho), cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(basis_labels)
    ax2.set_yticklabels(basis_labels)
    ax2.set_title('Density Matrix Re[ρ]', fontweight='bold')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Correlation visualization
    ax3 = fig.add_subplot(133)
    
    # Show entanglement correlations
    correlation_text = f"Bell State: {chosen_name}\n\n"
    correlation_text += "Entanglement Properties:\n"
    correlation_text += "• Perfect correlation\n"
    correlation_text += "• Non-local\n"
    correlation_text += "• Violates Bell inequality\n"
    correlation_text += "• Cannot be factored\n\n"
    correlation_text += "Measurement outcomes:\n"
    
    if chosen in ['phi_plus', 'phi_minus']:
        correlation_text += "If A measures 0 → B gets 0\n"
        correlation_text += "If A measures 1 → B gets 1"
    else:
        correlation_text += "If A measures 0 → B gets 1\n"
        correlation_text += "If A measures 1 → B gets 0"
    
    ax3.text(0.5, 0.5, correlation_text, ha='center', va='center',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax3.axis('off')
    
    plt.tight_layout()
    filename = f'entanglement_{chosen}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Quantum entanglement visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_entanglement()
