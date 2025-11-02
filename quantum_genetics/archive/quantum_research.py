
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, exp, factorial
import random
from scipy.linalg import expm
from scipy.special import sph_harm, genlaguerre

print("=" * 70)
print("  ADVANCED QUANTUM RESEARCH APPLICATIONS")
print("=" * 70)
print("\n🔬 Loading research modules...")
print("   • Quantum Chemistry (Molecular Orbitals)")
print("   • Condensed Matter (Band Structure)")
print("   • Quantum Computing (Grover's Algorithm)")
print()

# ============================================================================
# 1. QUANTUM CHEMISTRY - Molecular Orbital Theory
# ============================================================================

def hydrogen_molecular_orbital(x, y, z, R=2.0, orbital_type='bonding'):
    """
    H₂ molecular orbital using LCAO (Linear Combination of Atomic Orbitals)
    R: internuclear distance (Bohr radii)
    """
    # Two hydrogen nuclei at ±R/2 along z-axis
    r1 = np.sqrt(x**2 + y**2 + (z - R/2)**2)
    r2 = np.sqrt(x**2 + y**2 + (z + R/2)**2)
    
    # 1s atomic orbitals
    psi_1s_1 = (1/np.sqrt(pi)) * np.exp(-r1)
    psi_1s_2 = (1/np.sqrt(pi)) * np.exp(-r2)
    
    # Molecular orbitals
    if orbital_type == 'bonding':
        # σ bonding orbital
        psi = (psi_1s_1 + psi_1s_2) / np.sqrt(2)
    else:
        # σ* antibonding orbital
        psi = (psi_1s_1 - psi_1s_2) / np.sqrt(2)
    
    return psi

def plot_molecular_orbitals():
    """Visualize H₂ molecular orbitals"""
    print("\n[1] QUANTUM CHEMISTRY - H₂ Molecular Orbitals")
    print("=" * 70)
    print("📊 Computing bonding and antibonding orbitals...\n")
    
    # Create grid
    x = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)
    
    R = 2.0  # Internuclear distance
    
    # Compute orbitals
    psi_bonding = hydrogen_molecular_orbital(X, Y, Z, R, 'bonding')
    psi_antibonding = hydrogen_molecular_orbital(X, Y, Z, R, 'antibonding')
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Bonding orbital
    contour1 = axes[0, 0].contourf(X, Z, psi_bonding, levels=30, cmap='RdBu_r')
    axes[0, 0].plot([0, 0], [-R/2, R/2], 'ko', markersize=10, label='H nuclei')
    axes[0, 0].set_title('σ Bonding Orbital (ψ_g)', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('x (Bohr)', fontsize=10)
    axes[0, 0].set_ylabel('z (Bohr)', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')
    plt.colorbar(contour1, ax=axes[0, 0])
    
    # Antibonding orbital
    contour2 = axes[0, 1].contourf(X, Z, psi_antibonding, levels=30, cmap='RdBu_r')
    axes[0, 1].plot([0, 0], [-R/2, R/2], 'ko', markersize=10, label='H nuclei')
    axes[0, 1].set_title('σ* Antibonding Orbital (ψ_u)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('x (Bohr)', fontsize=10)
    axes[0, 1].set_ylabel('z (Bohr)', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].set_aspect('equal')
    plt.colorbar(contour2, ax=axes[0, 1])
    
    # Probability density
    prob_bonding = np.abs(psi_bonding)**2
    prob_antibonding = np.abs(psi_antibonding)**2
    
    axes[1, 0].contourf(X, Z, prob_bonding, levels=30, cmap='hot')
    axes[1, 0].plot([0, 0], [-R/2, R/2], 'wo', markersize=10)
    axes[1, 0].set_title('|ψ_g|² - Electron Density (Bonding)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('x (Bohr)', fontsize=10)
    axes[1, 0].set_ylabel('z (Bohr)', fontsize=10)
    axes[1, 0].set_aspect('equal')
    
    axes[1, 1].contourf(X, Z, prob_antibonding, levels=30, cmap='hot')
    axes[1, 1].plot([0, 0], [-R/2, R/2], 'wo', markersize=10)
    axes[1, 1].set_title('|ψ_u|² - Electron Density (Antibonding)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('x (Bohr)', fontsize=10)
    axes[1, 1].set_ylabel('z (Bohr)', fontsize=10)
    axes[1, 1].set_aspect('equal')
    
    plt.suptitle('H₂ Molecular Orbitals - Quantum Chemistry', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('research_molecular_orbitals.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: research_molecular_orbitals.png")
    plt.close()

# ============================================================================
# 2. CONDENSED MATTER - Band Structure
# ============================================================================

def kronig_penney_bands(k_range, V0=5.0, a=1.0):
    """
    Simplified band structure using Kronig-Penney model
    Periodic potential V(x) with period a
    """
    energies = []
    
    for k in k_range:
        # Simplified dispersion relation for 1D crystal
        E = (k**2 / 2.0) + V0 * (np.sin(k * a) / (k * a))**2
        energies.append(E)
    
    return np.array(energies)

def plot_band_structure():
    """Visualize electronic band structure"""
    print("\n[2] CONDENSED MATTER PHYSICS - Electronic Band Structure")
    print("=" * 70)
    print("📊 Computing band structure for 1D periodic crystal...\n")
    
    # First Brillouin zone
    k = np.linspace(-pi, pi, 500)
    
    # Multiple bands with different potentials
    E1 = kronig_penney_bands(k, V0=2.0, a=1.0)
    E2 = kronig_penney_bands(k, V0=4.0, a=1.0) + 5
    E3 = kronig_penney_bands(k, V0=3.0, a=1.0) + 12
    
    # Density of states (simplified)
    energy_range = np.linspace(0, 20, 200)
    dos = np.zeros_like(energy_range)
    
    for E_band in [E1, E2, E3]:
        for E in E_band:
            idx = np.argmin(np.abs(energy_range - E))
            if 0 <= idx < len(dos):
                dos[idx] += 1
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Band structure
    axes[0].plot(k, E1, 'b-', lw=2, label='Valence band')
    axes[0].plot(k, E2, 'r-', lw=2, label='Conduction band')
    axes[0].plot(k, E3, 'g-', lw=2, label='Higher band')
    axes[0].axhline(E1.max(), color='blue', linestyle='--', alpha=0.3)
    axes[0].axhline(E2.min(), color='red', linestyle='--', alpha=0.3)
    axes[0].fill_between(k, E1.max(), E2.min(), alpha=0.2, color='yellow', label='Band gap')
    axes[0].set_xlabel('Crystal momentum k (π/a)', fontsize=11)
    axes[0].set_ylabel('Energy (eV)', fontsize=11)
    axes[0].set_title('Electronic Band Structure', fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(-pi, pi)
    
    # Density of states
    axes[1].plot(dos, energy_range, 'k-', lw=2)
    axes[1].fill_betweenx(energy_range, 0, dos, alpha=0.3, color='blue')
    axes[1].axhline(E1.max(), color='blue', linestyle='--', alpha=0.5, label='Valence max')
    axes[1].axhline(E2.min(), color='red', linestyle='--', alpha=0.5, label='Conduction min')
    axes[1].set_xlabel('Density of States', fontsize=11)
    axes[1].set_ylabel('Energy (eV)', fontsize=11)
    axes[1].set_title('Density of States', fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('1D Crystal Band Structure - Condensed Matter Physics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('research_band_structure.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: research_band_structure.png")
    plt.close()

# ============================================================================
# 3. QUANTUM COMPUTING - Grover's Algorithm
# ============================================================================

def grover_diffusion_operator(n_qubits):
    """Grover diffusion operator D = 2|ψ⟩⟨ψ| - I"""
    N = 2**n_qubits
    psi = np.ones(N) / np.sqrt(N)
    D = 2 * np.outer(psi, psi) - np.eye(N)
    return D

def grover_oracle(target_state, n_qubits):
    """Oracle that flips the sign of the target state"""
    N = 2**n_qubits
    oracle = np.eye(N)
    oracle[target_state, target_state] = -1
    return oracle

def run_grover_algorithm(n_qubits=3, target_state=5):
    """Simulate Grover's search algorithm"""
    N = 2**n_qubits
    
    # Initial superposition
    state = np.ones(N) / np.sqrt(N)
    
    # Optimal number of iterations
    n_iterations = int(np.pi / 4 * np.sqrt(N))
    
    # Track evolution
    prob_evolution = []
    
    # Grover iteration
    oracle = grover_oracle(target_state, n_qubits)
    diffusion = grover_diffusion_operator(n_qubits)
    
    for _ in range(n_iterations + 1):
        prob_evolution.append(np.abs(state)**2)
        state = diffusion @ oracle @ state
    
    return np.array(prob_evolution), n_iterations

def plot_grover_algorithm():
    """Visualize Grover's quantum search"""
    print("\n[3] QUANTUM COMPUTING - Grover's Search Algorithm")
    print("=" * 70)
    print("📊 Simulating quantum search on 3-qubit system (8 states)...\n")
    
    n_qubits = 3
    target = 5
    prob_evolution, n_iter = run_grover_algorithm(n_qubits, target)
    
    N = 2**n_qubits
    
    print(f"   Search space: {N} states")
    print(f"   Target state: |{target}⟩ = |{bin(target)[2:].zfill(n_qubits)}⟩")
    print(f"   Grover iterations: {n_iter}")
    print(f"   Final probability: {prob_evolution[-1][target]:.3f}\n")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Evolution heatmap
    im = axes[0, 0].imshow(prob_evolution.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0, 0].set_xlabel('Grover Iteration', fontsize=11)
    axes[0, 0].set_ylabel('Quantum State', fontsize=11)
    axes[0, 0].set_title('Probability Evolution', fontweight='bold', fontsize=12)
    axes[0, 0].axhline(target - 0.5, color='cyan', linestyle='--', lw=2, label=f'Target |{target}⟩')
    axes[0, 0].legend()
    plt.colorbar(im, ax=axes[0, 0], label='Probability')
    
    # Target probability vs iteration
    axes[0, 1].plot(range(len(prob_evolution)), prob_evolution[:, target], 
                   'r-o', lw=2, markersize=6, label=f'Target state |{target}⟩')
    axes[0, 1].axhline(1/N, color='blue', linestyle='--', label='Random search')
    axes[0, 1].set_xlabel('Iteration', fontsize=11)
    axes[0, 1].set_ylabel('Probability', fontsize=11)
    axes[0, 1].set_title('Target State Amplification', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Initial vs Final distribution
    axes[1, 0].bar(range(N), prob_evolution[0], alpha=0.6, label='Initial', edgecolor='black')
    axes[1, 0].bar(range(N), prob_evolution[-1], alpha=0.6, label='Final', edgecolor='black')
    axes[1, 0].axvline(target, color='red', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Quantum State', fontsize=11)
    axes[1, 0].set_ylabel('Probability', fontsize=11)
    axes[1, 0].set_title('Initial vs Final Measurement Probabilities', fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Algorithm statistics
    speedup = N / n_iter
    stats_text = f"Grover's Algorithm:\n\n"
    stats_text += f"Qubits: {n_qubits}\n"
    stats_text += f"Search space: {N} states\n"
    stats_text += f"Target: |{target}⟩\n\n"
    stats_text += f"Iterations: {n_iter}\n"
    stats_text += f"Classical search: ~{N//2}\n"
    stats_text += f"Quantum speedup: {speedup:.1f}x\n\n"
    stats_text += f"Success probability:\n"
    stats_text += f"  Initial: {prob_evolution[0][target]:.3f}\n"
    stats_text += f"  Final: {prob_evolution[-1][target]:.3f}\n"
    
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.suptitle("Grover's Quantum Search Algorithm", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('research_grover_algorithm.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: research_grover_algorithm.png")
    plt.close()

# ============================================================================
# MAIN RESEARCH SUITE
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  QUANTUM RESEARCH APPLICATIONS SUITE")
    print("=" * 70)
    print("\nRunning advanced quantum simulations across three domains:\n")
    
    plot_molecular_orbitals()
    plot_band_structure()
    plot_grover_algorithm()
    
    print("\n" + "=" * 70)
    print("✨ RESEARCH SUITE COMPLETE!")
    print("=" * 70)
    print("\n🔬 Generated visualizations:")
    print("   • Molecular orbital theory (quantum chemistry)")
    print("   • Electronic band structure (condensed matter)")
    print("   • Grover's search algorithm (quantum computing)")
    print("\n💡 Applications:")
    print("   • Drug design & molecular modeling")
    print("   • Semiconductor device physics")
    print("   • Quantum algorithm development")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
