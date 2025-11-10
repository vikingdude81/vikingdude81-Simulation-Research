
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
import random

def pauli_x():
    """Pauli-X (NOT) gate"""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_y():
    """Pauli-Y gate"""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def pauli_z():
    """Pauli-Z gate"""
    return np.array([[1, 0], [0, -1]], dtype=complex)

def hadamard():
    """Hadamard gate"""
    return np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)

def phase_gate(phi):
    """Phase gate"""
    return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)

def rotation_x(theta):
    """Rotation around X axis"""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def rotation_y(theta):
    """Rotation around Y axis"""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def rotation_z(theta):
    """Rotation around Z axis"""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

def apply_gate(gate, state):
    """Apply quantum gate to state"""
    return gate @ state

def state_to_bloch(state):
    """Convert quantum state to Bloch sphere coordinates"""
    alpha, beta = state[0], state[1]
    
    # Bloch vector components
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return np.array([x, y, z])

def plot_quantum_circuit():
    """Visualize quantum circuit with gates"""
    # Random initial state
    theta = random.uniform(0, pi)
    phi = random.uniform(0, 2*pi)
    
    alpha = np.cos(theta/2)
    beta = np.exp(1j*phi) * np.sin(theta/2)
    initial_state = np.array([alpha, beta], dtype=complex)
    
    # Random sequence of gates
    gate_choices = [
        ('H', hadamard(), 'Hadamard'),
        ('X', pauli_x(), 'Pauli-X'),
        ('Y', pauli_y(), 'Pauli-Y'),
        ('Z', pauli_z(), 'Pauli-Z'),
        ('Rz(π/4)', rotation_z(pi/4), 'Z-Rotation')
    ]
    
    n_gates = random.randint(3, 5)
    selected_gates = random.sample(gate_choices, n_gates)
    
    print("=" * 60)
    print("  QUANTUM GATE CIRCUIT SIMULATION")
    print("=" * 60)
    print(f"\n🎲 Initial state: |ψ₀⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
    print(f"   Bloch angles: θ={theta*180/pi:.1f}°, φ={phi*180/pi:.1f}°")
    print(f"\n📊 Gate sequence ({n_gates} gates):")
    for i, (symbol, _, name) in enumerate(selected_gates):
        print(f"   {i+1}. {name} ({symbol})")
    print()
    
    # Track state evolution
    states = [initial_state]
    state = initial_state.copy()
    
    for symbol, gate, name in selected_gates:
        state = apply_gate(gate, state)
        states.append(state.copy())
    
    print("🎨 Creating circuit diagram...")
    
    # Create circuit visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Circuit diagram
    ax1 = axes[0]
    ax1.set_xlim(-0.5, n_gates + 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.axis('off')
    
    # Draw qubit line
    ax1.plot([0, n_gates + 1], [0.5, 0.5], 'k-', linewidth=2)
    
    # Initial state label
    ax1.text(-0.3, 0.5, '|ψ₀⟩', ha='right', va='center', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # Draw gates
    for i, (symbol, _, name) in enumerate(selected_gates):
        x_pos = i + 0.5
        
        # Gate box
        if symbol == 'H':
            color = 'yellow'
        elif symbol in ['X', 'Y', 'Z']:
            color = 'lightcoral'
        else:
            color = 'lightgreen'
        
        rect = plt.Rectangle((x_pos - 0.3, 0.2), 0.6, 0.6, 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x_pos, 0.5, symbol, ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    # Final state
    ax1.text(n_gates + 1.3, 0.5, '|ψf⟩', ha='left', va='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    ax1.set_title('Quantum Circuit Diagram', fontsize=14, fontweight='bold', pad=20)
    
    # State evolution on Bloch sphere projection
    ax2 = axes[1]
    
    # Convert all states to Bloch vectors
    bloch_vectors = [state_to_bloch(s) for s in states]
    
    # Plot XY projection
    x_coords = [v[0] for v in bloch_vectors]
    y_coords = [v[1] for v in bloch_vectors]
    z_coords = [v[2] for v in bloch_vectors]
    
    # Draw Bloch sphere circle
    theta_circle = np.linspace(0, 2*pi, 100)
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3, linewidth=1)
    
    # Plot state evolution
    for i in range(len(bloch_vectors) - 1):
        ax2.arrow(x_coords[i], y_coords[i], 
                 x_coords[i+1] - x_coords[i], 
                 y_coords[i+1] - y_coords[i],
                 head_width=0.08, head_length=0.08, 
                 fc=f'C{i}', ec=f'C{i}', linewidth=2)
    
    # Mark initial and final states
    ax2.scatter(x_coords[0], y_coords[0], s=200, c='blue', 
               marker='o', edgecolors='black', linewidth=2, 
               label='Initial |ψ₀⟩', zorder=5)
    ax2.scatter(x_coords[-1], y_coords[-1], s=200, c='red', 
               marker='s', edgecolors='black', linewidth=2, 
               label='Final |ψf⟩', zorder=5)
    
    # Add gate labels
    for i, (symbol, _, _) in enumerate(selected_gates):
        ax2.text(x_coords[i+1], y_coords[i+1] + 0.15, symbol, 
                ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('X (Bloch sphere)', fontsize=12)
    ax2.set_ylabel('Y (Bloch sphere)', fontsize=12)
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_title('State Evolution on Bloch Sphere (XY Projection)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    filename = 'quantum_gates_circuit.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Create detailed probability evolution
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    # Probability of |0⟩ and |1⟩
    prob_0 = [np.abs(s[0])**2 for s in states]
    prob_1 = [np.abs(s[1])**2 for s in states]
    
    steps = list(range(len(states)))
    gate_labels = ['Init'] + [symbol for symbol, _, _ in selected_gates]
    
    axes2[0, 0].plot(steps, prob_0, 'b-o', linewidth=2, markersize=8, label='P(|0⟩)')
    axes2[0, 0].plot(steps, prob_1, 'r-s', linewidth=2, markersize=8, label='P(|1⟩)')
    axes2[0, 0].set_xlabel('Gate Step', fontsize=11)
    axes2[0, 0].set_ylabel('Probability', fontsize=11)
    axes2[0, 0].set_xticks(steps)
    axes2[0, 0].set_xticklabels(gate_labels, rotation=45)
    axes2[0, 0].set_ylim(0, 1.1)
    axes2[0, 0].legend(fontsize=11)
    axes2[0, 0].grid(alpha=0.3)
    axes2[0, 0].set_title('Measurement Probabilities', fontweight='bold')
    
    # Bloch Z-coordinate
    axes2[0, 1].plot(steps, z_coords, 'g-^', linewidth=2, markersize=8)
    axes2[0, 1].set_xlabel('Gate Step', fontsize=11)
    axes2[0, 1].set_ylabel('Z (Bloch)', fontsize=11)
    axes2[0, 1].set_xticks(steps)
    axes2[0, 1].set_xticklabels(gate_labels, rotation=45)
    axes2[0, 1].set_ylim(-1.1, 1.1)
    axes2[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes2[0, 1].grid(alpha=0.3)
    axes2[0, 1].set_title('Bloch Z-coordinate Evolution', fontweight='bold')
    
    # Final state bar chart
    axes2[1, 0].bar(['|0⟩', '|1⟩'], [prob_0[-1], prob_1[-1]], 
                   color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
    axes2[1, 0].set_ylabel('Probability', fontsize=11)
    axes2[1, 0].set_ylim(0, 1.1)
    axes2[1, 0].set_title('Final State Measurement', fontweight='bold')
    axes2[1, 0].grid(alpha=0.3, axis='y')
    
    # Summary text
    final_state = states[-1]
    summary_text = f"Circuit Summary:\n\n"
    summary_text += f"Gates applied: {n_gates}\n\n"
    summary_text += f"Initial: |ψ₀⟩\n"
    summary_text += f"  P(|0⟩) = {prob_0[0]:.3f}\n"
    summary_text += f"  P(|1⟩) = {prob_1[0]:.3f}\n\n"
    summary_text += f"Final: |ψf⟩\n"
    summary_text += f"  P(|0⟩) = {prob_0[-1]:.3f}\n"
    summary_text += f"  P(|1⟩) = {prob_1[-1]:.3f}\n"
    
    axes2[1, 1].text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes2[1, 1].axis('off')
    
    plt.suptitle('Quantum Gate Circuit Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename2 = 'quantum_gates_analysis.png'
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Quantum gates visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_quantum_circuit()
