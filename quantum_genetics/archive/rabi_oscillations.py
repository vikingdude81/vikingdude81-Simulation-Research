
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt, cos, sin
import random

def rabi_oscillation(t, omega_rabi, delta=0):
    """
    Solve time evolution of driven two-level system
    omega_rabi: Rabi frequency (coupling strength)
    delta: Detuning from resonance
    """
    omega_eff = sqrt(omega_rabi**2 + delta**2)
    
    # Probability of being in excited state
    P_excited = (omega_rabi / omega_eff)**2 * sin(omega_eff * t / 2)**2
    P_ground = 1 - P_excited
    
    return P_ground, P_excited

def bloch_vector_rabi(t, omega_rabi, delta=0):
    """Bloch vector for Rabi oscillations"""
    omega_eff = sqrt(omega_rabi**2 + delta**2)
    
    theta_rabi = 2 * np.arcsin(omega_rabi / omega_eff)
    
    x = sin(theta_rabi) * sin(omega_eff * t)
    y = sin(theta_rabi) * cos(omega_eff * t)
    z = cos(theta_rabi) + (1 - cos(theta_rabi)) * cos(omega_eff * t)
    
    return np.array([x, y, z])

def plot_rabi_oscillations():
    """Visualize Rabi oscillations in driven two-level system"""
    
    # Random parameters
    omega_rabi = random.uniform(1.5, 3.0)
    delta = random.choice([0, random.uniform(-1, 1)])  # On/off resonance
    
    resonance_status = "ON resonance" if abs(delta) < 0.1 else "OFF resonance"
    
    print("=" * 60)
    print("  RABI OSCILLATIONS - DRIVEN TWO-LEVEL SYSTEM")
    print("=" * 60)
    print(f"\n🎲 System parameters:")
    print(f"   Rabi frequency Ω = {omega_rabi:.3f}")
    print(f"   Detuning δ = {delta:.3f}")
    print(f"   Status: {resonance_status}")
    print()
    print("🌊 Simulating coherent population transfer...")
    print("   |g⟩ ↔ |e⟩ oscillations driven by external field")
    print()
    
    # Time evolution
    t_max = 20
    times = np.linspace(0, t_max, 500)
    
    P_ground = []
    P_excited = []
    bloch_vecs = []
    
    for t in times:
        Pg, Pe = rabi_oscillation(t, omega_rabi, delta)
        P_ground.append(Pg)
        P_excited.append(Pe)
        bloch_vecs.append(bloch_vector_rabi(t, omega_rabi, delta))
    
    bloch_vecs = np.array(bloch_vecs)
    
    print("🎨 Creating Rabi oscillation visualization...")
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Population dynamics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, P_ground, 'b-', linewidth=2.5, label='Ground state |g⟩')
    ax1.plot(times, P_excited, 'r-', linewidth=2.5, label='Excited state |e⟩')
    ax1.fill_between(times, 0, P_ground, alpha=0.3, color='blue')
    ax1.fill_between(times, 0, P_excited, alpha=0.3, color='red')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Population', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.set_title(f'Rabi Oscillations: Ω={omega_rabi:.2f}, δ={delta:.2f} ({resonance_status})',
                 fontsize=14, fontweight='bold')
    
    # Calculate Rabi period
    omega_eff = sqrt(omega_rabi**2 + delta**2)
    T_rabi = 2 * pi / omega_eff
    
    # Mark Rabi periods
    for n in range(int(t_max / T_rabi) + 1):
        t_period = n * T_rabi
        if t_period <= t_max:
            ax1.axvline(t_period, color='green', linestyle='--', alpha=0.4)
    
    ax1.text(0.02, 0.98, f'Rabi period T = {T_rabi:.2f}',
            transform=ax1.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Bloch sphere trajectory (XY)
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Unit circle
    theta_circle = np.linspace(0, 2*pi, 100)
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)
    
    # Trajectory
    ax2.plot(bloch_vecs[:, 0], bloch_vecs[:, 1], 'purple', linewidth=2)
    ax2.scatter([bloch_vecs[0, 0]], [bloch_vecs[0, 1]], 
               s=150, c='green', marker='o', edgecolors='black', 
               linewidth=2, label='Start', zorder=5)
    ax2.scatter([bloch_vecs[-1, 0]], [bloch_vecs[-1, 1]], 
               s=150, c='red', marker='s', edgecolors='black',
               linewidth=2, label='End', zorder=5)
    
    ax2.set_xlabel('X (Bloch)', fontsize=11)
    ax2.set_ylabel('Y (Bloch)', fontsize=11)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('Bloch Sphere Trajectory (XY)', fontweight='bold')
    
    # Plot 3: Z-component evolution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(times, bloch_vecs[:, 2], 'g-', linewidth=2.5)
    ax3.axhline(1, color='b', linestyle='--', alpha=0.4, label='|g⟩')
    ax3.axhline(-1, color='r', linestyle='--', alpha=0.4, label='|e⟩')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.2)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Z (Bloch)', fontsize=11)
    ax3.set_ylim(-1.2, 1.2)
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_title('Population Inversion (Z-axis)', fontweight='bold')
    
    # Plot 4: Phase space
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(P_ground, P_excited, 'purple', linewidth=2)
    ax4.scatter([P_ground[0]], [P_excited[0]], s=150, c='green', 
               marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax4.scatter([P_ground[-1]], [P_excited[-1]], s=150, c='red',
               marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax4.set_xlabel('P(|g⟩)', fontsize=11)
    ax4.set_ylabel('P(|e⟩)', fontsize=11)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(alpha=0.3)
    ax4.set_title('Phase Space', fontweight='bold')
    
    # Plot 5: Energy level diagram
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Draw energy levels
    y_ground = 0.2
    y_excited = 0.8
    
    ax5.plot([0.2, 0.8], [y_ground, y_ground], 'b-', linewidth=4, label='|g⟩')
    ax5.plot([0.2, 0.8], [y_excited, y_excited], 'r-', linewidth=4, label='|e⟩')
    
    # Draw transition arrow
    ax5.annotate('', xy=(0.5, y_excited - 0.05), xytext=(0.5, y_ground + 0.05),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=3))
    
    ax5.text(0.55, (y_ground + y_excited) / 2, f'ℏΩ = {omega_rabi:.2f}',
            fontsize=12, va='center')
    
    ax5.text(0.1, y_ground, '|g⟩', fontsize=14, va='center', fontweight='bold')
    ax5.text(0.1, y_excited, '|e⟩', fontsize=14, va='center', fontweight='bold')
    
    if abs(delta) > 0.1:
        ax5.text(0.5, 0.05, f'Detuning δ = {delta:.2f}',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('Two-Level System', fontweight='bold')
    
    plt.suptitle('Rabi Oscillations in Driven Quantum System',
                fontsize=15, fontweight='bold')
    
    filename = 'rabi_oscillations.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Create animation
    print("🎬 Creating Rabi oscillation animation...")
    
    fig_anim, axes_anim = plt.subplots(1, 2, figsize=(14, 6))
    
    def update(frame):
        for ax in axes_anim:
            ax.clear()
        
        t_current = times[frame]
        
        # Population dynamics
        axes_anim[0].plot(times[:frame+1], P_ground[:frame+1], 'b-', linewidth=2.5)
        axes_anim[0].plot(times[:frame+1], P_excited[:frame+1], 'r-', linewidth=2.5)
        axes_anim[0].scatter([t_current], [P_ground[frame]], s=100, c='blue', zorder=5)
        axes_anim[0].scatter([t_current], [P_excited[frame]], s=100, c='red', zorder=5)
        axes_anim[0].set_xlabel('Time', fontsize=12)
        axes_anim[0].set_ylabel('Population', fontsize=12)
        axes_anim[0].set_xlim(0, t_max)
        axes_anim[0].set_ylim(-0.05, 1.05)
        axes_anim[0].grid(alpha=0.3)
        axes_anim[0].set_title(f'Population Dynamics: t={t_current:.2f}', fontweight='bold')
        axes_anim[0].legend(['|g⟩', '|e⟩'], fontsize=11)
        
        # Bloch sphere XZ projection
        theta_circle = np.linspace(0, 2*pi, 100)
        axes_anim[1].plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)
        
        axes_anim[1].plot(bloch_vecs[:frame+1, 0], bloch_vecs[:frame+1, 2], 
                         'purple', linewidth=2)
        axes_anim[1].scatter([bloch_vecs[frame, 0]], [bloch_vecs[frame, 2]],
                            s=200, c='yellow', edgecolors='black', linewidth=2, zorder=5)
        
        axes_anim[1].set_xlabel('X (Bloch)', fontsize=12)
        axes_anim[1].set_ylabel('Z (Bloch)', fontsize=12)
        axes_anim[1].set_xlim(-1.2, 1.2)
        axes_anim[1].set_ylim(-1.2, 1.2)
        axes_anim[1].set_aspect('equal')
        axes_anim[1].grid(alpha=0.3)
        axes_anim[1].axhline(1, color='b', linestyle='--', alpha=0.3)
        axes_anim[1].axhline(-1, color='r', linestyle='--', alpha=0.3)
        axes_anim[1].set_title('Bloch Vector (XZ)', fontweight='bold')
        
        return axes_anim
    
    anim = FuncAnimation(fig_anim, update, frames=len(times), 
                        interval=30, blit=False)
    
    writer = PillowWriter(fps=25)
    anim.save('rabi_oscillations_animation.gif', writer=writer)
    print(f"✓ Saved: rabi_oscillations_animation.gif")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Rabi oscillations visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_rabi_oscillations()
