import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt

def gaussian_wavepacket(x, x0, k0, sigma):
    """Gaussian wavepacket"""
    return (1 / (sigma * sqrt(2 * pi)))**0.5 * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)

def potential_barrier(x, barrier_start, barrier_end, V0):
    """Rectangular potential barrier"""
    V = np.zeros_like(x)
    V[(x >= barrier_start) & (x <= barrier_end)] = V0
    return V

def plot_tunneling():
    print("=" * 60)
    print("  QUANTUM TUNNELING SIMULATION")
    print("=" * 60)
    print("\n🎯 Simulating particle approaching potential barrier...")
    print()

    # Setup
    x = np.linspace(-10, 10, 500)
    barrier_start, barrier_end = -1, 1
    V0 = 5.0  # Barrier height

    # Initial wavepacket
    x0 = -5
    k0 = 3.0  # Momentum (kinetic energy = k0²/2)
    sigma = 0.5

    print(f"   Particle energy: E = {k0**2/2:.2f}")
    print(f"   Barrier height: V = {V0:.2f}")
    print(f"   Classical result: {'Reflected' if k0**2/2 < V0 else 'Transmitted'}")
    print(f"   Quantum result: Partial tunneling possible!")
    print()

    # Create animation
    print("🎬 Creating tunneling animation...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    V = potential_barrier(x, barrier_start, barrier_end, V0)

    frames = 100
    time_points = np.linspace(0, 5, frames)

    def update(frame):
        ax1.clear()
        ax2.clear()

        t = time_points[frame]

        # Time evolution (simplified - just translation)
        psi = gaussian_wavepacket(x, x0 + k0*t, k0, sigma)

        # Add damping near barrier (simulate scattering)
        damping = np.exp(-0.5 * ((x - 0)**2) / 4)
        psi = psi * (1 - 0.3 * (1 - damping))

        probability = np.abs(psi)**2
        real_part = np.real(psi)

        # Plot probability density
        ax1.fill_between(x, 0, probability, alpha=0.6, color='blue', label='|ψ|²')
        ax1.plot(x, V/10, 'r-', linewidth=3, label='Barrier (V/10)')
        ax1.axvline(barrier_start, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(barrier_end, color='r', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_ylim(0, 1.5)
        ax1.set_xlim(-10, 10)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title(f'Quantum Tunneling - Time: {t:.2f}', fontweight='bold', fontsize=12)

        # Plot real part
        ax2.plot(x, real_part, 'b-', linewidth=2, label='Re[ψ]')
        ax2.fill_between(x, barrier_start, barrier_end, alpha=0.3, color='red', 
                         transform=ax2.get_xaxis_transform(), label='Barrier')
        ax2.set_xlabel('Position x', fontsize=11)
        ax2.set_ylabel('Re[ψ(x,t)]', fontsize=11)
        ax2.set_ylim(-1, 1)
        ax2.set_xlim(-10, 10)
        ax2.legend()
        ax2.grid(alpha=0.3)

        return ax1, ax2

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    writer = PillowWriter(fps=20)
    anim.save('quantum_tunneling.gif', writer=writer)
    print(f"✓ Saved: quantum_tunneling.gif")
    plt.close()

    print("\n" + "=" * 60)
    print("Quantum tunneling visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_tunneling()