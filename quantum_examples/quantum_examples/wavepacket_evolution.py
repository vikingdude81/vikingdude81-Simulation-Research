import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt

def gaussian_wavepacket(x, t, x0, k0, sigma, m=1, hbar=1):
    """
    Free particle Gaussian wavepacket with time evolution
    Shows quantum spreading over time
    """
    sigma_t = sigma * sqrt(1 + (hbar * t / (2 * m * sigma**2))**2)

    norm = (1 / (sigma_t * sqrt(2 * pi)))**0.5
    phase = k0 * (x - x0) - (hbar * k0**2 * t) / (2 * m)
    envelope = np.exp(-((x - x0 - hbar * k0 * t / m)**2) / (4 * sigma_t**2))

    return norm * envelope * np.exp(1j * phase)

def plot_wavepacket_spreading():
    print("=" * 60)
    print("  QUANTUM WAVEPACKET SPREADING")
    print("=" * 60)
    print("\n🌊 Simulating free particle wavepacket evolution...")
    print("   (Demonstrates Heisenberg Uncertainty Principle)")
    print()

    # Setup
    x = np.linspace(-15, 15, 600)
    x0 = 0
    k0 = 2.0
    sigma0 = 1.0

    print(f"   Initial position uncertainty: Δx₀ = {sigma0:.2f}")
    print(f"   Initial momentum: k₀ = {k0:.2f}")
    print(f"   Δx · Δp ≥ ℏ/2 → Wavepacket must spread!")
    print()

    # Create animation
    print("🎬 Creating wavepacket evolution animation...")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    frames = 80
    time_points = np.linspace(0, 10, frames)

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        t = time_points[frame]

        psi = gaussian_wavepacket(x, t, x0, k0, sigma0)
        probability = np.abs(psi)**2
        real_part = np.real(psi)
        imag_part = np.imag(psi)

        sigma_t = sigma0 * sqrt(1 + (t / (2 * sigma0**2))**2)

        # Probability density
        ax1.fill_between(x, 0, probability, alpha=0.6, color='blue')
        ax1.plot(x, probability, 'b-', linewidth=2)
        ax1.set_ylabel('|ψ|²', fontsize=12)
        ax1.set_ylim(0, 0.5)
        ax1.set_xlim(-15, 15)
        ax1.grid(alpha=0.3)
        ax1.set_title(f'Wavepacket Spreading - Time: {t:.2f}, Width: {sigma_t:.2f}', 
                     fontweight='bold', fontsize=12)
        ax1.axvline(x0 + k0*t, color='red', linestyle='--', alpha=0.5, label='Classical position')
        ax1.legend()

        # Real part
        ax2.plot(x, real_part, 'b-', linewidth=2, label='Re[ψ]')
        ax2.set_ylabel('Re[ψ]', fontsize=12)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlim(-15, 15)
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax2.legend()

        # Imaginary part
        ax3.plot(x, imag_part, 'r-', linewidth=2, label='Im[ψ]')
        ax3.set_xlabel('Position x', fontsize=12)
        ax3.set_ylabel('Im[ψ]', fontsize=12)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlim(-15, 15)
        ax3.grid(alpha=0.3)
        ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax3.legend()

        return ax1, ax2, ax3

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    writer = PillowWriter(fps=20)
    anim.save('wavepacket_spreading.gif', writer=writer)
    print(f"✓ Saved: wavepacket_spreading.gif")
    plt.close()

    print("\n" + "=" * 60)
    print("Wavepacket evolution visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_wavepacket_spreading()