
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt, factorial
import random

def hermite_phys(n, x):
    """
    Compute Hermite polynomials using recurrence relation.
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    
    Hnm2 = np.ones_like(x)
    Hnm1 = 2 * x
    
    for k in range(2, n + 1):
        Hn = 2 * x * Hnm1 - 2 * (k - 1) * Hnm2
        Hnm2, Hnm1 = Hnm1, Hn
    
    return Hn

def psi_n(x, n):
    """
    1D harmonic oscillator eigenfunction
    """
    Hn = hermite_phys(n, x)
    norm = 1.0 / np.sqrt((2.0 ** n) * factorial(n) * np.sqrt(pi))
    return norm * np.exp(-x**2 / 2.0) * Hn

def psi_2d(x, y, nx, ny, t=0):
    """
    2D wavefunction with time evolution
    """
    spatial = psi_n(x, nx) * psi_n(y, ny)
    energy = nx + ny + 1
    time_phase = np.exp(-1j * energy * t)
    return spatial * time_phase

def plot_random_state():
    # Randomize quantum numbers (0 to 5 for variety)
    nx = random.randint(0, 5)
    ny = random.randint(0, 5)
    
    print("=" * 60)
    print("  RANDOM QUANTUM STATE GENERATOR")
    print("=" * 60)
    print(f"\n🎲 Randomly selected quantum state: ψ({nx},{ny})")
    print(f"   Energy level: E = {nx + ny + 1}ℏω")
    print(f"   Degeneracy at this level: {nx + ny + 1} states")
    print()
    
    # Create grid
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    
    # Compute wavefunction and probability
    Z_wavefunction = np.real(psi_2d(X, Y, nx, ny))
    Z_probability = np.abs(psi_2d(X, Y, nx, ny))**2
    
    # Create animation first
    print("🎬 Creating time evolution animation...")
    
    fig_anim = plt.figure(figsize=(10, 7))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    
    frames = 60
    time_points = np.linspace(0, 2*pi, frames)
    
    def update(frame):
        ax_anim.clear()
        t = time_points[frame]
        Z_time = np.real(psi_2d(X, Y, nx, ny, t))
        
        surf = ax_anim.plot_surface(X, Y, Z_time, cmap='viridis', 
                                    rstride=4, cstride=4, 
                                    linewidth=0, antialiased=True,
                                    alpha=0.8)
        
        ax_anim.set_title(f"Time Evolution: ψ({nx},{ny}) at t={t:.2f}", 
                         fontsize=14, fontweight='bold')
        ax_anim.set_xlabel("x", fontsize=12)
        ax_anim.set_ylabel("y", fontsize=12)
        ax_anim.set_zlabel("Re[ψ(x,y,t)]", fontsize=12)
        ax_anim.set_zlim(-0.5, 0.5)
        
        return surf,
    
    anim = FuncAnimation(fig_anim, update, frames=frames, interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save(f'random_quantum_{nx}_{ny}_animation.gif', writer=writer)
    print(f"✓ Saved: random_quantum_{nx}_{ny}_animation.gif")
    plt.close(fig_anim)
    
    # Create 3D surface plot of wavefunction
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    surf1 = ax1.plot_surface(X, Y, Z_wavefunction, cmap='viridis', 
                            rstride=4, cstride=4, 
                            linewidth=0, antialiased=True,
                            alpha=0.8)
    
    ax1.set_title(f"Random Quantum State ψ({nx},{ny})", 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_zlabel("ψ(x,y)", fontsize=12)
    
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(f'random_quantum_{nx}_{ny}_wavefunction.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: random_quantum_{nx}_{ny}_wavefunction.png")
    
    # Create 3D surface plot of probability density
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    surf2 = ax2.plot_surface(X, Y, Z_probability, cmap='plasma', 
                            rstride=4, cstride=4, 
                            linewidth=0, antialiased=True,
                            alpha=0.8)
    
    ax2.set_title(f"Random Quantum State ψ({nx},{ny}) - Probability Density |ψ|²", 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_zlabel("|ψ(x,y)|²", fontsize=12)
    
    fig2.colorbar(surf2, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(f'random_quantum_{nx}_{ny}_probability.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: random_quantum_{nx}_{ny}_probability.png")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("Run the 'Random Quantum State' workflow again for another!")
    print("=" * 60)

if __name__ == "__main__":
    plot_random_state()
