
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt, factorial

def hermite_phys(n, x):
    """
    Compute Hermite polynomials using recurrence relation.
    H_0(x) = 1
    H_1(x) = 2x
    H_n(x) = 2x*H_{n-1}(x) - 2(n-1)*H_{n-2}(x)
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
    1D harmonic oscillator eigenfunction (m=ω=ℏ=1 units):
    ψ_n(x) = (1 / sqrt(2^n n! sqrt(π))) * H_n(x) * exp(-x²/2)
    """
    Hn = hermite_phys(n, x)
    norm = 1.0 / np.sqrt((2.0 ** n) * factorial(n) * np.sqrt(pi))
    return norm * np.exp(-x**2 / 2.0) * Hn

def psi_2d(x, y, nx, ny, t=0):
    """
    2D wavefunction with time evolution: 
    ψ_{nx,ny}(x,y,t) = ψ_nx(x) * ψ_ny(y) * exp(-i*E*t)
    where E = (nx + ny + 1) in units where ℏ=ω=1
    """
    spatial = psi_n(x, nx) * psi_n(y, ny)
    energy = nx + ny + 1
    time_phase = np.exp(-1j * energy * t)
    return spatial * time_phase

def plot_3d_surface(X, Y, Z, nx, ny, title_suffix=""):
    """Create a 3D surface plot"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                           rstride=4, cstride=4, 
                           linewidth=0, antialiased=True,
                           alpha=0.8)
    
    ax.set_title(f"2D Quantum Harmonic Oscillator ψ({nx},{ny}){title_suffix}", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("ψ(x,y)", fontsize=12)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    return fig, ax

def plot_contour(X, Y, Z, nx, ny, title_suffix=""):
    """Create a contour plot"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    contour = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
    ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.3)
    
    ax.set_title(f"2D Quantum Harmonic Oscillator ψ({nx},{ny}){title_suffix}", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect('equal')
    
    fig.colorbar(contour, ax=ax)
    plt.tight_layout()
    return fig, ax

def create_animation(nx, ny, frames=60):
    """Create an animation of time evolution"""
    print("\nCreating animation... (this may take a moment)")
    
    x = np.linspace(-4, 4, 150)
    y = np.linspace(-4, 4, 150)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    time_points = np.linspace(0, 2*pi, frames)
    
    def update(frame):
        ax.clear()
        t = time_points[frame]
        Z = np.real(psi_2d(X, Y, nx, ny, t))
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                               rstride=4, cstride=4, 
                               linewidth=0, antialiased=True,
                               alpha=0.8)
        
        ax.set_title(f"Time Evolution: ψ({nx},{ny}) at t={t:.2f}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Re[ψ(x,y,t)]")
        ax.set_zlim(-0.5, 0.5)
        
        return surf,
    
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save('quantum_animation.gif', writer=writer)
    print("Animation saved as 'quantum_animation.gif'")
    
    plt.close()

def main():
    print("=" * 60)
    print("  2D QUANTUM HARMONIC OSCILLATOR VISUALIZATION")
    print("=" * 60)
    print("\nExamples: (0,0), (1,0), (1,1), (2,1), (3,2)")
    print()
    
    try:
        nx = int(input("Enter nx (quantum number for x): "))
        ny = int(input("Enter ny (quantum number for y): "))
        
        if nx < 0 or ny < 0:
            print("Quantum numbers must be non-negative!")
            return
    except ValueError:
        print("Please enter valid integers!")
        return
    
    print("\nVisualization options:")
    print("1. Wavefunction ψ(x,y) - 3D surface")
    print("2. Probability density |ψ(x,y)|² - 3D surface")
    print("3. Wavefunction ψ(x,y) - Contour plot")
    print("4. Probability density |ψ(x,y)|² - Contour plot")
    print("5. Time evolution animation (saves as GIF)")
    print("6. All static plots")
    
    try:
        choice = int(input("\nEnter your choice (1-6): "))
    except ValueError:
        choice = 1
    
    print()
    print(f"Quantum state: ψ({nx},{ny})")
    print(f"Energy level: E = {nx + ny + 1}ℏω")
    print(f"Degeneracy at this level: {nx + ny + 1} states")
    print()
    
    # Create grid
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    
    # Compute wavefunction and probability density
    psi = psi_2d(X, Y, nx, ny)
    Z_wavefunction = np.real(psi)
    Z_probability = np.abs(psi)**2
    
    if choice == 1:
        fig, ax = plot_3d_surface(X, Y, Z_wavefunction, nx, ny)
        plt.savefig('quantum_wavefunction_3d.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_wavefunction_3d.png")
        plt.show()
        
    elif choice == 2:
        fig, ax = plot_3d_surface(X, Y, Z_probability, nx, ny, " - Probability Density |ψ|²")
        plt.savefig('quantum_probability_3d.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_probability_3d.png")
        plt.show()
        
    elif choice == 3:
        fig, ax = plot_contour(X, Y, Z_wavefunction, nx, ny)
        plt.savefig('quantum_wavefunction_contour.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_wavefunction_contour.png")
        plt.show()
        
    elif choice == 4:
        fig, ax = plot_contour(X, Y, Z_probability, nx, ny, " - Probability Density |ψ|²")
        plt.savefig('quantum_probability_contour.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_probability_contour.png")
        plt.show()
        
    elif choice == 5:
        create_animation(nx, ny)
        
    elif choice == 6:
        # Create all static plots
        fig1, _ = plot_3d_surface(X, Y, Z_wavefunction, nx, ny)
        plt.savefig('quantum_wavefunction_3d.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_wavefunction_3d.png")
        
        fig2, _ = plot_3d_surface(X, Y, Z_probability, nx, ny, " - Probability Density |ψ|²")
        plt.savefig('quantum_probability_3d.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_probability_3d.png")
        
        fig3, _ = plot_contour(X, Y, Z_wavefunction, nx, ny)
        plt.savefig('quantum_wavefunction_contour.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_wavefunction_contour.png")
        
        fig4, _ = plot_contour(X, Y, Z_probability, nx, ny, " - Probability Density |ψ|²")
        plt.savefig('quantum_probability_contour.png', dpi=150, bbox_inches='tight')
        print("Saved: quantum_probability_contour.png")
        
        plt.show()
    
    print("\n" + "=" * 60)
    print("Run again to explore different quantum states!")
    print("=" * 60)

if __name__ == "__main__":
    main()
