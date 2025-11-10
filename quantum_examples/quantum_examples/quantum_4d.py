
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, sqrt, factorial
import random

def hermite_phys(n, x):
    """Compute Hermite polynomials"""
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
    """1D harmonic oscillator eigenfunction"""
    Hn = hermite_phys(n, x)
    norm = 1.0 / np.sqrt((2.0 ** n) * factorial(n) * np.sqrt(pi))
    return norm * np.exp(-x**2 / 2.0) * Hn

def psi_4d_projection(x, y, z, nx, ny, nz, nw, w_fixed=0):
    """
    4D quantum harmonic oscillator projected to 3D
    We fix the 4th dimension w at a constant value
    ψ(x,y,z,w) = ψ_nx(x) * ψ_ny(y) * ψ_nz(z) * ψ_nw(w)
    """
    return psi_n(x, nx) * psi_n(y, ny) * psi_n(z, nz) * psi_n(w_fixed, nw)

def plot_4d_quantum():
    # Random 4D quantum numbers
    nx = random.randint(0, 2)
    ny = random.randint(0, 2)
    nz = random.randint(0, 2)
    nw = random.randint(0, 2)
    
    total_n = nx + ny + nz + nw
    energy = total_n + 2  # E = (n + 2)ℏω for 4D
    
    print("=" * 60)
    print("  4D QUANTUM HARMONIC OSCILLATOR")
    print("=" * 60)
    print(f"\n🎲 4D quantum state: ψ({nx},{ny},{nz},{nw})")
    print(f"   Total quantum number: n = {total_n}")
    print(f"   Energy level: E = {energy}ℏω")
    print(f"   Degeneracy: {(total_n + 1) * (total_n + 2) * (total_n + 3) // 6} states")
    print(f"\n   Visualizing via 3D projections (fixing w coordinate)")
    print()
    
    # Create 3D grid
    grid_size = 40
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    z = np.linspace(-3, 3, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create projections at different w values
    w_values = [-1, 0, 1]
    
    print("📊 Computing 4D→3D projections...")
    fig = plt.figure(figsize=(16, 5))
    
    for i, w_fix in enumerate(w_values):
        print(f"   Computing slice at w = {w_fix}...")
        
        # Compute 3D projection
        psi_values = psi_4d_projection(X, Y, Z, nx, ny, nz, nw, w_fixed=w_fix)
        probability = np.abs(psi_values)**2
        probability = probability / (np.max(probability) + 1e-10)
        
        # Plot isosurface
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        threshold = 0.3
        mask = probability > threshold
        points = np.column_stack([X[mask], Y[mask], Z[mask]])
        
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=probability[mask], cmap='plasma', alpha=0.5, s=8)
        
        ax.set_title(f'4D→3D Projection at w={w_fix}\nψ({nx},{ny},{nz},{nw})',
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
    
    plt.suptitle(f'4D Quantum Harmonic Oscillator: ψ({nx},{ny},{nz},{nw}) - E = {energy}ℏω',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'quantum_4d_{nx}_{ny}_{nz}_{nw}_projections.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    # Create marginal probability distributions
    print("📊 Computing marginal distributions...")
    fig2, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Compute 4D grid point
    grid_1d = 30
    coords_1d = np.linspace(-3, 3, grid_1d)
    
    # X marginal (integrate over y, z, w)
    x_marginal = np.zeros(grid_1d)
    for i, x_val in enumerate(coords_1d):
        integrand = 0
        for y_val in coords_1d:
            for z_val in coords_1d:
                for w_val in coords_1d:
                    psi_val = (psi_n(x_val, nx) * psi_n(y_val, ny) * 
                              psi_n(z_val, nz) * psi_n(w_val, nw))
                    integrand += np.abs(psi_val)**2
        x_marginal[i] = integrand
    
    x_marginal /= np.max(x_marginal)
    
    # Similar for other dimensions (simplified - just show 1D wavefunctions)
    y_prob = np.abs(psi_n(coords_1d, ny))**2
    z_prob = np.abs(psi_n(coords_1d, nz))**2
    w_prob = np.abs(psi_n(coords_1d, nw))**2
    
    axes[0, 0].plot(coords_1d, y_prob / np.max(y_prob), 'b-', linewidth=2)
    axes[0, 0].fill_between(coords_1d, 0, y_prob / np.max(y_prob), alpha=0.4)
    axes[0, 0].set_title(f'X dimension (nx={nx})', fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Probability density')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(coords_1d, y_prob / np.max(y_prob), 'r-', linewidth=2)
    axes[0, 1].fill_between(coords_1d, 0, y_prob / np.max(y_prob), alpha=0.4, color='red')
    axes[0, 1].set_title(f'Y dimension (ny={ny})', fontweight='bold')
    axes[0, 1].set_xlabel('y')
    axes[0, 1].set_ylabel('Probability density')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(coords_1d, z_prob / np.max(z_prob), 'g-', linewidth=2)
    axes[1, 0].fill_between(coords_1d, 0, z_prob / np.max(z_prob), alpha=0.4, color='green')
    axes[1, 0].set_title(f'Z dimension (nz={nz})', fontweight='bold')
    axes[1, 0].set_xlabel('z')
    axes[1, 0].set_ylabel('Probability density')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(coords_1d, w_prob / np.max(w_prob), 'm-', linewidth=2)
    axes[1, 1].fill_between(coords_1d, 0, w_prob / np.max(w_prob), alpha=0.4, color='magenta')
    axes[1, 1].set_title(f'W dimension (nw={nw}) - 4th dimension!', fontweight='bold')
    axes[1, 1].set_xlabel('w')
    axes[1, 1].set_ylabel('Probability density')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'4D Harmonic Oscillator ψ({nx},{ny},{nz},{nw}) - 1D Marginals',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename2 = f'quantum_4d_{nx}_{ny}_{nz}_{nw}_marginals.png'
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("4D quantum oscillator visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_4d_quantum()
