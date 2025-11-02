
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, sqrt, factorial
import random

def hermite_phys(n, x):
    """Compute Hermite polynomials using recurrence relation."""
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

def psi_3d(x, y, z, nx, ny, nz):
    """3D quantum harmonic oscillator wavefunction"""
    return psi_n(x, nx) * psi_n(y, ny) * psi_n(z, nz)

def plot_3d_quantum_state():
    # Randomize quantum numbers
    nx = random.randint(0, 3)
    ny = random.randint(0, 3)
    nz = random.randint(0, 3)
    
    print("=" * 60)
    print("  3D QUANTUM HARMONIC OSCILLATOR")
    print("=" * 60)
    print(f"\n🎲 Random quantum state: ψ({nx},{ny},{nz})")
    print(f"   Energy level: E = {nx + ny + nz + 1.5}ℏω")
    print(f"   Total quantum number n = {nx + ny + nz}")
    print()
    
    # Create 3D grid (lower resolution for performance)
    grid_size = 50
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    z = np.linspace(-3, 3, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Compute probability density
    print("📊 Computing 3D probability density...")
    psi_values = psi_3d(X, Y, Z, nx, ny, nz)
    probability = np.abs(psi_values)**2
    
    # Normalize for better visualization
    probability = probability / np.max(probability)
    
    # Create isosurface plot
    print("🎨 Creating isosurface visualization...")
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: High probability isosurface (80% threshold)
    ax1 = fig.add_subplot(121, projection='3d')
    threshold_high = 0.8
    
    # Find points above threshold
    mask_high = probability > threshold_high
    points_high = np.column_stack([X[mask_high], Y[mask_high], Z[mask_high]])
    
    if len(points_high) > 0:
        ax1.scatter(points_high[:, 0], points_high[:, 1], points_high[:, 2],
                   c=probability[mask_high], cmap='hot', alpha=0.6, s=20)
    
    ax1.set_title(f'ψ({nx},{ny},{nz}) - High Probability (>80%)',
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_zlim(-3, 3)
    
    # Plot 2: Medium probability isosurface (30% threshold)
    ax2 = fig.add_subplot(122, projection='3d')
    threshold_med = 0.3
    
    mask_med = probability > threshold_med
    points_med = np.column_stack([X[mask_med], Y[mask_med], Z[mask_med]])
    
    if len(points_med) > 0:
        ax2.scatter(points_med[:, 0], points_med[:, 1], points_med[:, 2],
                   c=probability[mask_med], cmap='viridis', alpha=0.4, s=10)
    
    ax2.set_title(f'ψ({nx},{ny},{nz}) - Medium Probability (>30%)',
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-3, 3)
    
    plt.tight_layout()
    filename = f'quantum_3d_{nx}_{ny}_{nz}_isosurface.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    # Create slice plots
    print("🔪 Creating cross-sectional slices...")
    fig2, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # XY plane (z=0)
    z_idx = grid_size // 2
    prob_xy = probability[:, :, z_idx]
    im1 = axes[0, 0].imshow(prob_xy.T, origin='lower', extent=[-3, 3, -3, 3],
                            cmap='hot', interpolation='bilinear')
    axes[0, 0].set_title(f'XY Plane (z=0)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # XZ plane (y=0)
    y_idx = grid_size // 2
    prob_xz = probability[:, y_idx, :]
    im2 = axes[0, 1].imshow(prob_xz.T, origin='lower', extent=[-3, 3, -3, 3],
                            cmap='hot', interpolation='bilinear')
    axes[0, 1].set_title(f'XZ Plane (y=0)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('z')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # YZ plane (x=0)
    x_idx = grid_size // 2
    prob_yz = probability[x_idx, :, :]
    im3 = axes[1, 0].imshow(prob_yz.T, origin='lower', extent=[-3, 3, -3, 3],
                            cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title(f'YZ Plane (x=0)')
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('z')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 3D projection view
    axes[1, 1].text(0.5, 0.5, 
                    f'3D Quantum State\nψ({nx},{ny},{nz})\n\n'
                    f'Energy: {nx + ny + nz + 1.5}ℏω\n'
                    f'Nodes: nx={nx}, ny={ny}, nz={nz}',
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.suptitle(f'ψ({nx},{ny},{nz}) Probability Density |ψ|² - Cross Sections',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename2 = f'quantum_3d_{nx}_{ny}_{nz}_slices.png'
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("3D quantum state visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_3d_quantum_state()
