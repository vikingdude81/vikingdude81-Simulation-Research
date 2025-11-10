import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, sqrt, factorial, exp
from scipy.special import genlaguerre, sph_harm
import random

def hydrogen_wavefunction(r, theta, phi, n, l, m):
    """
    Hydrogen atom wavefunction in spherical coordinates
    n: principal quantum number (1, 2, 3, ...)
    l: angular momentum quantum number (0 to n-1)
    m: magnetic quantum number (-l to l)
    """
    a0 = 1.0  # Bohr radius in atomic units

    # Radial part
    rho = 2 * r / (n * a0)
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)
    norm = sqrt((2 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    radial = norm * np.exp(-rho / 2) * (rho**l) * laguerre(rho)

    # Angular part (spherical harmonic)
    angular = sph_harm(m, l, phi, theta)

    return radial * angular

def plot_hydrogen_orbital():
    # Random quantum numbers
    n = random.randint(1, 3)
    l = random.randint(0, n - 1)
    m = random.randint(-l, l)

    orbital_names = {
        (1, 0, 0): "1s",
        (2, 0, 0): "2s",
        (2, 1, -1): "2p_-1", (2, 1, 0): "2p_0", (2, 1, 1): "2p_1",
        (3, 0, 0): "3s",
        (3, 1, -1): "3p_-1", (3, 1, 0): "3p_0", (3, 1, 1): "3p_1",
        (3, 2, -2): "3d_-2", (3, 2, -1): "3d_-1", (3, 2, 0): "3d_0",
        (3, 2, 1): "3d_1", (3, 2, 2): "3d_2"
    }

    orbital_name = orbital_names.get((n, l, m), f"n={n},l={l},m={m}")

    print("=" * 60)
    print("  HYDROGEN ATOM ORBITAL VISUALIZATION")
    print("=" * 60)
    print(f"\n🎲 Random orbital: {orbital_name}")
    print(f"   Quantum numbers: n={n}, l={l}, m={m}")
    print(f"   Energy: E = -13.6/{n}² = {-13.6/n**2:.3f} eV")
    print()

    # Create spherical grid
    r = np.linspace(0.1, 15, 60)
    theta = np.linspace(0, pi, 60)
    phi = np.linspace(0, 2*pi, 60)

    R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')

    # Convert to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Compute wavefunction
    print("📊 Computing orbital probability density...")
    psi = hydrogen_wavefunction(R, THETA, PHI, n, l, m)
    probability = np.abs(psi)**2
    probability = probability / np.max(probability)

    # Plot isosurfaces
    print("🎨 Creating visualization...")
    fig = plt.figure(figsize=(15, 5))

    for i, threshold in enumerate([0.1, 0.3, 0.5]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        mask = probability > threshold
        points = np.column_stack([X[mask], Y[mask], Z[mask]])

        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=probability[mask], cmap='plasma', alpha=0.5, s=5)

        ax.set_title(f'{orbital_name} - Threshold {threshold*100:.0f}%', fontweight='bold')
        ax.set_xlabel('x (Bohr radii)')
        ax.set_ylabel('y (Bohr radii)')
        ax.set_zlabel('z (Bohr radii)')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

    plt.tight_layout()
    filename = f'hydrogen_{orbital_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

    print("\n" + "=" * 60)
    print("Hydrogen orbital visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_hydrogen_orbital()