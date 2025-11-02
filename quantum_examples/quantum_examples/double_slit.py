
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sqrt

def double_slit_wavefunction(x, y, slit1_y, slit2_y, wavelength, t=0):
    """
    Interference pattern from two coherent sources
    """
    k = 2 * pi / wavelength
    omega = k  # Assume v = 1
    
    # Distance from each slit
    r1 = np.sqrt(x**2 + (y - slit1_y)**2)
    r2 = np.sqrt(x**2 + (y - slit2_y)**2)
    
    # Wave from each slit
    psi1 = np.exp(1j * (k * r1 - omega * t)) / np.sqrt(r1 + 0.1)
    psi2 = np.exp(1j * (k * r2 - omega * t)) / np.sqrt(r2 + 0.1)
    
    return psi1 + psi2

def plot_double_slit():
    print("=" * 60)
    print("  QUANTUM DOUBLE-SLIT EXPERIMENT")
    print("=" * 60)
    print("\n🌊 Simulating wave-particle duality...")
    print("   Famous experiment showing quantum interference!")
    print()
    
    # Parameters
    wavelength = 1.0
    slit_separation = 3.0
    slit1_y = slit_separation / 2
    slit2_y = -slit_separation / 2
    
    print(f"   Wavelength λ = {wavelength:.2f}")
    print(f"   Slit separation d = {slit_separation:.2f}")
    print(f"   Expected fringe spacing: λL/d")
    print()
    
    # Create grid
    x = np.linspace(0.1, 15, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    
    # Static interference pattern
    print("🎨 Creating interference pattern...")
    psi = double_slit_wavefunction(X, Y, slit1_y, slit2_y, wavelength)
    intensity = np.abs(psi)**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D pattern
    im1 = ax1.imshow(intensity, extent=[0.1, 15, -10, 10], aspect='auto',
                     cmap='hot', origin='lower', interpolation='bilinear')
    ax1.axhline(slit1_y, color='cyan', linewidth=3, label='Slit 1', xmax=0.05)
    ax1.axhline(slit2_y, color='cyan', linewidth=3, label='Slit 2', xmax=0.05)
    ax1.axvline(0.1, color='white', linewidth=2, alpha=0.5, label='Barrier')
    ax1.set_xlabel('Distance from slits', fontsize=12)
    ax1.set_ylabel('Screen position', fontsize=12)
    ax1.set_title('Double-Slit Interference Pattern', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Detection screen (far-field pattern)
    screen_x = 14
    screen_idx = np.argmin(np.abs(x - screen_x))
    screen_intensity = intensity[:, screen_idx]
    
    ax2.plot(y, screen_intensity, 'b-', linewidth=2)
    ax2.fill_between(y, 0, screen_intensity, alpha=0.4)
    ax2.set_xlabel('Screen position', fontsize=12)
    ax2.set_ylabel('Detection probability', fontsize=12)
    ax2.set_title('Interference Fringes on Detection Screen', fontweight='bold', fontsize=13)
    ax2.grid(alpha=0.3)
    
    # Mark bright fringes
    peaks = []
    for i in range(1, len(screen_intensity) - 1):
        if screen_intensity[i] > screen_intensity[i-1] and screen_intensity[i] > screen_intensity[i+1]:
            if screen_intensity[i] > 0.1 * np.max(screen_intensity):
                peaks.append(y[i])
    
    for peak in peaks:
        ax2.axvline(peak, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('double_slit_interference.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: double_slit_interference.png")
    
    # Create animation
    print("🎬 Creating time evolution animation...")
    fig_anim, ax_anim = plt.subplots(figsize=(12, 8))
    
    frames = 60
    time_points = np.linspace(0, 2*pi, frames)
    
    def update(frame):
        ax_anim.clear()
        t = time_points[frame]
        
        psi_t = double_slit_wavefunction(X, Y, slit1_y, slit2_y, wavelength, t)
        intensity_t = np.abs(psi_t)**2
        
        im = ax_anim.imshow(intensity_t, extent=[0.1, 15, -10, 10], aspect='auto',
                           cmap='hot', origin='lower', interpolation='bilinear',
                           vmin=0, vmax=np.max(intensity))
        
        ax_anim.axhline(slit1_y, color='cyan', linewidth=3, xmax=0.05)
        ax_anim.axhline(slit2_y, color='cyan', linewidth=3, xmax=0.05)
        ax_anim.axvline(0.1, color='white', linewidth=2, alpha=0.5)
        
        ax_anim.set_xlabel('Distance from slits', fontsize=12)
        ax_anim.set_ylabel('Screen position', fontsize=12)
        ax_anim.set_title(f'Double-Slit Wave Propagation - Time: {t:.2f}',
                         fontweight='bold', fontsize=13)
        
        return im,
    
    anim = FuncAnimation(fig_anim, update, frames=frames, interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('double_slit_animation.gif', writer=writer)
    print("✓ Saved: double_slit_animation.gif")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Double-slit experiment visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    plot_double_slit()
