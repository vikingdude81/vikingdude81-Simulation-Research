
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
import json

def analyze_spectroscopy_data(wavelengths, intensities, element="hydrogen"):
    """
    Analyze spectroscopic data using quantum energy levels
    Example: Hydrogen emission spectrum analysis
    """
    print("=" * 60)
    print("  SPECTROSCOPY DATA ANALYSIS - QUANTUM APPROACH")
    print("=" * 60)
    print(f"\n🔬 Analyzing {element} spectrum data...")
    print(f"   Data points: {len(wavelengths)}")
    print()
    
    # Theoretical hydrogen energy levels (Rydberg formula)
    # 1/λ = R_H * (1/n_f² - 1/n_i²)
    R_H = 1.097e7  # Rydberg constant (m⁻¹)
    
    # Find peaks and match to transitions
    transitions = []
    for wl, intensity in zip(wavelengths, intensities):
        if intensity > 0.5 * max(intensities):  # Peak detection
            # Try to match to quantum transitions
            for n_i in range(2, 7):
                for n_f in range(1, n_i):
                    lambda_theory = 1 / (R_H * (1/n_f**2 - 1/n_i**2)) * 1e9  # nm
                    if abs(wl - lambda_theory) < 5:  # 5nm tolerance
                        transitions.append((n_i, n_f, wl, intensity, lambda_theory))
    
    print(f"✓ Identified {len(transitions)} quantum transitions:\n")
    for n_i, n_f, wl_exp, intensity, wl_theory in transitions:
        series = "Lyman" if n_f == 1 else "Balmer" if n_f == 2 else "Paschen"
        print(f"   {n_i} → {n_f} ({series}): λ_exp={wl_exp:.1f}nm, λ_theory={wl_theory:.1f}nm")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Spectrum
    ax1.plot(wavelengths, intensities, 'b-', linewidth=2)
    for n_i, n_f, wl_exp, intensity, _ in transitions:
        ax1.axvline(wl_exp, color='red', linestyle='--', alpha=0.5)
        ax1.text(wl_exp, intensity + 0.05, f'{n_i}→{n_f}', 
                rotation=90, va='bottom', fontsize=9)
    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Intensity', fontsize=11)
    ax1.set_title('Experimental Spectrum with Quantum Transitions', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Energy level diagram
    ax2.set_ylim(-14, 0)
    for n in range(1, 6):
        E_n = -13.6 / n**2  # eV
        ax2.axhline(E_n, color='blue', linewidth=2)
        ax2.text(0.05, E_n, f'n={n}', fontsize=11, va='center')
    
    # Draw observed transitions
    for n_i, n_f, _, _, _ in transitions:
        E_i = -13.6 / n_i**2
        E_f = -13.6 / n_f**2
        ax2.annotate('', xy=(0.5, E_f), xytext=(0.5, E_i),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax2.set_ylabel('Energy (eV)', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    ax2.set_title('Quantum Energy Levels', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('spectroscopy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: spectroscopy_analysis.png")
    plt.close()

def analyze_interference_pattern(image_data):
    """
    Analyze 2D interference pattern (double-slit, diffraction, etc.)
    Input: 2D array of intensity values
    """
    print("\n" + "=" * 60)
    print("  INTERFERENCE PATTERN ANALYSIS")
    print("=" * 60)
    print(f"\n📊 Image size: {image_data.shape}")
    
    # Extract fringe spacing
    center_row = image_data[image_data.shape[0]//2, :]
    
    # FFT to find periodicity
    fft = np.fft.fft(center_row)
    frequencies = np.fft.fftfreq(len(center_row))
    
    # Find dominant frequency (fringe spacing)
    peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
    fringe_spacing = 1 / abs(frequencies[peak_idx])
    
    print(f"✓ Detected fringe spacing: {fringe_spacing:.2f} pixels")
    
    # Visibility (contrast)
    I_max = np.max(center_row)
    I_min = np.min(center_row)
    visibility = (I_max - I_min) / (I_max + I_min)
    
    print(f"✓ Fringe visibility: {visibility:.3f}")
    print(f"   (1.0 = perfect coherence, 0.0 = no coherence)")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 2D pattern
    im = axes[0, 0].imshow(image_data, cmap='hot', aspect='auto')
    axes[0, 0].set_title('Interference Pattern', fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Cross-section
    axes[0, 1].plot(center_row, 'b-', linewidth=2)
    axes[0, 1].set_title('Central Cross-Section', fontweight='bold')
    axes[0, 1].set_xlabel('Position (pixels)')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].grid(alpha=0.3)
    
    # FFT spectrum
    axes[1, 0].plot(frequencies[:len(frequencies)//2], 
                   np.abs(fft[:len(fft)//2]), 'g-', linewidth=2)
    axes[1, 0].axvline(frequencies[peak_idx], color='red', linestyle='--')
    axes[1, 0].set_title('Spatial Frequency Spectrum', fontweight='bold')
    axes[1, 0].set_xlabel('Spatial frequency')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(alpha=0.3)
    
    # Statistics
    stats_text = f"Analysis Results:\n\n"
    stats_text += f"Fringe spacing: {fringe_spacing:.2f} px\n"
    stats_text += f"Visibility: {visibility:.3f}\n"
    stats_text += f"Max intensity: {I_max:.1f}\n"
    stats_text += f"Min intensity: {I_min:.1f}\n"
    stats_text += f"SNR: {I_max/I_min:.2f}\n"
    
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                   fontsize=12, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('interference_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: interference_analysis.png")
    plt.close()

def analyze_qubit_tomography(measurement_data):
    """
    Analyze quantum state tomography data
    Input: Dictionary with Pauli measurements {'X': prob, 'Y': prob, 'Z': prob}
    """
    print("\n" + "=" * 60)
    print("  QUANTUM STATE TOMOGRAPHY ANALYSIS")
    print("=" * 60)
    
    # Reconstruct density matrix from measurements
    # ρ = 1/2 * (I + <σ_x>σ_x + <σ_y>σ_y + <σ_z>σ_z)
    
    exp_x = measurement_data.get('X', 0)
    exp_y = measurement_data.get('Y', 0)
    exp_z = measurement_data.get('Z', 0)
    
    print(f"\n📊 Pauli expectation values:")
    print(f"   ⟨σ_x⟩ = {exp_x:.3f}")
    print(f"   ⟨σ_y⟩ = {exp_y:.3f}")
    print(f"   ⟨σ_z⟩ = {exp_z:.3f}")
    
    # Reconstruct density matrix
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)
    
    rho = 0.5 * (identity + exp_x * sigma_x + exp_y * sigma_y + exp_z * sigma_z)
    
    # Calculate purity
    purity = np.real(np.trace(rho @ rho))
    
    # Bloch vector length
    bloch_length = sqrt(exp_x**2 + exp_y**2 + exp_z**2)
    
    print(f"\n✓ Reconstructed state:")
    print(f"   Purity: {purity:.3f}")
    print(f"   Bloch vector length: {bloch_length:.3f}")
    
    if purity > 0.95:
        print(f"   → Pure state (quantum coherence maintained)")
    else:
        print(f"   → Mixed state (decoherence detected)")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Density matrix
    im = axes[0].imshow(np.real(rho), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0].set_title('Reconstructed Density Matrix', fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['|0⟩', '|1⟩'])
    axes[0].set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im, ax=axes[0])
    
    # Bloch sphere
    theta_circle = np.linspace(0, 2*pi, 100)
    axes[1].plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)
    axes[1].arrow(0, 0, exp_x, exp_y, head_width=0.1, head_length=0.1, 
                 fc='blue', ec='blue', linewidth=3)
    axes[1].scatter([exp_x], [exp_y], s=200, c='red', zorder=5)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].set_aspect('equal')
    axes[1].grid(alpha=0.3)
    axes[1].set_title('Bloch Vector (XY)', fontweight='bold')
    
    # Summary
    summary = f"Tomography Results:\n\n"
    summary += f"Purity: {purity:.3f}\n"
    summary += f"Fidelity to pure: {bloch_length:.3f}\n\n"
    summary += f"Eigenvalues:\n"
    eigenvalues = np.linalg.eigvalsh(rho)
    summary += f"  λ₁ = {eigenvalues[0]:.3f}\n"
    summary += f"  λ₂ = {eigenvalues[1]:.3f}\n"
    
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('tomography_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: tomography_analysis.png")
    plt.close()

# Example usage demonstrations
def demo_analysis():
    print("=" * 60)
    print("  QUANTUM DATA ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("\nDemonstrating 3 real-world applications:\n")
    
    # 1. Spectroscopy
    print("[1] Hydrogen Spectroscopy Analysis")
    wavelengths = np.array([656.3, 486.1, 434.0, 410.2, 397.0])  # Balmer series (nm)
    intensities = np.array([1.0, 0.8, 0.6, 0.4, 0.3])
    analyze_spectroscopy_data(wavelengths, intensities)
    
    # 2. Interference
    print("\n[2] Double-Slit Interference Analysis")
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    # Simulate interference pattern
    r1 = np.sqrt(X**2 + (Y - 2)**2)
    r2 = np.sqrt(X**2 + (Y + 2)**2)
    interference = np.abs(np.exp(1j * r1) + np.exp(1j * r2))**2
    analyze_interference_pattern(interference)
    
    # 3. Quantum state tomography
    print("\n[3] Qubit State Tomography Analysis")
    # Example: measure a qubit in |+⟩ state
    measurements = {'X': 1.0, 'Y': 0.0, 'Z': 0.0}
    analyze_qubit_tomography(measurements)
    
    print("\n" + "=" * 60)
    print("✨ All data analyses complete!")
    print("=" * 60)

if __name__ == "__main__":
    demo_analysis()
