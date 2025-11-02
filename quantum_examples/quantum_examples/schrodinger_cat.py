import os
# Configure PyVista for headless rendering BEFORE importing pyvista
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'osmesa'

import numpy as np
import pyvista as pv
from math import pi, sqrt
from scipy.special import factorial
import random

# Enable off-screen rendering
pv.OFF_SCREEN = True

def coherent_state(alpha, n_max=30):
    """Create coherent state |α⟩"""
    n = np.arange(n_max)
    coeffs = (alpha**n / np.sqrt(factorial(n))) * np.exp(-np.abs(alpha)**2 / 2)
    return coeffs

def cat_state(alpha, phase=0, n_max=30):
    """
    Schrödinger cat state: |cat⟩ = (|α⟩ + e^(iφ)|-α⟩)/N
    Superposition of two coherent states
    """
    coherent_plus = coherent_state(alpha, n_max)
    coherent_minus = coherent_state(-alpha, n_max)
    
    cat = coherent_plus + np.exp(1j * phase) * coherent_minus
    norm = np.sqrt(np.sum(np.abs(cat)**2))
    
    return cat / norm

def create_fock_state_bars(alpha, phase, n_max=30):
    """Create 3D bar chart of Fock state populations"""
    cat = cat_state(alpha, phase, n_max)
    coherent_p = coherent_state(alpha, n_max)
    coherent_m = coherent_state(-alpha, n_max)
    
    # Create 3D bar positions
    n_states = len(cat)
    points = []
    heights = np.abs(cat)**2
    
    for i in range(n_states):
        # Create a box for each bar
        if heights[i] > 0.001:  # Only show significant states
            box = pv.Cube(center=(i, 0, heights[i]/2), 
                         x_length=0.7, y_length=0.7, z_length=heights[i])
            points.append(box)
    
    if not points:
        # Create at least one bar
        box = pv.Cube(center=(0, 0, 0.01), x_length=0.7, y_length=0.7, z_length=0.02)
        points.append(box)
    
    bars = points[0]
    for p in points[1:]:
        bars = bars + p
    
    return bars, heights

def create_quantum_cloud(alpha, phase, n_points=50):
    """Create volumetric quantum probability cloud"""
    grid = pv.ImageData(dimensions=(n_points, n_points, n_points))
    grid.spacing = (10.0/n_points, 10.0/n_points, 10.0/n_points)
    grid.origin = (-5, -5, -5)
    
    # Get grid coordinates
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    z = np.linspace(-5, 5, n_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    alpha_real = np.real(alpha)
    
    # Two coherent states in phase space
    W_plus = np.exp(-2*((X - alpha_real)**2 + Y**2 + Z**2))
    W_minus = np.exp(-2*((X + alpha_real)**2 + Y**2 + Z**2))
    
    # Interference pattern
    interference = 2 * np.cos(4*alpha_real*X + phase)
    interference *= np.exp(-2*(X**2 + Y**2 + Z**2))
    
    # Cat state Wigner function
    W_cat = 0.5 * (W_plus + W_minus) + 0.3 * interference
    
    # Normalize
    W_cat = np.abs(W_cat)
    W_cat = W_cat / W_cat.max()
    
    grid['wigner'] = W_cat.flatten(order='F')
    
    return grid

def plot_photon_number_distribution(alpha, phase):
    """3D visualization of photon number distribution"""
    phase_names = {0: '0', pi/2: 'π/2', pi: 'π', 3*pi/2: '3π/2'}
    
    print("📊 Creating 3D photon number distribution...")
    
    cat = cat_state(alpha, phase, n_max=25)
    heights = np.abs(cat)**2
    
    pl = pv.Plotter(window_size=[1000, 800], off_screen=True)
    pl.set_background('#1a1a2e')
    
    # Create individual bars with colors
    import matplotlib
    import matplotlib.colors as mcolors
    
    cmap = matplotlib.colormaps.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=0, vmax=heights.max())
    
    for i, h in enumerate(heights):
        if h > 0.001:  # Only show significant states
            box = pv.Cube(center=(i, 0, h/2), 
                         x_length=0.7, y_length=0.7, z_length=h)
            color = cmap(norm(h))[:3]  # RGB only
            pl.add_mesh(box, color=color, smooth_shading=True)
    
    # Add grid
    pl.show_grid(color='white', xtitle='Photon Number n', 
                ztitle='Probability')
    
    # Title
    title = f"Photon Number Distribution\nα={alpha:.2f}, φ={phase_names.get(phase, f'{phase:.2f}')}"
    pl.add_text(title, position='upper_edge', color='white', font_size=12)
    
    # Camera position
    pl.camera.position = (12, -20, 10)
    pl.camera.focal_point = (12, 0, 0)
    
    filename = f'photon_dist_3d_alpha_{alpha:.1f}.png'
    pl.screenshot(filename)
    print(f"✓ Saved: {filename}")
    pl.close()

def plot_quantum_interference(alpha, phase):
    """3D volume rendering of quantum interference pattern"""
    phase_names = {0: '0', pi/2: 'pi_over_2', pi: 'pi', 3*pi/2: '3pi_over_2'}
    phase_display = {0: '0', pi/2: 'π/2', pi: 'π', 3*pi/2: '3π/2'}
    
    print("🌊 Creating quantum interference visualization...")
    
    grid = create_quantum_cloud(alpha, phase, n_points=60)
    
    pl = pv.Plotter(window_size=[1000, 1000], off_screen=True)
    pl.set_background('black')
    
    # Volume rendering with simple opacity
    pl.add_volume(grid, scalars='wigner', cmap='viridis',
                  opacity='sigmoid', shade=True)
    
    # Add contour surfaces
    contours = grid.contour([0.3, 0.5, 0.7], scalars='wigner')
    pl.add_mesh(contours, color='cyan', opacity=0.2, 
                style='wireframe', line_width=1.5)
    
    # Add axes
    pl.add_axes(color='white')
    
    # Title
    title = f"Quantum Wigner Function (3D)\nα={alpha:.2f}, φ={phase_display.get(phase, f'{phase:.2f}')}"
    pl.add_text(title, position='upper_edge', color='white', font_size=12)
    
    # Camera
    pl.camera.position = (10, 10, 10)
    pl.camera.focal_point = (0, 0, 0)
    
    filename = f'wigner_3d_alpha_{alpha:.1f}_phase_{phase_names.get(phase, "custom")}.png'
    pl.screenshot(filename)
    print(f"✓ Saved: {filename}")
    pl.close()

def create_orbital_animation(alpha, phase):
    """Create rotating animation of quantum state"""
    phase_names = {0: '0', pi/2: 'pi_over_2', pi: 'pi', 3*pi/2: '3pi_over_2'}
    phase_display = {0: '0', pi/2: 'π/2', pi: 'π', 3*pi/2: '3π/2'}
    
    print("🎬 Creating orbital animation...")
    
    grid = create_quantum_cloud(alpha, phase, n_points=45)
    
    pl = pv.Plotter(window_size=[800, 800], off_screen=True)
    pl.set_background('#0a0a1a')
    
    # Volume rendering
    pl.add_volume(grid, scalars='wigner', cmap='plasma',
                  opacity='sigmoid', shade=True)
    
    # Add surface mesh
    surface = grid.contour([0.5], scalars='wigner')
    pl.add_mesh(surface, color='yellow', opacity=0.3, 
                smooth_shading=True)
    
    # Title
    title = f"Cat State: α={alpha:.2f}, φ={phase_display.get(phase, f'{phase:.2f}')}"
    pl.add_text(title, position='upper_edge', color='white', font_size=12)
    
    # Setup camera
    pl.camera.focal_point = (0, 0, 0)
    
    # Create GIF
    filename = f'cat_orbit_alpha_{alpha:.1f}.gif'
    pl.open_gif(filename, fps=20)
    
    n_frames = 60
    for i in range(n_frames):
        angle = 2 * pi * i / n_frames
        radius = 12
        pl.camera.position = (
            radius * np.cos(angle),
            radius * np.sin(angle),
            5 + 3*np.sin(2*angle)
        )
        pl.write_frame()
    
    pl.close()
    print(f"✓ Saved: {filename}")

def create_phase_evolution(alpha):
    """Animate phase evolution of cat state"""
    print("🔄 Creating phase evolution animation...")
    
    pl = pv.Plotter(window_size=[800, 800], off_screen=True)
    pl.set_background('#121212')
    
    filename = f'phase_evolution_alpha_{alpha:.1f}.gif'
    pl.open_gif(filename, fps=15)
    
    n_frames = 30
    for i in range(n_frames):
        phase = 2 * pi * i / n_frames
        
        grid = create_quantum_cloud(alpha, phase, n_points=35)
        
        pl.clear()
        
        # Volume rendering
        pl.add_volume(grid, scalars='wigner', cmap='coolwarm',
                      opacity='linear', shade=True)
        
        # Contours
        contour = grid.contour([0.4], scalars='wigner')
        pl.add_mesh(contour, color='white', opacity=0.4)
        
        # Title
        title = f"Phase Evolution: φ = {phase:.2f} rad ({phase*180/pi:.0f}°)"
        pl.add_text(title, position='upper_edge', color='white', font_size=12)
        
        # Camera
        pl.camera.position = (12, 8, 8)
        pl.camera.focal_point = (0, 0, 0)
        
        pl.write_frame()
    
    pl.close()
    print(f"✓ Saved: {filename}")

def create_comparison_view(alpha, phase):
    """Create side-by-side comparison visualization"""
    phase_names = {0: '0', pi/2: 'pi_over_2', pi: 'pi', 3*pi/2: '3pi_over_2'}
    phase_display = {0: '0', pi/2: 'π/2', pi: 'π', 3*pi/2: '3π/2'}
    
    print("🔬 Creating comparison visualization...")
    
    # Create quantum clouds
    grid_cat = create_quantum_cloud(alpha, phase, n_points=40)
    grid_plus = create_quantum_cloud(alpha, 0, n_points=40)
    grid_minus = create_quantum_cloud(-alpha, 0, n_points=40)
    
    pl = pv.Plotter(shape=(1, 3), window_size=[1500, 500], off_screen=True)
    pl.set_background('black')
    
    # Left: |α⟩
    pl.subplot(0, 0)
    pl.add_volume(grid_plus, scalars='wigner', cmap='Blues', opacity='sigmoid')
    pl.add_text('|α⟩', color='white', font_size=14)
    pl.camera.position = (8, 8, 8)
    pl.camera.focal_point = (0, 0, 0)
    
    # Middle: Cat state
    pl.subplot(0, 1)
    pl.add_volume(grid_cat, scalars='wigner', cmap='plasma', opacity='sigmoid')
    pl.add_text(f'Cat State (φ={phase_display.get(phase, f"{phase:.2f}")})', 
                color='white', font_size=14)
    pl.camera.position = (8, 8, 8)
    pl.camera.focal_point = (0, 0, 0)
    
    # Right: |−α⟩
    pl.subplot(0, 2)
    pl.add_volume(grid_minus, scalars='wigner', cmap='Reds', opacity='sigmoid')
    pl.add_text('|−α⟩', color='white', font_size=14)
    pl.camera.position = (8, 8, 8)
    pl.camera.focal_point = (0, 0, 0)
    
    filename = f'cat_comparison_alpha_{alpha:.1f}.png'
    pl.screenshot(filename)
    print(f"✓ Saved: {filename}")
    pl.close()

def main():
    """Complete PyVista visualization suite"""
    alpha_mag = random.uniform(2.0, 2.8)
    phase = random.choice([0, pi/2, pi, 3*pi/2])
    alpha = alpha_mag
    
    phase_names = {0: '0', pi/2: 'π/2', pi: 'π', 3*pi/2: '3π/2'}
    
    print("=" * 75)
    print("  SCHRÖDINGER CAT STATE - COMPLETE PyVista VISUALIZATION SUITE")
    print("=" * 75)
    print(f"\n🐱 Cat state: |cat⟩ = (|α⟩ + e^(i{phase_names.get(phase, phase)})|−α⟩)/√2")
    print(f"   α = {alpha:.2f}")
    print(f"   Phase: φ = {phase_names.get(phase, phase)}")
    print(f"   Quantum superposition in stunning 3D!\n")
    
    # Create complete visualization suite
    plot_photon_number_distribution(alpha, phase)
    plot_quantum_interference(alpha, phase)
    create_comparison_view(alpha, phase)
    create_orbital_animation(alpha, phase)
    create_phase_evolution(alpha)
    
    print("\n" + "=" * 75)
    print("✨ COMPLETE VISUALIZATION SUITE CREATED!")
    print("   📊 3 high-resolution 3D images")
    print("   🎬 2 animated GIFs")
    print("   🌟 Total: 5 quantum visualizations!")
    print("=" * 75)

if __name__ == "__main__":
    main()
