"""
🔬 Deep Analysis - Parameter Space Exploration

Comprehensive visualization and analysis of the quantum genetic evolution
parameter space, fitness landscape, and convergence patterns.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from quantum_genetic_agents import QuantumAgent
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import time


def explore_parameter_space_2d(param1_name, param1_range, param2_name, param2_range, 
                                fixed_params, resolution=20, timesteps=100):
    """
    Explore 2D parameter space by varying two parameters while keeping others fixed.
    
    Args:
        param1_name: Name of first parameter ('mu', 'omega', 'd', 'phi')
        param1_range: (min, max) for parameter 1
        param2_name: Name of second parameter
        param2_range: (min, max) for parameter 2
        fixed_params: Dict of fixed parameter values
        resolution: Grid resolution
        timesteps: Simulation timesteps
        
    Returns:
        dict: Results including parameter grid and fitness values
    """
    param_map = {'mu': 0, 'omega': 1, 'd': 2, 'phi': 3}
    
    # Create parameter grids
    p1_values = np.linspace(param1_range[0], param1_range[1], resolution)
    p2_values = np.linspace(param2_range[0], param2_range[1], resolution)
    
    # Initialize fitness grid
    fitness_grid = np.zeros((resolution, resolution))
    
    print(f"\n🔍 Exploring {param1_name} vs {param2_name} parameter space...")
    print(f"   Resolution: {resolution}x{resolution} = {resolution*resolution} simulations")
    
    total = resolution * resolution
    completed = 0
    start_time = time.time()
    
    for i, p1_val in enumerate(p1_values):
        for j, p2_val in enumerate(p2_values):
            # Create genome with current parameter values
            genome = list(fixed_params.values())
            genome[param_map[param1_name]] = p1_val
            genome[param_map[param2_name]] = p2_val
            
            # Simulate
            agent = QuantumAgent(agent_id=0, genome=genome, environment='standard')
            for t in range(timesteps):
                agent.evolve(t)
            
            fitness_grid[j, i] = agent.get_final_fitness()
            
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total - completed)
                print(f"   Progress: {completed}/{total} ({100*completed/total:.1f}%) - ETA: {eta:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"   ✅ Complete! Time: {elapsed:.1f}s")
    
    return {
        'param1_name': param1_name,
        'param2_name': param2_name,
        'param1_values': p1_values,
        'param2_values': p2_values,
        'fitness_grid': fitness_grid,
        'fixed_params': fixed_params
    }


def visualize_parameter_space_2d(results, output_path):
    """Create comprehensive 2D parameter space visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    p1_name = results['param1_name']
    p2_name = results['param2_name']
    p1_vals = results['param1_values']
    p2_vals = results['param2_values']
    fitness = results['fitness_grid']
    
    # Greek symbols for parameter names
    symbols = {'mu': 'μ', 'omega': 'ω', 'd': 'd', 'phi': 'φ'}
    p1_label = symbols.get(p1_name, p1_name)
    p2_label = symbols.get(p2_name, p2_name)
    
    # 1. Heatmap
    im1 = ax1.imshow(fitness, origin='lower', aspect='auto', cmap='viridis',
                     extent=[p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]])
    ax1.set_xlabel(f'{p1_label} ({p1_name})', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{p2_label} ({p2_name})', fontsize=12, fontweight='bold')
    ax1.set_title(f'Fitness Landscape: {p1_label} vs {p2_label}', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Fitness')
    
    # Mark champion location if available
    if 'champion' in results.get('fixed_params', {}):
        ax1.plot(results['fixed_params']['champion'][p1_name], 
                results['fixed_params']['champion'][p2_name],
                'r*', markersize=20, label='Champion')
        ax1.legend()
    
    # 2. Contour plot
    levels = 20
    contour = ax2.contour(p1_vals, p2_vals, fitness, levels=levels, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    contourf = ax2.contourf(p1_vals, p2_vals, fitness, levels=levels, cmap='viridis', alpha=0.6)
    ax2.set_xlabel(f'{p1_label} ({p1_name})', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{p2_label} ({p2_name})', fontsize=12, fontweight='bold')
    ax2.set_title(f'Fitness Contours', fontsize=14, fontweight='bold')
    plt.colorbar(contourf, ax=ax2, label='Fitness')
    
    # 3. Cross-sections at optimal points
    # Find max fitness location
    max_idx = np.unravel_index(np.argmax(fitness), fitness.shape)
    
    # Horizontal cross-section (varying p1)
    ax3.plot(p1_vals, fitness[max_idx[0], :], 'b-', linewidth=2, label=f'At {p2_label}={p2_vals[max_idx[0]]:.3f}')
    ax3.axvline(p1_vals[max_idx[1]], color='r', linestyle='--', label='Optimum')
    ax3.set_xlabel(f'{p1_label} ({p1_name})', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax3.set_title(f'Cross-Section: Varying {p1_label}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Vertical cross-section (varying p2)
    ax4.plot(p2_vals, fitness[:, max_idx[1]], 'g-', linewidth=2, label=f'At {p1_label}={p1_vals[max_idx[1]]:.3f}')
    ax4.axvline(p2_vals[max_idx[0]], color='r', linestyle='--', label='Optimum')
    ax4.set_xlabel(f'{p2_label} ({p2_name})', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax4.set_title(f'Cross-Section: Varying {p2_label}', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def analyze_parameter_sensitivity(base_genome, param_name, param_range, num_points=50, timesteps=100):
    """
    Analyze sensitivity of fitness to a single parameter.
    
    Returns detailed sensitivity analysis including gradients.
    """
    param_map = {'mu': 0, 'omega': 1, 'd': 2, 'phi': 3}
    param_idx = param_map[param_name]
    
    param_values = np.linspace(param_range[0], param_range[1], num_points)
    fitness_values = []
    
    print(f"\n📈 Analyzing sensitivity to {param_name}...")
    
    for param_val in param_values:
        genome = base_genome.copy()
        genome[param_idx] = param_val
        
        agent = QuantumAgent(agent_id=0, genome=genome, environment='standard')
        for t in range(timesteps):
            agent.evolve(t)
        
        fitness_values.append(agent.get_final_fitness())
    
    fitness_values = np.array(fitness_values)
    
    # Calculate gradients
    gradients = np.gradient(fitness_values, param_values)
    
    # Find critical points
    sign_changes = np.where(np.diff(np.sign(gradients)))[0]
    
    results = {
        'param_name': param_name,
        'param_values': param_values,
        'fitness_values': fitness_values,
        'gradients': gradients,
        'critical_points': param_values[sign_changes] if len(sign_changes) > 0 else [],
        'max_fitness': np.max(fitness_values),
        'max_param': param_values[np.argmax(fitness_values)],
        'mean_gradient': np.mean(np.abs(gradients)),
        'max_gradient': np.max(np.abs(gradients))
    }
    
    print(f"   Max fitness: {results['max_fitness']:.2f} at {param_name}={results['max_param']:.4f}")
    print(f"   Mean |gradient|: {results['mean_gradient']:.2f}")
    print(f"   Max |gradient|: {results['max_gradient']:.2f}")
    
    return results


def visualize_all_parameter_sensitivities(sensitivity_results, output_path):
    """Visualize sensitivity analysis for all parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    symbols = {'mu': 'μ', 'omega': 'ω', 'd': 'd', 'phi': 'φ'}
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (results, color) in enumerate(zip(sensitivity_results, colors)):
        ax = axes[idx]
        param_name = results['param_name']
        symbol = symbols.get(param_name, param_name)
        
        # Plot fitness
        ax.plot(results['param_values'], results['fitness_values'], 
               color=color, linewidth=2, label='Fitness')
        
        # Mark maximum
        ax.axvline(results['max_param'], color='red', linestyle='--', 
                  alpha=0.7, label=f"Max at {results['max_param']:.4f}")
        
        # Highlight critical points
        if len(results['critical_points']) > 0:
            for cp in results['critical_points']:
                ax.axvline(cp, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(f'{symbol} ({param_name})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax.set_title(f'Sensitivity to {symbol}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with statistics
        stats_text = f"Max: {results['max_fitness']:.0f}\n"
        stats_text += f"Mean |∇|: {results['mean_gradient']:.1f}\n"
        stats_text += f"Max |∇|: {results['max_gradient']:.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def create_3d_fitness_landscape(param1_name, param1_range, param2_name, param2_range,
                                 fixed_params, resolution=30, timesteps=100):
    """Create 3D fitness landscape visualization."""
    print(f"\n🎨 Creating 3D fitness landscape...")
    
    # Get 2D data
    results = explore_parameter_space_2d(
        param1_name, param1_range, param2_name, param2_range,
        fixed_params, resolution, timesteps
    )
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    X, Y = np.meshgrid(results['param1_values'], results['param2_values'])
    Z = results['fitness_grid']
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    symbols = {'mu': 'μ', 'omega': 'ω', 'd': 'd', 'phi': 'φ'}
    ax1.set_xlabel(f"{symbols[param1_name]} ({param1_name})", fontsize=11, fontweight='bold')
    ax1.set_ylabel(f"{symbols[param2_name]} ({param2_name})", fontsize=11, fontweight='bold')
    ax1.set_zlabel('Fitness', fontsize=11, fontweight='bold')
    ax1.set_title('3D Fitness Landscape', fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Wireframe view
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_wireframe(X, Y, Z, color='blue', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel(f"{symbols[param1_name]}", fontsize=11, fontweight='bold')
    ax2.set_ylabel(f"{symbols[param2_name]}", fontsize=11, fontweight='bold')
    ax2.set_zlabel('Fitness', fontsize=11, fontweight='bold')
    ax2.set_title('Wireframe View', fontsize=14, fontweight='bold')
    
    # Top view with contours
    ax3 = fig.add_subplot(223)
    contour = ax3.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax3.set_xlabel(f"{symbols[param1_name]}", fontsize=11, fontweight='bold')
    ax3.set_ylabel(f"{symbols[param2_name]}", fontsize=11, fontweight='bold')
    ax3.set_title('Top View (Contours)', fontsize=14, fontweight='bold')
    fig.colorbar(contour, ax=ax3)
    
    # Gradient magnitude
    ax4 = fig.add_subplot(224)
    gy, gx = np.gradient(Z)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    im = ax4.imshow(gradient_mag, origin='lower', aspect='auto', cmap='hot',
                    extent=[X.min(), X.max(), Y.min(), Y.max()])
    ax4.set_xlabel(f"{symbols[param1_name]}", fontsize=11, fontweight='bold')
    ax4.set_ylabel(f"{symbols[param2_name]}", fontsize=11, fontweight='bold')
    ax4.set_title('Gradient Magnitude (Sensitivity)', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax4, label='|∇Fitness|')
    
    plt.tight_layout()
    
    return fig, results


def main():
    """Run comprehensive deep analysis."""
    print("\n" + "="*70)
    print("🔬 DEEP ANALYSIS - PARAMETER SPACE EXPLORATION")
    print("="*70)
    
    output_dir = Path(__file__).parent / "deep_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Champion genome for reference
    champion_genome = [5.0, 0.1, 0.0001, 6.283185307179586]
    
    # ===== ANALYSIS 1: Parameter Sensitivity Analysis =====
    print("\n" + "="*70)
    print("📊 ANALYSIS 1: Parameter Sensitivity")
    print("="*70)
    
    sensitivity_results = []
    
    # Analyze each parameter
    sensitivity_results.append(analyze_parameter_sensitivity(
        champion_genome, 'mu', (1.0, 5.0), num_points=50
    ))
    
    sensitivity_results.append(analyze_parameter_sensitivity(
        champion_genome, 'omega', (0.05, 0.5), num_points=50
    ))
    
    sensitivity_results.append(analyze_parameter_sensitivity(
        champion_genome, 'd', (0.0001, 0.01), num_points=50
    ))
    
    sensitivity_results.append(analyze_parameter_sensitivity(
        champion_genome, 'phi', (0, 2*np.pi), num_points=50
    ))
    
    # Visualize all sensitivities
    visualize_all_parameter_sensitivities(
        sensitivity_results,
        output_dir / "parameter_sensitivity_analysis.png"
    )
    
    # Save sensitivity data
    sensitivity_data = {
        param['param_name']: {
            'max_fitness': float(param['max_fitness']),
            'max_param': float(param['max_param']),
            'mean_gradient': float(param['mean_gradient']),
            'max_gradient': float(param['max_gradient'])
        }
        for param in sensitivity_results
    }
    
    with open(output_dir / "sensitivity_analysis.json", 'w') as f:
        json.dump(sensitivity_data, f, indent=2)
    
    print(f"\n💾 Sensitivity analysis saved")
    
    # ===== ANALYSIS 2: 2D Parameter Space (d vs phi) =====
    print("\n" + "="*70)
    print("📊 ANALYSIS 2: d vs φ Parameter Space")
    print("="*70)
    
    fixed_params = {'mu': 5.0, 'omega': 0.1, 'd': 0.0001, 'phi': 6.283}
    
    results_d_phi = explore_parameter_space_2d(
        'd', (0.0001, 0.005),
        'phi', (5.5, 7.0),
        fixed_params,
        resolution=25,
        timesteps=100
    )
    
    visualize_parameter_space_2d(
        results_d_phi,
        output_dir / "parameter_space_d_vs_phi.png"
    )
    
    # ===== ANALYSIS 3: 2D Parameter Space (mu vs omega) =====
    print("\n" + "="*70)
    print("📊 ANALYSIS 3: μ vs ω Parameter Space")
    print("="*70)
    
    results_mu_omega = explore_parameter_space_2d(
        'mu', (3.0, 5.0),
        'omega', (0.05, 0.3),
        fixed_params,
        resolution=25,
        timesteps=100
    )
    
    visualize_parameter_space_2d(
        results_mu_omega,
        output_dir / "parameter_space_mu_vs_omega.png"
    )
    
    # ===== ANALYSIS 4: 3D Landscape (d vs phi) =====
    print("\n" + "="*70)
    print("📊 ANALYSIS 4: 3D Fitness Landscape (d vs φ)")
    print("="*70)
    
    fig_3d, _ = create_3d_fitness_landscape(
        'd', (0.0001, 0.005),
        'phi', (5.5, 7.0),
        fixed_params,
        resolution=20,
        timesteps=100
    )
    
    fig_3d.savefig(output_dir / "fitness_landscape_3d_d_phi.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: fitness_landscape_3d_d_phi.png")
    plt.close(fig_3d)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DEEP ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Generated files:")
    print(f"   1. parameter_sensitivity_analysis.png")
    print(f"   2. sensitivity_analysis.json")
    print(f"   3. parameter_space_d_vs_phi.png")
    print(f"   4. parameter_space_mu_vs_omega.png")
    print(f"   5. fitness_landscape_3d_d_phi.png")
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
