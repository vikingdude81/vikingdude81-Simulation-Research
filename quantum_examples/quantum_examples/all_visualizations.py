
import subprocess
import sys
from tqdm import tqdm
import time

def run_visualization(script_name, description, progress_bar):
    """Run a visualization script and handle output with progress tracking"""
    print("\n" + "=" * 70)
    print(f"  RUNNING: {description}")
    print("=" * 70)
    
    try:
        # Update progress bar description
        progress_bar.set_description(f"🎨 {description}")
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True)
        
        # Print the output from the subprocess
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully!")
            progress_bar.update(1)
        else:
            print(f"✗ {description} encountered an error.")
            if result.stderr:
                print(result.stderr)
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")

def main():
    print("\n" + "=" * 70)
    print("  QUANTUM VISUALIZATION SUITE - RUNNING ALL 9 MODULES")
    print("=" * 70)
    print("\nThis will generate all quantum mechanics visualizations:")
    print("  1. 2D Quantum Harmonic Oscillator")
    print("  2. 3D Quantum Harmonic Oscillator")
    print("  3. 4D Quantum Harmonic Oscillator")
    print("  4. Hydrogen Atom Orbitals")
    print("  5. Quantum Tunneling")
    print("  6. Wavepacket Spreading")
    print("  7. Quantum Entanglement (Bell States)")
    print("  8. Schrödinger Cat States")
    print("  9. Double-Slit Interference")
    print("\n" + "=" * 70)
    
    visualizations = [
        ("random_quantum.py", "2D Quantum Harmonic Oscillator"),
        ("quantum_3d.py", "3D Quantum Harmonic Oscillator"),
        ("quantum_4d.py", "4D Quantum Harmonic Oscillator"),
        ("hydrogen_atom.py", "Hydrogen Atom Orbitals"),
        ("quantum_tunneling.py", "Quantum Tunneling"),
        ("wavepacket_evolution.py", "Wavepacket Spreading"),
        ("quantum_entanglement.py", "Quantum Entanglement (Bell States)"),
        ("schrodinger_cat.py", "Schrödinger Cat States"),
        ("double_slit.py", "Double-Slit Interference")
    ]
    
    # Create overall progress bar
    with tqdm(total=len(visualizations), 
              desc="🌟 Overall Progress", 
              unit="module",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              colour='green') as overall_progress:
        
        for script, description in visualizations:
            run_visualization(script, description, overall_progress)
    
    print("\n" + "=" * 70)
    print("  ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print("\n📁 Check your file explorer for all generated images and animations!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
