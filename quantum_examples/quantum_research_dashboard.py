#!/usr/bin/env python3
"""
🔬 Quantum Research Dashboard
Interactive menu system for quantum simulations, analysis, and evolution research
"""

import os
import sys
import subprocess
from termcolor import colored

def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header():
    """Print the dashboard header"""
    print(colored("=" * 80, "cyan"))
    print(colored("🔬 QUANTUM RESEARCH DASHBOARD", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan"))
    print()

def print_menu(title, options):
    """Print a formatted menu"""
    print(colored(f"\n{title}", "yellow", attrs=["bold"]))
    print(colored("-" * 60, "yellow"))
    for key, value in options.items():
        print(f"  {colored(key, 'green')} - {value}")
    print()

def run_script(script_path, description):
    """Run a Python script with proper error handling"""
    print(colored(f"\n{'=' * 60}", "cyan"))
    print(colored(f"🚀 Running: {description}", "cyan", attrs=["bold"]))
    print(colored(f"{'=' * 60}\n", "cyan"))
    
    try:
        result = subprocess.run([sys.executable, script_path], check=False)
        if result.returncode != 0:
            print(colored(f"\n⚠️  Script exited with code {result.returncode}", "yellow"))
        else:
            print(colored(f"\n✅ {description} completed successfully!", "green"))
    except FileNotFoundError:
        print(colored(f"\n❌ Error: Script not found at {script_path}", "red"))
    except KeyboardInterrupt:
        print(colored("\n\n⏸️  Interrupted by user", "yellow"))
    except Exception as e:
        print(colored(f"\n❌ Error running script: {e}", "red"))
    
    input(colored("\nPress Enter to continue...", "cyan"))

def quantum_examples_menu():
    """Quantum examples submenu"""
    while True:
        clear_screen()
        print_header()
        print(colored("📊 QUANTUM EXAMPLES & SIMULATIONS", "magenta", attrs=["bold"]))
        
        examples = {
            "1": "Bloch Sphere - Visualize qubit states on the Bloch sphere",
            "2": "Double Slit Experiment - Classic quantum interference",
            "3": "Hydrogen Atom - Atomic orbital visualizations",
            "4": "Quantum 3D - 3D harmonic oscillator states",
            "5": "Quantum 4D - 4D quantum system projections",
            "6": "Quantum Decoherence - Decoherence effects visualization",
            "7": "Quantum Entanglement - Entangled state evolution",
            "8": "Quantum Gates - Quantum gate operations",
            "9": "Quantum Tunneling - Tunneling through potential barriers",
            "10": "Rabi Oscillations - Two-level system dynamics",
            "11": "Random Quantum States - Generate random quantum states",
            "12": "Schrödinger's Cat - Superposition and measurement",
            "13": "Wavepacket Evolution - Time evolution of wavepackets",
            "14": "All Visualizations - Run comprehensive visualization suite",
            "b": "Back to main menu"
        }
        
        print_menu("Select a quantum example:", examples)
        choice = input(colored("Enter your choice: ", "cyan")).strip().lower()
        
        script_map = {
            "1": ("quantum_examples/bloch_sphere.py", "Bloch Sphere"),
            "2": ("quantum_examples/double_slit.py", "Double Slit Experiment"),
            "3": ("quantum_examples/hydrogen_atom.py", "Hydrogen Atom"),
            "4": ("quantum_examples/quantum_3d.py", "Quantum 3D"),
            "5": ("quantum_examples/quantum_4d.py", "Quantum 4D"),
            "6": ("quantum_examples/quantum_decoherence.py", "Quantum Decoherence"),
            "7": ("quantum_examples/quantum_entanglement.py", "Quantum Entanglement"),
            "8": ("quantum_examples/quantum_gates.py", "Quantum Gates"),
            "9": ("quantum_examples/quantum_tunneling.py", "Quantum Tunneling"),
            "10": ("quantum_examples/rabi_oscillations.py", "Rabi Oscillations"),
            "11": ("quantum_examples/random_quantum.py", "Random Quantum States"),
            "12": ("quantum_examples/schrodinger_cat.py", "Schrödinger's Cat"),
            "13": ("quantum_examples/wavepacket_evolution.py", "Wavepacket Evolution"),
            "14": ("quantum_examples/all_visualizations.py", "All Visualizations"),
        }
        
        if choice == "b":
            break
        elif choice in script_map:
            script_path, description = script_map[choice]
            run_script(script_path, description)
        else:
            print(colored("Invalid choice. Please try again.", "red"))
            input(colored("Press Enter to continue...", "cyan"))

def analysis_menu():
    """Analysis tools submenu"""
    while True:
        clear_screen()
        print_header()
        print(colored("📈 ANALYSIS & RESEARCH TOOLS", "magenta", attrs=["bold"]))
        
        tools = {
            "1": "Analyze Evolution Dynamics - Study evolutionary trends",
            "2": "Compare All Genomes - Benchmark genome performance",
            "3": "Comprehensive Analysis - Full system analysis",
            "4": "Deep Genome Analysis - In-depth genome inspection",
            "5": "Quantum Data Analysis - Analyze quantum datasets",
            "6": "Quantum Research - General quantum research tools",
            "7": "Genome App Tester - Test genome applications",
            "8": "Quantum Genome Tester - Test quantum genome fitness",
            "9": "Extreme Frontier Test - Test extreme conditions",
            "b": "Back to main menu"
        }
        
        print_menu("Select an analysis tool:", tools)
        choice = input(colored("Enter your choice: ", "cyan")).strip().lower()
        
        script_map = {
            "1": ("analysis_tools/analyze_evolution_dynamics.py", "Evolution Dynamics Analysis"),
            "2": ("analysis_tools/compare_all_genomes.py", "Genome Comparison"),
            "3": ("analysis_tools/comprehensive_analysis.py", "Comprehensive Analysis"),
            "4": ("analysis_tools/deep_genome_analysis.py", "Deep Genome Analysis"),
            "5": ("analysis_tools/quantum_data_analysis.py", "Quantum Data Analysis"),
            "6": ("analysis_tools/quantum_research.py", "Quantum Research"),
            "7": ("analysis_tools/genome_app_tester.py", "Genome App Tester"),
            "8": ("analysis_tools/quantum_genome_tester.py", "Quantum Genome Tester"),
            "9": ("analysis_tools/extreme_frontier_test.py", "Extreme Frontier Test"),
        }
        
        if choice == "b":
            break
        elif choice in script_map:
            script_path, description = script_map[choice]
            run_script(script_path, description)
        else:
            print(colored("Invalid choice. Please try again.", "red"))
            input(colored("Press Enter to continue...", "cyan"))

def evolution_menu():
    """Evolution engine submenu"""
    while True:
        clear_screen()
        print_header()
        print(colored("🧬 EVOLUTION ENGINE", "magenta", attrs=["bold"]))
        
        options = {
            "1": "Quantum Genetic Agents - Main evolution engine",
            "2": "Multi-Objective Evolution - Optimize multiple objectives",
            "3": "Phase-Focused Evolution - Focus on phase parameters",
            "4": "Quantum Evolution Agents - Quantum-specific evolution",
            "5": "Quantum ML - Machine learning genome prediction",
            "b": "Back to main menu"
        }
        
        print_menu("Select an evolution experiment:", options)
        choice = input(colored("Enter your choice: ", "cyan")).strip().lower()
        
        script_map = {
            "1": ("evolution_engine/quantum_genetic_agents.py", "Quantum Genetic Agents"),
            "2": ("evolution_engine/multi_objective_evolution.py", "Multi-Objective Evolution"),
            "3": ("evolution_engine/phase_focused_evolution.py", "Phase-Focused Evolution"),
            "4": ("evolution_engine/quantum_evolution_agents.py", "Quantum Evolution Agents"),
            "5": ("evolution_engine/quantum_ml.py", "Quantum ML"),
        }
        
        if choice == "b":
            break
        elif choice in script_map:
            script_path, description = script_map[choice]
            print(colored(f"\n⚠️  Note: Evolution experiments can take several minutes to complete.", "yellow"))
            confirm = input(colored("Continue? (y/n): ", "cyan")).strip().lower()
            if confirm == 'y':
                run_script(script_path, description)
        else:
            print(colored("Invalid choice. Please try again.", "red"))
            input(colored("Press Enter to continue...", "cyan"))

def server_menu():
    """Server management submenu"""
    while True:
        clear_screen()
        print_header()
        print(colored("🚀 SERVER MANAGEMENT", "magenta", attrs=["bold"]))
        
        options = {
            "1": "Start Genome Deployment Server - Launch web dashboard (port 5000)",
            "2": "Start Co-Evolution Server - Launch co-evolution server",
            "3": "View Server Documentation - How to use the servers",
            "b": "Back to main menu"
        }
        
        print_menu("Select a server option:", options)
        choice = input(colored("Enter your choice: ", "cyan")).strip().lower()
        
        if choice == "1":
            print(colored("\n" + "=" * 60, "cyan"))
            print(colored("🚀 Starting Genome Deployment Server...", "cyan", attrs=["bold"]))
            print(colored("=" * 60, "cyan"))
            print(colored("\n💡 The server will start on http://0.0.0.0:5000", "yellow"))
            print(colored("   Use the Webview panel to access the dashboard", "yellow"))
            print(colored("\n   Press Ctrl+C to stop the server\n", "yellow"))
            input(colored("Press Enter to start the server...", "cyan"))
            run_script("server/genome_deployment_server.py", "Genome Deployment Server")
            
        elif choice == "2":
            print(colored("\n" + "=" * 60, "cyan"))
            print(colored("🚀 Starting Co-Evolution Server...", "cyan", attrs=["bold"]))
            print(colored("=" * 60, "cyan"))
            print(colored("\n   Press Ctrl+C to stop the server\n", "yellow"))
            input(colored("Press Enter to start the server...", "cyan"))
            run_script("server/co_evolution_server.py", "Co-Evolution Server")
            
        elif choice == "3":
            print(colored("\n" + "=" * 60, "yellow"))
            print(colored("📚 SERVER DOCUMENTATION", "yellow", attrs=["bold"]))
            print(colored("=" * 60 + "\n", "yellow"))
            print("🌐 Genome Deployment Server:")
            print("   - Web dashboard for deploying and monitoring evolved genomes")
            print("   - Real-time performance tracking")
            print("   - Deploy best individual, averaged ensemble, or custom genomes")
            print("   - Access at: http://0.0.0.0:5000")
            print("\n🔄 Co-Evolution Server:")
            print("   - Server for co-evolutionary experiments")
            print("   - Multiple populations evolving together")
            print("   - Advanced competition and cooperation dynamics")
            input(colored("\nPress Enter to continue...", "cyan"))
            
        elif choice == "b":
            break
        else:
            print(colored("Invalid choice. Please try again.", "red"))
            input(colored("Press Enter to continue...", "cyan"))

def main_menu():
    """Main menu loop"""
    while True:
        clear_screen()
        print_header()
        print(colored("Welcome to the Quantum Research Platform!", "white"))
        print(colored("Explore quantum simulations, run analysis, and evolve quantum agents.\n", "white"))
        
        main_options = {
            "1": "🔬 Quantum Examples - Explore quantum phenomena simulations",
            "2": "📈 Analysis Tools - Research and data analysis",
            "3": "🧬 Evolution Engine - Run evolutionary experiments",
            "4": "🚀 Server Management - Deploy and monitor genomes",
            "5": "📖 About - Project information",
            "q": "Quit"
        }
        
        print_menu("Main Menu:", main_options)
        choice = input(colored("Enter your choice: ", "cyan")).strip().lower()
        
        if choice == "1":
            quantum_examples_menu()
        elif choice == "2":
            analysis_menu()
        elif choice == "3":
            evolution_menu()
        elif choice == "4":
            server_menu()
        elif choice == "5":
            clear_screen()
            print_header()
            print(colored("📖 ABOUT THE QUANTUM RESEARCH PLATFORM\n", "yellow", attrs=["bold"]))
            print("This platform combines quantum mechanics simulations with")
            print("genetic algorithms to create an integrated research environment.")
            print("\nKey Features:")
            print("  • 14 quantum physics simulation examples")
            print("  • 9 analysis and research tools")
            print("  • 5 evolutionary computation engines")
            print("  • Web-based genome deployment dashboard")
            print("\nProject Structure:")
            print("  quantum_examples/  - Quantum physics simulations")
            print("  analysis_tools/    - Research and analysis scripts")
            print("  evolution_engine/  - Genetic algorithm implementations")
            print("  server/           - Web servers and APIs")
            print("\nFor more information, see replit.md")
            input(colored("\nPress Enter to continue...", "cyan"))
        elif choice == "q":
            clear_screen()
            print(colored("\n👋 Thank you for using the Quantum Research Platform!\n", "cyan"))
            sys.exit(0)
        else:
            print(colored("Invalid choice. Please try again.", "red"))
            input(colored("Press Enter to continue...", "cyan"))

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        clear_screen()
        print(colored("\n\n👋 Goodbye!\n", "cyan"))
        sys.exit(0)
