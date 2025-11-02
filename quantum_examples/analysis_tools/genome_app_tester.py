
"""
🧪 Genome App Tester - Test your apps with evolved genomes
Use Gen 2117 or other genomes as parameter sets for testing your applications
"""

import json
import sys
import subprocess
from quantum_genetic_agents import load_genome

class GenomeAppTester:
    """Test applications using genome parameters"""
    
    def __init__(self, genome_file):
        self.genome, self.metadata = load_genome(genome_file)
        self.results = []
        
    def get_params_as_dict(self, param_names=None):
        """Convert genome to dictionary for app testing"""
        if param_names is None:
            param_names = ['mutation_rate', 'oscillation_freq', 'decoherence_rate', 'phase_offset']
        
        return {
            name: value for name, value in zip(param_names, self.genome)
        }
    
    def get_params_as_env_vars(self):
        """Get genome parameters as environment variable dict"""
        return {
            'GENOME_MUTATION_RATE': str(self.genome[0]),
            'GENOME_OSCILLATION_FREQ': str(self.genome[1]),
            'GENOME_DECOHERENCE_RATE': str(self.genome[2]),
            'GENOME_PHASE_OFFSET': str(self.genome[3]),
            'GENOME_FITNESS': str(self.metadata.get('fitness', 0))
        }
    
    def test_command(self, command, description=""):
        """Test a shell command with genome params as env vars"""
        import os
        
        env = os.environ.copy()
        env.update(self.get_params_as_env_vars())
        
        print(f"\n🧪 Testing: {description or command}")
        print(f"   Using genome: {self.genome}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            
            self.results.append({
                'command': command,
                'description': description,
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'genome': self.genome
            })
            
            if success:
                print(f"   ✓ Success!")
            else:
                print(f"   ✗ Failed (exit code: {result.returncode})")
            
            return success, result
            
        except subprocess.TimeoutExpired:
            print(f"   ⏱ Timeout (30s)")
            return False, None
    
    def test_python_function(self, func, description=""):
        """Test a Python function with genome parameters"""
        print(f"\n🧪 Testing function: {description or func.__name__}")
        print(f"   Using genome: {self.genome}")
        
        try:
            params = self.get_params_as_dict()
            result = func(**params)
            
            self.results.append({
                'function': func.__name__,
                'description': description,
                'success': True,
                'result': result,
                'genome': self.genome
            })
            
            print(f"   ✓ Success! Result: {result}")
            return True, result
            
        except Exception as e:
            self.results.append({
                'function': func.__name__,
                'description': description,
                'success': False,
                'error': str(e),
                'genome': self.genome
            })
            
            print(f"   ✗ Error: {e}")
            return False, None
    
    def export_results(self, filename='test_results.json'):
        """Export test results to JSON"""
        with open(filename, 'w') as f:
            json.dump({
                'genome_file': self.metadata,
                'genome': self.genome,
                'tests': self.results
            }, f, indent=2)
        
        print(f"\n✓ Results exported to: {filename}")

def example_app_test_function(**genome_params):
    """Example: Use genome parameters to configure your app"""
    learning_rate = genome_params['mutation_rate'] * 0.001
    momentum = genome_params['oscillation_freq'] / 5.0
    dropout = 1 - genome_params['decoherence_rate'] * 10
    
    # Simulate ML training
    accuracy = 0.7 + (learning_rate * 100) + (momentum * 0.1)
    
    return {
        'accuracy': min(accuracy, 1.0),
        'learning_rate': learning_rate,
        'momentum': momentum,
        'dropout': dropout
    }

def main():
    print("\n" + "=" * 80)
    print("  🧪 GENOME APP TESTER")
    print("=" * 80)
    
    # Test with Gen 2117
    print("\n📊 Testing with Gen 2117 (Production Genome)...")
    tester = GenomeAppTester('co_evolved_best_gen_2117.json')
    
    # Example 1: Test Python function
    tester.test_python_function(
        example_app_test_function,
        description="ML hyperparameter tuning"
    )
    
    # Example 2: Test shell command with env vars
    tester.test_command(
        'echo "Mutation rate: $GENOME_MUTATION_RATE, Fitness: $GENOME_FITNESS"',
        description="Environment variable access"
    )
    
    # Export results
    tester.export_results('gen2117_app_tests.json')
    
    print("\n" + "=" * 80)
    print("✨ HOW TO USE IN YOUR WORKFLOW:")
    print("=" * 80)
    print("""
1. In your Python app:
   from genome_app_tester import GenomeAppTester
   
   tester = GenomeAppTester('co_evolved_best_gen_2117.json')
   params = tester.get_params_as_dict(['lr', 'momentum', 'decay', 'batch_norm'])
   
   # Use params in your ML model
   model.compile(
       learning_rate=params['lr'] * 0.001,
       momentum=params['momentum'] / 2.0
   )

2. In shell scripts:
   export $(python -c "from genome_app_tester import GenomeAppTester; \\
           t = GenomeAppTester('co_evolved_best_gen_2117.json'); \\
           print(' '.join([f'{k}={v}' for k,v in t.get_params_as_env_vars().items()]))")
   
   # Use $GENOME_MUTATION_RATE, etc. in your script

3. Test different genomes:
   for genome in specialized_*.json; do
       tester = GenomeAppTester($genome)
       tester.test_python_function(my_app_function)
   done
    """)
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
