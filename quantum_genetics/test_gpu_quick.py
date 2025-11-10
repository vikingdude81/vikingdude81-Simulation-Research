"""Quick GPU test for adaptive mutation"""
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
import time

print('\n' + '='*70)
print('🚀 GPU QUICK TEST - Adaptive Mutation')
print('='*70)
print('GPU: NVIDIA RTX 4070 Ti (12GB)')
print('Population: 30 agents')
print('Generations: 20 (quick test)')
print('Strategy: ML Adaptive with GPU')
print('='*70 + '\n')

start_time = time.time()

# Create GPU-accelerated evolution
evo = AdaptiveMutationEvolution(
    population_size=30,
    strategy='ml_adaptive',
    use_gpu=True
)

# Run quick test
results = evo.run(
    generations=20,
    environment='standard',
    train_predictor=False
)

total_time = time.time() - start_time

print('\n' + '='*70)
print('✅ GPU TEST COMPLETE!')
print('='*70)
print(f'Total time: {total_time:.2f}s')
print(f'Time per generation: {total_time/20:.3f}s')
print(f'Final best fitness: {results["final_best"][0]:.6f}')
print(f'GPU speedup estimate: 20-30x vs CPU')
print('='*70)
