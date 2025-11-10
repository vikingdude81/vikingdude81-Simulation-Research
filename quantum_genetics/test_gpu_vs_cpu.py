"""Compare CPU vs GPU performance"""
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
import time

print('\n' + '='*80)
print('⚡ GPU vs CPU PERFORMANCE COMPARISON')
print('='*80)
print('Hardware: NVIDIA RTX 4070 Ti (12GB)')
print('Test: 30 agents, 50 generations, standard environment')
print('='*80 + '\n')

# Test 1: GPU
print('\n' + '-'*80)
print('🔥 TEST 1: GPU-ACCELERATED (ML Adaptive)')
print('-'*80)
gpu_start = time.time()

evo_gpu = AdaptiveMutationEvolution(
    population_size=30,
    strategy='ml_adaptive',
    use_gpu=True
)

results_gpu = evo_gpu.run(
    generations=50,
    environment='standard',
    train_predictor=False
)

gpu_time = time.time() - gpu_start
gpu_fitness = results_gpu['final_best'][0]

print(f'\n✅ GPU Complete!')
print(f'   Time: {gpu_time:.2f}s')
print(f'   Final fitness: {gpu_fitness:.6f}')
print(f'   Avg time/gen: {gpu_time/50:.3f}s')

# Test 2: CPU (for comparison)
print('\n' + '-'*80)
print('🖥️  TEST 2: CPU-ONLY (ML Adaptive)')
print('-'*80)
cpu_start = time.time()

evo_cpu = AdaptiveMutationEvolution(
    population_size=30,
    strategy='ml_adaptive',
    use_gpu=False  # Force CPU
)

results_cpu = evo_cpu.run(
    generations=50,
    environment='standard',
    train_predictor=False
)

cpu_time = time.time() - cpu_start
cpu_fitness = results_cpu['final_best'][0]

print(f'\n✅ CPU Complete!')
print(f'   Time: {cpu_time:.2f}s')
print(f'   Final fitness: {cpu_fitness:.6f}')
print(f'   Avg time/gen: {cpu_time/50:.3f}s')

# Comparison
print('\n' + '='*80)
print('📊 PERFORMANCE SUMMARY')
print('='*80)
print(f'GPU Time:     {gpu_time:6.2f}s  |  Fitness: {gpu_fitness:.6f}')
print(f'CPU Time:     {cpu_time:6.2f}s  |  Fitness: {cpu_fitness:.6f}')
print('-'*80)
speedup = cpu_time / gpu_time
print(f'SPEEDUP:      {speedup:6.2f}x  |  GPU is {speedup:.1f}x FASTER! 🚀')
print(f'Time Saved:   {cpu_time - gpu_time:6.2f}s  |  {((cpu_time - gpu_time) / cpu_time * 100):.1f}% faster')
print('='*80)

# What this means
print('\n💡 WHAT THIS MEANS:')
print(f'   • For 100 generations: GPU = {gpu_time*2:.1f}s vs CPU = {cpu_time*2:.1f}s')
print(f'   • For 500 generations: GPU = {gpu_time*10:.1f}s vs CPU = {cpu_time*10:.1f}s')
print(f'   • In 1 hour on GPU: ~{3600/(gpu_time/50):.0f} generations')
print(f'   • In 1 hour on CPU: ~{3600/(cpu_time/50):.0f} generations')
print(f'   • GPU allows {speedup:.1f}x MORE experiments in same time!')
print('='*80 + '\n')
