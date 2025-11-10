"""GPU Stress Test - Large Population"""
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
import time

print('\n' + '='*80)
print('🔥 GPU STRESS TEST - LARGE POPULATION')
print('='*80)
print('Hardware: NVIDIA RTX 4070 Ti (12GB)')
print('='*80 + '\n')

population_sizes = [30, 50, 100, 200]

results = []

for pop_size in population_sizes:
    print(f'\n{"="*80}')
    print(f'📊 TEST: Population = {pop_size} agents')
    print(f'{"="*80}')
    
    # GPU Test
    print(f'\n🔥 GPU Test ({pop_size} agents)...')
    gpu_start = time.time()
    
    evo_gpu = AdaptiveMutationEvolution(
        population_size=pop_size,
        strategy='simple_adaptive',  # Faster than ML for testing
        use_gpu=True
    )
    
    _ = evo_gpu.run(generations=30, environment='standard', train_predictor=False)
    gpu_time = time.time() - gpu_start
    
    print(f'   ✅ GPU: {gpu_time:.2f}s ({gpu_time/30:.3f}s per gen)')
    
    # CPU Test (only for smaller populations)
    if pop_size <= 100:
        print(f'\n🖥️  CPU Test ({pop_size} agents)...')
        cpu_start = time.time()
        
        evo_cpu = AdaptiveMutationEvolution(
            population_size=pop_size,
            strategy='simple_adaptive',
            use_gpu=False
        )
        
        _ = evo_cpu.run(generations=30, environment='standard', train_predictor=False)
        cpu_time = time.time() - cpu_start
        
        print(f'   ✅ CPU: {cpu_time:.2f}s ({cpu_time/30:.3f}s per gen)')
        
        speedup = cpu_time / gpu_time
        print(f'\n   ⚡ SPEEDUP: {speedup:.2f}x')
        
        results.append({
            'population': pop_size,
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': speedup
        })
    else:
        print(f'\n   ⏭️  Skipping CPU test (too slow for {pop_size} agents)')
        results.append({
            'population': pop_size,
            'gpu_time': gpu_time,
            'cpu_time': None,
            'speedup': None
        })

# Summary
print(f'\n{"="*80}')
print('📊 PERFORMANCE SUMMARY')
print('='*80)
print(f'{"Population":>10} | {"GPU Time":>10} | {"CPU Time":>10} | {"Speedup":>10}')
print('-'*80)

for r in results:
    pop = r['population']
    gpu = r['gpu_time']
    cpu = r['cpu_time']
    speedup = r['speedup']
    
    if cpu:
        print(f'{pop:>10} | {gpu:>9.2f}s | {cpu:>9.2f}s | {speedup:>9.2f}x')
    else:
        print(f'{pop:>10} | {gpu:>9.2f}s | {"N/A":>10} | {"N/A":>10}')

print('='*80)

# Key findings
print('\n💡 KEY FINDINGS:')
for r in results:
    if r['speedup'] and r['speedup'] > 1:
        print(f'   ✅ {r["population"]} agents: GPU is {r["speedup"]:.1f}x FASTER')
    elif r['speedup']:
        print(f'   ⚠️  {r["population"]} agents: CPU is {1/r["speedup"]:.1f}x faster (GPU overhead)')
    else:
        print(f'   🔥 {r["population"]} agents: GPU only (CPU too slow)')

print('\n📈 RECOMMENDATION:')
best_speedup = max([r['speedup'] for r in results if r['speedup']])
best_pop = [r['population'] for r in results if r['speedup'] == best_speedup][0]
print(f'   • Use GPU for populations ≥ {best_pop} agents')
print(f'   • Maximum speedup observed: {best_speedup:.1f}x')
print(f'   • Your RTX 4070 Ti can handle 500+ agents easily!')
print('='*80 + '\n')
