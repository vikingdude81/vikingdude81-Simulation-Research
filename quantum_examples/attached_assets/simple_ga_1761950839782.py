"""
🧬 Simple Genetic Algorithm - String Evolution
Based on "Hidden Order" by John Holland

This is the "Hello World" of GAs. The goal is to evolve a random string 
until it matches a target string (like evolving chromosomes in nature).

Concepts demonstrated:
- Chromosomes (strings of genes)
- Fitness (how close to target)
- Selection (keeping the best)
- Crossover (breeding)
- Mutation (random changes)
"""
import random

# === CONFIGURATION ===
TARGET = "HIDDENORDER"
GENES = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
ELITE_COUNT = 10

print("=" * 80)
print("🧬 GENETIC ALGORITHM - STRING EVOLUTION")
print("=" * 80)
print(f"\n🎯 Target: '{TARGET}'")
print(f"📊 Population Size: {POPULATION_SIZE}")
print(f"🧪 Mutation Rate: {MUTATION_RATE}")
print(f"🏆 Elite Count: {ELITE_COUNT}")
print("\n" + "=" * 80)

# === AGENT (CHROMOSOME) FUNCTIONS ===

def create_individual():
    """Creates a random chromosome string."""
    return ''.join(random.choice(GENES) for _ in range(len(TARGET)))

def calculate_fitness(individual):
    """
    Calculates fitness = how many characters match the target.
    Higher fitness = closer to target = better adaptation.
    """
    score = 0
    for i in range(len(TARGET)):
        if individual[i] == TARGET[i]:
            score += 1
    return score

def crossover(parent1, parent2):
    """
    Performs crossover (breeding).
    Takes genes from parent1 and parent2 to create offspring.
    Like mixing DNA from two parents.
    """
    midpoint = random.randint(0, len(TARGET) - 1)
    child = parent1[:midpoint] + parent2[midpoint:]
    return child

def mutate(individual):
    """
    Performs mutation on an individual.
    Each gene has a small chance to randomly change.
    This maintains diversity and enables exploration.
    """
    individual_list = list(individual)
    for i in range(len(individual_list)):
        if random.random() < MUTATION_RATE:
            individual_list[i] = random.choice(GENES)
    return "".join(individual_list)

# === MAIN EVOLUTION LOOP ===

# 1. INITIALIZATION: Create random first generation
population = [create_individual() for _ in range(POPULATION_SIZE)]

generation = 1
best_history = []

while True:
    # 2. FITNESS CALCULATION: Score every individual
    fitness_scores = [(calculate_fitness(ind), ind) for ind in population]
    
    # Sort by fitness (highest first)
    sorted_population = sorted(fitness_scores, key=lambda x: x[0], reverse=True)
    
    best_fitness, best_individual = sorted_population[0]
    avg_fitness = sum(f[0] for f in fitness_scores) / len(fitness_scores)
    
    best_history.append(best_fitness)
    
    # Display progress every 10 generations or when best improves
    if generation == 1 or generation % 10 == 0 or best_fitness > best_history[-2] if len(best_history) > 1 else False:
        print(f"Gen {generation:4d} | Best: {best_fitness:2d}/{len(TARGET)} | Avg: {avg_fitness:5.2f} | '{best_individual}'")
    
    # 3. TERMINATION: Stop if target reached
    if best_fitness == len(TARGET):
        print("\n" + "=" * 80)
        print("✅ TARGET EVOLVED SUCCESSFULLY!")
        print("=" * 80)
        print(f"🏆 Solution: '{best_individual}'")
        print(f"📊 Generations: {generation}")
        print(f"🧬 Total evaluations: {generation * POPULATION_SIZE}")
        print("=" * 80)
        break
    
    # 4. SELECTION & REPRODUCTION: Create next generation
    next_generation = []
    
    # Keep the elites (best individuals) unchanged
    # This is "elitism" - ensures we don't lose good solutions
    elites = [ind[1] for ind in sorted_population[:ELITE_COUNT]]
    next_generation.extend(elites)
    
    # 5. CROSSOVER & MUTATION: Fill rest of population
    while len(next_generation) < POPULATION_SIZE:
        # Select two fit parents from top half
        # This is "tournament selection"
        parent1 = random.choice(sorted_population[:POPULATION_SIZE // 2])[1]
        parent2 = random.choice(sorted_population[:POPULATION_SIZE // 2])[1]
        
        # Create child through crossover
        child = crossover(parent1, parent2)
        
        # Mutate child (random exploration)
        child = mutate(child)
        
        next_generation.append(child)
    
    # New generation replaces old
    population = next_generation
    generation += 1

# === ANALYSIS ===
print("\n📈 Evolution Statistics:")
print(f"   Initial best fitness: {best_history[0]}")
print(f"   Final best fitness: {best_history[-1]}")
print(f"   Improvement: {best_history[-1] - best_history[0]} characters")
print(f"   Generations to solve: {generation}")

# Show evolution progress
print("\n📊 Fitness Over Time:")
milestones = [0, len(best_history)//4, len(best_history)//2, 3*len(best_history)//4, -1]
for i in milestones:
    gen = i if i >= 0 else len(best_history) - 1
    print(f"   Gen {gen:4d}: Fitness = {best_history[gen]:2d}/{len(TARGET)}")

print("\n💡 Key Concepts Demonstrated:")
print("   ✅ Chromosome: String of genes")
print("   ✅ Fitness: Match quality metric")
print("   ✅ Selection: Best survive")
print("   ✅ Crossover: Breeding/mixing genes")
print("   ✅ Mutation: Random exploration")
print("   ✅ Elitism: Preserve best solutions")
print("   ✅ Evolution: Gradual improvement")

print("\n🎯 Ready for trading strategies? This same process will evolve")
print("   buy/sell/hold decisions instead of letters!")
print("=" * 80)
