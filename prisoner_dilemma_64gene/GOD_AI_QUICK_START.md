# 🚀 GOD-AI QUICK START GUIDE

## ⚡ 30-Second Start

```python
cd prisoner_dilemma_64gene
python prisoner_echo_god.py
```

Watch the God-AI monitor and intervene in real-time!

## 📖 What You'll See

```
🧠👁️  ECHO MODEL WITH GOD-AI CONTROLLER (Mode: RULE_BASED) 👁️🧠
================================================================================

🧠 GOD: 💰 STIMULUS: Gave 50 resources to all 247 agents | Reason: Economic crisis!

Generation: 147 | Elapsed: 23.4s | Speed: 6.28 gen/s

POPULATION STATS
Size: 247 | Avg Age: 34.2 | Clustering: 61.3%
Resources: Avg=143.5, Min=12, Max=8742, Total=35444
Cooperation: 62.3% [████████████████░░░░]

🧠 GOD-AI INTERVENTIONS
Total: 8 | stimulus: 3 | welfare: 2 | spawn_tribe: 2 | emergency_revival: 1

SPATIAL GRID (30×30 sample)
[Colored grid showing wealth distribution]
```

## 🎮 Interactive Options

### 1. Quick Test (50 generations)
```python
from prisoner_echo_god import run_god_echo_simulation

# Fast test - see interventions in action
run_god_echo_simulation(
    generations=50,
    god_mode="RULE_BASED",
    update_frequency=5  # Update every 5 generations
)
```

### 2. Full Simulation (500 generations)
```python
# Full run with all features
run_god_echo_simulation(
    generations=500,
    initial_size=100,
    god_mode="RULE_BASED",
    update_frequency=10
)
```

### 3. Baseline (No God)
```python
# Pure evolution without interventions
run_god_echo_simulation(
    generations=500,
    god_mode="DISABLED"
)
```

### 4. Comparative Experiment
```python
from compare_god_ai import run_controlled_experiment

# Run 5 trials of God vs No-God
results = run_controlled_experiment(
    generations=500,
    trials=5
)
```

## 🎯 What Interventions Will You See?

### 💰 STIMULUS (Universal Basic Income)
**Trigger**: Average wealth < 50  
**Action**: Give everyone 50 resources  
**Effect**: Prevents economic collapse

### 🏥 WELFARE (Social Safety Net)
**Trigger**: Wealth inequality > 10:1 ratio  
**Action**: Give poorest 10% extra 100 resources  
**Effect**: Helps struggling tribes survive

### 🌟 SPAWN TRIBE (Immigration/Innovation)
**Trigger**: One tribe dominates >90%  
**Action**: Introduce 15 agents with new genetics  
**Effect**: Breaks stagnation, increases diversity

### 🚨 EMERGENCY REVIVAL (Crisis Response)
**Trigger**: Population < 5% of max  
**Action**: Massive resource boost + spawn 50 new agents  
**Effect**: Prevents extinction

## 📊 Results Location

All results saved to:
```
outputs/god_ai/
├── god_echo_results_20251031_123456.json
├── experiment_results_20251031_123456.json
└── god_comparison_20251031_123456.png
```

## 🔍 Understanding the Dashboard

### Color Legend:
- 🟢 **Green** = Healthy (high wealth, cooperation)
- 🟡 **Yellow** = Medium
- 🔴 **Red** = Struggling (low wealth, defectors)
- ⚪ **White dots** = Empty grid cells

### Key Metrics:
- **Population**: Current number of agents
- **Avg Wealth**: Average resources per agent
- **Cooperation**: % of cooperative moves
- **Clustering**: % of same-tribe neighbors
- **God Interventions**: Total times God has acted

## ⚙️ Customization

### Change Intervention Thresholds:

Edit these values in `prisoner_echo_god.py`:

```python
# At top of file
STAGNATION_THRESHOLD = 0.90      # Default: 90% dominance
LOW_WEALTH_THRESHOLD = 50        # Default: 50 resources
STIMULUS_AMOUNT = 50             # Default: 50 resources
WELFARE_AMOUNT = 100             # Default: 100 resources
GOD_INTERVENTION_COOLDOWN = 10   # Default: 10 generations
```

### Change Simulation Parameters:

```python
run_god_echo_simulation(
    generations=1000,        # Run longer
    initial_size=200,        # Start with more agents
    god_mode="RULE_BASED",   # or "DISABLED"
    update_frequency=20      # Update less often (faster)
)
```

## 🧪 Experiments to Try

### Experiment 1: God Dependency
**Question**: Do agents become dependent on God?

```python
# Run 1: God active for 500 generations
pop1 = run_god_echo_simulation(generations=500, god_mode="RULE_BASED")

# Run 2: Then disable God and see what happens
pop2 = run_god_echo_simulation(generations=500, god_mode="DISABLED")

# Compare cooperation rates - did they drop?
```

### Experiment 2: Intervention Frequency
**Question**: What's the optimal cooldown?

```python
# Try different cooldowns
for cooldown in [5, 10, 20, 50]:
    # Edit GOD_INTERVENTION_COOLDOWN in file
    results = run_god_echo_simulation(generations=500)
    print(f"Cooldown {cooldown}: {results.history['cooperation'][-1]}")
```

### Experiment 3: Stimulus vs Welfare
**Question**: Which is more effective?

```python
# Disable welfare, test stimulus only
# (Comment out WELFARE logic in _rule_based_decision)

# Then disable stimulus, test welfare only
# Compare outcomes
```

## 📈 Analyzing Results

### Load Results JSON:
```python
import json

with open('outputs/god_ai/god_echo_results_*.json', 'r') as f:
    data = json.load(f)

# Check final stats
print(data['final_stats'])

# Look at intervention history
for intervention in data['god_interventions']:
    print(f"Gen {intervention['generation']}: {intervention['intervention_type']}")
    print(f"  Reason: {intervention['reason']}")
```

### Plot Intervention Timeline:
```python
import matplotlib.pyplot as plt

generations = [i['generation'] for i in data['god_interventions']]
types = [i['intervention_type'] for i in data['god_interventions']]

plt.figure(figsize=(12, 6))
plt.scatter(generations, types, s=100, alpha=0.7)
plt.xlabel('Generation')
plt.ylabel('Intervention Type')
plt.title('God Intervention Timeline')
plt.show()
```

## 🐛 Troubleshooting

### "Population went extinct too fast"
- Increase `INITIAL_RESOURCES` (default: 100)
- Decrease `METABOLISM_COST` (default: 1)
- Lower intervention thresholds to trigger earlier

### "God never intervenes"
- Lower `GOD_INTERVENTION_COOLDOWN` (default: 10)
- Raise intervention thresholds to trigger more easily
- Check `god_mode` is not "DISABLED"

### "Simulation runs too slow"
- Increase `update_frequency` (update less often)
- Reduce `GRID_WIDTH` and `GRID_HEIGHT` (default: 50×50)
- Reduce `MAX_POPULATION` (default: 1000)

### "Out of memory"
- Reduce `generations` (default: 500)
- Reduce `MAX_POPULATION` (default: 1000)
- Reduce grid size (default: 50×50)

## 📚 Next Steps

### 1. **Run Comparative Experiment**
```bash
python compare_god_ai.py
```
This will take ~30 minutes but gives you full statistical analysis.

### 2. **Modify Intervention Logic**
Open `prisoner_echo_god.py` and edit `_rule_based_decision()` to test your own policies!

### 3. **Implement ML-Based God**
Follow `GOD_AI_README.md` section "ML-Based God (Todo #4)" to train a learning controller.

### 4. **Implement API-Based God**
Follow `GOD_AI_README.md` section "API-Based God (Todo #5)" to use GPT-4/Claude as governor.

### 5. **Write Research Paper**
Use the results from comparative experiments to write up your findings!

## 💡 Tips

- **Start Small**: Run 50-100 generations first to understand behavior
- **Compare Baselines**: Always run God vs No-God to see true impact
- **Check Logs**: Read `outputs/god_ai/*.json` for detailed intervention records
- **Visualize**: Use `compare_god_ai.py` for automatic plot generation
- **Experiment**: Change thresholds and see what happens!

## 🎓 Learning Path

1. ✅ **Day 1**: Run basic simulation, understand dashboard
2. ✅ **Day 2**: Run comparative experiment, analyze results
3. ✅ **Day 3**: Modify intervention thresholds, test effects
4. ✅ **Day 4**: Add new intervention type (see README)
5. ✅ **Week 2**: Implement ML-based God
6. ✅ **Week 3**: Implement API-based God
7. ✅ **Month 2**: Write and publish research paper

## 🚀 You're Ready!

Just run:
```bash
cd prisoner_dilemma_64gene
python prisoner_echo_god.py
```

Watch the God-AI reshape a complex adaptive system in real-time! 🧠👁️

---

**Questions?** Check `GOD_AI_README.md` for full documentation.

**Issues?** See troubleshooting section above.

**Ready for more?** Try the comparative experiment: `python compare_god_ai.py`
