# 🏛️ Enhanced Government Systems - Documentation

## Overview

We've expanded from **5 basic government types** to **12 comprehensive political-economic systems** with granular parameters that capture real ideological differences.

---

## New Government Types

### 🚩 **COMMUNIST**
**Philosophy**: State ownership, wealth equality, collective redistribution

**Parameters**:
- **Tax Rate**: 90% (near-total redistribution)
- **UBI**: 15 (strong universal basic income)
- **Enforcement**: "Jail" (re-educate defectors)
- **Severity**: 0.8 (severe)
- **Cooperator Bonus**: +10 (reward collective behavior)
- **Defector Penalty**: -15 (punish individualism)
- **Equality Enforcement**: YES (force wealth equality every generation)

**Key Features**:
- Aggressive wealth leveling (total wealth / num_agents = equal_share)
- Strong universal basic income from collective
- Harsh punishment of "individualist" defectors
- Rewards "collective" cooperators
- Anti-meritocratic (equality > performance)

**Expected Outcomes**:
- Very high equality (low Gini coefficient)
- Moderate cooperation (enforcement-based)
- Potential diversity loss (conformity pressure)

---

### ⚡ **FASCIST**
**Philosophy**: Corporatist authoritarianism, state-directed economy, nationalist purges

**Parameters**:
- **Tax Rate**: 40% (corporatist - state + industry partnership)
- **Welfare**: 12 (only for "insiders")
- **Enforcement**: "Execute" (extreme)
- **Severity**: 1.0 (maximum)
- **Frequency**: Every 2 generations (very frequent)
- **Cooperator Bonus**: +8 (reward "patriots")
- **Defector Penalty**: -30 (punish "traitors")
- **Nationalist Purge**: YES (remove "outsiders")

**Key Features**:
1. **Corporatist redistribution**: Welfare only for poor cooperators (insiders)
2. **Extreme enforcement**: Execute ALL defectors periodically
3. **Reward patriots**: Bonus for cooperators (nationalist loyalty)
4. **Nationalist purge**: Remove poorest 10% every few generations

**Expected Outcomes**:
- Very high cooperation (fear-based)
- Low diversity (genetic + behavioral conformity)
- Stable but oppressive system

---

### 🌟 **SOCIAL DEMOCRACY** (Nordic Model)
**Philosophy**: High tax, strong welfare, cooperation incentives

**Parameters**:
- **Tax Rate**: 50% (Nordic model)
- **Income Tax**: 40%
- **UBI**: 8 (universal basic income)
- **Welfare**: 20 (strong safety net)
- **Enforcement**: "Tax" (moderate)
- **Cooperator Bonus**: +5 (positive reinforcement)
- **Meritocratic**: YES (reward efficiency)

**Key Features**:
- Universal basic income (always)
- Progressive taxation with redistribution
- Bonus for cooperators (positive incentive)
- Moderate enforcement (tax defectors, don't kill them)

**Expected Outcomes**:
- High cooperation (carrot > stick)
- High diversity (softer approach)
- Moderate equality (balanced)

---

### 🦅 **LIBERTARIAN**
**Philosophy**: Pure non-intervention, absolute individual freedom

**Parameters**:
- **Tax Rate**: 0% (no taxation)
- **UBI**: 0
- **Welfare**: 0
- **Enforcement**: None
- **Stimulus**: Never

**Key Features**:
- ZERO government intervention
- Pure agent-driven evolution
- Market fundamentalism

**Expected Outcomes**:
- Variable cooperation (unstable)
- High diversity (no conformity pressure)
- High inequality (no redistribution)

---

### 🙏 **THEOCRACY**
**Philosophy**: Morality-based governance (cooperation = virtue, defection = sin)

**Parameters**:
- **Tax Rate**: 25% (tithe)
- **Welfare**: 15 (charity for faithful poor)
- **Enforcement**: "Jail" (punish sinners)
- **Severity**: 0.7 (harsh but fair)
- **Cooperator Bonus**: +12 (reward virtue)
- **Defector Penalty**: -18 (punish sin)

**Key Features**:
- Charity (help the faithful poor)
- Reward virtue (cooperators get bonus)
- Punish sin (defectors lose wealth, but not executed)
- Moral enforcement (less frequent than fascist, more severe than welfare)

**Expected Outcomes**:
- High cooperation (moral pressure)
- Moderate diversity (conformity to values)
- Moderate equality (charity-based)

---

### 💰 **OLIGARCHY**
**Philosophy**: Rule by wealthy elite, inverse welfare state

**Parameters**:
- **Tax Rate**: 5% (minimal tax on rich)
- **Welfare**: 2 (almost none for poor)
- **Enforcement**: None (rich aren't punished)
- **Stimulus**: Only for top 20% (bailouts)
- **Meritocratic**: YES (wealth = merit)

**Key Features**:
- Bailouts for rich during crisis
- Almost no welfare for poor
- Wealth concentration encouraged
- Meritocratic bonus (rich get richer)

**Expected Outcomes**:
- Low cooperation (no incentives)
- High diversity (no enforcement)
- Extreme inequality (Gini → 1.0)

---

### 🤖 **TECHNOCRACY**
**Philosophy**: Data-driven, algorithmic, optimal governance

**Parameters**:
- **Tax Rate**: 35% (optimal calculated)
- **UBI**: 10 (algorithmic)
- **Welfare**: 18 (targeted)
- **Enforcement**: "Tax" (rational, proportional)
- **Cooperator Bonus**: +7 (efficiency reward)
- **Defector Penalty**: -10 (disincentivize inefficiency)
- **Meritocratic**: YES (reward performance)

**Key Features**:
- Algorithmic UBI (always)
- Optimal redistribution (only if inequality > 5)
- Rational enforcement (proportional to defection rate)
- Efficiency bonuses (meritocratic cooperator rewards)

**Expected Outcomes**:
- High cooperation (data-optimized)
- High diversity (rational balance)
- Moderate equality (optimized)

---

## Granular Parameters by Type

### Tax Rates
```
Libertarian:       0%  ← Pure free market
Oligarchy:         5%  ← Minimal state
Laissez-Faire:    20%  ← Classical liberalism
Authoritarian:    20%  ← State power
Theocracy:        25%  ← Tithe-based
Mixed Economy:    25%  ← Pragmatic
Welfare State:    30%  ← Safety net
Technocracy:      35%  ← Optimal
Fascist:          40%  ← Corporatist
Social Democracy: 50%  ← Nordic model
Communist:        90%  ← Near-total redistribution
```

### Enforcement Severity (0.0 = none, 1.0 = total)
```
Libertarian:       0.0  ← No enforcement
Laissez-Faire:     0.0  ← No enforcement
Welfare State:     0.0  ← No enforcement
Oligarchy:         0.0  ← No enforcement
Central Banker:    0.0  ← No enforcement
Social Democracy:  0.3  ← Soft (tax defectors)
Mixed Economy:     0.4  ← Moderate
Technocracy:       0.5  ← Rational
Theocracy:         0.7  ← Harsh (moral)
Communist:         0.8  ← Severe (ideological)
Authoritarian:     0.9  ← Very severe
Fascist:           1.0  ← Total (execution)
```

### Universal Basic Income (UBI)
```
Libertarian:       0  ← None
Oligarchy:         0  ← None
Laissez-Faire:     0  ← None
Authoritarian:     0  ← None
Central Banker:    0  ← None
Theocracy:         5  ← Charity
Mixed Economy:     5  ← Minimal
Social Democracy:  8  ← Moderate
Technocracy:      10  ← Algorithmic
Communist:        15  ← Strong collective provision
```

---

## Key Comparisons

### Authoritarian Variants
Compare **Authoritarian**, **Fascist**, and **Communist**:
- All use harsh enforcement
- Fascist: Nationalist + corporatist + execution
- Communist: Egalitarian + collective + re-education
- Authoritarian: Pure enforcement (baseline)

### Redistributive Variants
Compare **Welfare State**, **Social Democracy**, and **Communist**:
- Tax rates: 30% → 50% → 90%
- Welfare State: Reactive (help poor when needed)
- Social Democracy: Proactive (UBI + cooperation bonus)
- Communist: Aggressive (forced wealth equality)

### Minimal Intervention Variants
Compare **Libertarian**, **Laissez-Faire**, and **Oligarchy**:
- Libertarian: ZERO intervention (pure theory)
- Laissez-Faire: Minimal (crisis-only)
- Oligarchy: Minimal for poor, bailouts for rich (inverse welfare)

---

## Research Questions

1. **Does enforcement level correlate with cooperation?**
   - Test: Libertarian (0.0) vs Fascist (1.0)
   
2. **Is there an optimal tax rate for cooperation + diversity + equality?**
   - Test: 0% → 10% → 20% → 30% → 50% → 90%
   
3. **Do positive incentives (bonuses) outperform negative (punishments)?**
   - Social Democracy (+5 cooperator bonus) vs Authoritarian (-20 defector penalty)
   
4. **Does wealth equality enforcement (Communist) harm diversity?**
   - Communist (forced equality) vs Social Democracy (incentivized equality)
   
5. **Can data-driven governance (Technocracy) outperform ideological systems?**
   - Technocracy vs all others

---

## How to Use

### Test Single Government
```python
from enhanced_government_styles import EnhancedGovernmentController, EnhancedGovernmentStyle

# Create controller
gov = EnhancedGovernmentController(EnhancedGovernmentStyle.COMMUNIST)

# Apply policy each generation
action = gov.apply_policy(agents, grid_size)

# Remove marked agents
agents = [a for a in agents if a.wealth > -9999]

# Get summary
summary = gov.get_summary()
```

### Compare All 12 Governments
```bash
cd prisoner_dilemma_64gene
python compare_all_governments.py
```

This will:
- Test all 12 government types
- Run 300 generations with 200 agents each
- Save results to JSON
- Print comparison table
- Identify best performers

---

## Expected Results

### Predictions

**Highest Cooperation**:
1. Fascist (1.0 severity, execution)
2. Authoritarian (0.9 severity, jail)
3. Communist (0.8 severity + equality)

**Highest Diversity**:
1. Libertarian (no conformity pressure)
2. Laissez-Faire (minimal intervention)
3. Social Democracy (soft approach)

**Highest Equality (Lowest Gini)**:
1. Communist (forced equality)
2. Social Democracy (50% tax + UBI)
3. Welfare State (30% tax + welfare)

**Best Overall Balance**:
- Probably **Technocracy** or **Social Democracy**
- Optimal mix of cooperation, diversity, and equality

---

## Files Created

1. **enhanced_government_styles.py** (1000+ lines)
   - `EnhancedGovernmentStyle` enum (12 types)
   - `GovernmentParameters` dataclass (granular settings)
   - `EnhancedGovernmentController` class (policy application)

2. **compare_all_governments.py** (~300 lines)
   - Test runner for all 12 government types
   - Comprehensive metrics tracking
   - Comparison tables and insights

3. **ENHANCED_GOVERNMENT_SYSTEMS.md** (this file)
   - Documentation for all government types
   - Parameter comparisons
   - Research questions

---

## Next Steps

1. ✅ **Run comparison**: `python compare_all_governments.py`
2. ⏳ **Wait for ML training** to complete (currently at Episode 90/100)
3. ⏳ **Run variable tax experiment**: `python test_variable_tax.py`
4. 📊 **Compare ML vs all human governments** (including new types)
5. 🔬 **Analyze which parameters** matter most for cooperation/diversity/equality
6. 📝 **Write research paper** with statistical significance tests

---

## Integration with ML Agent

The ML agent currently learns to choose from 7 actions with **fixed parameters**:
- Action 1: Welfare (30% tax) ← FIXED
- Action 2: Universal Stimulus (+10 wealth) ← FIXED
- Action 6: Boost Cooperators (+5 wealth) ← FIXED

**Future Enhancement**: Train ML agent with **parameterized actions**:
- Action 1: (welfare, tax_rate ∈ [0.1, 0.9])
- Action 2: (stimulus, amount ∈ [5, 20])
- Action 6: (boost, amount ∈ [3, 15])

Then compare:
- Fixed parameters (current)
- ML-optimized parameters (future)
- Human-designed parameters (12 government types)

**Research question**: Can ML discover better parameter values than human ideologies?

---

**Status**: System ready for comprehensive testing! 🚀
