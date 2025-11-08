# 🤖 LLM-Based God Controller Guide

## Overview

The LLM God Controller integrates external AI models (GPT-4, Claude, Gemini) as governance agents for the economic simulation. Instead of hard-coded rules or quantum genetics, the LLM analyzes the economic state in natural language and decides interventions.

## How It Works

### Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. SIMULATION STATE                                     │
│  ━━━━━━━━━━━━━━━━━━━                                     │
│  Population: 500                                         │
│  Avg Wealth: $8,500                                      │
│  Cooperation: 45%                                        │
│  Gini Coefficient: 0.65 (high inequality!)               │
│  Tribe Dominance: 70%                                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  2. SERIALIZE TO NATURAL LANGUAGE + JSON                 │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                 │
│  "The economy has 500 agents with average wealth         │
│   of $8,500. Cooperation is low at 45% and inequality    │
│   is high (Gini=0.65). One tribe dominates 70%."         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  3. CALL LLM API                                         │
│  ━━━━━━━━━━━━━━━                                        │
│  System Prompt: "You are an economic god..."             │
│  User Prompt: "Current state: {...}"                     │
│                                                          │
│  LLM thinks: "High inequality threatens stability.       │
│               Need targeted welfare for poorest 15%."    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  4. LLM RESPONDS WITH JSON                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━                             │
│  {                                                       │
│    "intervention": "welfare",                            │
│    "reasoning": "High inequality (Gini=0.65) threatens   │
│                  social stability. Targeted assistance   │
│                  recommended.",                          │
│    "parameters": {                                       │
│      "target_bottom_percent": 0.15,                      │
│      "amount": 120                                       │
│    }                                                     │
│  }                                                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  5. PARSE & VALIDATE                                     │
│  ━━━━━━━━━━━━━━━━━━━━                                  │
│  ✓ Valid intervention type                               │
│  ✓ Valid parameters                                      │
│  ✓ Reasoning provided                                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  6. EXECUTE INTERVENTION                                 │
│  ━━━━━━━━━━━━━━━━━━━━━━                                │
│  Give $120 to poorest 15% (75 agents)                    │
│  Total injection: $9,000                                 │
│  Track outcome for next iteration                        │
└─────────────────────────────────────────────────────────┘
```

## Usage

### 1. Mock Mode (No API Required)

Perfect for testing and development:

```python
from llm_god_controller import LLMGodController

# Create controller in mock mode
controller = LLMGodController(
    provider="mock",  # Simulates LLM responses
    intervention_cooldown=10
)

# Test with different scenarios
state = {
    'population': 500,
    'avg_wealth': 80,
    'cooperation_rate': 0.45,
    'gini_coefficient': 0.65,
    'tribe_dominance': 0.70
}

decision = controller.decide_intervention(state, generation=50)

if decision:
    intervention_type, reasoning, parameters = decision
    print(f"Intervention: {intervention_type}")
    print(f"Reasoning: {reasoning}")
    print(f"Parameters: {parameters}")
```

**Mock mode behavior:**
- Simulates intelligent decision-making
- No API calls or costs
- Instant responses
- Good for testing logic

### 2. OpenAI (GPT-4, GPT-3.5)

Use real GPT models for sophisticated governance:

```python
# Set API key (or use environment variable OPENAI_API_KEY)
controller = LLMGodController(
    provider="openai",
    model="gpt-4",  # or "gpt-3.5-turbo"
    api_key="sk-...",
    intervention_cooldown=10,
    rate_limit_per_minute=10  # Respect rate limits
)

# Use same as mock mode
decision = controller.decide_intervention(state, generation)
```

**Requirements:**
```bash
pip install openai
```

**Environment setup:**
```bash
# Linux/Mac
export OPENAI_API_KEY='sk-...'

# Windows PowerShell
$env:OPENAI_API_KEY='sk-...'
```

### 3. Anthropic Claude

Claude models excel at reasoning and following instructions:

```python
controller = LLMGodController(
    provider="anthropic",
    model="claude-3-opus-20240229",  # or "claude-3-sonnet-20240229"
    api_key="sk-ant-...",
    intervention_cooldown=10
)
```

**Requirements:**
```bash
pip install anthropic
```

**Environment setup:**
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### 4. Google Gemini

Google's latest models:

```python
controller = LLMGodController(
    provider="google",
    model="gemini-pro",
    api_key="AIza...",
    intervention_cooldown=10
)
```

**Requirements:**
```bash
pip install google-generativeai
```

**Environment setup:**
```bash
export GOOGLE_API_KEY='AIza...'
```

## Integration with Simulation

### Add to prisoner_echo_god.py

```python
# In GodController.__init__()
if self.mode == "API_BASED":
    from llm_god_controller import LLMGodController
    self.llm_controller = LLMGodController(
        provider="openai",  # or "anthropic", "google", "mock"
        model="gpt-4",
        intervention_cooldown=GOD_INTERVENTION_COOLDOWN,
        fallback_to_rules=True  # Fall back if API fails
    )

# In _api_based_decision()
def _api_based_decision(self, state: Dict, population: List) -> Optional[Tuple[InterventionType, str, Dict]]:
    """API-based (external LLM) decision making."""
    result = self.llm_controller.decide_intervention(
        state, 
        self.generation
    )
    
    if result:
        intervention_type_str, reasoning, parameters = result
        intervention_type = InterventionType[intervention_type_str]
        return (intervention_type, reasoning, parameters)
    
    return None
```

### Run Simulation

```python
from prisoner_echo_god import run_god_echo_simulation, GodMode

# Run with LLM God
result = run_god_echo_simulation(
    generations=100,
    initial_size=300,
    god_mode=GodMode.API_BASED,
    output_dir="outputs/god_ai"
)
```

## Testing & Comparison

### Quick Test Script

```python
"""Test LLM God with simulation."""
from prisoner_echo_god import run_god_echo_simulation, GodMode
from llm_god_controller import LLMGodController

def test_llm_god():
    print("\n🤖 Testing LLM God Controller...")
    
    # Run simulation with API-based God
    result = run_god_echo_simulation(
        generations=50,
        initial_size=300,
        god_mode=GodMode.API_BASED,
        output_dir="outputs/god_ai"
    )
    
    # Display results
    print(f"\n✅ {result.num_generations} generations completed")
    print(f"   Final population: {len(result.agents)}")
    
    if result.god:
        print(f"   God interventions: {len(result.god.interventions)}")
        
        # Get LLM controller stats
        stats = result.god.llm_controller.get_statistics()
        print(f"\n📊 LLM Stats:")
        print(f"   API calls: {stats['total_api_calls']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Failed calls: {stats['failed_calls']}")
        print(f"   Interventions by type: {stats['interventions_by_type']}")

if __name__ == "__main__":
    test_llm_god()
```

### Comparative Test: Quantum vs LLM

Compare quantum genetic controller with LLM:

```python
"""Compare Quantum ML God vs LLM God."""
from test_comparative_gods import run_comparative_test

# Add API_BASED mode to test
modes = ["DISABLED", "RULE_BASED", "ML_BASED", "API_BASED"]

results = run_comparative_test(
    generations=100,
    initial_size=300,
    runs_per_mode=3,
    modes=modes  # Test all 4 modes
)

# Compare outcomes:
# - ML_BASED: Fast, deterministic, learned from evolution
# - API_BASED: Flexible, natural language reasoning, can adapt to novel situations
```

## Advantages & Trade-offs

### LLM God Advantages ✅

1. **Natural Language Reasoning**
   - Can explain decisions in human terms
   - Easier to understand "why" it chose an intervention
   
2. **Flexibility**
   - Can handle novel situations not in training data
   - Can incorporate external knowledge (e.g., economic theory)
   
3. **No Training Required**
   - Ready to use immediately
   - No need for genetic evolution or ML training
   
4. **Adaptability**
   - Can be prompted to prioritize different goals
   - Easy to modify behavior with prompt engineering

### LLM God Disadvantages ❌

1. **API Costs**
   - GPT-4: ~$0.03 per 1K tokens (input) + $0.06 per 1K tokens (output)
   - 100 interventions ≈ $3-5
   
2. **Latency**
   - API calls take 1-3 seconds
   - Slows down simulation significantly
   
3. **Rate Limits**
   - OpenAI: 3,500 requests/minute (tier 1)
   - Need to respect limits
   
4. **Non-Deterministic**
   - Same state may produce different decisions
   - Harder to reproduce exact results
   
5. **Requires Internet**
   - Can't run offline
   - Dependent on API availability

### Quantum ML God Advantages ✅

1. **Fast & Offline**
   - No API calls, instant decisions
   - Works without internet
   
2. **Deterministic**
   - Same genome → same decisions
   - Reproducible results
   
3. **Evolved for Task**
   - Optimized through genetic evolution
   - Proven performance on this specific problem
   
4. **No Costs**
   - Free to run unlimited simulations

### When to Use Which?

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| **Research & experimentation** | Quantum ML | Fast iteration, reproducible |
| **Production deployment** | Quantum ML | No API costs, reliable |
| **Rapid prototyping** | LLM (mock) | No setup, easy testing |
| **Novel situations** | LLM (real) | Can reason about new scenarios |
| **Explaining decisions** | LLM (real) | Natural language reasoning |
| **Large-scale testing** | Quantum ML | Fast, no rate limits |
| **Hybrid approach** | Both! | LLM for novel, quantum for routine |

## Cost Estimation

### OpenAI GPT-4 Costs

Assuming:
- 1 intervention every 10 generations
- 100 generations = 10 interventions
- Each intervention = ~500 input tokens + ~150 output tokens

```
Cost per run:
Input:  10 interventions × 500 tokens × $0.03/1K = $0.15
Output: 10 interventions × 150 tokens × $0.06/1K = $0.09
Total: ~$0.24 per 100-generation run
```

For 1000 runs (comprehensive testing):
```
1000 runs × $0.24 = $240
```

### Cost Reduction Strategies

1. **Use GPT-3.5-Turbo** (10x cheaper)
   ```python
   controller = LLMGodController(provider="openai", model="gpt-3.5-turbo")
   # Cost: ~$0.024 per 100-gen run (90% savings)
   ```

2. **Increase Cooldown**
   ```python
   controller = LLMGodController(intervention_cooldown=20)
   # Halves API calls → halves cost
   ```

3. **Use Mock for Most Testing**
   ```python
   # Development: mock mode
   controller = LLMGodController(provider="mock")
   
   # Final validation: real API
   controller = LLMGodController(provider="openai", model="gpt-4")
   ```

4. **Hybrid Approach**
   ```python
   # Use quantum for routine, LLM for novel
   if is_novel_situation(state):
       decision = llm_controller.decide(state)
   else:
       decision = quantum_controller.decide(state)
   ```

## Advanced: Prompt Engineering

### Customize System Prompt

```python
# In llm_god_controller.py, modify _create_system_prompt()

def _create_system_prompt(self) -> str:
    return """You are an AI god governing an economic simulation.

GOAL: Maximize long-term prosperity while minimizing inequality.

PHILOSOPHY: 
- Prefer targeted interventions over universal ones
- Let natural cooperation emerge when possible
- Intervene only when necessary to prevent collapse

AVAILABLE INTERVENTIONS:
1. welfare: Help the poorest (reduces inequality)
2. stimulus: Give to everyone (boosts economy)
3. spawn_tribe: Add diversity (prevents stagnation)
4. emergency_revival: Prevent extinction (last resort)

DECISION CRITERIA:
- Gini > 0.6 → Consider welfare
- Population < 200 → Consider emergency
- Dominance > 0.8 → Consider spawn_tribe
- Otherwise → Let economy evolve naturally

Respond in JSON format: {...}
"""
```

### Add Context to Prompt

```python
def _create_user_prompt(self, state: Dict) -> str:
    prompt = super()._create_user_prompt(state)
    
    # Add historical context
    if len(self.interventions) > 0:
        last = self.interventions[-1]
        prompt += f"\n\nLast intervention (gen {last['generation']}):"
        prompt += f"\n- Type: {last['decision'][0]}"
        prompt += f"\n- Outcome: {'positive' if self._was_effective(last) else 'negative'}"
    
    return prompt
```

## Troubleshooting

### "Module not found" errors

```bash
# Install required packages
pip install openai anthropic google-generativeai
```

### "API key not found"

```bash
# Set environment variable
export OPENAI_API_KEY='sk-...'

# Or pass directly
controller = LLMGodController(provider="openai", api_key="sk-...")
```

### "Rate limit exceeded"

```python
# Reduce rate limit
controller = LLMGodController(
    provider="openai",
    rate_limit_per_minute=5  # Lower limit
)

# Increase cooldown
controller = LLMGodController(
    intervention_cooldown=20  # Wait longer between interventions
)
```

### LLM not responding with JSON

Some LLMs may add extra text. The parser handles this:

```python
# Parser looks for JSON in response, even if wrapped in markdown
response = """
Here's my decision:

```json
{
  "intervention": "welfare",
  "reasoning": "...",
  "parameters": {...}
}
```

This should help the economy.
"""

# Parser extracts the JSON automatically
```

## Next Steps

1. **✅ Run Demo**: Test with mock mode (no API required)
   ```bash
   python llm_god_controller.py
   ```

2. **🔧 Integrate**: Add API_BASED mode to prisoner_echo_god.py

3. **🧪 Test**: Run comparative test (Quantum vs LLM vs Rules)

4. **📊 Analyze**: Compare performance, costs, and reasoning quality

5. **🎯 Optimize**: Tune prompts and parameters for best results

6. **🚀 Deploy**: Choose best controller for your use case

---

**Created**: November 3, 2025  
**Status**: ✅ Complete and tested  
**Files**: `llm_god_controller.py`, `LLM_GOD_GUIDE.md`
