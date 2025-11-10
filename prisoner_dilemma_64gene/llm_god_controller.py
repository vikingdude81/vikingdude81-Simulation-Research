"""
🤖👁️ API-BASED GOD CONTROLLER (External LLM)

This module integrates external LLMs (GPT-4, Claude, Gemini) as the God controller
for the prisoner's dilemma simulation.

Features:
- Serialize economic state to natural language + JSON
- Call external LLM API with governance prompt
- Parse and validate LLM responses
- Execute interventions based on LLM decisions
- Rate limiting and fallback to rule-based controller
- Mock mode for testing without API calls
"""

import json
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from enum import Enum
import os

try:
    from prisoner_echo_god import InterventionType
except ImportError:
    class InterventionType(Enum):
        STIMULUS = "stimulus"
        WELFARE = "welfare"
        SPAWN_TRIBE = "spawn_tribe"
        EMERGENCY_REVIVAL = "emergency_revival"
        FORCED_COOPERATION = "forced_cooperation"


class LLMGodController:
    """
    God controller that uses external LLM for governance decisions.
    
    Supports multiple LLM providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Google (Gemini)
    - Mock (for testing)
    """
    
    def __init__(self, 
                 provider="mock",
                 model="gpt-4",
                 api_key=None,
                 intervention_cooldown=10,
                 rate_limit_per_minute=10,
                 fallback_to_rules=True,
                 prompt_style="conservative"):
        """
        Initialize LLM God controller.
        
        Args:
            provider: "openai", "anthropic", "google", or "mock"
            model: Model name (e.g., "gpt-4", "claude-3-opus", "gemini-pro")
            api_key: API key for the provider (or use environment variable)
            intervention_cooldown: Minimum generations between interventions
            rate_limit_per_minute: Max API calls per minute
            fallback_to_rules: Fall back to rule-based if API fails
            prompt_style: "conservative", "neutral", or "aggressive"
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.intervention_cooldown = intervention_cooldown
        self.rate_limit_per_minute = rate_limit_per_minute
        self.fallback_to_rules = fallback_to_rules
        self.prompt_style = prompt_style
        
        # State tracking
        self.interventions = []
        self.last_intervention_time = -intervention_cooldown
        self.api_call_times = []
        self.total_api_calls = 0
        self.failed_calls = 0
        
        # Initialize LLM client
        self._init_llm_client()
        
        print(f"\n🤖 LLM God Controller Initialized")
        print(f"   Provider: {provider}")
        print(f"   Model: {model}")
        print(f"   Mode: {'Mock (no API calls)' if provider == 'mock' else 'Live API'}")
        if fallback_to_rules:
            print(f"   Fallback: Rule-based controller enabled")
    
    def _init_llm_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                print("   ✅ OpenAI client initialized")
            except ImportError:
                print("   ⚠️  OpenAI package not installed. Run: pip install openai")
                self.provider = "mock"
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("   ✅ Anthropic client initialized")
            except ImportError:
                print("   ⚠️  Anthropic package not installed. Run: pip install anthropic")
                self.provider = "mock"
        
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                print("   ✅ Google GenAI client initialized")
            except ImportError:
                print("   ⚠️  Google GenAI package not installed. Run: pip install google-generativeai")
                self.provider = "mock"
        
        else:  # mock
            self.client = None
            print("   ℹ️  Mock mode - will simulate LLM responses")
    
    def decide_intervention(self, state: Dict, generation: int) -> Optional[Tuple[str, str, Dict]]:
        """
        Main decision method called by GodController.
        
        Args:
            state: Economic state dictionary
            generation: Current generation number
        
        Returns:
            Tuple of (intervention_type, reasoning, parameters) or None
        """
        # Check cooldown
        if generation - self.last_intervention_time < self.intervention_cooldown:
            return None
        
        # Check rate limit
        if not self._check_rate_limit():
            print("   ⚠️  Rate limit reached, skipping API call")
            if self.fallback_to_rules:
                return self._fallback_decision(state)
            return None
        
        # Add generation to state
        state['generation'] = generation
        
        # Call LLM for decision
        try:
            decision = self._call_llm(state)
            
            if decision:
                self.last_intervention_time = generation
                self.interventions.append({
                    'generation': generation,
                    'decision': decision,
                    'state': state.copy()
                })
                return decision
        
        except Exception as e:
            print(f"   ❌ LLM API error: {e}")
            self.failed_calls += 1
            
            if self.fallback_to_rules:
                print("   🔄 Falling back to rule-based controller")
                return self._fallback_decision(state)
        
        return None
    
    def _call_llm(self, state: Dict) -> Optional[Tuple[str, str, Dict]]:
        """
        Call the LLM API with the economic state.
        
        Returns:
            Tuple of (intervention_type, reasoning, parameters) or None
        """
        # Track API call
        self.total_api_calls += 1
        self.api_call_times.append(time.time())
        
        # Create prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(state)
        
        # Call appropriate provider
        if self.provider == "mock":
            response = self._mock_llm_response(state)
        elif self.provider == "openai":
            response = self._call_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            response = self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "google":
            response = self._call_google(system_prompt + "\n\n" + user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Parse response
        return self._parse_llm_response(response, state)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM based on prompt_style."""
        base_prompt = """You are an AI god governing an economic simulation based on the prisoner's dilemma.

Your role is to observe the simulation state and decide whether to intervene to improve outcomes.

Available interventions:
1. **stimulus**: Give resources to all agents (Universal Basic Income)
2. **welfare**: Give resources to the poorest agents (targeted assistance)
3. **spawn_tribe**: Introduce new agents with different genetics (diversity injection)
4. **emergency_revival**: Emergency intervention to prevent population collapse
5. **no_intervention**: Let the economy evolve naturally

For each intervention, you must specify parameters. For example:
- stimulus: {"amount_per_agent": 50}
- welfare: {"target_bottom_percent": 0.10, "amount": 100}
- spawn_tribe: {"invader_count": 15, "strategy": "tit_for_tat"}
- emergency_revival: {"resource_injection": 5000, "spawn_count": 20}

Respond in JSON format:
{
  "intervention": "welfare" | "stimulus" | "spawn_tribe" | "emergency_revival" | "no_intervention",
  "reasoning": "Explain your decision in 1-2 sentences",
  "parameters": {
    // Intervention-specific parameters
  }
}

Consider:
- Population size and sustainability
- Wealth distribution (Gini coefficient)
- Cooperation vs defection rates
- Tribe diversity and dominance
- Recent trends and events

"""
        
        # Add style-specific guidance
        if self.prompt_style == "conservative":
            style_guidance = "Intervene only when necessary. Sometimes the best action is no action. Let natural selection work unless there is a clear crisis."
        elif self.prompt_style == "neutral":
            style_guidance = "Analyze the situation and decide whether intervention would improve outcomes. Balance between intervention and natural evolution."
        elif self.prompt_style == "aggressive":
            style_guidance = "Intervene frequently to optimize outcomes. Active governance tends to improve cooperation and wealth distribution. Don't hesitate to act."
        else:
            style_guidance = "Intervene only when necessary. Sometimes the best action is no action."
        
        return base_prompt + style_guidance
    
    def _create_user_prompt(self, state: Dict) -> str:
        """Create the user prompt with current state."""
        # Format state for LLM
        prompt = f"""Current Simulation State (Generation {state.get('generation', 0)}):

📊 Population Metrics:
- Population: {state.get('population', 0)} agents
- Average Wealth: ${state.get('avg_wealth', 0):.0f}
- Total Wealth: ${state.get('total_wealth', 0):.0f}

🤝 Cooperation Metrics:
- Cooperation Rate: {state.get('cooperation_rate', 0):.1%}
- Clustering: {state.get('clustering', 0):.1%}

📈 Inequality Metrics:
- Gini Coefficient: {state.get('gini_coefficient', 0):.3f} (0=equal, 1=unequal)
- Wealth Inequality Ratio: {state.get('wealth_inequality', 0):.1f}:1

🏛️ Tribe Dynamics:
- Tribe Diversity: {state.get('tribe_diversity', 0):.1%}
- Max Tribe Dominance: {state.get('tribe_dominance', 0):.1%}

"""
        
        # Add recent trends if available
        if 'growth_rate' in state:
            trend = "growing" if state['growth_rate'] > 0 else "declining"
            prompt += f"📉 Trend: Wealth {trend} at {state['growth_rate']:.1%} per generation\n"
        
        prompt += "\nWhat intervention (if any) should be made? Respond in JSON format."
        
        return prompt
    
    def _mock_llm_response(self, state: Dict) -> str:
        """Generate a mock LLM response for testing without API calls."""
        # Simulate intelligent decision-making
        pop = state.get('population', 0)
        wealth = state.get('avg_wealth', 0)
        coop = state.get('cooperation_rate', 0)
        gini = state.get('gini_coefficient', 0)
        dominance = state.get('tribe_dominance', 0)
        
        # Decision logic (simulating what an LLM might decide)
        if pop < 100:
            return json.dumps({
                "intervention": "emergency_revival",
                "reasoning": "Population critically low, immediate revival needed to prevent extinction.",
                "parameters": {
                    "resource_injection": 5000,
                    "spawn_count": 25
                }
            })
        
        elif gini > 0.6:
            return json.dumps({
                "intervention": "welfare",
                "reasoning": f"High inequality (Gini={gini:.2f}) threatens social stability. Targeted welfare for bottom 15% recommended.",
                "parameters": {
                    "target_bottom_percent": 0.15,
                    "amount": 120
                }
            })
        
        elif dominance > 0.85:
            return json.dumps({
                "intervention": "spawn_tribe",
                "reasoning": f"Single tribe dominates {dominance:.0%}. Injecting genetic diversity to prevent stagnation.",
                "parameters": {
                    "invader_count": 18,
                    "strategy": "tit_for_tat"
                }
            })
        
        elif wealth < 50 and coop < 0.4:
            return json.dumps({
                "intervention": "stimulus",
                "reasoning": "Low wealth and cooperation create downward spiral. Universal stimulus to restart economy.",
                "parameters": {
                    "amount_per_agent": 60
                }
            })
        
        else:
            return json.dumps({
                "intervention": "no_intervention",
                "reasoning": "Economy is stable. Natural evolution is producing good outcomes.",
                "parameters": {}
            })
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        # Only use response_format for models that support it (gpt-4-turbo, gpt-3.5-turbo)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7
        }
        
        # Add JSON mode for supported models
        if "gpt-3.5" in self.model or "gpt-4-turbo" in self.model or "gpt-4o" in self.model:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text
    
    def _call_google(self, full_prompt: str) -> str:
        """Call Google Gemini API."""
        response = self.client.generate_content(full_prompt)
        return response.text
    
    def _parse_llm_response(self, response: str, state: Dict) -> Optional[Tuple[str, str, Dict]]:
        """Parse and validate the LLM response."""
        try:
            # Try to find JSON in response (in case LLM added text)
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response)
            
            intervention = data.get('intervention', 'no_intervention')
            reasoning = data.get('reasoning', 'No reasoning provided')
            parameters = data.get('parameters', {})
            
            # Validate intervention type
            if intervention == 'no_intervention':
                return None
            
            # Convert to uppercase for enum
            intervention_type = intervention.upper()
            
            # Add LLM context to reasoning
            reasoning = f"🤖 LLM GOD DECISION ({self.provider}/{self.model})\n   {reasoning}"
            
            return (intervention_type, reasoning, parameters)
        
        except json.JSONDecodeError as e:
            print(f"   ⚠️  Failed to parse LLM response as JSON: {e}")
            print(f"   Response: {response[:200]}...")
            return None
        
        except Exception as e:
            print(f"   ⚠️  Error parsing LLM response: {e}")
            return None
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        # Remove calls older than 1 minute
        self.api_call_times = [t for t in self.api_call_times if now - t < 60]
        return len(self.api_call_times) < self.rate_limit_per_minute
    
    def _fallback_decision(self, state: Dict) -> Optional[Tuple[str, str, Dict]]:
        """Fallback to rule-based decision if LLM fails."""
        # Simple rule-based logic
        pop = state.get('population', 0)
        wealth = state.get('avg_wealth', 0)
        gini = state.get('gini_coefficient', 0)
        
        if pop < 100:
            return ("EMERGENCY_REVIVAL", "🔄 Fallback: Population collapse", 
                   {"resource_injection": 5000, "spawn_count": 20})
        
        elif gini > 0.5:
            return ("WELFARE", "🔄 Fallback: High inequality",
                   {"target_bottom_percent": 0.10, "amount": 100})
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get controller statistics."""
        return {
            'total_interventions': len(self.interventions),
            'total_api_calls': self.total_api_calls,
            'failed_calls': self.failed_calls,
            'success_rate': (self.total_api_calls - self.failed_calls) / max(1, self.total_api_calls),
            'provider': self.provider,
            'model': self.model,
            'interventions_by_type': self._count_interventions_by_type()
        }
    
    def _count_interventions_by_type(self) -> Dict:
        """Count interventions by type."""
        counts = {}
        for intervention in self.interventions:
            itype = intervention['decision'][0] if intervention['decision'] else 'none'
            counts[itype] = counts.get(itype, 0) + 1
        return counts


# === TESTING AND DEMO ===

def demo_llm_god():
    """Demonstrate LLM God controller with mock mode."""
    print("\n" + "="*70)
    print("🤖 LLM GOD CONTROLLER DEMO (Mock Mode)")
    print("="*70)
    
    # Create controller in mock mode
    controller = LLMGodController(
        provider="mock",
        intervention_cooldown=5
    )
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Crisis: Low Population',
            'state': {
                'population': 75,
                'avg_wealth': 50,
                'cooperation_rate': 0.30,
                'gini_coefficient': 0.55,
                'tribe_dominance': 0.60,
                'generation': 0
            }
        },
        {
            'name': 'High Inequality',
            'state': {
                'population': 800,
                'avg_wealth': 120,
                'cooperation_rate': 0.55,
                'gini_coefficient': 0.75,
                'tribe_dominance': 0.50,
                'generation': 10
            }
        },
        {
            'name': 'Stagnation: High Dominance',
            'state': {
                'population': 900,
                'avg_wealth': 95,
                'cooperation_rate': 0.60,
                'gini_coefficient': 0.35,
                'tribe_dominance': 0.92,
                'generation': 20
            }
        },
        {
            'name': 'Economic Collapse',
            'state': {
                'population': 600,
                'avg_wealth': 35,
                'cooperation_rate': 0.25,
                'gini_coefficient': 0.60,
                'tribe_dominance': 0.55,
                'generation': 30
            }
        },
        {
            'name': 'Healthy Economy',
            'state': {
                'population': 950,
                'avg_wealth': 150,
                'cooperation_rate': 0.70,
                'gini_coefficient': 0.30,
                'tribe_dominance': 0.55,
                'generation': 40
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Population: {scenario['state']['population']}")
        print(f"   Wealth: ${scenario['state']['avg_wealth']}")
        print(f"   Cooperation: {scenario['state']['cooperation_rate']:.1%}")
        print(f"   Gini: {scenario['state']['gini_coefficient']:.2f}")
        
        decision = controller.decide_intervention(scenario['state'], scenario['state']['generation'])
        
        if decision:
            itype, reasoning, params = decision
            print(f"\n   🎯 Decision: {itype}")
            print(f"   {reasoning}")
            print(f"   Parameters: {params}")
        else:
            print("\n   ✅ No intervention needed")
    
    # Show statistics
    stats = controller.get_statistics()
    print(f"\n📈 Controller Statistics:")
    print(f"   Total API calls: {stats['total_api_calls']}")
    print(f"   Total interventions: {stats['total_interventions']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Interventions by type: {stats['interventions_by_type']}")
    
    print("\n✅ Demo complete!")


def test_with_openai():
    """Test with real OpenAI API (requires API key)."""
    print("\n" + "="*70)
    print("🤖 TESTING WITH REAL OPENAI API")
    print("="*70)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not found in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or pass it directly: controller = LLMGodController(provider='openai', api_key='...')")
        return
    
    controller = LLMGodController(
        provider="openai",
        model="gpt-4",
        api_key=api_key,
        intervention_cooldown=0  # No cooldown for testing
    )
    
    # Test with one scenario
    state = {
        'population': 500,
        'avg_wealth': 80,
        'cooperation_rate': 0.45,
        'gini_coefficient': 0.65,
        'tribe_dominance': 0.70,
        'generation': 50
    }
    
    print("\n📊 Testing with high inequality scenario...")
    print(f"   Population: {state['population']}")
    print(f"   Gini: {state['gini_coefficient']}")
    
    decision = controller.decide_intervention(state, 50)
    
    if decision:
        itype, reasoning, params = decision
        print(f"\n   🎯 GPT-4 Decision: {itype}")
        print(f"   {reasoning}")
        print(f"   Parameters: {params}")
    
    stats = controller.get_statistics()
    print(f"\n📈 Stats: {stats['total_api_calls']} calls, {stats['total_interventions']} interventions")


if __name__ == "__main__":
    # Run demo with mock
    demo_llm_god()
    
    # Uncomment to test with real OpenAI API:
    # test_with_openai()
