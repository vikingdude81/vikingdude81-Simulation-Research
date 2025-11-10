"""Quick test of Ultimate Echo simulation."""

from ultimate_echo_simulation import run_ultimate_echo
from government_styles import GovernmentStyle

# Run quick test
print("\n🧪 TESTING ULTIMATE ECHO SIMULATION")
print("="*60)

run_ultimate_echo(
    generations=50,
    initial_size=50,
    government_style=GovernmentStyle.LAISSEZ_FAIRE,
    update_frequency=10
)
