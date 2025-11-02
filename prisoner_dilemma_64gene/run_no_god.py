"""Quick script to run NO-GOD baseline simulation for comparison."""

from prisoner_echo_god import run_god_echo_simulation

# Run with God DISABLED
run_god_echo_simulation(
    generations=500,
    initial_size=100,
    god_mode="DISABLED",
    update_frequency=10
)
