"""
Direct ML Governance Training Launcher
Bypasses menu for automated training
"""

from ml_governance_gpu import train_governance_agent

if __name__ == "__main__":
    print("🚀 Starting ML Governance Training...")
    print("📊 100 episodes, 300 generations each")
    print("⏱️  Estimated time: ~30 minutes")
    print()
    
    # Run full training directly
    agent, history = train_governance_agent(episodes=100)
    
    print("\n✅ Training complete!")
    print(f"📈 Final performance:")
    print(f"   Avg Cooperation (last 10): {sum(history['episode_cooperation'][-10:])/10:.1f}%")
    print(f"   Avg Diversity (last 10): {sum(history['episode_diversity'][-10:])/10:.3f}")
    print(f"   Total rewards: {sum(history['episode_rewards']):.2f}")
