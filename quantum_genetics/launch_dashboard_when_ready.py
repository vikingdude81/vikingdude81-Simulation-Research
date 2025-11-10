"""
Auto-Launch Mega-Long Dashboard After Current Tests
Simple script that waits for ultra-long to finish, then launches dashboard
"""

import time
import subprocess
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("🎯 AUTO-LAUNCHER FOR MEGA-LONG DASHBOARD")
print("=" * 80)
print()
print("This script will:")
print("  1. Wait for ultra-long evolution to complete (~18:25)")
print("  2. Optionally restart island model if needed")
print("  3. Launch mega-long 1000-gen with beautiful dashboard")
print()
print("Current time:", datetime.now().strftime('%H:%M:%S'))
print()

# Check if ultra-long is still running
ultra_long_files = list(Path('.').glob('ultra_long_analysis_*.json'))
if ultra_long_files:
    latest = max(ultra_long_files, key=lambda p: p.stat().st_mtime)
    file_time = datetime.fromtimestamp(latest.stat().st_mtime)
    
    # If file is from today and recent, ultra-long might be done
    if file_time.date() == datetime.now().date():
        if file_time.hour == 18 and file_time.minute >= 20:
            print("✅ Ultra-long appears to be complete!")
            ultra_complete = True
        else:
            print("🔄 Ultra-long still running...")
            ultra_complete = False
    else:
        print("⏳ Waiting for ultra-long to start/complete...")
        ultra_complete = False
else:
    print("⏳ No ultra-long results found yet, waiting...")
    ultra_complete = False

if not ultra_complete:
    print()
    print("Monitoring for completion... (checking every 2 minutes)")
    print("Expected completion: ~18:25")
    print()
    
    while not ultra_complete:
        time.sleep(120)  # Check every 2 minutes
        
        ultra_long_files = list(Path('.').glob('ultra_long_analysis_*.json'))
        if ultra_long_files:
            latest = max(ultra_long_files, key=lambda p: p.stat().st_mtime)
            file_age = time.time() - latest.stat().st_mtime
            
            if file_age < 300:  # File created in last 5 minutes
                print(f"✅ Detected new file: {latest.name}")
                ultra_complete = True
                break
        
        print(f"⏳ Still waiting... ({datetime.now().strftime('%H:%M:%S')})")

print()
print("=" * 80)
print("✅ ULTRA-LONG COMPLETE!")
print("=" * 80)
print()

# Automatically run island model (no prompt)
print("🏝️  AUTO-LAUNCHING ISLAND MODEL EVOLUTION...")
print("Expected runtime: ~5 minutes")
print()
time.sleep(2)

subprocess.run(['python', 'run_island_evolution.py'])

print()
print("✅ Island model complete!")
print()
time.sleep(2)

print()
print("=" * 80)
print("🎨 NOW LAUNCHING MEGA-LONG WITH DASHBOARD...")
print("=" * 80)
print()

# Launch mega-long with dashboard
print()
print("=" * 80)
print("🎨 LAUNCHING MEGA-LONG 1000-GEN WITH LIVE DASHBOARD")
print("=" * 80)
print()
print("Expected runtime: ~45-50 minutes")
print("Dashboard will update every 0.5 seconds")
print()
print("Features:")
print("  🏆 Champion genome visualization")
print("  📊 Population statistics")
print("  📈 Fitness evolution sparklines")
print("  🌊 Diversity tracking")
print("  ⚡ Performance metrics")
print("  🚀 Innovation event tracker")
print()
print("Starting in 3 seconds...")
time.sleep(3)

# Run the dashboard experiment
subprocess.run(['python', 'run_mega_long_with_dashboard.py'])

print()
print("=" * 80)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("=" * 80)
print()
