#!/bin/bash

# Push Migrated Branches Script
# This script pushes the locally created migration branches to the remote repository

echo "=== Branch Migration - Push Script ==="
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

echo "Current branches:"
git branch -v
echo ""

# Push government-simulation-research branch
echo "Pushing government-simulation-research branch..."
if git show-ref --verify --quiet refs/heads/government-simulation-research; then
    git push origin government-simulation-research
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed government-simulation-research"
    else
        echo "❌ Failed to push government-simulation-research"
    fi
else
    echo "⚠️  Branch government-simulation-research not found locally"
fi

echo ""

# Push copilot/integrate-multiscale-dynamics branch
echo "Pushing copilot/integrate-multiscale-dynamics branch..."
if git show-ref --verify --quiet refs/heads/copilot/integrate-multiscale-dynamics; then
    git push origin copilot/integrate-multiscale-dynamics
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed copilot/integrate-multiscale-dynamics"
    else
        echo "❌ Failed to push copilot/integrate-multiscale-dynamics"
    fi
else
    echo "⚠️  Branch copilot/integrate-multiscale-dynamics not found locally"
fi

echo ""
echo "=== Push Complete ==="
echo ""
echo "Verify branches on GitHub:"
echo "https://github.com/vikingdude81/vikingdude81-Simulation-Research/branches"
