#!/bin/bash

# Branch Migration Script
# This script creates and pushes the migrated branches to the remote repository

set -e  # Exit on error

echo "=== Branch Migration Script ==="
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Add source remote if it doesn't exist
if ! git remote | grep -q "^source$"; then
    echo "Adding source remote..."
    git remote add source https://github.com/vikingdude81/crypto-ml-trading-system.git
fi

echo "Fetching from source repository..."
git fetch source
echo ""

# Function to create and push a branch
create_and_push_branch() {
    local branch_name=$1
    local source_branch=$2
    
    echo "Processing branch: $branch_name"
    echo "----------------------------------------"
    
    # Check if branch already exists remotely
    if git ls-remote --heads origin "$branch_name" | grep -q "$branch_name"; then
        echo "⚠️  Branch $branch_name already exists remotely - skipping"
        return 0
    fi
    
    # Check if branch exists locally
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        echo "Branch exists locally, checking it out..."
        git checkout "$branch_name"
    else
        echo "Creating branch from source..."
        git checkout -b "$branch_name" "$source_branch"
        
        # Update references
        echo "Updating repository references..."
        find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.toml" -o -name "*.cfg" -o -name "*.ini" \) -not -path "./.git/*" -exec sed -i 's/crypto-ml-trading-system/vikingdude81-Simulation-Research/g' {} + 2>/dev/null || true
        
        # Commit if there are changes
        if ! git diff --quiet || ! git diff --staged --quiet; then
            git add -A
            git commit -m "Update repository references from crypto-ml-trading-system to vikingdude81-Simulation-Research"
            echo "✓ Committed reference updates"
        fi
    fi
    
    # Push the branch
    echo "Pushing branch to origin..."
    if git push -u origin "$branch_name"; then
        echo "✅ Successfully pushed $branch_name"
    else
        echo "❌ Failed to push $branch_name"
        return 1
    fi
    
    echo ""
}

# Migrate branches
create_and_push_branch "government-simulation-research" "source/government-simulation-research"
create_and_push_branch "copilot/integrate-multiscale-dynamics" "source/copilot/integrate-multiscale-dynamics"

# Return to original branch
echo "Returning to original branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH"

echo ""
echo "=== Branch Migration Complete ==="
echo ""

# Get remote URL dynamically
REMOTE_URL=$(git config --get remote.origin.url | sed 's/\.git$//')
if [ -n "$REMOTE_URL" ]; then
    echo "Migrated branches:"
    echo "  1. government-simulation-research"
    echo "  2. copilot/integrate-multiscale-dynamics"
    echo ""
    echo "Verify at: ${REMOTE_URL}/branches"
else
    echo "Check branches with: git branch -r"
fi
