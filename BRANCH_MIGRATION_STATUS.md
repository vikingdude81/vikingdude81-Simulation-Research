# Branch Migration Status

## Overview
This PR prepares the migration of two branches from `vikingdude81/crypto-ml-trading-system` to `vikingdude81/vikingdude81-Simulation-Research`.

## Branches to Migrate
1. `government-simulation-research`
2. `copilot/integrate-multiscale-dynamics`

## Current Status

### ✅ Completed Steps
1. **Source repository added as remote** - The crypto-ml-trading-system repository has been added as a remote named `source`
2. **Branches fetched** - Both target branches have been successfully fetched from the source repository
3. **Local branches created** - Both branches have been created locally with all files preserved
4. **References updated** - All references to "crypto-ml-trading-system" have been updated to "vikingdude81-Simulation-Research" in both branches
5. **GitHub Actions workflow created** - A workflow has been added to automate pushing the branches

### 📋 How to Complete Migration

After this PR is merged, the branches can be pushed using one of these methods:

#### Option 1: GitHub Actions Workflow (Recommended)
1. Go to the Actions tab in the repository
2. Select the "Push Migrated Branches" workflow
3. Click "Run workflow"
4. The workflow will create and push both branches automatically

#### Option 2: Manual Push via Script
Run the existing `push_branches.sh` script:
```bash
./push_branches.sh
```

#### Option 3: Manual Git Commands
```bash
# Fetch the source repository
git remote add source https://github.com/vikingdude81/crypto-ml-trading-system.git
git fetch source

# Push government-simulation-research
git checkout -b government-simulation-research source/government-simulation-research
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" \) -not -path "./.git/*" -exec sed -i 's/crypto-ml-trading-system/vikingdude81-Simulation-Research/g' {} +
git add -A
git commit -m "Update repository references"
git push origin government-simulation-research

# Push copilot/integrate-multiscale-dynamics
git checkout -b copilot/integrate-multiscale-dynamics source/copilot/integrate-multiscale-dynamics
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" \) -not -path "./.git/*" -exec sed -i 's/crypto-ml-trading-system/vikingdude81-Simulation-Research/g' {} +
git add -A
git commit -m "Update repository references"
git push origin copilot/integrate-multiscale-dynamics
```

## Branch Contents

### government-simulation-research
- **Purpose**: Agent-based modeling for government systems, policy dynamics, and institutional behavior
- **Key Files**: 
  - `government_simulation/` - Core simulation module
  - `prisoner_dilemma_64gene/` - Prisoner's dilemma research
  - Various research documentation files

### copilot/integrate-multiscale-dynamics
- **Purpose**: Integration of multiscale dynamics into the trading system
- **Key Files**:
  - Extended ML pipeline components
  - Fourier integration analysis
  - Multiscale SNN integration

## Verification
After pushing, verify the branches exist at:
https://github.com/vikingdude81/vikingdude81-Simulation-Research/branches

Both branches should appear with all files from the source repository and updated references.
