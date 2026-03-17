#!/bin/bash
# =============================================================================
# deploy/upload_code.sh
# =============================================================================
# Pushes the latest code to GitHub so the EC2 instance can git clone it.
#
# Usage:
#   bash deploy/upload_code.sh
#   bash deploy/upload_code.sh "your commit message"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GITHUB_REPO="https://github.com/VariPhiGen/crowd_management.git"
COMMIT_MSG="${1:-deploy: update pipeline code}"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Crimenabi — Push code to GitHub"
echo "============================================================"
echo "  Repo   : $GITHUB_REPO"
echo "  Message: $COMMIT_MSG"
echo ""

# Init git if not already a repo
if [ ! -d ".git" ]; then
    echo "  Initializing git repository..."
    git init
    git remote add origin "$GITHUB_REPO"
fi

# Make sure remote is correct
git remote set-url origin "$GITHUB_REPO" 2>/dev/null || true

git add -A
git commit -m "$COMMIT_MSG" || echo "  (nothing new to commit)"
git branch -M main
git push -u origin main

echo ""
echo "  ✅ Code pushed to $GITHUB_REPO"
echo ""
echo "  EC2 will run: git clone $GITHUB_REPO"
echo "============================================================"
