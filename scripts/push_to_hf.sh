#!/bin/bash
# Push current state to HuggingFace Space as a clean squashed commit.
# Required because git history contains large PDFs that HF rejects.
# Usage: bash scripts/push_to_hf.sh

set -e

export PATH="$HOME/bin:$PATH"   # picks up ~/bin/git-lfs

HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2 | tr -d '[:space:]')
HF_REMOTE="https://azhar15c:${HF_TOKEN}@huggingface.co/spaces/azhar15c/AuditPilot"

echo "Creating clean orphan branch..."
git checkout --orphan hf-deploy
git add .
git commit -m "AuditPilot — deploy $(date '+%Y-%m-%d %H:%M')" --quiet

echo "Pushing to HuggingFace Space..."
git push --force "$HF_REMOTE" hf-deploy:main

echo "Cleaning up..."
git checkout main
git branch -D hf-deploy

echo "Done. Space will rebuild automatically."
