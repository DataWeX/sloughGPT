#!/bin/bash
# CI/CD Status Checker for SloughGPT
# Usage: ./check_ci_status.sh [commit_sha]

set -e

REPO="iamtowbee/sloughGPT"

echo "=========================================="
echo "Checking CI/CD Status for SloughGPT"
echo "=========================================="

# Get latest commit if not provided
if [ -z "$1" ]; then
    COMMIT=$(gh run list --repo $REPO --limit 1 --json headRefName,status,conclusion -q '.[0].headRefName' 2>/dev/null || echo "main")
    echo "Using latest commit on: $COMMIT"
else
    COMMIT=$1
    echo "Checking commit: $COMMIT"
fi

echo ""
echo "📊 Recent Workflow Runs:"
echo "-------------------------------------------"

# List recent workflow runs
gh run list --repo $REPO --limit 5 --json name,status,conclusion,createdAt \
    --jq '.[] | "\(.name) | \(.status) | \(.conclusion // "pending") | \(.createdAt[0:10])"'

echo ""
echo "=========================================="
echo "✅ CI/CD status check complete!"
echo "=========================================="
echo ""
echo "To view full details:"
echo "  gh run list --repo $REPO"
echo "  gh run view <run-id> --repo $REPO"
