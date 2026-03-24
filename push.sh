#!/data/data/com.termux/files/usr/bin/bash

# Universal auto-push script for Termux
# Works in any Git repository

set -e  # Stop on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Universal Git Auto-Push Script ===${NC}"

# 1. Ensure we're in a Git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a Git repository. Please run this script from inside a Git repo.${NC}"
    exit 1
fi

# 2. Check if remote 'origin' exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo -e "${YELLOW}No remote 'origin' found.${NC}"
    echo -n "Enter the GitHub repository URL (e.g., https://github.com/username/repo.git): "
    read -r repo_url
    if [ -z "$repo_url" ]; then
        echo -e "${RED}No URL provided. Exiting.${NC}"
        exit 1
    fi
    # Optionally embed a personal access token
    echo -n "Do you want to embed a personal access token to avoid password prompts? (y/n): "
    read -r embed_token
    if [[ "$embed_token" == "y" || "$embed_token" == "Y" ]]; then
        echo -n "Enter your personal access token (with repo scope): "
        read -rs token
        echo
        # Embed token into URL
        repo_url_with_token=$(echo "$repo_url" | sed "s#https://#https://${token}@#")
        git remote add origin "$repo_url_with_token"
        echo -e "${GREEN}Remote added with embedded token.${NC}"
    else
        git remote add origin "$repo_url"
        echo -e "${GREEN}Remote added without token. You will be prompted for credentials.${NC}"
    fi
else
    echo -e "${GREEN}Remote 'origin' already set: $(git remote get-url origin)${NC}"
fi

# 3. Stage all changes
echo -e "${GREEN}Staging all changes...${NC}"
git add .

# 4. Check if there is anything to commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit. Exiting.${NC}"
    exit 0
fi

# 5. Commit with timestamp
commit_msg="Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${GREEN}Committing: $commit_msg${NC}"
git commit -m "$commit_msg"

# 6. Push to current branch
current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
echo -e "${GREEN}Pushing to origin/$current_branch...${NC}"
git push origin "$current_branch"

echo -e "${GREEN}Push completed successfully!${NC}"
