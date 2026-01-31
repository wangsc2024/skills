---
name: git-workflow
description: |
  Manage Git workflows, branching strategies, commit conventions, and pull request processes. Provides guidance on Git best practices, conflict resolution, and team collaboration.
  Use when: setting up Git workflows, writing commit messages, creating branches, managing PRs, resolving merge conflicts, or when user mentions git, commit, branch, merge, PR, pull request, 分支, 合併, 提交.
  Triggers: "git", "commit", "branch", "merge", "PR", "pull request", "分支", "合併", "提交訊息", "conflict"
version: 1.0.0
---

# Git Workflow

Best practices for Git version control, branching strategies, and team collaboration.

## Branching Strategies

### GitHub Flow (Recommended for most projects)

```
main ─────●─────●─────●─────●─────●─────
           \         /     \       /
feature-a   ●───●───●       \     /
                             \   /
feature-b                     ●─●
```

```bash
# Create feature branch
git checkout -b feature/user-auth

# Work and commit
git add .
git commit -m "feat: add login form"

# Push and create PR
git push -u origin feature/user-auth
gh pr create

# After merge, clean up
git checkout main
git pull
git branch -d feature/user-auth
```

### Git Flow (For versioned releases)

```
main     ─────●─────────────────●───── (releases only)
              │                 │
hotfix        │     ●───●       │
              │    /     \      │
develop ─●───●───●───●───●───●──●───
          \     /         \   /
feature    ●───●           ●─●
```

### Trunk-Based Development (For CI/CD)

```
main ─────●─────●─────●─────●─────●─────
         / \   / \   / \
        ●   ● ●   ● ●   ●  (short-lived branches)
```

## Branch Naming Convention

```bash
# Format: type/description
feature/user-authentication
feature/JIRA-123-add-login
bugfix/fix-null-pointer
hotfix/critical-security-patch
refactor/cleanup-user-service
docs/update-readme
test/add-unit-tests

# Bad examples (avoid)
my-branch
fix
update
john-working
```

## Commit Message Convention

### Conventional Commits

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change, no feature/fix |
| `perf` | Performance improvement |
| `test` | Adding/fixing tests |
| `chore` | Build, CI, dependencies |
| `revert` | Revert previous commit |

### Examples

```bash
# Simple feature
git commit -m "feat: add user registration endpoint"

# With scope
git commit -m "feat(auth): implement JWT token refresh"

# With body
git commit -m "fix(api): handle null response from payment gateway

The payment gateway occasionally returns null instead of an error object.
Added null check and proper error handling.

Fixes #123"

# Breaking change
git commit -m "feat(api)!: change response format for /users endpoint

BREAKING CHANGE: Response now uses 'data' wrapper object.
Migration: Update all clients to access response.data instead of response."
```

### Commit Message Rules

```markdown
1. Subject line ≤ 50 characters
2. Use imperative mood ("add" not "added")
3. No period at end of subject
4. Separate subject from body with blank line
5. Body explains what and why, not how
6. Reference issues in footer
```

## Pull Request Template

```markdown
## Summary
[Brief description of changes]

## Type of Change
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 💥 Breaking change
- [ ] 📝 Documentation
- [ ] 🔧 Refactoring

## Changes Made
- [Change 1]
- [Change 2]
- [Change 3]

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
[Add screenshots]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Dependent changes merged

## Related Issues
Fixes #[issue number]
```

## Common Git Commands

### Daily Workflow

```bash
# Start new work
git checkout main
git pull origin main
git checkout -b feature/my-feature

# Save progress
git add .
git commit -m "feat: work in progress"

# Update with main
git fetch origin
git rebase origin/main

# Push changes
git push -u origin feature/my-feature
```

### Stashing

```bash
# Save uncommitted changes
git stash
git stash save "work in progress on feature X"

# List stashes
git stash list

# Apply and remove
git stash pop

# Apply and keep
git stash apply stash@{0}

# Drop stash
git stash drop stash@{0}
```

### Interactive Rebase

```bash
# Squash last 3 commits
git rebase -i HEAD~3

# In editor, change 'pick' to:
# - squash (s): Combine with previous
# - fixup (f): Combine, discard message
# - reword (r): Change commit message
# - drop (d): Remove commit
```

### Undo Operations

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo specific commit (create new commit)
git revert <commit-hash>

# Discard all local changes
git checkout -- .
git clean -fd

# Undo pushed commit (safe)
git revert <commit-hash>
git push
```

## Conflict Resolution

### Step-by-Step

```bash
# 1. Update your branch
git fetch origin
git rebase origin/main

# 2. If conflicts, Git will pause
# 3. Open conflicted files, look for:
<<<<<<< HEAD
your changes
=======
incoming changes
>>>>>>> main

# 4. Resolve by choosing or merging both
# 5. Mark resolved
git add <resolved-file>

# 6. Continue rebase
git rebase --continue

# Or abort if needed
git rebase --abort
```

### Conflict Prevention

```markdown
1. Pull/rebase frequently from main
2. Keep branches short-lived
3. Communicate with team about shared files
4. Break large changes into smaller PRs
```

## Git Hooks

### Pre-commit Hook

```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run linter
npm run lint
if [ $? -ne 0 ]; then
  echo "❌ Lint failed. Fix errors before committing."
  exit 1
fi

# Run tests
npm test
if [ $? -ne 0 ]; then
  echo "❌ Tests failed. Fix tests before committing."
  exit 1
fi
```

### Commit-msg Hook

```bash
#!/bin/sh
# .git/hooks/commit-msg

# Validate conventional commit format
commit_regex='^(feat|fix|docs|style|refactor|perf|test|chore|revert)(\(.+\))?(!)?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
  echo "❌ Invalid commit message format."
  echo "Expected: type(scope): subject"
  echo "Example: feat(auth): add login endpoint"
  exit 1
fi
```

## Team Practices

### Code Review Etiquette

```markdown
**For Authors:**
- Keep PRs small (<400 lines)
- Write clear descriptions
- Self-review before requesting
- Respond to feedback promptly

**For Reviewers:**
- Review within 24 hours
- Be constructive, not critical
- Approve when satisfied
- Use suggestions feature
```

### Protected Branch Rules

```yaml
main:
  - Require pull request reviews (1-2)
  - Require status checks to pass
  - Require linear history
  - No force push
  - No deletion

develop:
  - Require pull request reviews (1)
  - Require status checks to pass
```

## Checklist

### Before Creating PR
- [ ] Rebased on latest main
- [ ] All tests pass locally
- [ ] Linter shows no errors
- [ ] Commits are clean and logical
- [ ] PR description is complete

### Before Merging
- [ ] All CI checks pass
- [ ] Required reviews obtained
- [ ] No unresolved conversations
- [ ] Branch is up to date
- [ ] Squash commits if needed
