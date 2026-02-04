# Releases

Version history for this repository (5 releases).

## v2.1.1: v2.1.1 - Fix Plugin Template Paths
**Published:** 2026-01-10

## Fixed

- **Plugin Template Path Issue** (Fixes #15)
  - Templates weren't found when installed via plugin marketplace
  - Plugin cache expected templates at repo root
  - Added planning-with-files/ folder at root level for plugin installs
  - Kept skills/planning-with-files/ for legacy ~/.claude/skills/ installs

## Structure

| Folder | Purpose |
|--------|---------|
| planning-with-files/ | For plugin marketplace installs |
| skills/planning-with-files/ | For manual ~/.claude/skills/ installs |

## Upgrade

```
/plugin update planning-with-files
```

Or reinstall:
```
/plugin uninstall planning-with-files
/plugin marketplace add OthmanAdi/planning-with-files
/plugin install planning-with-files@planning-with-files
```

**Full Changelog**: https://github.com/OthmanAdi/planning-with-files/compare/v2.1.0...v2.1.1


[View on GitHub](https://github.com/OthmanAdi/planning-with-files/releases/tag/v2.1.1)

---

## v2.1.0: v2.1.0 - Claude Code v2.1 Compatibility
**Published:** 2026-01-10

## What's New

### Claude Code v2.1 Compatibility
- **SessionStart hook** - Displays ready message when Claude Code session begins
- **PostToolUse hook** - Reminds to update task_plan.md after Write/Edit operations
- **user-invocable: true** - Skill now appears in slash command menu (`/planning-with-files`)
- **YAML list format** - Updated allowed-tools to new YAML list syntax

### Documentation Restructure
- New `docs/` folder with organized documentation:
  - `installation.md` - All installation methods
  - `quickstart.md` - 5-step usage guide
  - `workflow.md` - Visual workflow diagram
  - `troubleshooting.md` - Common issues and solutions
  - `cursor.md` - Cursor IDE setup
  - `windows.md` - Windows-specific notes
- Shortened README with links to detailed docs

### Install
```bash
/plugin marketplace add OthmanAdi/planning-with-files
/plugin install planning-with-files@planning-with-files
```

**Full Changelog**: https://github.com/OthmanAdi/planning-with-files/compare/v2.0.1...v2.1.0

[View on GitHub](https://github.com/OthmanAdi/planning-with-files/releases/tag/v2.1.0)

---

## v2.0.1: v2.0.1 - Working Directory Fix
**Published:** 2026-01-09

## What's Changed

### Fixed: Working Directory Clarity

Planning files (`task_plan.md`, `findings.md`, `progress.md`) should be created in your project directory, not in the skill installation folder. The skill instructions now make this explicit.

**Before:** Instructions said "Create task_plan.md" without specifying where.

**After:** Instructions clearly state that planning files go in your project directory, with templates referenced from the skill folder.

### Added: Troubleshooting Section in README

New troubleshooting section covers common issues:
- Planning files created in wrong directory
- Files not persisting between sessions
- Hooks not triggering

### Documentation

- Added "Important: Where Files Go" section to SKILL.md
- Updated Quick Start with clearer file location guidance
- Added "Create files in your project" to Anti-Patterns table

---

## Upgrade

Re-install to get the latest version:

```bash
# One-line installer
curl -L https://github.com/OthmanAdi/planning-with-files/archive/master.tar.gz | tar -xzv --strip-components=2 "planning-with-files-master/skills/planning-with-files"

# Move to skills directory
mv planning-with-files ~/.claude/skills/
```

Or if you cloned the repo:
```bash
git pull origin master
```

---

## Thanks

Thanks to [@wqh17101](https://github.com/wqh17101) for reporting this issue and confirming the fix.

---

**Full Changelog:** [v2.0.0...v2.0.1](https://github.com/OthmanAdi/planning-with-files/compare/v2.0.0...v2.0.1)


[View on GitHub](https://github.com/OthmanAdi/planning-with-files/releases/tag/v2.0.1)

---

## v2.0.0: v2.0.0 - Hooks, Templates & Scripts
**Published:** 2026-01-08

## v2.0.0 - Major Update

### New Features

**Hooks Integration** (Claude Code 2.1.0+)
- `PreToolUse` hook: Auto-reads task_plan.md before Write/Edit/Bash
- `Stop` hook: Verifies all phases complete before stopping

**Templates**
- `templates/task_plan.md` - Structured phase tracking
- `templates/findings.md` - Research storage
- `templates/progress.md` - Session logging with test results

**Scripts**
- `scripts/init-session.sh` - Initialize all planning files
- `scripts/check-complete.sh` - Verify task completion

**Enhanced Documentation**
- The 2-Action Rule
- The 3-Strike Error Protocol
- Read vs Write Decision Matrix
- The 5-Question Reboot Test

### Install
```
/plugin marketplace add OthmanAdi/planning-with-files
/plugin install planning-with-files@planning-with-files
```

### Migration
See [MIGRATION.md](https://github.com/OthmanAdi/planning-with-files/blob/master/MIGRATION.md) for upgrade guide.

v1.0.0 users: your existing workflow still works. The `legacy` branch preserves v1.0.0.

[View on GitHub](https://github.com/OthmanAdi/planning-with-files/releases/tag/v2.0.0)

---

## v1.0.0: v1.0.0 - Initial Release
**Published:** 2026-01-08

## Initial Release

The core Manus-style planning pattern for Claude Code.

### Features
- 3-file pattern: task_plan.md, notes.md, deliverable.md
- SKILL.md with core workflow instructions
- reference.md with 6 Manus principles
- examples.md with real-world usage patterns

### Install
```bash
git clone -b legacy https://github.com/OthmanAdi/planning-with-files.git ~/.claude/skills/planning-with-files
```

For v2.0.0 with hooks and templates, use the `master` branch.

[View on GitHub](https://github.com/OthmanAdi/planning-with-files/releases/tag/v1.0.0)

---

