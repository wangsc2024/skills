---
name: skill-creator
description: |
  Create and manage Claude Code Skills. Generates SKILL.md files, validates skill structure, and provides templates for building custom skills.
  Use when: creating new skills, setting up skill directories, troubleshooting skill configuration, or when user mentions 建立skill, create skill, SKILL.md, 新增技能, skill模板.
  Triggers: "create skill", "建立skill", "SKILL.md", "新增技能", "skill模板", "自訂skill"
---

# Skill Creator

A meta-skill for creating, validating, and managing Claude Code Skills.

## Quick Start

### Create a New Skill

```bash
# Project-level skill (shared via git)
mkdir -p .claude/skills/<skill-name>

# Personal skill (available across all projects)
mkdir -p ~/.claude/skills/<skill-name>
```

## SKILL.md Template

```yaml
---
name: your-skill-name
description: Clear description of what this skill does. Explain when to use it and what triggers it. Include keywords for auto-discovery.
---

# Your Skill Name

## Instructions

[Step-by-step guidance for Claude to follow]

## Examples

[Concrete usage examples]

## Reference

[Links to additional documentation if needed]
```

## Validation Checklist

### Frontmatter Requirements

| Field | Requirement |
|-------|-------------|
| `name` | ≤64 chars, lowercase, alphanumeric + hyphens only |
| `description` | ≤1024 chars, must include "what" and "when" |
| `allowed-tools` | Optional: comma-separated list of tools |

### Common Validation Errors

1. **Name too long**: Keep under 64 characters
2. **Invalid characters**: Use only `a-z`, `0-9`, `-`
3. **Missing description**: Must be non-empty
4. **Reserved words**: Cannot contain "anthropic" or "claude"

## Directory Structure

### Simple Skill

```
.claude/skills/my-skill/
└── SKILL.md
```

### Complex Skill

```
.claude/skills/my-skill/
├── SKILL.md              # Main instructions (required)
├── REFERENCE.md          # API documentation
├── EXAMPLES.md           # Usage examples
├── TROUBLESHOOTING.md    # Common issues
├── scripts/              # Executable scripts
│   ├── setup.py
│   └── validate.py
└── templates/            # Template files
    └── component.txt
```

## Writing Effective Descriptions

### Good Examples

```yaml
# Specific with trigger words
description: Generate algorithmic art using p5.js and SVG. Creates flow fields, fractals, and generative patterns. Use when creating visual art, generative designs, or creative coding projects.

# Clear action and context
description: Review code for security vulnerabilities, performance issues, and best practices. Use when reviewing pull requests, auditing code, or checking for common security flaws.
```

### Bad Examples

```yaml
# Too vague
description: Helps with code

# Missing "when to use"
description: Creates components

# Too long and rambling
description: This skill is designed to help users who want to...
```

## Skill Categories

### Analysis Skills (Read-Only)

```yaml
---
name: code-analyzer
description: Analyze code quality and patterns
allowed-tools: Read, Grep, Glob
---
```

### Generation Skills

```yaml
---
name: component-generator
description: Generate code components and boilerplate
allowed-tools: Read, Write, Bash
---
```

### Workflow Skills

```yaml
---
name: deployment-helper
description: Guide through deployment processes
allowed-tools: Read, Bash, WebFetch
---
```

## Tool Restrictions

Use `allowed-tools` to limit what Claude can do:

| Use Case | Tools |
|----------|-------|
| Code review | `Read, Grep, Glob` |
| Documentation | `Read, Write` |
| Git operations | `Read, Bash` |
| Full access | (omit field) |

## Best Practices

### Content Guidelines

1. **Keep SKILL.md < 500 lines** for token efficiency
2. **Use progressive disclosure**: Main instructions in SKILL.md, details in separate files
3. **Write in third person**: "Generates components" not "I generate components"
4. **Include concrete examples**: Real code, not abstract descriptions
5. **List required dependencies**: npm packages, Python modules, etc.

### File Organization

1. **One concept per file**: Don't mix unrelated instructions
2. **Shallow references**: Only reference files one level deep
3. **Executable scripts**: Put in `scripts/` directory
4. **Templates**: Put in `templates/` directory

## Debugging Skills

### Skill Not Triggering?

1. Check description contains relevant keywords
2. Verify YAML frontmatter syntax:
   ```bash
   head -n 10 ~/.claude/skills/my-skill/SKILL.md
   ```
3. Ensure file is named exactly `SKILL.md`
4. Check directory location is correct

### Test Your Skill

Ask Claude questions that should trigger your skill:
- "Can you help me with [keyword from description]?"
- "I need to [action mentioned in description]"

## Skill vs Slash Command

| Feature | Skill | Slash Command |
|---------|-------|---------------|
| Invocation | Automatic | Manual (`/command`) |
| Location | `.claude/skills/` | `.claude/commands/` |
| Structure | Directory + SKILL.md | Single .md file |
| Complexity | Multi-file workflows | Simple prompts |

## Workflow for Creating a Skill

1. **Define purpose**: What problem does it solve?
2. **Identify triggers**: What keywords/contexts should activate it?
3. **Create directory**: `mkdir -p ~/.claude/skills/<name>`
4. **Write SKILL.md**: Start with template above
5. **Add resources**: Reference docs, scripts, templates
6. **Test thoroughly**: Try various prompts
7. **Iterate**: Refine based on behavior

## Example: Creating a Testing Skill

```yaml
---
name: test-writer
description: Generate unit tests for JavaScript and TypeScript code. Creates Jest or Vitest test files with proper mocking and assertions. Use when writing tests, adding test coverage, or setting up testing infrastructure.
---

# Test Writer

## Instructions

1. Analyze the source file to understand functions and classes
2. Identify testable units and edge cases
3. Generate test file with proper imports
4. Include positive and negative test cases
5. Add mocking for external dependencies

## Test Template

```typescript
import { describe, it, expect, vi } from 'vitest';
import { functionName } from './source';

describe('functionName', () => {
  it('should handle normal input', () => {
    expect(functionName(input)).toBe(expected);
  });

  it('should handle edge cases', () => {
    expect(functionName(edge)).toBe(expected);
  });

  it('should throw on invalid input', () => {
    expect(() => functionName(invalid)).toThrow();
  });
});
```

## Mocking Patterns

```typescript
// Mock module
vi.mock('./dependency', () => ({
  fetchData: vi.fn().mockResolvedValue(mockData)
}));

// Mock implementation
const spy = vi.spyOn(object, 'method').mockReturnValue(value);
```
```

This example shows a complete, well-structured skill.
