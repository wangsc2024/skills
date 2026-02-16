# CI/CD 整合指南

## GitHub Actions

### 基本整合

```yaml
name: Scan Skills Security

on: [push, pull_request]

jobs:
  skill-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install scanner
        run: pip install cisco-ai-skill-scanner

      - name: Scan skills
        run: |
          skill-scanner scan-all ./skills \
            --fail-on-findings \
            --format sarif \
            --output results.sarif

      - name: Upload SARIF to GitHub Security
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### 完整整合（含 LLM 分析）

```yaml
name: Comprehensive Skill Security Scan

on:
  push:
    paths:
      - "skills/**"
      - ".claude/skills/**"
  pull_request:
    paths:
      - "skills/**"
      - ".claude/skills/**"

jobs:
  skill-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install scanner
        run: pip install cisco-ai-skill-scanner[all]

      - name: Run static + behavioral scan
        run: |
          skill-scanner scan-all ./skills \
            --use-behavioral \
            --recursive \
            --fail-on-findings \
            --format sarif \
            --output static-results.sarif

      - name: Run LLM + meta scan
        if: github.event_name == 'pull_request'
        env:
          SKILL_SCANNER_LLM_API_KEY: ${{ secrets.SKILL_SCANNER_LLM_API_KEY }}
          SKILL_SCANNER_LLM_MODEL: claude-3-5-sonnet-20241022
        run: |
          skill-scanner scan-all ./skills \
            --use-behavioral \
            --use-llm \
            --enable-meta \
            --recursive \
            --format markdown \
            --detailed \
            --output llm-report.md

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: static-results.sarif

      - name: Comment PR with report
        if: github.event_name == 'pull_request' && always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            if (fs.existsSync('llm-report.md')) {
              const report = fs.readFileSync('llm-report.md', 'utf8');
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: '## Skill Security Scan Report\n\n' + report
              });
            }
```

## Pre-commit Hook

### 設定

```bash
# 從 skill-scanner 專案複製 hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 自訂 Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# 檢查是否有 skill 檔案變更
SKILL_FILES=$(git diff --cached --name-only | grep -E "(SKILL\.md|\.py|\.sh)$" | head -20)

if [ -z "$SKILL_FILES" ]; then
    exit 0
fi

echo "Scanning modified skills for security issues..."

# 找出變更的 skill 目錄
SKILL_DIRS=$(echo "$SKILL_FILES" | xargs -I {} dirname {} | sort -u)

EXIT_CODE=0
for dir in $SKILL_DIRS; do
    if [ -f "$dir/SKILL.md" ]; then
        echo "Scanning: $dir"
        skill-scanner scan "$dir" --fail-on-findings --format summary
        if [ $? -ne 0 ]; then
            EXIT_CODE=1
        fi
    fi
done

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Security issues detected! Fix before committing."
    echo "Use --disable-rule to suppress false positives."
fi

exit $EXIT_CODE
```

### .pre-commit-config.yaml

```yaml
repos:
  - repo: local
    hooks:
      - id: skill-scanner
        name: Skill Security Scanner
        entry: skill-scanner scan-all
        args: ["--fail-on-findings", "--format", "summary"]
        language: python
        additional_dependencies: ["cisco-ai-skill-scanner"]
        files: "(SKILL\\.md|\\.py|\\.sh)$"
        pass_filenames: false
```

## Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install cisco-ai-skill-scanner[all]

ENTRYPOINT ["skill-scanner"]
```

### 使用

```bash
# Build
docker build -t skill-scanner .

# 掃描本地 skill 目錄
docker run -v /path/to/skills:/skills skill-scanner scan-all /skills --format table

# 含 LLM 分析
docker run \
  -e SKILL_SCANNER_LLM_API_KEY="your_key" \
  -e SKILL_SCANNER_LLM_MODEL="claude-3-5-sonnet-20241022" \
  -v /path/to/skills:/skills \
  skill-scanner scan-all /skills --use-llm --enable-meta
```

### API Server Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install cisco-ai-skill-scanner[all]

EXPOSE 8000

CMD ["skill-scanner-api", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t skill-scanner-api -f Dockerfile.api .
docker run -p 8000:8000 \
  -e SKILL_SCANNER_LLM_API_KEY="your_key" \
  -e SKILL_SCANNER_LLM_MODEL="claude-3-5-sonnet-20241022" \
  skill-scanner-api
```

## 輸出格式選擇

| 格式 | 適用場景 | 命令 |
|------|---------|------|
| **summary** | 終端機快速查看 | `--format summary` |
| **table** | 批量掃描概覽 | `--format table` |
| **json** | CI/CD 自動化處理 | `--format json` |
| **sarif** | GitHub Code Scanning | `--format sarif` |
| **markdown** | PR 報告、文件 | `--format markdown --detailed` |

## 退出碼

| 退出碼 | 意義 |
|--------|------|
| 0 | 掃描完成，無 HIGH/CRITICAL 發現 |
| 1 | 發現 HIGH 或 CRITICAL 威脅（配合 `--fail-on-findings`） |
| 2 | 掃描錯誤（路徑不存在、設定錯誤等） |
