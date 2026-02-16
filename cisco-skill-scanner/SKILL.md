---
name: cisco-skill-scanner
description: |
  Cisco AI Defense Skill Scanner — AI Agent Skills 安全掃描工具。結合靜態分析 (YAML + YARA)、行為數據流分析、LLM 語義分析與 Meta 分析器，偵測 Prompt Injection、資料外洩、惡意程式碼等威脅。支援 OpenAI Codex Skills 與 Cursor Agent Skills 格式。
  Use when: scanning Agent Skills for security threats, detecting prompt injection in skills, checking skills for data exfiltration, auditing SKILL.md files, integrating skill security into CI/CD, running YARA rules on skills, or when user mentions Skill 安全掃描, Agent Skill 安全, skill 弱點掃描, 惡意 skill 偵測, prompt injection 偵測.
  Triggers: "skill scanner", "cisco skill scanner", "cisco-ai-skill-scanner", "Agent Skill 安全", "skill security", "skill 安全掃描", "prompt injection 偵測", "data exfiltration detection", "YARA rules", "skill audit", "malicious skill", "惡意 skill", "skill 弱點", "skill vulnerability", "AI agent security", "LLM security", "behavioral analysis", "SARIF", "skill-scanner scan", "scan-all", "AITech taxonomy", "Codex Skills", "Cursor Skills"
version: 1.0.0
last_updated: 2026-02-13
---

# Cisco AI Defense Skill Scanner

Cisco 開源的 AI Agent Skills 安全掃描器，用於偵測 Prompt Injection、資料外洩、指令注入、混淆及其他安全威脅。結合多引擎分析架構，提供全面的 Agent Skills 安全評估。

- **GitHub**: https://github.com/cisco-ai-defense/skill-scanner
- **PyPI**: `cisco-ai-skill-scanner`
- **授權**: Apache 2.0

## When to Use This Skill

- 需要掃描 AI Agent Skills（Claude Code Skills / OpenAI Codex Skills / Cursor Agent Skills）是否安全
- 需要偵測 Skill 中的 Prompt Injection、資料外洩、指令注入等威脅
- 需要在 CI/CD 中整合 Skill 安全掃描（SARIF 輸出支援 GitHub Code Scanning）
- 需要稽核 SKILL.md 檔案的安全性（YAML frontmatter + 程式碼）
- 需要使用 YARA / 靜態分析 / LLM 語義分析進行多引擎威脅偵測
- 需要進行 Python 腳本的行為數據流分析（source → sink 追蹤）
- 需要降低誤報率（Meta-Analyzer 整合多引擎結果）

## Quick Start

### 安裝

```bash
# 使用 uv（推薦）
uv pip install cisco-ai-skill-scanner

# 使用 pip
pip install cisco-ai-skill-scanner

# 安裝所有雲端供應商支援
pip install cisco-ai-skill-scanner[all]
```

### 基本掃描

```bash
# 掃描單一 Skill（僅靜態分析）
skill-scanner scan /path/to/skill

# 掃描含行為分析
skill-scanner scan /path/to/skill --use-behavioral

# 掃描含所有引擎
skill-scanner scan /path/to/skill --use-behavioral --use-llm --use-aidefense

# 啟用 Meta-Analyzer 降低誤報
skill-scanner scan /path/to/skill --use-llm --enable-meta

# 遞迴掃描多個 Skills
skill-scanner scan-all /path/to/skills --recursive --use-behavioral

# CI/CD：發現威脅時 build 失敗
skill-scanner scan-all ./skills --fail-on-findings --format sarif --output results.sarif
```

### Python SDK

```python
from skill_scanner import SkillScanner
from skill_scanner.core.analyzers import StaticAnalyzer, BehavioralAnalyzer

# 建立掃描器
scanner = SkillScanner(analyzers=[
    StaticAnalyzer(),
    BehavioralAnalyzer(use_static_analysis=True),
])

# 掃描 Skill
result = scanner.scan_skill("/path/to/skill")

print(f"安全: {result.is_safe}")
print(f"發現數: {len(result.findings)}")
for finding in result.findings:
    print(f"  [{finding.severity}] {finding.title}")
```

## Core Concepts

### 多引擎分析架構

```
┌─────────────────────────────────────────────────────────────┐
│                   Cisco Skill Scanner                        │
├───────────┬───────────┬──────────┬──────────┬───────────────┤
│  Static   │ Behavioral│   LLM    │   Meta   │  VirusTotal / │
│  Analyzer │ Analyzer  │ Analyzer │ Analyzer │  AI Defense   │
├───────────┼───────────┼──────────┼──────────┼───────────────┤
│ YAML 規則 │ AST 解析  │ 語義分析 │ 誤報過濾 │ 二進位/雲端   │
│ YARA 模式 │ CFG 建構  │ 意圖偵測 │ 結果整合 │ 惡意軟體掃描  │
│ Python    │ 數據流追蹤│ 結構化   │ 優先排序 │ 威脅情資      │
│ 驗證檢查  │ source→sink│ 輸出強制 │ 信心評分 │               │
├───────────┼───────────┼──────────┼──────────┼───────────────┤
│ 免費      │ 免費      │ 需 API   │ 需 API   │ 需 API Key    │
│ ~30ms     │ ~50-100ms │ ~5-10s   │ ~5-15s   │               │
└───────────┴───────────┴──────────┴──────────┴───────────────┘
```

### 安全分析器

| 分析器 | 偵測方式 | 分析範圍 | 需求 |
|--------|---------|---------|------|
| **Static** | YAML + YARA 模式匹配 | 所有檔案 | 無 |
| **Behavioral** | AST 數據流分析 | Python 檔案 | 無 |
| **LLM** | 語義分析 (Claude/GPT/Gemini) | SKILL.md + 腳本 | API Key |
| **Meta** | 誤報過濾、結果整合 | 所有發現 | API Key |
| **VirusTotal** | Hash 惡意軟體偵測 | 二進位檔案 | API Key |
| **AI Defense** | Cisco 雲端 AI 分析 | 文字內容 | API Key |

### AITech 威脅分類法

Skill Scanner 使用 Cisco AITech 威脅分類法（對齊 MITRE ATLAS / OWASP Top 10）：

| AITech Code | 威脅類別 | 風險等級 | 說明 |
|-------------|---------|---------|------|
| AITech-1.1 | Prompt Injection（直接） | HIGH-CRITICAL | 指令覆蓋嘗試 |
| AITech-1.2 | Prompt Injection（間接） | HIGH | 從外部資料源嵌入惡意指令 |
| AITech-4.3 | 能力膨脹 | LOW-HIGH | Skill 描述與行為不符、品牌冒充 |
| AITech-8.2 | 資料外洩 | CRITICAL | 憑證竊取、未授權資料傳輸 |
| AITech-9.1 | 指令/程式碼注入 | CRITICAL | eval()、os.system()、SQL injection |
| AITech-12.1 | 工具濫用 | MEDIUM-CRITICAL | 工具中毒、未宣告的工具使用 |
| AITech-13.1 | 可用性破壞 | LOW-MEDIUM | 資源耗盡、無限迴圈 |

## CLI 完整參考

### 命令

```bash
# 掃描單一 Skill
skill-scanner scan <DIRECTORY> [OPTIONS]

# 掃描多個 Skills
skill-scanner scan-all <DIRECTORY> [OPTIONS]

# 列出可用分析器
skill-scanner list-analyzers

# 驗證規則簽名
skill-scanner validate-rules
```

### 選項

| 選項 | 說明 |
|------|------|
| `--use-behavioral` | 啟用行為分析器（數據流分析） |
| `--use-llm` | 啟用 LLM 分析器（需 API Key） |
| `--use-virustotal` | 啟用 VirusTotal 二進位掃描 |
| `--use-aidefense` | 啟用 Cisco AI Defense 分析器 |
| `--enable-meta` | 啟用 Meta-Analyzer 降低誤報 |
| `--format` | 輸出格式: `summary`, `json`, `markdown`, `table`, `sarif` |
| `--output PATH` | 儲存報告至檔案 |
| `--fail-on-findings` | 發現 HIGH/CRITICAL 時以錯誤碼退出 |
| `--yara-mode` | 偵測模式: `strict`, `balanced`(預設), `permissive` |
| `--custom-rules PATH` | 使用自訂 YARA 規則目錄 |
| `--disable-rule RULE` | 停用特定規則（可重複使用） |
| `--recursive` | 遞迴掃描子目錄 |
| `--detailed` | 包含詳細報告（snippet + 修復建議） |
| `--check-overlap` | 檢查 Skills 間的重疊 |

## 環境設定

### LLM 分析器

```bash
# 通用設定（LLM + Meta 共用）
export SKILL_SCANNER_LLM_API_KEY="your_api_key"
export SKILL_SCANNER_LLM_MODEL="claude-3-5-sonnet-20241022"

# Meta-Analyzer 專用覆蓋（選用）
export SKILL_SCANNER_META_LLM_API_KEY="different_key"
export SKILL_SCANNER_META_LLM_MODEL="gpt-4o"
```

### 支援的 LLM 供應商

| 供應商 | 模型設定範例 | 備註 |
|--------|------------|------|
| **Anthropic** | `claude-3-5-sonnet-20241022` | 推薦 |
| **OpenAI** | `gpt-4o` | |
| **Azure OpenAI** | `azure/gpt-4` + BASE_URL | 需額外設定 |
| **AWS Bedrock** | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | 支援 IAM |
| **Google Gemini** | `gemini-2.0-flash-exp` | 直接 SDK |
| **Vertex AI** | `vertex_ai/gemini-1.5-pro` | 需服務帳號 |

### 其他 API Key

```bash
# VirusTotal
export VIRUSTOTAL_API_KEY="your_virustotal_key"

# Cisco AI Defense
export AI_DEFENSE_API_KEY="your_aidefense_key"
```

## 偵測能力詳解

### Static Analyzer（靜態分析器）

58 條規則，6 個掃描階段：

1. **Manifest Pass** — 驗證 YAML 結構、描述品質、品牌冒充
2. **Instruction Pass** — 掃描 SKILL.md 中的 prompt injection
3. **Code Pass** — 掃描 Python/Bash 腳本中的危險函式
4. **Consistency Pass** — 交叉檢查 manifest vs 行為
5. **Reference Pass** — 遞迴掃描連結的檔案
6. **Binary Pass** — 標記二進位執行檔

**規則定義格式**：

```yaml
# skill_scanner/data/rules/signatures.yaml
- id: RULE_ID
  category: threat_category
  severity: CRITICAL|HIGH|MEDIUM|LOW
  patterns:
    - "regex_pattern_1"
    - "regex_pattern_2"
  file_types: [python, bash, markdown]
  description: "偵測什麼"
  remediation: "如何修復"
```

### Behavioral Analyzer（行為分析器）

基於 AST 與 CFG 的數據流分析：

```
Python 腳本 → AST 解析 → 函式提取 → CFG 建構 → 數據流追蹤 → source→sink 分析
```

偵測模式：
- 參數 → eval（指令注入）
- 憑證檔案 → 網路呼叫（外洩）
- 環境變數 → 網路呼叫（憑證竊取）
- 多檔案跨檔分析

```python
from skill_scanner.core.analyzers import BehavioralAnalyzer

behavioral = BehavioralAnalyzer(use_static_analysis=True)
# ~50-100ms 每腳本，純 Python，無需 Docker
```

### LLM Analyzer（LLM 分析器）

使用 LLM 作為安全評審（LLM-as-a-Judge）：

- 透過 LiteLLM 支援 100+ 模型
- 隨機分隔符防注入保護
- 結構化輸出強制 AITech 分類
- 87KB Cisco 分析提示框架

```python
from skill_scanner.core.analyzers.llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer(
    model="claude-3-5-sonnet-20241022",
    api_key="your_key"
)
findings = analyzer.analyze(skill)
# 或非同步（批量推薦）
findings = await analyzer.analyze_async(skill)
```

### Meta-Analyzer（後設分析器）

二次審查降低誤報：

- **誤報過濾** — 根據完整 Skill 上下文移除誤報
- **結果整合** — 合併重複的模式匹配結果
- **優先排序** — 按可利用性和影響排序
- **信心評分** — 添加 meta_confidence / meta_exploitability / meta_impact

**分析器權威階層**：

| 分析器 | 權威等級 |
|--------|---------|
| LLM | 最高 |
| Behavioral | 高 |
| AI Defense | 中高 |
| Static | 中 |
| Trigger | 較低 |

## CI/CD 整合

### GitHub Actions

```yaml
name: Scan Skills Security

on: [push, pull_request]

jobs:
  skill-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install scanner
        run: pip install cisco-ai-skill-scanner

      - name: Scan skills
        run: |
          skill-scanner scan-all ./skills \
            --fail-on-findings \
            --format sarif \
            --output results.sarif

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### Pre-commit Hook

```bash
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## REST API Server

```bash
# 啟動 API Server
skill-scanner-api                    # 預設 localhost:8000
skill-scanner-api --port 8080       # 自訂埠
skill-scanner-api --reload          # 開發模式

# 掃描（curl）
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"skill_directory": "/path/to/skill", "use_behavioral": true}'

# 上傳 ZIP 掃描
curl -X POST http://localhost:8000/scan-upload \
  -F "file=@skill.zip" -F "use_llm=true"

# 批量非同步掃描
curl -X POST http://localhost:8000/scan-batch \
  -H "Content-Type: application/json" \
  -d '{"skills_directory": "/path/to/skills", "recursive": true}'
```

**API 端點**：

| 端點 | 方法 | 說明 |
|------|------|------|
| `/health` | GET | 健康檢查 |
| `/scan` | POST | 掃描單一 Skill |
| `/scan-upload` | POST | 上傳 ZIP 掃描 |
| `/scan-batch` | POST | 批量非同步掃描 |
| `/scan-batch/{id}` | GET | 取得批量掃描結果 |
| `/analyzers` | GET | 列出可用分析器 |

API 文件: `http://localhost:8000/docs` (Swagger UI)

## 輸出格式

```bash
# 摘要（預設）
skill-scanner scan /path/to/skill

# JSON（CI/CD 整合）
skill-scanner scan /path/to/skill --format json --output results.json

# SARIF（GitHub Code Scanning）
skill-scanner scan /path/to/skill --format sarif --output results.sarif

# Markdown（人類可讀報告）
skill-scanner scan /path/to/skill --format markdown --detailed --output report.md

# 表格（終端機友好）
skill-scanner scan-all /path/to/skills --format table
```

## 自訂規則

### 自訂 YARA 規則

```bash
skill-scanner scan /path/to/skill --custom-rules /path/to/my-rules/
```

### 停用特定規則

```bash
skill-scanner scan /path/to/skill \
  --disable-rule YARA_script_injection \
  --disable-rule MANIFEST_MISSING_LICENSE
```

### 偵測模式

```bash
# 嚴格模式（更多發現，較高誤報率）
skill-scanner scan /path/to/skill --yara-mode strict

# 平衡模式（預設）
skill-scanner scan /path/to/skill --yara-mode balanced

# 寬鬆模式（較少發現，可能遺漏）
skill-scanner scan /path/to/skill --yara-mode permissive
```

## 擴展自訂分析器

```python
from skill_scanner.core.analyzers.base import BaseAnalyzer
from skill_scanner.core.models import Skill, Finding

class MyAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("my_analyzer")

    def analyze(self, skill: Skill) -> list[Finding]:
        findings = []
        # ... 自訂分析邏輯 ...
        return findings

# 加入掃描器
from skill_scanner import SkillScanner
scanner = SkillScanner(analyzers=[MyAnalyzer()])
```

## Skill 作者安全最佳實務

- 不要硬編碼秘密（API Key、密碼）
- 不要使用 `eval()` / `exec()`
- 驗證所有輸入
- 在 `allowed-tools` 中宣告所有工具權限
- 描述準確反映實際行為
- 不要混淆程式碼
- 發布前先使用 skill-scanner 掃描

## Reference Files

| File | Content |
|------|---------|
| [index.md](references/index.md) | 參考文件索引 |
| [threat-taxonomy.md](references/threat-taxonomy.md) | AITech 完整威脅分類法 |
| [ci-cd-integration.md](references/ci-cd-integration.md) | CI/CD 整合詳細指南 |

## Resources

- [GitHub Repository](https://github.com/cisco-ai-defense/skill-scanner)
- [PyPI Package](https://pypi.org/project/cisco-ai-skill-scanner/)
- [Quick Start Guide](https://github.com/cisco-ai-defense/skill-scanner/blob/main/docs/quickstart.md)
- [Architecture](https://github.com/cisco-ai-defense/skill-scanner/blob/main/docs/architecture.md)
- [Threat Taxonomy](https://github.com/cisco-ai-defense/skill-scanner/blob/main/docs/threat-taxonomy.md)
- [Cisco AI Security Framework (arxiv)](https://arxiv.org/html/2512.12921v1)
- [Discord Community](https://discord.com/invite/nKWtDcXxtx)
