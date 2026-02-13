# Cisco Skill Scanner - 掃描報告

**掃描時間:** 2026-02-13T13:53:42.002977
**掃描工具:** cisco-ai-skill-scanner v1.0.2 (Static Analyzer)
**掃描目錄:** /home/user/skills

## 總覽

| 指標 | 數值 |
|------|------|
| 掃描 Skills 總數 | 66 |
| 安全 Skills | 61 |
| 不安全 Skills | 5 |
| 總發現數 | 120 |
| CRITICAL | 6 |
| HIGH | 21 |
| MEDIUM | 27 |
| INFO | 66 |

## 無法掃描的 Skills (3)

| Skill | 原因 |
|-------|------|
| knowledge-query | SKILL.md 缺少必填欄位: name |
| code-assistant | SKILL.md 缺少必填欄位: name |
| pingtung-news | YAML frontmatter 解析失敗 (格式錯誤) |

## 需優化的 Skills

### 高優先級 (Unsafe - 含 CRITICAL/HIGH)

#### cisco-skill-scanner
- **路徑:** `/home/user/skills/cisco-skill-scanner`
- **嚴重性分佈:** CRITICAL: 2, HIGH: 5, MEDIUM: 1
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| HIGH | PROMPT_INJECTION_IGNORE_INSTRUCTIONS | Attempts to override previous system instructions | threat-taxonomy.md:16 | Remove instructions that attempt to override system behavior |
| HIGH | PROMPT_INJECTION_UNRESTRICTED_MODE | Attempts to enable unrestricted or dangerous modes | threat-taxonomy.md:19 | Remove mode-switching instructions that bypass safety |
| CRITICAL | SECRET_PRIVATE_KEY | Private key block detected | threat-taxonomy.md:79 | Remove hardcoded private keys |
| HIGH | PROMPT_INJECTION_IGNORE_INSTRUCTIONS | Attempts to override previous system instructions | references/threat-taxonomy.md:16 | Remove instructions that attempt to override system behavior |
| HIGH | PROMPT_INJECTION_UNRESTRICTED_MODE | Attempts to enable unrestricted or dangerous modes | references/threat-taxonomy.md:19 | Remove mode-switching instructions that bypass safety |
| CRITICAL | SECRET_PRIVATE_KEY | Private key block detected | references/threat-taxonomy.md:79 | Remove hardcoded private keys |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/threat-taxonomy.md:16 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_PROMPT_INJECTION | Role reassignment pattern in asset file | references/threat-taxonomy.md:19 | Review the asset file and remove any malicious or unnecessary dynamic patterns |

#### mistral
- **路徑:** `/home/user/skills/mistral`
- **嚴重性分佈:** CRITICAL: 1
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| CRITICAL | YARA_prompt_injection_generic | PROMPT INJECTION detected by YARA | SKILL.md:272 | Review and remove prompt injection pattern |

#### daily-digest-notifier
- **路徑:** `/home/user/skills/daily-digest-notifier`
- **嚴重性分佈:** CRITICAL: 1, MEDIUM: 3
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/todoist_client.py:9 | Ensure network access is necessary and documented. Review all URLs |
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/digest.py:9 | Ensure network access is necessary and documented. Review all URLs |
| CRITICAL | DATA_EXFIL_HTTP_POST | HTTP POST request that may send data externally | scripts/digest.py:130 | Review all POST requests. Ensure they don't send sensitive data |
| MEDIUM | TOOL_ABUSE_UNDECLARED_NETWORK | Undeclared network usage |  | Declare network usage in compatibility field or remove network calls |

#### groq
- **路徑:** `/home/user/skills/groq`
- **嚴重性分佈:** HIGH: 16, MEDIUM: 4
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:1961 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2006 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2425 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2440 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2467 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2502 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2517 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/llms-full.md:2543 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_PROMPT_INJECTION | Role reassignment pattern in asset file | references/llms-full.md:2467 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_PROMPT_INJECTION | Role reassignment pattern in asset file | references/llms-full.md:2543 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:1732 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:1773 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:5084 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:5099 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:5126 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:5156 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:5171 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| HIGH | ASSET_PROMPT_INJECTION | Prompt injection pattern in asset file | references/models.md:5197 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_PROMPT_INJECTION | Role reassignment pattern in asset file | references/models.md:5126 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_PROMPT_INJECTION | Role reassignment pattern in asset file | references/models.md:5197 | Review the asset file and remove any malicious or unnecessary dynamic patterns |

#### account-management-system
- **路徑:** `/home/user/skills/account-management-system`
- **嚴重性分佈:** CRITICAL: 2
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| CRITICAL | SECRET_PRIVATE_KEY | Private key block detected | SKILL.md:343 | Remove hardcoded private keys |
| CRITICAL | SECRET_PRIVATE_KEY | Private key block detected | references/README.md:66 | Remove hardcoded private keys |

### 中優先級 (需審查 - MEDIUM)

#### skill-seekers
- **路徑:** `/home/user/skills/skill-seekers`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/scraper.py:14 | Ensure network access is necessary and documented. Review all URLs |
| MEDIUM | TOOL_ABUSE_UNDECLARED_NETWORK | Undeclared network usage |  | Declare network usage in compatibility field or remove network calls |

#### context7
- **路徑:** `/home/user/skills/context7`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/get_docs.py:4 | Ensure network access is necessary and documented. Review all URLs |
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/search_library.py:5 | Ensure network access is necessary and documented. Review all URLs |
| MEDIUM | TOOL_ABUSE_UNDECLARED_NETWORK | Undeclared network usage |  | Declare network usage in compatibility field or remove network calls |

#### ai-tech-digest
- **路徑:** `/home/user/skills/ai-tech-digest`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | SOCIAL_ENG_ANTHROPIC_IMPERSONATION | Potential Anthropic brand impersonation | SKILL.md: | Do not impersonate official skills or use unauthorized branding |

#### hackernews-ai-digest
- **路徑:** `/home/user/skills/hackernews-ai-digest`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/fetch_ai_news.py:21 | Ensure network access is necessary and documented. Review all URLs |
| MEDIUM | TOOL_ABUSE_UNDECLARED_NETWORK | Undeclared network usage |  | Declare network usage in compatibility field or remove network calls |
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | fetch_ai_news.py:21 | Ensure network access is necessary and documented. Review all URLs |

#### langchain
- **路徑:** `/home/user/skills/langchain`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | ASSET_SUSPICIOUS_URL | Suspicious free domain URL in asset | references/agents.md:10191 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_SUSPICIOUS_URL | Suspicious free domain URL in asset | references/agents.md:10224 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_SUSPICIOUS_URL | Suspicious free domain URL in asset | references/agents.md:10261 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_SUSPICIOUS_URL | Suspicious free domain URL in asset | references/llms-full.md:12872 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_SUSPICIOUS_URL | Suspicious free domain URL in asset | references/llms-full.md:12967 | Review the asset file and remove any malicious or unnecessary dynamic patterns |
| MEDIUM | ASSET_SUSPICIOUS_URL | Suspicious free domain URL in asset | references/llms-full.md:13284 | Review the asset file and remove any malicious or unnecessary dynamic patterns |

#### todoist
- **路徑:** `/home/user/skills/todoist`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | DATA_EXFIL_NETWORK_REQUESTS | HTTP client library imports that enable external communication | scripts/todoist.py:9 | Ensure network access is necessary and documented. Review all URLs |
| MEDIUM | TOOL_ABUSE_UNDECLARED_NETWORK | Undeclared network usage |  | Declare network usage in compatibility field or remove network calls |

#### react
- **路徑:** `/home/user/skills/react`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | ASSET_PROMPT_INJECTION | Role reassignment pattern in asset file | references/other.md:152 | Review the asset file and remove any malicious or unnecessary dynamic patterns |

#### frontend-design
- **路徑:** `/home/user/skills/frontend-design`
- **發現:**

| 嚴重性 | 規則 ID | 標題 | 檔案 | 修復建議 |
|--------|---------|------|------|----------|
| MEDIUM | YARA_capability_inflation_generic | SKILL DISCOVERY ABUSE detected by YARA | SKILL.md:67 | Review and remove skill discovery abuse pattern |
