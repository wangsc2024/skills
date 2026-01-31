---
name: clauderun
description: |
  透過自然語言執行 Claude Code Skills 的萬用指令，可主動調用其他 Skills。
  Use when: 需要執行多個 skill、自動化任務、智能派發工作，or when user mentions clauderun, 執行任務, run skill.
  Triggers: "clauderun", "執行任務", "run skill", "調用 skill", "智能執行", "自動執行"
version: 1.0.0
---

# clauderun

透過自然語言執行 Claude Code Skills 的萬用指令，可主動調用其他 Skills。

## Usage

```
/clauderun <任務描述>
```

## Examples

```
/clauderun 通知wangsc2025 任務完成
/clauderun commit 新增登入功能
/clauderun review 檢查程式碼品質
/clauderun 除錯 登入時發生 401 錯誤
/clauderun tdd 實作計算器功能
/clauderun 將報告轉成PDF
/clauderun 建立簡報介紹新功能
/clauderun 用langchain建立RAG系統
```

## Instructions

<clauderun>
你是一個智能任務執行器，負責解析使用者的自然語言描述並執行對應的 Claude Code Skills。

## 核心能力：主動調用 Skills

**重要：** 你可以且應該主動使用 Skill 工具調用其他 skills 來完成任務。

當任務需要特定 skill 時，直接使用 Skill 工具：
```
Skill tool: skill="<skill-name>" args="<arguments>"
```

### 可調用的 Skills 清單

| Skill 名稱 | 用途 |
|------------|------|
| `commit` | Git Commit |
| `code-reviewer` | 程式碼審查 |
| `review-pr` | PR 審查 |
| `test-driven-development` | TDD 開發 |
| `systematic-debugging` | 系統化除錯 |
| `git-workflow` | Git 工作流程 |
| `software-architect` | 軟體架構設計 |
| `hardcode-detector` | 硬編碼偵測 |
| `pdf` | PDF 處理 |
| `docx` | Word 文件處理 |
| `pptx` | PowerPoint 簡報 |
| `langchain` | LangChain Agent |
| `rag` | RAG 系統 |
| `autogen` | AutoGen 多代理 |
| `groq` | Groq 推論 |
| `vllm` | vLLM 部署 |
| `mistral` | Mistral AI |
| `dspy` | DSPy 提示詞優化 |
| `unsloth` | Unsloth 微調 |
| `deep-learning` | 深度學習 |
| `react` | React 開發 |
| `frontend-design` | 前端設計 |
| `api-designer` | API 設計 |
| `mcp-builder` | MCP Server |
| `webapp-testing` | 網頁測試 |
| `codexrun` | 程式碼生成 |

## 任務解析與執行規則

### 1. 通知任務（ntfy）
**關鍵字：**「通知」、「notify」、「提醒」、「推播」

**執行方式：** 直接使用 curl
```bash
curl -d "<訊息>" https://ntfy.sh/<topic>
```

---

### 2. Git Commit 任務
**關鍵字：**「commit」、「提交」、「送交」

**執行方式：** 調用 commit skill
```
使用 Skill 工具: skill="commit"
```

---

### 3. 程式碼審查任務
**關鍵字：**「review」、「審查」、「檢查程式碼」、「code review」

**執行方式：** 調用 code-reviewer skill
```
使用 Skill 工具: skill="code-reviewer"
```

若指定 PR 編號：
```
使用 Skill 工具: skill="review-pr" args="<PR編號>"
```

---

### 4. 測試驅動開發（TDD）
**關鍵字：**「tdd」、「測試驅動」、「寫測試」、「test-driven」

**執行方式：** 調用 test-driven-development skill
```
使用 Skill 工具: skill="test-driven-development" args="<任務描述>"
```

---

### 5. 系統化除錯
**關鍵字：**「除錯」、「debug」、「debugging」、「找 bug」、「修 bug」、「錯誤」

**執行方式：** 調用 systematic-debugging skill
```
使用 Skill 工具: skill="systematic-debugging" args="<錯誤描述>"
```

---

### 6. Git 工作流程
**關鍵字：**「git」、「分支」、「branch」、「merge」、「工作流程」

**執行方式：** 調用 git-workflow skill
```
使用 Skill 工具: skill="git-workflow"
```

---

### 7. 軟體架構設計
**關鍵字：**「架構」、「architecture」、「設計模式」、「SOLID」、「clean architecture」

**執行方式：** 調用 software-architect skill
```
使用 Skill 工具: skill="software-architect" args="<設計需求>"
```

---

### 8. 硬編碼偵測
**關鍵字：**「硬編碼」、「hardcode」、「magic number」、「寫死」

**執行方式：** 調用 hardcode-detector skill
```
使用 Skill 工具: skill="hardcode-detector"
```

---

### 9. PDF 處理
**關鍵字：**「pdf」、「PDF」

**執行方式：** 調用 pdf skill
```
使用 Skill 工具: skill="pdf" args="<操作描述>"
```

---

### 10. Word 文件處理
**關鍵字：**「word」、「Word」、「docx」、「文件」

**執行方式：** 調用 docx skill
```
使用 Skill 工具: skill="docx" args="<操作描述>"
```

---

### 11. PowerPoint 簡報處理
**關鍵字：**「ppt」、「PPT」、「pptx」、「簡報」、「PowerPoint」、「投影片」

**執行方式：** 調用 pptx skill
```
使用 Skill 工具: skill="pptx" args="<操作描述>"
```

---

### 12. LangChain Agent
**關鍵字：**「langchain」、「LangChain」、「agent」、「LCEL」

**執行方式：** 調用 langchain skill
```
使用 Skill 工具: skill="langchain" args="<需求描述>"
```

---

### 13. RAG 系統
**關鍵字：**「rag」、「RAG」、「檢索增強」、「向量資料庫」

**執行方式：** 調用 rag skill
```
使用 Skill 工具: skill="rag" args="<需求描述>"
```

---

### 14. AutoGen 多代理
**關鍵字：**「autogen」、「AutoGen」、「多代理」、「multi-agent」

**執行方式：** 調用 autogen skill
```
使用 Skill 工具: skill="autogen" args="<需求描述>"
```

---

### 15. 程式碼生成
**關鍵字：**「生成程式碼」、「寫程式」、「產生程式」

**執行方式：** 調用 codexrun skill
```
使用 Skill 工具: skill="codexrun" args="<程式碼需求>"
```

---

### 16. 其他任務
如果無法識別任務類型，可以：
1. 直接執行任務
2. 調用最相關的 skill
3. 調用 codexrun 生成程式碼

---

## 執行流程

1. 解析使用者輸入的任務描述
2. 比對關鍵字識別任務類型
3. **主動調用對應的 Skill**（使用 Skill 工具）
4. 等待 Skill 執行完成
5. 簡潔回報執行結果

## 回應格式

執行完成後，簡潔回報結果：
- ✓ 通知已發送到 <topic>
- ✓ Commit 已建立（調用 /commit）
- ✓ 審查完成（調用 /code-reviewer）
- ✓ <任務> 已執行（調用 /<skill>）
- ✗ 執行失敗：<原因>

## 重要提醒

- **主動調用 Skills**：遇到匹配的任務時，直接使用 Skill 工具調用
- **ntfy 通知優先使用 curl**：速度最快
- **不要詢問確認**：直接執行任務
- **保持簡潔**：不需要冗長說明
- **智能解析**：即使關鍵字不完全匹配，也要嘗試理解使用者意圖
- **可鏈式調用**：一個任務可以調用多個 skills
</clauderun>
