# Claude Code Skills 索引

此索引幫助快速選擇正確的 Skill，並提供組合使用建議與自動觸發規則。

> **更新時間:** 2026-02-03
> **版本:** v4.2
> **新增:** pingtung-news（屏東新聞 MCP 服務）

---

## Skills 類型說明

| 類型 | 說明 | 範例 |
|------|------|------|
| **目錄型** | 包含 SKILL.md + references/ 子目錄 | langchain, react, vllm |
| **指令型** | 獨立的 *_instruction.md 檔案 | ai_daily_report_instruction.md |
| **目錄+指令** | 兩者都有，提供更完整指引 | autogen, tiptap, ui-color-optimizer |
| **整合型** | 多個指令整合在一個檔案 | claude_project_instructions.md |
| **插件型** | 透過 plugin 系統提供的 skills | episodic-memory, pinecone, sentry |

---

## 自動化與任務執行

智能任務執行、自主開發迴圈、程式碼生成、問題診斷相關技能。

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **clauderun** | 執行, @fullreview, @tddflow, 組合任務 | 智能任務執行器，組合調用多個 Skills | 目錄 |
| **codexrun** | 生成程式碼, #api, #scraper, 寫程式 | 程式碼生成專家，自動偵測專案類型 | 目錄 |
| **rloop** | ralph, loop, 自主開發, PRD, 迭代 | Ralph 自主開發迴圈（英文版） | 目錄 |
| **ralph-loop** | 自主迭代, 多輪開發, 持續精進 | Ralph 自主迭代開發（中文詳細版） | 目錄 |
| **issue-resolver** | 處理問題, 修復bug, 待辦, 回報修復 | 自動化問題診斷與修復，支援 Gun.js relay | 目錄 |

---

## AI/ML 框架與推理

建構 AI 應用、多代理系統、LLM 訓練、推理與部署相關技能。

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **langchain** | LangChain, LCEL, agents, RAG, memory | LangChain AI 代理與 RAG 管道 | 目錄 |
| **dspy** | DSPy, signatures, teleprompters, MIPRO | Stanford 宣告式 Prompt 優化 | 目錄 |
| **autogen** | AutoGen, multi-agent, AgentChat, teams, Swarm, MagenticOne | Microsoft 多代理協作框架 | 目錄+指令 |
| **unsloth** | fine-tune, LoRA, QLoRA, GRPO, vision | 2-5x 快速 LLM 微調 | 目錄 |
| **vllm** | vLLM, inference, serving, PagedAttention, tensor parallel | 高吞吐量 LLM 推理與部署 | 目錄 |
| **mistral** | Mistral AI, pixtral, embeddings | Mistral 模型 API 與 SDK | 目錄 |
| **groq** | Groq, LPU, fast inference, Whisper | Groq 超快推理與語音轉文字 | 目錄 |
| **rag** | RAG, 向量資料庫, embeddings, chunking | 檢索增強生成完整指南 | 目錄 |
| **deep-learning** | CNN, RNN, LSTM, Transformer, Dropout | 深度學習核心架構與技術 | 目錄 |

---

## 開發流程與調試

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **test-driven-development** | TDD, 測試驅動, 寫測試, unit test | 測試先行開發 | 目錄 |
| **systematic-debugging** | debug, bug, error, 除錯, exception | 系統化除錯 | 目錄 |
| **writing-plans** | 計畫, plan, 規劃, 步驟, 任務拆解 | 建立實作計畫 | 目錄 |
| **executing-plans** | 執行計畫, follow plan, 按照計畫 | 執行實作計畫 | 目錄 |
| **software-architect** | 架構, architecture, SOLID, design pattern | 架構設計 | 目錄 |
| **code-reviewer** | review, 審查, PR review, 檢查程式碼 | 程式碼審查 | 目錄 |
| **hardcode-detector** | hardcode, 寫死, magic number, secrets | 硬編碼檢測 | 目錄 |
| **git-workflow** | git, commit, branch, merge, PR | Git 工作流程 | 目錄 |
| **mcp-builder** | MCP, Model Context Protocol, Claude擴展 | MCP Server 建構 | 目錄 |
| **webapp-testing** | Playwright, E2E, 端對端測試, browser test | 網頁自動化測試 | 目錄 |
| **playwright** | Playwright, E2E, 瀏覽器自動化, web testing | Microsoft Playwright 瀏覽器自動化框架 | 目錄 |

---

## 瀏覽器開發者工具

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **chrome-devtools** | Chrome DevTools, devtools, 開發者工具, console, 網路面板 | Chrome 瀏覽器調試、效能分析、Core Web Vitals | 目錄 |
| **vue-devtools** | Vue DevTools, Vue 調試, Vuex, Pinia | Vue.js 應用調試、狀態管理、效能監控 | 目錄 |

---

## 前端與設計

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **react** | React, hooks, components, state, JSX | React 框架開發 | 目錄 |
| **tiptap** | Tiptap, ProseMirror, rich text, WYSIWYG, Node Views | 富文本編輯器 | 目錄+指令 |
| **frontend-ui-tools** | React, component, 元件, Button, Modal | React 元件生成 | 目錄 |
| **frontend-design** | web design, UI設計, 網頁設計, landing page | 獨特前端設計 | 目錄 |
| **ui-color-optimizer** | color, 配色, 顏色, dark mode, 對比度, WCAG | 配色優化與無障礙設計 | 目錄+指令 |
| **theme-factory** | theme, 主題, 風格, style | 主題套用 | 目錄 |
| **web-artifacts-builder** | shadcn, web app, 複雜應用, React專案 | 複雜前端應用 | 目錄 |
| **gemini-interactive-learning** | Gemini 學習, 互動式教學, learning website, AI 教學網站 | Gemini AI 互動式學習網站開發（Next.js + Shadcn） | 目錄 |

---

## 視覺藝術

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **algorithmic-art** | p5.js, SVG, generative art, 生成藝術 | 演算法藝術 | 目錄 |
| **canvas-design** | poster, 海報, 視覺設計, artwork | 視覺設計作品 | 目錄 |

---

## 文件處理

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **docx** | Word, docx, 文件, 報告, 合約 | Word 文件 | 目錄 |
| **pdf** | PDF, 合併PDF, OCR, 表單 | PDF 處理 | 目錄 |
| **pptx** | PowerPoint, 簡報, slides, 投影片 | 簡報製作 | 目錄 |

---

## 帳號與權限

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **account-manager** | 帳號管理, RBAC, 權限, 認證, 登入登出 | 帳號與權限管理基礎指南 | 目錄 |
| **account-management-system** | JWT, Firebase Auth, 用戶 CRUD, FastAPI 認證 | 完整帳號管理系統（FastAPI + Firebase + RBAC） | 目錄 |

---

## 寫作與溝通

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **writing-masters** | 海明威, 極簡, 冰山理論, 寫作技巧 | 西方文學大師寫作技巧 | 目錄 |
| **storytelling-masters** | 說故事, 簡報, 英雄旅程, 影響力 | 說故事與簡報技藝 | 目錄 |
| **internal-comms** | status report, 報告, 週報, incident report | 內部溝通 | 目錄 |

---

## 學習與自我成長

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **learning-mastery** | 學習方法, 記憶, 費曼技巧, 間隔複習, Anki | 楊大輝《深度學習的技術》五層學習框架 | 目錄 |
| **atomic-habits** | 習慣, 原子習慣, 行為改變, 兩分鐘法則 | James Clear 原子習慣方法論 | 目錄 |
| **course-architect** | 課程設計, 學習路徑, 教學設計, 評估標準 | 課程安排大師（靈感來自 Mr. Ranedeer） | 目錄 |

---

## 資訊整合與追蹤

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **ai-daily-report** | AI新聞, 日報, Claude更新, OpenAI, 技術週報 | 每日 AI 技術新聞整理（基礎版） | 指令 |
| **ai-tech-digest** | AI 技術日報, ai tech digest, Claude 新聞, Gemini 新聞 | AI 技術日報追蹤與分析（Anthropic, OpenAI, Google） | 目錄 |
| **hackernews-ai-digest** | Hacker News, HN, AI news, 新聞摘要 | HN AI 新聞抓取、完整內容翻譯 | 目錄 |

---

## 任務管理與通知

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **ntfy-notify** | 通知, 提醒, 完成後通知, ntfy | 透過 ntfy.sh 發送任務完成通知 | 目錄 |
| **todoist** | todoist, 待辦事項, todo, 任務, 今日任務 | Todoist API 整合（查詢、新增、完成任務） | 目錄 |
| **daily-digest-notifier** | daily digest, 今日摘要, 行事曆通知, today's schedule | 整合 Google Calendar + Todoist + ntfy 每日摘要 | 目錄 |

---

## 知識庫建構與查詢

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **knowledge-query** | 查詢知識庫, 筆記查詢, RAG 查詢 | 個人知識庫查詢（PowerShell API） | 目錄 |
| **code-assistant** | 程式輔助, 程式碼查詢, useCallback | 基於知識庫的程式開發輔助 | 目錄 |
| **skill-creator** | create skill, 建立skill, SKILL.md | Skill 建立指南 | 目錄 |
| **skill-seekers** | 從文件建立 Skill, 文件轉 Skill | 技術文件轉 AI 知識庫（通用版） | 目錄 |
| **skill-seekers-claude-code** | 抓取文件, GitHub 倉庫分析 | Skill Seekers Claude Code 版本（含程式碼） | 目錄 |
| **skill-seekers-claude-chat** | 整理文件, 知識庫助手 | Skill Seekers Claude Chat 版本（對話式） | 目錄 |

---

## Claude Code 配置

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **claude-md-guide** | CLAUDE.md, 配置文件, Claude Code 配置, rules, commands | CLAUDE.md 三層配置架構設計指南 | 目錄 |

---

## 政府與地方資訊

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **pingtung-news** | 屏東新聞, 屏東縣政府, 周春米, 屏東縣府公告 | 屏東縣政府新聞查詢（MCP 即時服務） | 目錄 |
| **pingtung-policy-expert** | 屏東政策, 屏東施政, 縣政方針 | 屏東縣政策與施政專家 | 目錄 |

---

## 專業領域

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **rental-contract-guide** | 租賃契約, 租約, 房東, 房客, 押金, 電費 | 台灣租賃契約專家（法規、電費新制、糾紛處理） | 目錄 |

---

## 旅遊

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **osaka-travel** | 大阪, 日本旅遊, 關西, 京都, 奈良 | 大阪日本旅遊指南 | 目錄 |

---

## 插件生態系統 (Plugin Skills)

透過 Claude Code 插件系統提供的 skills，需要對應插件啟用。

### 記憶與對話

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **episodic-memory:search-conversations** | 之前的對話, 上次討論, 記得嗎 | 搜尋過往對話記錄 |
| **episodic-memory:remembering-conversations** | 最佳做法, 之前怎麼做 | 恢復對話上下文 |

### Git 與版本控制

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **commit-commands:commit** | commit, 提交 | 建立 git commit |
| **commit-commands:commit-push-pr** | commit push PR, 提交推送 | 提交、推送並開 PR |
| **commit-commands:clean_gone** | 清理分支, gone branches | 清理已刪除的遠端分支 |

### 程式碼審查

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **code-review:code-review** | review PR, 審查 PR | 審查 Pull Request |

### 功能開發

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **feature-dev:feature-dev** | 功能開發, 新功能, feature | 引導式功能開發 |

### Agent SDK 開發

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **agent-sdk-dev:new-sdk-app** | Agent SDK, 建立代理應用 | 建立 Claude Agent SDK 應用 |

### Plugin 開發

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **plugin-dev:create-plugin** | 建立插件, create plugin | 端對端插件建立工作流 |
| **plugin-dev:plugin-structure** | 插件結構, scaffold plugin | 插件結構腳手架 |
| **plugin-dev:skill-development** | 建立 skill, add skill | 在插件中建立 skill |
| **plugin-dev:agent-development** | 建立 agent, add agent | 在插件中建立 agent |
| **plugin-dev:command-development** | 建立 command, slash command | 建立斜線命令 |
| **plugin-dev:hook-development** | 建立 hook, PreToolCall | 建立 hook |
| **plugin-dev:mcp-integration** | 整合 MCP, add MCP server | 整合 MCP 伺服器 |

### Sentry 監控

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **sentry:getIssues** | Sentry issues, 錯誤列表 | 取得 Sentry issues |
| **sentry:seer** | Sentry 分析, 自然語言查詢 | 自然語言查詢 Sentry |
| **sentry:sentry-code-review** | Sentry PR review | 分析 PR 上的 Sentry 評論 |
| **sentry:sentry-setup-tracing** | Sentry tracing, 效能監控 | 設定 Sentry 效能追蹤 |
| **sentry:sentry-setup-logging** | Sentry logging | 設定 Sentry 日誌 |

### Pinecone 向量資料庫

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **pinecone:help** | Pinecone 幫助 | Pinecone 使用說明 |
| **pinecone:quickstart** | Pinecone 快速開始 | Pinecone 快速入門 |
| **pinecone:query** | Pinecone 查詢 | 查詢 Pinecone 索引 |
| **pinecone:assistant** | Pinecone Assistant | Pinecone Assistant 操作 |

### Hugging Face

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **huggingface-skills:hugging-face-cli** | hf CLI, Hugging Face CLI | 執行 Hugging Face Hub 操作 |
| **huggingface-skills:hugging-face-model-trainer** | 訓練模型, fine-tune HF | 訓練或微調語言模型 |
| **huggingface-skills:hugging-face-jobs** | HF Jobs, 運算任務 | 在 HF 上執行運算任務 |

### Figma 設計

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **figma:implement-design** | Figma 實作, 設計轉程式碼 | 將 Figma 設計轉為程式碼 |
| **figma:code-connect-components** | Code Connect, 連接元件 | 連接 Figma 與程式碼元件 |

### Superpowers 增強技能

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **superpowers:brainstorming** | 腦力激盪, 創意發想 | 創意工作前必用 |
| **superpowers:writing-plans** | 撰寫計畫 | 撰寫實作計畫 |
| **superpowers:test-driven-development** | TDD | 測試驅動開發 |
| **superpowers:systematic-debugging** | 系統化除錯 | 遇到 bug 時使用 |

---

## 快速選擇指南

### 自動化開發任務
1. **clauderun** - 組合執行多個 skills（@fullreview、@tddflow）
2. **codexrun** - 快速生成程式碼（#api、#scraper）
3. **rloop** - 自主迭代開發迴圈（PRD 轉專案）
4. **issue-resolver** - 自動診斷修復問題並回報

### 建構 RAG 系統
1. **rag** - RAG 核心概念與最佳實踐
2. **langchain** - LangChain 實作範例
3. **pinecone:quickstart** - Pinecone 向量資料庫
4. **mistral** 或 **groq** - Embeddings 與推理

### 開發 AI 代理
1. **langchain** - 單代理與工具整合
2. **autogen** - 多代理協作（Teams, Swarm, MagenticOne）
3. **dspy** - Prompt 自動優化

### 微調 LLM
1. **unsloth** - 快速 LoRA/QLoRA 微調
2. **deep-learning** - 理解底層架構
3. **huggingface-skills:hugging-face-model-trainer** - HF 模型訓練

### 部署 LLM 推理服務
1. **vllm** - 高吞吐量推理（PagedAttention）
2. **groq** - 超低延遲推理（LPU 硬體）
3. **mistral** - Mistral 模型 API

### 遇到 Bug
1. **systematic-debugging** - 系統化除錯
2. **chrome-devtools** - 前端 JavaScript 調試
3. **vue-devtools** - Vue.js 應用調試
4. **sentry:seer** - 使用 Sentry 分析錯誤

### 審查程式碼
1. **code-reviewer** - 品質/安全審查
2. **hardcode-detector** - 硬編碼檢測
3. **code-review:code-review** - 審查 PR

### 建立 UI
1. **react** - React 框架
2. **tiptap** - 富文本編輯器
3. **ui-color-optimizer** - 配色與無障礙（WCAG）
4. **frontend-design** - 獨特設計
5. **figma:implement-design** - Figma 設計轉程式碼

### 追蹤 AI 技術動態
1. **ai-tech-digest** - AI 公司官方新聞（Anthropic, OpenAI, Google）
2. **hackernews-ai-digest** - Hacker News AI 新聞（含中文翻譯）
3. **ai-daily-report** - 每日/每週 AI 新聞整理

### 任務管理與通知
1. **todoist** - Todoist 任務查詢與管理
2. **daily-digest-notifier** - 整合行事曆 + 待辦 + 通知
3. **ntfy-notify** - 任務完成推播通知

### 查詢個人知識庫
1. **knowledge-query** - 通用知識庫查詢（PowerShell API）
2. **code-assistant** - 程式開發相關查詢

### 查詢屏東縣政府新聞
1. **pingtung-news** - 即時新聞查詢（MCP 服務）
2. **pingtung-policy-expert** - 縣政政策分析

### 建立知識庫 / Skill
1. **skill-seekers** - 從文件網站建立 Skill
2. **skill-seekers-claude-code** - 完整程式碼版本
3. **skill-creator** - Skill 建立指南

### 配置 Claude Code
1. **claude-md-guide** - CLAUDE.md 三層架構設計
2. **plugin-dev:create-plugin** - 建立 Claude Code 插件

---

## 組合使用建議

| 任務類型 | 建議 Skill 組合 |
|---------|----------------|
| 自動化開發迴圈 | rloop + test-driven-development |
| 問題診斷修復 | issue-resolver + systematic-debugging |
| 組合任務執行 | clauderun + ntfy-notify |
| 建構 RAG 系統 | rag + langchain + pinecone |
| 個人知識庫查詢 | knowledge-query + code-assistant |
| 多代理系統 | autogen + groq/mistral |
| 模型微調後部署 | unsloth + vllm |
| 富文本編輯器 | tiptap + react |
| AI 技術追蹤系統 | ai-tech-digest + hackernews-ai-digest + rag |
| 每日任務通知 | daily-digest-notifier + todoist + ntfy-notify |
| 前端調試工作流 | chrome-devtools + vue-devtools + playwright |
| 帳號系統開發 | account-management-system + account-manager |
| Gemini 學習網站 | gemini-interactive-learning + react + course-architect |

---

## 使用方式

### 手動調用

```
/rag /langchain /tiptap /react /vllm /atomic-habits /autogen
/clauderun /codexrun /rloop /ntfy-notify /todoist /chrome-devtools
/playwright /claude-md-guide /issue-resolver /hackernews-ai-digest
```

### 自動觸發

Skills 會根據 description 中的關鍵字自動觸發。例如：
- 提到 **查詢知識庫** 或 **筆記查詢** → 自動觸發 knowledge-query skill
- 提到 **程式輔助** 或 **useCallback** → 自動觸發 code-assistant skill
- 提到 **RAG** 或 **向量資料庫** → 自動觸發 rag skill
- 提到 **DevTools** 或 **開發者工具** → 自動觸發 chrome-devtools skill
- 提到 **Hacker News** 或 **HN** → 自動觸發 hackernews-ai-digest skill
- 提到 **CLAUDE.md** 或 **配置文件** → 自動觸發 claude-md-guide skill
- 提到 **租賃契約** 或 **租約** → 自動觸發 rental-contract-guide skill
- 提到 **處理問題** 或 **修復bug** → 自動觸發 issue-resolver skill
- 提到 **todoist** 或 **待辦事項** → 自動觸發 todoist skill
- 提到 **屏東新聞** 或 **屏東縣政府** → 自動觸發 pingtung-news skill

---

## 維護說明

### 新增 Skill

**目錄型 Skill：**
1. 在 skills/ 目錄下建立新資料夾
2. 建立 SKILL.md 並填寫 frontmatter
3. 建立 references/ 子目錄存放參考文件
4. 更新此索引文件

### Skill 結構

```
skills/
├── skill-name/              # 目錄型
│   ├── SKILL.md               # 主要說明（必須）
│   └── references/            # 參考文件（選用）
├── skill-name_instruction.md  # 指令型（獨立檔案）
└── SKILLS_INDEX.md            # 本索引檔
```

---

## 版本歷史

| 版本 | 日期 | 更新內容 |
|-----|------|---------|
| v4.2 | 2026-02-03 | 新增 pingtung-news（屏東新聞 MCP 服務）；新增「政府與地方資訊」分類 |
| v4.1 | 2026-02-03 | 新增 knowledge-query（RAG 知識庫查詢）、code-assistant（程式輔助）；優化知識庫建構分類 |
| v4.0 | 2026-01-31 | 新增 14 個 skills：issue-resolver, skill-seekers 多版本, todoist, daily-digest-notifier, hackernews-ai-digest, ai-tech-digest, chrome-devtools, vue-devtools, playwright, claude-md-guide, rental-contract-guide, gemini-interactive-learning |
| v3.0 | 2026-01-31 | 新增自動化/任務執行類別、插件生態系統文件 |
| v2.0 | 2026-01-10 | 新增 rloop、course-architect、ntfy-notify |
| v1.0 | 2025-12-01 | 初始版本 |
