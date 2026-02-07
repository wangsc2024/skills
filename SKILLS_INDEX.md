# Claude Code Skills 索引

此索引幫助快速選擇正確的 Skill，並提供組合使用建議與自動觸發規則。

> **更新時間:** 2026-02-07
> **版本:** v5.1
> **路徑:** `D:\Source\skills\`

---

## 目錄

- [AI/ML 框架與推理](#aiml-框架與推理)
- [開發流程與調試](#開發流程與調試)
- [瀏覽器開發者工具](#瀏覽器開發者工具)
- [前端與設計](#前端與設計)
- [視覺藝術](#視覺藝術)
- [文件處理](#文件處理)
- [帳號與權限](#帳號與權限)
- [寫作與溝通](#寫作與溝通)
- [學習與自我成長](#學習與自我成長)
- [資訊整合與追蹤](#資訊整合與追蹤)
- [任務管理與通知](#任務管理與通知)
- [資料擷取與處理](#資料擷取與處理)
- [知識庫建構與查詢](#知識庫建構與查詢)
- [Claude Code 配置](#claude-code-配置)
- [政府與地方資訊](#政府與地方資訊)
- [專業領域](#專業領域)
- [插件生態系統](#插件生態系統-plugin-skills)

---

## AI/ML 框架與推理

建構 AI 應用、多代理系統、LLM 訓練、推理與部署相關技能。

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **langchain** | LangChain, LCEL, agents, RAG, memory | LangChain AI 代理與 RAG 管道 |
| **dspy** | DSPy, signatures, teleprompters, MIPRO | Stanford 宣告式 Prompt 優化 |
| **autogen** | AutoGen, multi-agent, AgentChat, teams, Swarm | Microsoft 多代理協作框架 |
| **unsloth** | fine-tune, LoRA, QLoRA, GRPO, vision | 2-5x 快速 LLM 微調 |
| **vllm** | vLLM, inference, serving, PagedAttention | 高吞吐量 LLM 推理與部署 |
| **mistral** | Mistral AI, pixtral, embeddings | Mistral 模型 API 與 SDK |
| **groq** | Groq, LPU, fast inference, Whisper | Groq 超快推理與語音轉文字 |
| **rag** | RAG, 向量資料庫, embeddings, chunking | 檢索增強生成完整指南 |
| **llamaindex-ts** | LlamaIndex, RAG TypeScript, LlamaParse | TypeScript LLM 應用框架 |
| **deep-learning** | CNN, RNN, LSTM, Transformer, Dropout | 深度學習核心架構與技術 |
| **context7** | context7, 查詢文檔, 查最新文件 | 即時查詢最新函式庫文檔與範例 |

---

## 開發流程與調試

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **test-driven-development** | TDD, 測試驅動, 寫測試, unit test | 測試先行開發 |
| **systematic-debugging** | debug, bug, error, 除錯, exception | 系統化除錯 |
| **writing-plans** | 計畫, plan, 規劃, 步驟, 任務拆解 | 建立實作計畫 |
| **executing-plans** | 執行計畫, follow plan, 按照計畫 | 執行實作計畫 |
| **planning-with-files** | task plan, manus workflow, 三檔案模式 | Manus 風格檔案規劃 |
| **software-architect** | 架構, architecture, SOLID, design pattern | 架構設計 |
| **code-reviewer** | review, 審查, PR review, 檢查程式碼 | 程式碼審查 |
| **hardcode-detector** | hardcode, 寫死, magic number, secrets | 硬編碼檢測 |
| **git-workflow** | git, commit, branch, merge, PR | Git 工作流程 |
| **mcp-builder** | MCP, Model Context Protocol, Claude擴展 | MCP Server 建構 |
| **webapp-testing** | E2E, 端對端測試, browser test | 網頁自動化測試 |
| **playwright** | Playwright, 瀏覽器自動化, web testing | Playwright 瀏覽器自動化框架 |
| **issue-resolver-skill** | 處理問題, 修復bug, 待辦, 回報修復 | 自動化問題診斷與修復 |

---

## 瀏覽器開發者工具

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **chrome-devtools** | Chrome DevTools, 開發者工具, console | Chrome 瀏覽器調試、效能分析 |
| **vue-devtools** | Vue DevTools, Vue 調試, Vuex, Pinia | Vue.js 應用調試與監控 |

---

## 前端與設計

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **react** | React, hooks, components, state, JSX | React 框架開發 |
| **tiptap** | Tiptap, ProseMirror, rich text, WYSIWYG | 富文本編輯器 |
| **frontend-ui-tools** | React component, 元件, Button, Modal | React 元件生成 |
| **frontend-design** | web design, UI設計, 網頁設計, landing page | 獨特前端設計 |
| **ui-color-optimizer** | color, 配色, dark mode, WCAG | 配色優化與無障礙設計 |
| **theme-factory** | theme, 主題, 風格, style | 主題套用 |
| **web-artifacts-builder** | shadcn, web app, 複雜應用, React專案 | 複雜前端應用 |
| **gemini-interactive-learning** | Gemini 學習, 互動式教學, AI 教學網站 | Gemini 互動學習網站（Next.js） |

---

## 視覺藝術

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **algorithmic-art** | p5.js, SVG, generative art, 生成藝術 | 演算法藝術 |
| **canvas-design** | poster, 海報, 視覺設計, artwork | 視覺設計作品 |

---

## 文件處理

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **docx** | Word, docx, 文件, 報告, 合約 | Word 文件 |
| **pdf** | PDF, 合併PDF, OCR, 表單 | PDF 處理 |
| **qpdf** | qpdf, PDF split, PDF merge, 分割PDF | 命令行 PDF 操作（分割、合併、加密） |
| **pptx** | PowerPoint, 簡報, slides, 投影片 | 簡報製作 |

---

## 帳號與權限

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **account-manager** | 帳號管理, RBAC, 權限, 認證 | 帳號與權限管理基礎 |
| **account-management-system** | JWT, Firebase Auth, 用戶 CRUD | 完整帳號管理系統 |

---

## 寫作與溝通

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **writing-masters** | 海明威, 極簡, 冰山理論, 寫作技巧 | 西方文學大師寫作技巧 |
| **storytelling-masters** | 說故事, 簡報, 英雄旅程, 影響力 | 說故事與簡報技藝 |
| **internal-comms** | status report, 報告, 週報 | 內部溝通 |

---

## 學習與自我成長

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **learning-mastery** | 學習方法, 記憶, 費曼技巧, Anki | 楊大輝五層學習框架 |
| **atomic-habits** | 習慣, 原子習慣, 行為改變, 兩分鐘法則 | James Clear 原子習慣 |
| **course-architect** | 課程設計, 學習路徑, 教學設計 | 課程安排大師 |
| **entropy-theory** | 熵, entropy, 信息熵, 交叉熵, KL散度 | 熵理論與應用 |

---

## 資訊整合與追蹤

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **ai-tech-digest** | AI 技術日報, Claude 新聞, Gemini 新聞 | AI 技術日報追蹤（Anthropic, OpenAI, Google） |
| **hackernews-ai-digest** | Hacker News, HN, AI news, 新聞摘要 | HN AI 新聞抓取與中文翻譯 |

---

## 任務管理與通知

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **ntfy-notify** | 通知, 提醒, 完成後通知, ntfy | 透過 ntfy.sh 發送任務完成通知 |
| **todoist** | todoist, 待辦事項, todo, 今日任務 | Todoist API 整合 |
| **daily-digest-notifier** | daily digest, 今日摘要, 行事曆通知 | Google Calendar + Todoist + ntfy |

---

## 資料擷取與處理

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **web-reader** | 讀取網頁, 抓取網頁, fetch, scrape | Python requests 網頁讀取 |

---

## 知識庫建構與查詢

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **knowledge-query** | 查詢知識庫, 筆記查詢, RAG 查詢 | 個人知識庫查詢 |
| **code-assistant** | 程式輔助, 程式碼查詢 | 基於知識庫的程式開發輔助 |
| **skill-creator** | create skill, 建立skill, SKILL.md | Skill 建立指南 |
| **skill-seekers** | 從文件建立 Skill, 文件轉 Skill | 技術文件轉 AI 知識庫（通用版） |
| **skill-seekers-claude-code** | 抓取文件, GitHub 倉庫分析 | Skill Seekers Claude Code 版本 |
| **skill-seekers-claude-chat** | 整理文件, 知識庫助手 | Skill Seekers Claude Chat 版本 |

---

## Claude Code 配置

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **claude-md-guide** | CLAUDE.md, 配置文件, Claude Code 配置 | CLAUDE.md 三層配置架構設計 |

---

## 政府與地方資訊

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **pingtung-news** | 屏東新聞, 屏東縣政府, 周春米 | 屏東縣政府新聞查詢（MCP） |
| **pingtung-policy-expert** | 屏東政策, 屏東施政, 縣政方針 | 屏東縣政策與施政專家 |
| **government-document-expert** | 公文, 簽, 函, 簽呈, 公文撰寫 | 台灣政府機關公文撰寫 |
| **government-policy-report** | 施政總報告, 議會報告, 五箭齊發 | 地方政府施政總報告 |

---

## 專業領域

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **rental-contract-guide** | 租賃契約, 租約, 房東, 押金, 電費 | 台灣租賃契約專家 |
| **taiwan-cybersecurity** | 台灣資安, 資通安全管理法, TWCERT | 台灣資通訊安全法規與防護 |
| **wsc-relay** | Gun.js, relay server, 分散式同步 | Gun.js 中繼伺服器指南 |
| **api-designer** | API, RESTful, GraphQL, OpenAPI | API 設計與文檔 |

---

## 插件生態系統 (Plugin Skills)

透過 Claude Code 插件系統提供的 skills，需要對應插件啟用。

### Git 與版本控制

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **commit-commands:commit** | commit, 提交 | 建立 git commit |
| **commit-commands:commit-push-pr** | commit push PR | 提交、推送並開 PR |
| **commit-commands:clean_gone** | 清理分支, gone branches | 清理已刪除的遠端分支 |

### 程式碼審查與功能開發

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **code-review:code-review** | review PR, 審查 PR | 審查 Pull Request |
| **feature-dev:feature-dev** | 功能開發, 新功能, feature | 引導式功能開發 |

### Agent SDK 與 Plugin 開發

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **agent-sdk-dev:new-sdk-app** | Agent SDK, 建立代理應用 | 建立 Claude Agent SDK 應用 |
| **plugin-dev:create-plugin** | 建立插件, create plugin | 端對端插件建立工作流 |
| **plugin-dev:skill-development** | 建立 skill, add skill | 在插件中建立 skill |
| **plugin-dev:agent-development** | 建立 agent, add agent | 在插件中建立 agent |

### 外部服務整合

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **pinecone:assistant** | Pinecone Assistant | Pinecone Assistant 操作 |
| **pinecone:quickstart** | Pinecone 快速開始 | Pinecone 快速入門 |
| **huggingface-skills:hugging-face-model-trainer** | 訓練模型, fine-tune HF | 訓練或微調語言模型 |

### Superpowers 增強技能

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **superpowers:brainstorming** | 腦力激盪, 創意發想 | 創意工作前必用 |
| **superpowers:writing-plans** | 撰寫計畫 | 撰寫實作計畫 |
| **superpowers:test-driven-development** | TDD | 測試驅動開發 |
| **superpowers:systematic-debugging** | 系統化除錯 | 遇到 bug 時使用 |

---

## 快速選擇指南

### 建構 RAG 系統
1. **rag** - 核心概念與最佳實踐
2. **langchain** - Python 實作
3. **llamaindex-ts** - TypeScript 實作
4. **pinecone:quickstart** - 向量資料庫
5. **context7** - 即時查詢最新文檔

### 開發 AI 代理
1. **langchain** - 單代理與工具整合
2. **autogen** - 多代理協作
3. **dspy** - Prompt 自動優化

### 微調與部署 LLM
1. **unsloth** - 快速 LoRA/QLoRA 微調
2. **vllm** - 高吞吐量推理
3. **groq** - 超低延遲推理

### 遇到 Bug
1. **systematic-debugging** - 系統化除錯
2. **chrome-devtools** - 前端調試
3. **vue-devtools** - Vue.js 調試

### 建立 UI
1. **react** - React 框架
2. **tiptap** - 富文本編輯器
3. **ui-color-optimizer** - 配色與 WCAG
4. **frontend-design** - 獨特設計

### 任務管理與通知
1. **todoist** - 任務管理
2. **daily-digest-notifier** - 每日摘要
3. **ntfy-notify** - 推播通知

### 撰寫政府公文
1. **government-document-expert** - 公文撰寫
2. **government-policy-report** - 施政總報告

---

## 組合使用建議

| 任務類型 | 建議 Skill 組合 |
|---------|----------------|
| 建構 RAG 系統（Python） | rag + langchain + pinecone |
| 建構 RAG 系統（TypeScript） | llamaindex-ts + pinecone + context7 |
| 多代理系統 | autogen + groq/mistral |
| 模型微調後部署 | unsloth + vllm |
| 富文本編輯器 | tiptap + react |
| 每日任務通知 | daily-digest-notifier + todoist + ntfy-notify |
| 前端調試工作流 | chrome-devtools + vue-devtools + playwright |
| 帳號系統開發 | account-management-system + account-manager |
| 複雜任務規劃 | planning-with-files + writing-plans + executing-plans |
| 政府公文系統 | government-document-expert + government-policy-report |
| PDF 批量處理 | pdf + qpdf |
| 機器學習理論 | deep-learning + entropy-theory |
| 網頁資料抓取 | web-reader + hackernews-ai-digest |
| 問題診斷修復 | issue-resolver-skill + systematic-debugging |

---

## 使用方式

### 手動調用

```
/rag /langchain /llamaindex-ts /tiptap /react /vllm /autogen
/ntfy-notify /todoist /chrome-devtools /playwright /claude-md-guide
/context7 /planning-with-files /qpdf /taiwan-cybersecurity /web-reader
/government-document-expert /government-policy-report /entropy-theory
```

### 自動觸發

Skills 會根據 description 中的關鍵字自動觸發：
- **RAG** / **向量資料庫** → rag skill
- **LlamaIndex** / **LlamaParse** → llamaindex-ts skill
- **context7** / **查最新文件** → context7 skill
- **DevTools** / **開發者工具** → chrome-devtools skill
- **公文** / **簽呈** → government-document-expert skill
- **熵** / **entropy** → entropy-theory skill
- **todoist** / **待辦事項** → todoist skill

---

## 維護說明

### Skill 結構

```
D:\Source\skills\
├── skill-name/              # 目錄型
│   ├── SKILL.md               # 主要說明（必須）
│   └── references/            # 參考文件（選用）
└── SKILLS_INDEX.md            # 本索引檔
```

### 新增 Skill

1. 在 skills/ 目錄下建立新資料夾
2. 建立 SKILL.md 並填寫 frontmatter
3. 建立 references/ 子目錄存放參考文件
4. 更新此索引文件

---

## 版本歷史

| 版本 | 日期 | 更新內容 |
|-----|------|---------|
| v5.1 | 2026-02-07 | 路徑調整至 D:\Source\skills；移除不存在的 skills（clauderun, codexrun, rloop, ralph-loop, ai-daily-report, osaka-travel）；修正 issue-resolver → issue-resolver-skill；精簡索引結構 |
| v5.0 | 2026-02-07 | 新增 9 個 skills：context7、entropy-theory、llamaindex-ts 等 |
| v4.2 | 2026-02-03 | 新增 pingtung-news（屏東新聞 MCP 服務） |
| v4.0 | 2026-01-31 | 新增 14 個 skills |
| v3.0 | 2026-01-31 | 新增自動化/任務執行類別、插件生態系統文件 |
| v1.0 | 2025-12-01 | 初始版本 |
