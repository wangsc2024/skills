# Claude Code Skills 索引

此索引幫助快速選擇正確的 Skill，並提供組合使用建議與自動觸發規則。

> **更新時間:** 2026-02-07
> **版本:** v6.1
> **路徑:** `D:\Source\skills\`

---

## 目錄

### 開發專案用
- [AI/ML 框架](#aiml-框架)
- [開發流程](#開發流程)
- [測試與調試](#測試與調試)
- [前端開發](#前端開發)
- [後端與 API](#後端與-api)
- [文件處理](#文件處理)
- [知識庫建構](#知識庫建構)

### 系統執行時用
- [任務自動化](#任務自動化)
- [資訊追蹤與通知](#資訊追蹤與通知)
- [政府與公文](#政府與公文)
- [專業領域](#專業領域)
- [個人成長](#個人成長)

### 共用
- [Claude Code 配置](#claude-code-配置)
- [插件生態系統](#插件生態系統-plugin-skills)

---

# 開發專案用

開發軟體專案時使用的技能，包含框架、流程、測試、前後端開發。

---

## AI/ML 框架

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

## 開發流程

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **writing-plans** | 計畫, plan, 規劃, 步驟, 任務拆解 | 建立實作計畫 |
| **executing-plans** | 執行計畫, follow plan, 按照計畫 | 執行實作計畫 |
| **planning-with-files** | task plan, manus workflow, 三檔案模式 | Manus 風格檔案規劃 |
| **software-architect** | 架構, architecture, SOLID, design pattern | 架構設計 |
| **code-reviewer** | review, 審查, PR review, 檢查程式碼 | 程式碼審查 |
| **hardcode-detector** | hardcode, 寫死, magic number, secrets | 硬編碼檢測 |
| **git-workflow** | git, commit, branch, merge, PR | Git 工作流程 |
| **mcp-builder** | MCP, Model Context Protocol, Claude擴展 | MCP Server 建構 |

---

## 測試與調試

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **test-driven-development** | TDD, 測試驅動, 寫測試, unit test | 測試先行開發 |
| **systematic-debugging** | debug, bug, error, 除錯, exception | 系統化除錯 |
| **playwright** | Playwright, E2E, 瀏覽器自動化, web testing, browser test | Playwright 瀏覽器自動化框架（支援 TS/Python） |
| **chrome-devtools** | Chrome DevTools, 開發者工具, console | Chrome 瀏覽器調試、效能分析 |
| **vue-devtools** | Vue DevTools, Vue 調試, Vuex, Pinia | Vue.js 應用調試與監控 |
| **issue-resolver-skill** | 處理問題, 修復bug, 待辦, 回報修復 | 自動化問題診斷與修復 |

---

## 前端開發

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **react** | React, hooks, components, state, JSX | React 框架開發 |
| **tiptap** | Tiptap, ProseMirror, rich text, WYSIWYG | 富文本編輯器 |
| **frontend-ui-tools** | React component, 元件, Button, Modal | React 元件生成 |
| **frontend-design** | web design, UI設計, 網頁設計, landing page | 獨特前端設計 |
| **ui-color-optimizer** | color, 配色, dark mode, WCAG | 配色優化與無障礙設計 |
| **theme-factory** | theme, 主題, 風格, style | 主題套用 |
| **web-artifacts-builder** | shadcn, web app, 複雜應用, React專案 | 複雜前端應用 |
| **algorithmic-art** | p5.js, SVG, generative art, 生成藝術 | 演算法藝術 |
| **canvas-design** | poster, 海報, 視覺設計, artwork | 視覺設計作品 |
| **ui-ux-pro-max** | UI/UX, design system, glassmorphism, landing page | 完整 UI/UX 設計智慧（50+ 風格、97 色盤、9 框架） |

---

## 後端與 API

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **api-designer** | API, RESTful, GraphQL, OpenAPI | API 設計與文檔 |
| **account-management-system** | JWT, Firebase Auth, 用戶 CRUD, RBAC, 權限 | 完整帳號管理系統（含 RBAC 權限） |
| **wsc-relay** | Gun.js, relay server, 分散式同步 | Gun.js 中繼伺服器指南 |

---

## 文件處理

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **docx** | Word, docx, 文件, 報告, 合約 | Word 文件 |
| **pdf** | PDF, 合併PDF, OCR, 表單 | PDF 處理 |
| **qpdf** | qpdf, PDF split, PDF merge, 分割PDF | 命令行 PDF 操作（分割、合併、加密） |
| **pptx** | PowerPoint, 簡報, slides, 投影片 | 簡報製作 |

---

## 知識庫建構

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **skill-creator** | create skill, 建立skill, SKILL.md | Skill 建立指南 |
| **skill-seekers** | 從文件建立 Skill, 文件轉 Skill, 抓取文件, GitHub 倉庫分析 | 技術文件轉 AI 知識庫（整合版） |

---

# 系統執行時用

日常工作與系統運行時使用的技能，包含自動化、通知、資訊追蹤。

---

## 任務自動化

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **nanoclaw-skills** | nanoclaw, agent, 自動化代理, 知識助手, 市場分析 | AI Agent 技能集合（安全強化版） |
| **todoist** | todoist, 待辦事項, todo, 今日任務 | Todoist API 整合 |
| **daily-greeting-digest** | hi, hello, 嗨, 哈囉, 早安, 午安, 安安 | 時段摘要（早：待辦+天氣+新聞；午：HN+股價；下午：新聞+天氣） |
| **daily-digest-notifier** | daily digest, 今日摘要, 行事曆通知 | Google Calendar + Todoist + ntfy |
| **ntfy-notify** | 通知, 提醒, 完成後通知, ntfy | 透過 ntfy.sh 發送任務完成通知 |
| **gemini-interactive-learning** | Gemini 學習, 互動式教學, AI 教學網站 | Gemini 互動學習網站（Next.js） |

---

## 資訊追蹤與通知

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **ai-tech-digest** | AI 技術日報, Claude 新聞, Gemini 新聞 | AI 技術日報追蹤（Anthropic, OpenAI, Google） |
| **hackernews-ai-digest** | Hacker News, HN, AI news, 新聞摘要 | HN AI 新聞抓取與中文翻譯 |

---

## 政府與公文

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **government-document-expert** | 公文, 簽, 函, 簽呈, 公文撰寫 | 台灣政府機關公文撰寫 |
| **government-policy-report** | 施政總報告, 議會報告, 五箭齊發 | 地方政府施政總報告 |
| **pingtung-policy-expert** | 屏東政策, 屏東施政, 縣政方針 | 屏東縣政策與施政專家 |

---

## 專業領域

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **rental-contract-guide** | 租賃契約, 租約, 房東, 押金, 電費 | 台灣租賃契約專家 |
| **taiwan-cybersecurity** | 台灣資安, 資通安全管理法, TWCERT | 台灣資通訊安全法規與防護 |
| **internal-comms** | status report, 報告, 週報 | 內部溝通 |

---

## 個人成長

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **learning-mastery** | 學習方法, 記憶, 費曼技巧, Anki | 楊大輝五層學習框架 |
| **atomic-habits** | 習慣, 原子習慣, 行為改變, 兩分鐘法則 | James Clear 原子習慣 |
| **course-architect** | 課程設計, 學習路徑, 教學設計 | 課程安排大師 |
| **entropy-theory** | 熵, entropy, 信息熵, 交叉熵, KL散度 | 熵理論與應用 |
| **writing-masters** | 海明威, 極簡, 冰山理論, 寫作技巧 | 西方文學大師寫作技巧 |
| **storytelling-masters** | 說故事, 簡報, 英雄旅程, 影響力 | 說故事與簡報技藝 |

---

# 共用

---

## Claude Code 配置

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **claude-md-guide** | CLAUDE.md, 配置文件, Claude Code 配置 | CLAUDE.md 三層配置架構設計 |

---

## 插件生態系統 (Plugin Skills)

透過 Claude Code 插件系統提供的 skills，需要對應插件啟用。

### 開發專案用插件

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **commit-commands:commit** | commit, 提交 | 建立 git commit |
| **commit-commands:commit-push-pr** | commit push PR | 提交、推送並開 PR |
| **commit-commands:clean_gone** | 清理分支, gone branches | 清理已刪除的遠端分支 |
| **code-review:code-review** | review PR, 審查 PR | 審查 Pull Request |
| **feature-dev:feature-dev** | 功能開發, 新功能, feature | 引導式功能開發 |
| **agent-sdk-dev:new-sdk-app** | Agent SDK, 建立代理應用 | 建立 Claude Agent SDK 應用 |
| **plugin-dev:create-plugin** | 建立插件, create plugin | 端對端插件建立工作流 |
| **plugin-dev:skill-development** | 建立 skill, add skill | 在插件中建立 skill |
| **plugin-dev:agent-development** | 建立 agent, add agent | 在插件中建立 agent |
| **superpowers:test-driven-development** | TDD | 測試驅動開發 |
| **superpowers:systematic-debugging** | 系統化除錯 | 遇到 bug 時使用 |
| **superpowers:writing-plans** | 撰寫計畫 | 撰寫實作計畫 |

### 系統執行時用插件

| Skill | 觸發詞 | 用途 |
|-------|--------|------|
| **pinecone:assistant** | Pinecone Assistant | Pinecone Assistant 操作 |
| **pinecone:quickstart** | Pinecone 快速開始 | Pinecone 快速入門 |
| **huggingface-skills:hugging-face-model-trainer** | 訓練模型, fine-tune HF | 訓練或微調語言模型 |
| **superpowers:brainstorming** | 腦力激盪, 創意發想 | 創意工作前必用 |

---

## Skill 依賴圖譜

Skills 之間的依賴與關聯關係，幫助理解如何組合使用。

### 開發流程依賴鏈

```
writing-plans ──┬──▶ executing-plans ──▶ code-reviewer ──▶ git-workflow
                │
planning-with-files ◀──┘

software-architect ──▶ writing-plans
                  └──▶ api-designer
```

### 測試與調試依賴鏈

```
test-driven-development ◀──▶ systematic-debugging
        │                           │
        ▼                           ▼
  playwright ◀─────────────── chrome-devtools
        │                           │
        └───────▶ vue-devtools ◀────┘

issue-resolver-skill ──▶ systematic-debugging ──▶ test-driven-development
```

### AI/ML 開發依賴鏈

```
                    ┌──── langchain ────┐
                    │                   │
rag ────────────────┼──── autogen  ────▶ dspy
                    │                   │
                    └── llamaindex-ts ──┘
                            │
                            ▼
                        context7 (文檔查詢)

unsloth ──▶ vllm ──▶ groq (訓練→推理→部署)
```

### 前端開發依賴鏈

```
ui-ux-pro-max ──┬──▶ frontend-design ──┬──▶ react
                │                      │
                ├──▶ ui-color-optimizer│
                │                      ▼
                └──▶ theme-factory ──▶ web-artifacts-builder
                                           │
                                           ▼
                                    frontend-ui-tools
                                           │
                                           ▼
                                       tiptap (富文本)
```

### 知識庫建構依賴鏈

```
skill-seekers ──▶ skill-creator ──▶ context7
      │
      └──────────────────────────▶ rag (RAG 系統整合)
```

### 通知系統依賴鏈

```
daily-greeting-digest ──┬──▶ todoist
                        │
                        ├──▶ hackernews-ai-digest
                        │
                        └──▶ ai-tech-digest

daily-digest-notifier ──▶ todoist ──▶ ntfy-notify
```

---

## Skill 組合模式

### 模式一：TDD 開發循環

適用於：新功能開發、Bug 修復

```
1. writing-plans        # 規劃實作步驟
2. test-driven-development  # 寫測試（紅燈）
3. [實作程式碼]             # 實作（綠燈）
4. systematic-debugging     # 如遇問題則除錯
5. code-reviewer           # 程式碼審查
6. git-workflow            # 提交變更
```

### 模式二：RAG 系統開發

適用於：建構檢索增強生成應用

```
Python 版：
1. rag                # 了解 RAG 架構
2. langchain          # 使用 LangChain 框架
3. context7           # 查詢最新文檔
4. pinecone:assistant # 向量資料庫操作

TypeScript 版：
1. rag                # 了解 RAG 架構
2. llamaindex-ts      # 使用 LlamaIndex.TS
3. context7           # 查詢最新文檔
```

### 模式三：AI 代理開發

適用於：多代理協作系統

```
1. autogen            # Microsoft AutoGen 多代理
2. langchain          # LangChain 代理架構
3. dspy               # Prompt 優化
4. software-architect # 架構設計
```

### 模式四：前端專案完整開發

適用於：從零開始的前端專案

```
1. ui-ux-pro-max      # 設計系統選擇
2. frontend-design    # UI 設計方向
3. ui-color-optimizer # 配色方案
4. react              # React 開發
5. web-artifacts-builder # 組件開發
6. playwright         # E2E 測試
7. chrome-devtools    # 調試優化
```

### 模式五：LLM 微調與部署

適用於：自訂模型訓練與部署

```
1. unsloth            # 高效微調（LoRA/QLoRA）
2. vllm               # 推理服務部署
3. groq               # 生產環境推理
4. deep-learning      # 底層原理參考
```

### 模式六：每日工作開始

適用於：開啟一天的工作

```
1. daily-greeting-digest  # 時段摘要（自動觸發）
   └── 自動整合：todoist + 天氣 + 新聞 + HN
2. ai-tech-digest         # 查看 AI 最新動態（如需要）
```

### 模式七：問題診斷修復

適用於：生產環境問題處理

```
1. issue-resolver-skill   # 從待辦取得問題
2. systematic-debugging   # 四階段診斷法
3. test-driven-development # 寫回歸測試
4. code-reviewer          # 修復審查
5. ntfy-notify            # 完成後通知
```

### 模式八：政府公文作業

適用於：公務文書處理

```
1. government-document-expert  # 公文撰寫
2. government-policy-report    # 施政報告
3. pingtung-policy-expert      # 屏東縣特定（如適用）
4. docx                        # Word 文件處理
5. pdf                         # PDF 處理
```

### 模式九：知識庫建立

適用於：技術文件轉換

```
1. skill-seekers    # 抓取文檔/GitHub
2. skill-creator    # 生成 SKILL.md
3. context7         # 驗證文檔品質
```

### 模式十：學習效能提升

適用於：學習新技術/技能

```
1. learning-mastery   # 五層學習框架
2. atomic-habits      # 習慣養成
3. course-architect   # 課程規劃
4. entropy-theory     # 資訊理論應用
```

---

## 快速選擇指南

### 開發專案用

| 場景 | 建議 Skill 組合 |
|------|----------------|
| 建構 RAG 系統（Python） | rag + langchain + pinecone |
| 建構 RAG 系統（TypeScript） | llamaindex-ts + pinecone + context7 |
| 開發 AI 代理 | langchain + autogen + dspy |
| 微調與部署 LLM | unsloth + vllm + groq |
| 前端開發 | react + tiptap + ui-color-optimizer + frontend-design |
| 前端調試 | chrome-devtools + vue-devtools + playwright |
| 帳號系統開發 | account-management-system |
| 複雜任務規劃 | planning-with-files + writing-plans + executing-plans |
| 問題診斷修復 | issue-resolver-skill + systematic-debugging |
| PDF 批量處理 | pdf + qpdf |
| 知識庫建立 | skill-seekers + skill-creator |

### 系統執行時用

| 場景 | 建議 Skill 組合 |
|------|----------------|
| 每日時段問候 | daily-greeting-digest（自動整合 todoist + 天氣 + 新聞 + HN） |
| 每日任務通知 | daily-digest-notifier + todoist + ntfy-notify |
| AI 技術追蹤 | ai-tech-digest + hackernews-ai-digest |
| 政府公文系統 | government-document-expert + government-policy-report |
| 個人學習 | learning-mastery + atomic-habits + entropy-theory |

---

## 使用方式

### 開發專案用
```
/rag /langchain /llamaindex-ts /react /vllm /autogen /context7
/test-driven-development /systematic-debugging /playwright
/writing-plans /executing-plans /planning-with-files
/skill-seekers /skill-creator
```

### 系統執行時用
```
/todoist /ntfy-notify /daily-digest-notifier /daily-greeting-digest
/ai-tech-digest /hackernews-ai-digest
/government-document-expert /government-policy-report
/learning-mastery /atomic-habits
```

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

---

## 版本歷史

| 版本 | 日期 | 更新內容 |
|-----|------|---------|
| v6.1 | 2026-02-07 | 新增 Skill 依賴圖譜與 10 種組合模式 |
| v6.0 | 2026-02-07 | 整合 skills：移除 account-manager、web-reader；合併 webapp-testing→playwright、skill-seekers-claude-chat/code→skill-seekers |
| v5.4 | 2026-02-07 | 新增 ui-ux-pro-max 完整 UI/UX 設計智慧 |
| v5.3 | 2026-02-07 | 新增 daily-greeting-digest 時段摘要助手 |
| v5.2 | 2026-02-07 | 重新分類為「開發專案用」與「系統執行時用」兩大類 |
| v5.1 | 2026-02-07 | 路徑調整至 D:\Source\skills；移除不存在的 skills |
| v5.0 | 2026-02-07 | 新增 9 個 skills：context7、entropy-theory、llamaindex-ts 等 |
| v4.0 | 2026-01-31 | 新增 14 個 skills |
| v1.0 | 2025-12-01 | 初始版本 |
