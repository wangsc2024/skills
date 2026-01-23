# Claude Code Skills 索引

此索引幫助快速選擇正確的 Skill，並提供組合使用建議與自動觸發規則。

> **更新時間:** 2026-01-15
> **優化重點:** 強化觸發機制、補充使用指南、完善最佳實踐

---

## 🔄 Skills 類型說明

| 類型 | 說明 | 範例 |
|------|------|------|
| **目錄型** | 包含 SKILL.md + references/ 子目錄 | langchain, react, vllm |
| **指令型** | 獨立的 *_instruction.md 檔案 | ai_daily_report_instruction.md |
| **目錄+指令** | 兩者都有，提供更完整指引 | autogen, tiptap, ui-color-optimizer |
| **整合型** | 多個指令整合在一個檔案 | claude_project_instructions.md |

---

## 🤖 AI/ML 框架與推理

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

## 🔧 開發流程

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

## 🎨 前端與設計

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **react** | React, hooks, components, state, JSX | React 框架開發 | 目錄 |
| **tiptap** | Tiptap, ProseMirror, rich text, WYSIWYG, Node Views | 富文本編輯器 | 目錄+指令 |
| **frontend-ui-tools** | React, component, 元件, Button, Modal | React 元件生成 | 目錄 |
| **frontend-design** | web design, UI設計, 網頁設計, landing page | 獨特前端設計 | 目錄 |
| **ui-color-optimizer** | color, 配色, 顏色, dark mode, 對比度, WCAG, Material Design, Fluent | 配色優化與無障礙設計 | 目錄+指令 |
| **theme-factory** | theme, 主題, 風格, style | 主題套用 | 目錄 |
| **web-artifacts-builder** | shadcn, web app, 複雜應用, React專案 | 複雜前端應用 | 目錄 |

## 🖼️ 視覺藝術

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **algorithmic-art** | p5.js, SVG, generative art, 生成藝術 | 演算法藝術 | 目錄 |
| **canvas-design** | poster, 海報, 視覺設計, artwork | 視覺設計作品 | 目錄 |

## 📄 文件處理

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **docx** | Word, docx, 文件, 報告, 合約 | Word 文件 | 目錄 |
| **pdf** | PDF, 合併PDF, OCR, 表單 | PDF 處理 | 目錄 |
| **pptx** | PowerPoint, 簡報, slides, 投影片 | 簡報製作 | 目錄 |
| **rental-contract-expert** | 租賃契約, 租約, 電費新制, 押金上限 | 房屋租賃專業規範與檢核 | 目錄+指令 |

## 🛠️ 工具與整合

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **api-designer** | API, REST, endpoint, GraphQL, OpenAPI | API 設計 | 目錄 |
| **mcp-builder** | MCP, Model Context Protocol, Claude擴展 | MCP Server | 目錄 |
| **webapp-testing** | Playwright, E2E, 端對端測試, browser test | 網頁自動化測試 | 目錄 |
| **account-manager** | 帳號管理, RBAC, 權限, 認證, 登入登出 | 帳號與權限管理 | 目錄 |

## ✍️ 寫作與溝通

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **writing-masters** | 海明威, 極簡, 冰山理論, 寫作技巧 | 西方文學大師寫作技巧 | 目錄 |
| **storytelling-masters** | 說故事, 簡報, 英雄旅程, 影響力 | 說故事與簡報技藝 | 目錄 |
| **internal-comms** | status report, 報告, 週報, incident report | 內部溝通 | 目錄 |
| **skill-creator** | create skill, 建立skill, SKILL.md | Skill 建立 | 目錄 |

## 📚 學習與自我成長

個人成長、習慣養成、學習方法論相關技能。

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **learning-mastery** | 學習方法, 記憶, 費曼技巧, 間隔複習, Anki | 楊大輝《深度學習的技術》五層學習框架 | 目錄 |
| **atomic-habits** | 習慣, 原子習慣, 行為改變, 兩分鐘法則 | James Clear 原子習慣方法論 | 目錄 |

## 📰 資訊整合與追蹤

AI 技術新聞、產業動態追蹤相關技能。

| Skill | 觸發詞 | 用途 | 類型 |
|-------|--------|------|------|
| **ai-daily-report** | AI新聞, 日報, Claude更新, OpenAI, Gemini, Unsloth, 技術週報 | 每日 AI 技術新聞整理 | 指令 |

---

## 📋 指令型 Skills (Instruction Files)

獨立的指令檔案，提供完整的操作流程與模板。

| 指令檔案 | 對應 Skill | 用途 |
|---------|-----------|------|
| ai_daily_report_instruction.md | ai-daily-report | AI 技術日報整理流程與模板 |
| autogen_skill_instruction.md | autogen | AutoGen 多代理框架開發指南 |
| tiptap_skill_instruction.md | tiptap | Tiptap 富文本編輯器開發指南 |
| ui_color_optimizer_skill_instruction.md | ui-color-optimizer | UI 配色與無障礙設計指南 |
| claude_project_instructions.md | 多個 | 整合 React/前端/測試/主題指令 |

---

## 🚀 快速選擇指南

### 建構 RAG 系統
1. rag - RAG 核心概念與最佳實踐
2. langchain - LangChain 實作範例
3. mistral 或 groq - Embeddings 與推理

### 開發 AI 代理
1. langchain - 單代理與工具整合
2. autogen - 多代理協作（Teams, Swarm, MagenticOne）
3. dspy - Prompt 自動優化

### 微調 LLM
1. unsloth - 快速 LoRA/QLoRA 微調
2. deep-learning - 理解底層架構

### 部署 LLM 推理服務
1. vllm - 高吞吐量推理（PagedAttention）
2. groq - 超低延遲推理（LPU 硬體）
3. mistral - Mistral 模型 API

### 寫新功能
1. writing-plans - 建立計畫
2. test-driven-development - TDD 實作
3. executing-plans - 執行計畫

### 遇到 Bug
1. systematic-debugging - 系統化除錯

### 審查程式碼
1. code-reviewer - 品質/安全審查
2. hardcode-detector - 硬編碼檢測

### 建立 UI
1. react - React 框架
2. tiptap - 富文本編輯器
3. frontend-ui-tools - React 元件
4. ui-color-optimizer - 配色與無障礙（WCAG）
5. frontend-design - 獨特設計

### 處理文件
1. docx - Word
2. pdf - PDF
3. pptx - PowerPoint

### 商業溝通
1. storytelling-masters - 說故事技巧
2. writing-masters - 文字精煉

### 個人成長與學習
1. atomic-habits - 習慣養成四大法則
2. learning-mastery - 深度學習五層框架

### 追蹤 AI 技術動態
1. ai-daily-report - 每日/每週 AI 新聞整理

---

## 📊 AI Provider / 推理引擎 選擇指南

| 需求 | 推薦方案 | 原因 |
|-----|---------|------|
| 最低延遲 | Groq | LPU 硬體，280-1000 tps |
| 高吞吐量自建 | vLLM | PagedAttention, continuous batching |
| 複雜推理 | Mistral Large 或 OpenAI | 更強邏輯能力 |
| 成本敏感 | Groq 或 vLLM 自建 | 低成本推理 |
| Vision/OCR | Mistral (pixtral) 或 vLLM | 專門視覺模型 |
| 語音轉文字 | Groq (Whisper) | 216x realtime |
| Embeddings | Mistral 或 OpenAI | 高品質向量 |
| 量化部署 | vLLM (AWQ, GPTQ, FP8) | 支援多種量化格式 |
| 分散式部署 | vLLM (tensor/pipeline parallel) | 多 GPU/多節點支援 |

---

## 🔗 組合使用建議

| 任務類型 | 建議 Skill 組合 |
|---------|----------------|
| 建構 RAG 系統 | rag + langchain + mistral/groq |
| 快速原型 | react + groq |
| 多代理系統 | autogen + groq/mistral |
| 模型微調後部署 | unsloth + vllm |
| 語音助理 | groq (Whisper + Compound) |
| 富文本編輯器 | tiptap + react |
| 知識庫應用 | rag + tiptap + langchain |
| 習慣追蹤應用 | atomic-habits + react + account-manager |
| 學習平台開發 | learning-mastery + rag + langchain |
| 商業簡報製作 | storytelling-masters + pptx + canvas-design |
| 高吞吐量 LLM 服務 | vllm + langchain/autogen |
| 生產環境 LLM 部署 | vllm (Docker/K8s) + mistral/groq (fallback) |
| 前端配色優化 | ui-color-optimizer + theme-factory + frontend-design |
| AI 技術追蹤系統 | ai-daily-report + rag + langchain |

---

## 📚 技能詳細資訊

### AI/ML 技能

| Skill | 參考文件 | 說明 |
|-------|---------|------|
| langchain | references/ | LCEL, RAG, LangGraph, MCP |
| dspy | references/ | Signatures, Modules, Teleprompters |
| autogen | references/ + instruction | AgentChat, Teams, Swarm, MagenticOne |
| unsloth | references/ | LoRA, GRPO, Vision |
| vllm | references/ | PagedAttention, Quantization, Deployment |
| mistral | references/ | Chat, Embeddings, Function Calling |
| groq | references/ | LPU, Whisper, Compound |
| rag | references/ | Chunking, Hybrid Search, RAGAS |
| deep-learning | - | CNN, RNN, LSTM, Transformer |

### 前端技能

| Skill | 參考文件 | 說明 |
|-------|---------|------|
| react | references/ | Hooks, Components, State |
| tiptap | references/ + instruction | Nodes, Marks, Extensions, Node Views |
| ui-color-optimizer | references/ + instruction | Material Design, Fluent, WCAG |

### 寫作技能

| Skill | 說明 |
|-------|------|
| writing-masters | 海明威冰山理論、史蒂芬金寫作法、奧威爾六原則 |
| storytelling-masters | 安奈特西蒙斯六種故事、杜乔爾特簡報、坎貝爾英雄旅程 |

### 學習與成長技能

| Skill | 說明 |
|-------|------|
| learning-mastery | 楊大輝《深度學習的技術》五層框架 |
| atomic-habits | James Clear 原子習慣四大法則 |

### 資訊追蹤技能

| Skill | 說明 |
|-------|------|
| ai-daily-report | 每日 AI 技術新聞整理：Anthropic、OpenAI、Google、Unsloth |

---

## 使用方式

### 手動調用

調用特定 skill：/rag /langchain /tiptap /react /vllm /atomic-habits /learning-mastery /autogen /ui-color-optimizer

### 自動觸發

Skills 會根據 description 中的關鍵字自動觸發。例如：
- 提到 RAG 或 向量資料庫 → 自動觸發 rag skill
- 提到 Tiptap 或 rich text editor → 自動觸發 tiptap skill
- 提到 vLLM 或 inference serving → 自動觸發 vllm skill
- 提到 習慣 或 原子習慣 → 自動觸發 atomic-habits skill
- 提到 學習方法 或 費曼技巧 → 自動觸發 learning-mastery skill
- 提到 AutoGen 或 multi-agent 或 Swarm → 自動觸發 autogen skill
- 提到 配色 或 WCAG 或 Material Design → 自動觸發 ui-color-optimizer skill
- 提到 AI新聞 或 技術日報 → 自動觸發 ai-daily-report skill

### 自動觸發原理

Claude Code 系統會掃描 SKILL.md 的 `description` 欄位，當用戶的問題或任務包含 description 中提到的關鍵詞時，系統會自動建議或觸發對應的 skill。

**最佳實踐：**
- ✅ 在 description 中明確列出 "Use when:" 使用場景
- ✅ 在 description 中明確列出 "Triggers:" 觸發關鍵詞
- ✅ 同時包含中英文關鍵詞以支援跨語言觸發
- ✅ 包含技術名詞、縮寫、常見別名（如 vLLM, inference serving, 推理部署）
- ❌ 避免過於通用的詞彙（如 "code", "programming"）可能造成誤觸發

**範例（卓越級 Description）：**
```yaml
---
name: ui-color-optimizer
description: |
  Optimize UI color schemes for accessibility, aesthetics, and brand consistency. Generates harmonious color palettes, checks WCAG contrast ratios, and provides CSS/Tailwind variables.
  Use when: designing color schemes, fixing contrast issues, creating dark mode, building design systems, or when user mentions 配色, color, 顏色, palette, dark mode, 深色模式, contrast, 對比度, WCAG.
  Triggers: "color", "配色", "顏色", "palette", "dark mode", "深色模式", "對比度", "WCAG", "調色盤"
---
```

---

## ❓ 常見問題

### Q1: 為什麼我的 Skill 沒有被自動觸發？

**可能原因：**
1. **Description 缺少關鍵觸發詞** - 檢查 description 是否包含用戶可能使用的詞彙
2. **關鍵詞過於通用** - 如 "code", "programming" 等詞彙過於廣泛
3. **缺少中英文雙語支援** - 用戶可能用中文或英文描述需求
4. **沒有明確的 Triggers 區段** - 建議明確列出 "Triggers:" 觸發詞

**解決方案：**
```yaml
# ❌ 不佳範例
description: React framework for building user interfaces.

# ✅ 優秀範例
description: |
  React framework for building user interfaces. Use for React components, hooks, state management, JSX, and modern frontend development.
  Use when: working with React, creating components, managing state, or when user mentions React, component, 元件, hook, useState, useEffect, JSX.
  Triggers: "React", "component", "元件", "hook", "useState", "useEffect", "JSX", "前端開發"
```

### Q2: 一個 Skill 應該涵蓋多廣的範圍？

**原則：Single Responsibility Principle (SRP)**
- ✅ 每個 skill 專注於特定領域或工具
- ✅ langchain skill 專注於 LangChain 框架
- ✅ vllm skill 專注於 vLLM 推理引擎
- ❌ 避免創建過於寬泛的 "AI 開發" skill

**例外：學習方法論 Skills**
- atomic-habits, learning-mastery 等可以較為通用，因為它們是方法論而非技術工具

### Q3: 如何決定是否需要 references/ 子目錄？

**需要 references/ 的情況：**
- 技術框架/函式庫文件（langchain, react, vllm）
- 需要詳細 API 參考的工具
- 包含大量範例程式碼的 skill

**不需要 references/ 的情況：**
- 方法論/最佳實踐 skills（atomic-habits, systematic-debugging）
- 簡單工具或流程指引
- 已有完整 instruction.md 的 skills

### Q4: Instruction 檔案 vs SKILL.md，何時使用哪個？

| 特性 | SKILL.md (必須) | instruction.md (選用) |
|-----|----------------|---------------------|
| **用途** | 定義 skill 元資料與觸發機制 | 提供詳細操作流程與範本 |
| **長度** | 簡短（通常 < 100 行） | 可以很長（包含完整教學） |
| **內容** | name, description, 快速參考 | 步驟式指引、範例、模板 |
| **範例** | 所有 skills 都有 | autogen, tiptap, ui-color-optimizer |

**建議：**
- 簡單 skill：僅需 SKILL.md
- 複雜流程：SKILL.md + instruction.md
- 技術文件：SKILL.md + references/

### Q5: 如何測試 Skill 是否正確觸發？

**測試方法：**
1. **手動測試** - 在對話中使用觸發詞，觀察 Claude Code 是否建議該 skill
2. **檢查 description** - 確認 description 包含 "Use when:" 和 "Triggers:"
3. **跨語言測試** - 分別用中文和英文觸發詞測試
4. **邊界測試** - 測試相關但不應觸發的詞彙

**範例測試案例（vllm skill）：**
- ✅ "如何部署 vLLM 推理服務？" → 應觸發
- ✅ "need high-throughput inference" → 應觸發
- ✅ "PagedAttention 的原理" → 應觸發
- ❌ "什麼是機器學習？" → 不應觸發

---

## 維護說明

### 新增 Skill

**目錄型 Skill：**
1. 在 skills/ 目錄下建立新資料夾
2. 建立 SKILL.md 並填寫 frontmatter
3. 建立 references/ 子目錄存放參考文件
4. 更新此索引文件

**指令型 Skill：**
1. 在 skills/ 目錄下建立 skill-name_instruction.md
2. 撰寫完整操作指引與模板
3. 更新此索引文件的「指令型 Skills」區段

### Skill 結構

```
skills/
├── skill-name/              # 目錄型
│   ├── SKILL.md               # 主要說明（必須）
│   └── references/            # 參考文件（選用）
│       ├── index.md
│       └── *.md
├── skill-name_instruction.md  # 指令型（獨立檔案）
└── SKILLS_INDEX.md            # 本索引檔
```
