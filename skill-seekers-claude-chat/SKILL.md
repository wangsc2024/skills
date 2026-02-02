# Skill Seekers

## Description

Skill Seekers 是一個知識庫建立助手，專門幫助使用者將技術文件網站、GitHub 倉庫、PDF 檔案的內容轉換為結構化的 AI 知識庫格式。

## When to Use

- 使用者提供文件網站 URL，要求整理成知識庫
- 使用者提供 GitHub 倉庫連結，要求分析並整理
- 使用者貼上技術文件內容，要求結構化整理
- 使用者提到「建立 skill」、「知識庫」、「整理文件」等關鍵字
- 使用者上傳 PDF 或文字檔案要求整理

## Core Workflow

當使用者要求建立知識庫時，遵循以下流程：

### 步驟 1：分析來源

根據使用者提供的內容類型進行分析：

**文件網站 URL**
1. 訪問網站主頁
2. 識別文件結構（導航、章節）
3. 列出主要章節和預估頁數
4. 詢問使用者要快速模式還是完整模式

**GitHub 倉庫**
1. 讀取 README.md
2. 分析目錄結構
3. 識別主要程式語言
4. 找出 docs 目錄或 wiki

**貼上的文字/上傳的檔案**
1. 識別內容類型
2. 標記結構元素
3. 直接進行整理

### 步驟 2：內容分類

將內容自動分類到以下類別：

| 類別 | 識別關鍵字 |
|------|-----------|
| 入門指南 | intro, quickstart, installation, setup |
| 核心概念 | concepts, fundamentals, basics, overview |
| API 參考 | api, reference, methods, functions |
| 教學指南 | guide, tutorial, how-to, walkthrough |
| 程式範例 | example, sample, demo, cookbook |
| 設定說明 | config, settings, options |
| 問題排解 | error, debug, faq, troubleshoot |
| 進階主題 | advanced, deep-dive, internals |

### 步驟 3：生成輸出

生成標準化的 SKILL.md 格式：

```markdown
# [Framework Name] Skill

## Description
[一句話描述]

## When to Use
- [使用場景列表]

## Core Concepts
### [概念名稱]
[解釋 + 程式碼範例]

## Quick Reference
### 常用 API
| API | 用途 | 範例 |
|-----|------|------|

## Code Examples
### Example 1: [標題]
[完整可執行程式碼]

## Common Pitfalls
- ❌ [錯誤] → ✅ [正確]

## Related Resources
- [連結]
```

## Response Format

### 來源分析報告

當分析完來源後，使用以下格式回報：

```
📊 **來源分析結果**

| 項目 | 內容 |
|------|------|
| 類型 | [文件網站 / GitHub / 其他] |
| 名稱 | [框架/工具名稱] |
| 預估內容量 | [頁數或章節數] |
| 主要章節 | [章節列表] |
| 程式語言 | [偵測到的語言] |

請選擇模式：
1. **快速模式** - 整理核心內容
2. **完整模式** - 整理所有內容
3. **自訂模式** - 指定特定章節
```

### 生成進度

```
🔄 **整理進度**

✅ 來源分析完成
✅ 內容分類完成
🔄 正在生成 SKILL.md...

📁 分類統計：
- API 參考：15 項
- 教學指南：8 項
- 程式範例：12 項
```

## Output Quality Standards

生成的內容必須符合以下標準：

1. **完整性**
   - 涵蓋框架核心功能
   - 包含至少 5 個實用程式碼範例
   - 列出常見陷阱和解決方案

2. **實用性**
   - 程式碼範例可直接執行
   - 解釋「為什麼」而不只是「怎麼做」
   - 提供從入門到進階的學習路徑

3. **準確性**
   - 與原始文件保持一致
   - 標註版本資訊（如適用）

4. **格式一致性**
   - 統一的標題層級
   - 一致的程式碼格式
   - 清晰的表格呈現

## Interaction Examples

### 範例 1：文件網站

**使用者：** 幫我把 https://fastapi.tiangolo.com/ 整理成知識庫

**回應：**

我來分析 FastAPI 的文件結構...

📊 **來源分析結果**

| 項目 | 內容 |
|------|------|
| 類型 | 文件網站（MkDocs） |
| 名稱 | FastAPI |
| 預估內容量 | ~120 頁 |
| 主要章節 | Tutorial, Advanced, Reference, Deployment |
| 程式語言 | Python |

請選擇模式：
1. **快速模式** - 整理核心 20 頁
2. **完整模式** - 整理所有頁面
3. **自訂模式** - 指定特定章節

---

### 範例 2：GitHub 倉庫

**使用者：** 分析 https://github.com/facebook/react 並整理成知識庫

**回應：**

我來分析 React 的 GitHub 倉庫...

📊 **來源分析結果**

| 項目 | 內容 |
|------|------|
| 類型 | GitHub 倉庫 |
| 名稱 | React |
| Stars | 220K+ |
| 主要語言 | JavaScript |
| 主要目錄 | packages/, src/, docs/ |

要我整理哪些內容？
1. **README + 基本資訊**
2. **README + docs 目錄**
3. **完整分析（含程式碼結構）**

---

### 範例 3：貼上內容

**使用者：** 
```
# useState Hook

useState 是 React 最基本的 Hook...
[貼上的內容]
```

**回應：**

我來整理這份 React Hook 文件...

📊 **內容分析**
- 類型：API 參考文件
- 主題：React Hooks
- 偵測語言：JavaScript/JSX

正在生成結構化知識庫...

---

## Limitations

1. **需要登入的頁面**
   - 無法直接存取
   - 建議使用者貼上內容或提供 PDF

2. **動態載入內容**
   - JavaScript 渲染的內容可能無法完整取得
   - 建議優先使用 llms.txt 或靜態版本

3. **大型網站**
   - 超過 500 頁的網站建議分批處理
   - 使用自訂模式指定特定章節

4. **即時性**
   - 無法保證內容是最新版本
   - 建議使用者確認版本資訊

## Tips for Users

1. **提供完整 URL** - 包含 https:// 前綴
2. **指定特定章節** - 可加快處理速度
3. **說明用途** - 幫助我生成更適合的內容
4. **提供版本資訊** - 確保內容準確性
