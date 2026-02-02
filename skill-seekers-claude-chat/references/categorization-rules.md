# 內容分類規則

## 分類系統

將抓取的內容分類到 8 大類別：

## 類別定義

### 1. 入門指南 (getting_started)

**識別關鍵字：**
- intro, introduction
- quickstart, quick-start
- installation, install
- setup, getting-started
- start, begin, first-steps

**內容特徵：**
- 環境安裝步驟
- 第一個範例程式
- 基本設定說明
- 前置需求列表

---

### 2. 核心概念 (core_concepts)

**識別關鍵字：**
- concepts, concept
- fundamentals, fundamental
- basics, basic
- overview
- principles, principle
- understanding

**內容特徵：**
- 專有名詞解釋
- 架構說明
- 設計理念
- 基本原理

---

### 3. API 參考 (api_reference)

**識別關鍵字：**
- api, apis
- reference, ref
- methods, method
- functions, function
- classes, class
- modules, module
- interface, interfaces
- endpoints, endpoint

**內容特徵：**
- 函數/方法簽章
- 參數說明
- 回傳值描述
- 型別定義

---

### 4. 教學指南 (guides)

**識別關鍵字：**
- guide, guides
- tutorial, tutorials
- how-to, howto
- walkthrough
- learn, learning
- step-by-step
- building

**內容特徵：**
- 步驟式教學
- 實作專案
- 完整流程說明
- 從零開始建置

---

### 5. 程式範例 (examples)

**識別關鍵字：**
- example, examples
- sample, samples
- demo, demos
- cookbook
- recipes, recipe
- patterns, pattern
- snippets, snippet
- showcase

**內容特徵：**
- 完整可執行程式碼
- 使用案例
- 常見情境實作
- 程式碼片段集

---

### 6. 設定說明 (configuration)

**識別關鍵字：**
- config, configuration
- settings, setting
- options, option
- parameters, parameter
- environment, env
- customize, customization
- preferences

**內容特徵：**
- 設定檔格式
- 環境變數列表
- 選項說明
- 自訂化方法

---

### 7. 問題排解 (troubleshooting)

**識別關鍵字：**
- error, errors
- debug, debugging
- faq, faqs
- troubleshoot, troubleshooting
- issue, issues
- problem, problems
- fix, fixes
- common-issues
- solutions

**內容特徵：**
- 錯誤訊息解釋
- 常見問題列表
- 解決方案步驟
- 除錯技巧

---

### 8. 進階主題 (advanced)

**識別關鍵字：**
- advanced
- deep-dive
- internals, internal
- architecture
- performance
- optimization, optimize
- scaling, scale
- best-practices

**內容特徵：**
- 進階用法
- 效能優化
- 內部實作原理
- 最佳實踐

---

## 分類邏輯

### 優先順序

1. **URL 路徑** - 最高權重
2. **頁面標題** - 次高權重
3. **內容關鍵字** - 輔助判斷

### 判斷流程

```
1. 檢查 URL 是否包含分類關鍵字
   ↓ 有 → 使用該分類
   ↓ 無 → 繼續

2. 檢查標題是否包含分類關鍵字
   ↓ 有 → 使用該分類
   ↓ 無 → 繼續

3. 分析內容特徵
   ↓ 有明確特徵 → 使用對應分類
   ↓ 無 → 標記為 "general"
```

### 多重匹配處理

當內容符合多個分類時，使用以下優先順序：

1. api_reference（最具體）
2. examples
3. guides
4. getting_started
5. configuration
6. troubleshooting
7. core_concepts
8. advanced

## 自訂分類

可根據特定框架新增分類：

**React 專用：**
- hooks：useState, useEffect, useContext
- components：component, props, children
- state：state, redux, context

**Python 專用：**
- decorators：@decorator, wrapper
- async：async, await, asyncio
- typing：type hints, annotations
