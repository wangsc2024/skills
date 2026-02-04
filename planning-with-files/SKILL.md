---
name: planning-with-files
version: "2.1"
description: |
  Claude Code 規劃工作流程技能，實現 Manus 風格的持久化 markdown 規劃模式。
  透過 3 檔案模式（task_plan.md, findings.md, progress.md）解決上下文遺失問題。
triggers:
  - "planning with files"
  - "manus workflow"
  - "task plan"
  - "三檔案模式"
  - "任務規劃"
  - "專案規劃"
  - "context engineering"
  - "上下文工程"
---

# Planning with Files

> **Work like Manus** — 實現 Meta 以 $20 億美元收購的 AI 代理公司的工作模式

## 核心概念

這是一個 Claude Code 技能，將你的工作流程轉換為使用持久化 markdown 檔案進行規劃、進度追蹤和知識存儲——這正是 Manus 成功的秘密。

### 核心原則

```
Context Window = RAM（易失性、有限）
Filesystem = Disk（持久性、無限）

→ 任何重要的東西都要寫入檔案。
```

## 問題與解決方案

### 問題
Claude Code（和大多數 AI 代理）存在以下問題：
- **易失性記憶** — TodoWrite 工具在上下文重置時消失
- **目標漂移** — 經過 50+ 次工具調用後，原始目標被遺忘
- **隱藏錯誤** — 失敗未被追蹤，導致相同錯誤重複發生
- **上下文填塞** — 所有內容都塞入上下文而非存儲

### 解決方案：3 檔案模式

對於每個複雜任務，創建三個檔案：

```
task_plan.md      → 追蹤階段和進度
findings.md       → 存儲研究和發現
progress.md       → 會話日誌和測試結果
```

## 關鍵規則

1. **先建立計劃** — 永遠不要在沒有 `task_plan.md` 的情況下開始
2. **2-Action 規則** — 每 2 次查看/瀏覽器操作後保存發現
3. **記錄所有錯誤** — 它們有助於避免重複
4. **永不重複失敗** — 追蹤嘗試，改變方法

## task_plan.md 模板

```markdown
# 任務規劃：[任務名稱]

## 目標
[明確說明任務目標]

## 階段

### 階段 1：[名稱]
- [ ] 步驟 1.1
- [ ] 步驟 1.2
- [ ] 步驟 1.3

### 階段 2：[名稱]
- [ ] 步驟 2.1
- [ ] 步驟 2.2

### 階段 3：[名稱]
- [ ] 步驟 3.1
- [ ] 步驟 3.2

## 當前狀態
- 正在進行：階段 X
- 已完成：X/Y 步驟

## 阻塞/問題
- [列出任何阻塞或問題]

## 錯誤日誌
| 時間 | 錯誤 | 嘗試的解決方案 | 結果 |
|------|------|----------------|------|
```

## findings.md 模板

```markdown
# 研究發現：[任務名稱]

## 發現摘要

### [主題 1]
- 發現內容
- 來源/參考

### [主題 2]
- 發現內容
- 來源/參考

## 重要代碼片段

```[語言]
// 重要代碼
```

## 決策記錄
| 決策 | 原因 | 替代方案 |
|------|------|----------|
```

## progress.md 模板

```markdown
# 進度日誌：[任務名稱]

## 會話記錄

### 會話 1 - [日期]

#### 完成的工作
- [x] 任務 1
- [x] 任務 2

#### 測試結果
- 測試 A：通過 ✅
- 測試 B：失敗 ❌（原因：...）

#### 下一步
- [ ] 下一個任務

### 會話 2 - [日期]
...
```

## 使用時機

**適用於：**
- 多步驟任務（3+ 步驟）
- 研究任務
- 建立/創建專案
- 跨越多次工具調用的任務

**不適用於：**
- 簡單問題
- 單檔案編輯
- 快速查詢

## Manus 原則

| 原則 | 實現方式 |
|------|----------|
| 檔案系統作為記憶 | 存儲在檔案中，而非上下文 |
| 注意力操控 | 在決策前重新讀取計劃 |
| 錯誤持久化 | 在計劃檔案中記錄失敗 |
| 目標追蹤 | 複選框顯示進度 |
| 完成驗證 | 檢查所有階段是否完成 |

## 5 步快速指南

1. **收到複雜任務時** → 創建 `task_plan.md`
2. **開始研究時** → 創建 `findings.md` 記錄發現
3. **每完成一個步驟** → 更新 `task_plan.md` 勾選完成
4. **遇到錯誤時** → 在錯誤日誌中記錄
5. **完成任務前** → 驗證所有階段已完成

## 安裝方式

```bash
# Plugin 安裝（推薦）
/plugin marketplace add OthmanAdi/planning-with-files
/plugin install planning-with-files@planning-with-files

# 手動調用
/planning-with-files
```

## 參考資料

- [GitHub 倉庫](https://github.com/OthmanAdi/planning-with-files)
- [安裝指南](https://github.com/OthmanAdi/planning-with-files/blob/master/docs/installation.md)
- [快速入門](https://github.com/OthmanAdi/planning-with-files/blob/master/docs/quickstart.md)
- [疑難排解](https://github.com/OthmanAdi/planning-with-files/blob/master/docs/troubleshooting.md)

---

**基於 [OthmanAdi/planning-with-files](https://github.com/OthmanAdi/planning-with-files)** | Stars: 9,203 | MIT License
