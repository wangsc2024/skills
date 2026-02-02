---
name: issue-resolver
description: |
  自動化問題診斷與修復 Skill。使用時機: (1) 從 Gun.js relay 取得特定專案的待處理問題, (2) 診斷系統問題並產生測試計畫, (3) 找出問題根因並修復, (4) 批次處理多個專案的問題, (5) 修復完成後回寫結果到問題系統。觸發詞: "處理表單系統的問題", "修復公文系統的bug", "取得人事系統的待辦", "回報修復結果"。
---

# Issue Resolver Skill

針對特定專案取得問題清單，診斷、修復，並**回寫結果到問題系統**。

## 安裝

```bash
cd issue-resolver-skill
npm install
```

## 🚀 快速開始

```bash
# 1. 列出所有可用專案
node scripts/resolve-project.js --list

# 2. 處理「表單系統」的問題
node scripts/resolve-project.js --project form-system

# 3. 修復完成後，回報結果
node scripts/report-fix.js --id issue-xxx --status resolved \
    --fix-summary "修復登入按鈕問題" \
    --root-cause "事件監聽器未綁定"
```

## 📝 修復回報 (重要!)

**問題修復後必須回寫到問題系統**，使用 `report-fix.js`：

### 基本用法
```bash
# 標記為已解決，並說明修復內容
node scripts/report-fix.js \
    --id issue-xxx \
    --status resolved \
    --fix-summary "修復登入按鈕無反應問題"
```

### 完整用法
```bash
node scripts/report-fix.js \
    --id issue-xxx \
    --status resolved \
    --root-cause "事件監聽器在 DOM 載入前註冊" \
    --fix-summary "將腳本移至 DOMContentLoaded 事件中" \
    --fix-details "1. 移動事件綁定到 DOMContentLoaded\n2. 增加元素存在檢查" \
    --files-changed "src/app.js,src/utils.js" \
    --commit "abc1234" \
    --time-spent "2h"
```

### 參數說明

| 參數 | 說明 | 範例 |
|------|------|------|
| `--id` | 問題 ID (必填) | `--id issue-xxx` |
| `--status` | 更新狀態 | `--status resolved` |
| `--root-cause` | 根本原因 | `--root-cause "N+1 查詢"` |
| `--fix-summary` | 修復摘要 | `--fix-summary "實作批次查詢"` |
| `--fix-details` | 詳細說明 | `--fix-details "..."` |
| `--files-changed` | 變更檔案 | `--files-changed "a.js,b.js"` |
| `--commit` | Git commit | `--commit "abc1234"` |
| `--time-spent` | 花費時間 | `--time-spent "2h"` |
| `--author` | 修復者 | `--author "Claude"` |

## 完整工作流程

```bash
# Step 1: 取得專案的待處理問題
node scripts/resolve-project.js --project form-system

# Step 2: 查看診斷報告
cat ./issue-reports/form-system/issue-xxx.txt

# Step 3: 根據報告進行修復...
# (修改程式碼、測試、提交)

# Step 4: 回報修復結果 ⭐
node scripts/report-fix.js \
    --id issue-xxx \
    --status resolved \
    --root-cause "發現的根本原因" \
    --fix-summary "修復摘要" \
    --files-changed "修改的檔案" \
    --commit "git-commit-hash"
```

## 📋 可用專案

| ID | 名稱 |
|----|------|
| `form-system` | 📝 表單系統 |
| `document-system` | 📄 公文系統 |
| `hr-system` | 👥 人事系統 |
| `finance-system` | 💰 財務系統 |
| `portal` | 🌐 入口網站 |
| `mobile-app` | 📱 行動應用 |
| `api-service` | 🔌 API 服務 |

## Scripts

| Script | 說明 |
|--------|------|
| `resolve-project.js` | 處理指定專案的問題 |
| `report-fix.js` | **回報修復結果** ⭐ |
| `fetch-issues.js` | 取得問題清單 |
| `process-issue.js` | 診斷單一問題 |
| `update-status.js` | 更新問題狀態 |
| `batch-process.js` | 批次處理 |

## 範例

```
User: 處理表單系統的問題，修復後回報結果

Claude:

# 1. 取得問題
$ node scripts/resolve-project.js --project form-system

找到 2 個待處理問題:
  1. 🔴 [ui] 表單送出按鈕無反應
  2. 🟠 [performance] 列表載入緩慢

# 2. 處理第一個問題
診斷報告指出可能是事件監聽器問題...

# 3. 修復程式碼
修改 src/form.js，將事件綁定移至 DOMContentLoaded

# 4. 回報修復結果
$ node scripts/report-fix.js \
    --id issue-xxx \
    --status resolved \
    --root-cause "事件監聽器在 DOM 載入前註冊" \
    --fix-summary "將腳本移至 DOMContentLoaded" \
    --files-changed "src/form.js" \
    --commit "fix: 修復表單送出按鈕"

✅ 修復回報已寫入!

# 5. 繼續處理下一個問題...
```
