# Issue Resolver Skill - 使用指南

## 📋 概述

這個 Skill 讓 Claude Code 能夠：
1. 連接到 Gun.js Relay Server 取得問題清單
2. **指定特定專案**（如：表單系統、公文系統）篩選問題
3. 自動診斷問題並產生測試計畫
4. 協助找出根本原因並修復

---

## 🚀 安裝

```bash
# 1. 解壓縮 Skill
unzip issue-resolver-skill.zip

# 2. 安裝依賴
cd issue-resolver-skill
npm install
```

---

## 📌 指定專案處理問題

### 列出所有可用專案

```bash
node scripts/resolve-project.js --list
```

輸出：
```
📋 可用的專案:

ID                  名稱              技術棧
------------------------------------------------------------
form-system         表單系統          
document-system     公文系統          
hr-system           人事系統          
finance-system      財務系統          
portal              入口網站          
mobile-app          行動應用          
api-service         API 服務          
```

### 處理特定專案的問題

```bash
# 處理「表單系統」的所有待處理問題
node scripts/resolve-project.js --project form-system

# 處理「公文系統」的緊急問題
node scripts/resolve-project.js --project document-system --priority critical

# 處理「人事系統」的高優先級 UI 問題
node scripts/resolve-project.js --project hr-system --priority critical,high --group ui

# 僅分析，不產生報告（乾跑模式）
node scripts/resolve-project.js --project finance-system --dry-run

# 限制只處理前 5 個問題
node scripts/resolve-project.js --project form-system --limit 5
```

---

## 🔧 命令列參數

### resolve-project.js（主要腳本）

| 參數 | 說明 | 範例 |
|------|------|------|
| `-p, --project ID` | 指定專案 ID（必填） | `--project form-system` |
| `--priority LEVEL` | 優先級篩選 | `--priority critical,high` |
| `--group GROUP` | 問題類型篩選 | `--group ui,system` |
| `--limit N` | 限制處理數量 | `--limit 10` |
| `--dry-run` | 僅分析不產生報告 | `--dry-run` |
| `--output-dir DIR` | 報告輸出目錄 | `--output-dir ./reports` |
| `-l, --list` | 列出所有專案 | `--list` |

### fetch-issues.js（取得問題清單）

```bash
# 取得所有問題
node scripts/fetch-issues.js

# 取得特定專案的問題
node scripts/fetch-issues.js --system form-system

# 輸出為 Markdown
node scripts/fetch-issues.js --system document-system --format markdown

# 儲存到檔案
node scripts/fetch-issues.js --system hr-system --output issues.json
```

### update-status.js（更新問題狀態）

```bash
# 標記問題為已解決
node scripts/update-status.js --id issue-xxx --status resolved

# 標記為處理中
node scripts/update-status.js --id issue-xxx --status in-progress
```

---

## 📊 工作流程範例

### 場景：處理表單系統的問題

```bash
# 1. 列出專案，確認 ID
$ node scripts/resolve-project.js --list

# 2. 處理表單系統的緊急問題
$ node scripts/resolve-project.js --project form-system --priority critical,high

============================================================
🔧 專案問題解決器
============================================================
專案: 表單系統 (form-system)
優先級篩選: critical,high
============================================================

🔍 正在從 Relay 取得 "form-system" 的問題...

📋 找到 3 個待處理問題:

  1. 🔴 [ui] 表單送出按鈕無反應
  2. 🔴 [data] 必填欄位驗證失效
  3. 🟠 [performance] 表單載入緩慢

------------------------------------------------------------
開始處理問題...
------------------------------------------------------------

[1/3] 處理: 表單送出按鈕無反應
  優先級: CRITICAL
  類型: ui
  診斷計畫: UI 問題診斷
  📄 報告: ./issue-reports/form-system/issue-xxx.txt
  ✅ 完成

[2/3] 處理: 必填欄位驗證失效
  優先級: CRITICAL
  類型: data
  診斷計畫: 資料問題診斷
  📄 報告: ./issue-reports/form-system/issue-yyy.txt
  ✅ 完成

[3/3] 處理: 表單載入緩慢
  優先級: HIGH
  類型: performance
  診斷計畫: 效能問題診斷
  📄 報告: ./issue-reports/form-system/issue-zzz.txt
  ✅ 完成

============================================================
📊 處理摘要
============================================================
專案: 表單系統
處理問題數: 3
輸出目錄: ./issue-reports/form-system

✅ 處理完成！

# 3. 查看診斷報告
$ cat ./issue-reports/form-system/issue-xxx.txt

# 4. 根據報告進行修復...

# 5. 修復完成後更新狀態
$ node scripts/update-status.js --id issue-xxx --status resolved
```

---

## 📁 輸出報告格式

每個問題會產生一份診斷報告：

```
======================================================================
問題診斷報告
======================================================================

【專案資訊】
專案名稱: 表單系統
專案路徑: /path/to/project

【問題資訊】
ID: issue-1706345678901-abc123
標題: 表單送出按鈕無反應
類型: ui
優先級: critical
回報者: 王先生
描述: 點擊送出按鈕後沒有任何反應，表單資料無法提交

【診斷計畫】
計畫: UI 問題診斷
重點檢查:
  • console
  • css
  • dom
  • events

【技術建議】
  • 檢查 React 元件狀態和生命週期
  • 使用 React DevTools 檢查元件樹

【建議檢查檔案】
  • src/client/**/*.{jsx,vue,tsx,css,scss}

【行動項目】
[ ] 1. 複製問題描述，在本地環境重現
[ ] 2. 根據診斷計畫檢查相關程式碼
[ ] 3. 找出根本原因
[ ] 4. 實作修復
[ ] 5. 撰寫或更新測試
[ ] 6. 驗證修復有效
[ ] 7. 提交變更並更新問題狀態

【根本原因】
(調查後填寫)

【修復方案】
(實作後填寫)

======================================================================
報告產生時間: 2026/1/27 下午3:30:00
======================================================================
```

---

## ⚙️ 自訂專案配置

編輯 `projects.json` 新增或修改專案：

```json
{
  "projects": {
    "my-custom-project": {
      "name": "我的專案",
      "description": "專案描述",
      "path": "/home/user/projects/my-project",
      "repo": "https://github.com/user/my-project.git",
      "techStack": ["node", "express", "react", "mongodb"]
    }
  }
}
```

然後在問題回報系統的 `app.js` 中同步更新 `CONFIG.systems`：

```javascript
systems: [
    { id: 'my-custom-project', name: '我的專案', icon: '🚀', path: '' },
    // ... 其他專案
]
```

---

## 🔗 相關檔案

| 檔案 | 說明 |
|------|------|
| `SKILL.md` | Skill 主文件 |
| `package.json` | Node.js 套件設定 |
| `projects.json` | 專案配置檔 |
| `scripts/resolve-project.js` | 主程式 - 指定專案處理 |
| `scripts/fetch-issues.js` | 取得問題清單 |
| `scripts/process-issue.js` | 處理單一問題 |
| `scripts/batch-process.js` | 批次處理 |
| `scripts/update-status.js` | 更新問題狀態 |
| `references/diagnosis-guide.md` | 診斷指南 |
| `references/config-reference.md` | 配置參考 |

---

## ❓ 常見問題

### Q: 找不到問題？

確認：
1. 問題回報時有選擇正確的「所屬系統」
2. `--project` 參數與問題的系統 ID 一致
3. Relay Server 連線正常

### Q: 如何新增專案？

1. 編輯 `projects.json` 新增專案
2. 編輯問題回報系統的 `app.js` 中的 `CONFIG.systems`
3. 重新部署問題回報系統

### Q: 如何連接到其他 Relay Server？

設定環境變數：
```bash
export ISSUE_RELAY_URL="https://your-relay.com/gun"
export ISSUE_NODE_PREFIX="your-prefix"
```

或在 `projects.json` 中修改：
```json
{
  "relay": {
    "url": "https://your-relay.com/gun",
    "prefix": "your-prefix"
  }
}
```
