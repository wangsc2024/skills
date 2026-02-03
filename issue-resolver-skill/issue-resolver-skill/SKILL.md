---
name: issue-resolver-skill
description: |
  查詢並管理 BugBox 問題追蹤系統中的問題。連接 Gun.js P2P 資料庫，列出待處理問題、更新問題狀態、分析問題優先級。
  Use when: 查詢待修問題、列出 issue、管理問題狀態、追蹤 Bug 進度，或當用戶提到 issue-skill, issue, BugBox, 問題清單, 待修, 待處理問題, 問題追蹤, bug tracker。
  Triggers: "issue-skill", "issue", "BugBox", "問題清單", "待修", "待處理問題", "問題追蹤", "bug tracker", "查問題", "列出問題"
allowed-tools: Read, Bash, WebFetch, mcp__plugin_playwright_playwright__browser_navigate, mcp__plugin_playwright_playwright__browser_snapshot, mcp__plugin_playwright_playwright__browser_click, mcp__plugin_playwright_playwright__browser_select_option
---

# Issue Resolver Skill

透過 Playwright 瀏覽器自動化連接 BugBox 問題追蹤系統，查詢並管理問題。

## 系統資訊

- **BugBox URL**: `http://localhost:8080` (本地) 或部署站點
- **資料庫**: Gun.js P2P 資料庫，Relay Server: `https://relay-o.oopdoo.org.ua/gun`
- **資料路徑**: `bugbox/issues`

## 核心功能

### 1. 列出待處理問題

使用 Playwright 瀏覽器自動化：

```typescript
// 1. 導航至 BugBox
await browser_navigate({ url: "http://localhost:8080" });

// 2. 點擊問題列表
await browser_click({ ref: "問題列表按鈕ref" });

// 3. 篩選待處理問題
await browser_select_option({
  ref: "狀態篩選ref",
  values: ["📬 待處理"]
});

// 4. 擷取快照獲取問題列表
await browser_snapshot();
```

### 2. 更新問題狀態

```typescript
// 找到問題的狀態下拉選單並更新
await browser_select_option({
  ref: "問題狀態ref",
  values: ["✅ 已解決"]  // 或 "🔧 處理中", "🔒 已關閉"
});
```

### 3. 篩選問題

支援的篩選條件：
- **專案**: 問題回報系統, P2P計數器, TipTag編輯器, 帳號管理系統, Gun.js測試
- **類型**: 系統問題, 介面問題, 帳號問題, 資料問題, 效能問題, 功能建議
- **狀態**: 待處理, 處理中, 已解決, 已關閉
- **優先級**: 緊急, 高, 中, 低

## 問題狀態流程

```
📬 待處理 → 🔧 處理中 → ✅ 已解決 → 🔒 已關閉
```

## 使用範例

### 查詢所有待處理問題

```
用戶: 請列出有哪些問題待修
助手: [使用 Playwright 導航至 BugBox，篩選待處理問題，擷取問題列表]
```

### 依專案篩選

```
用戶: P2P計數器有什麼問題？
助手: [導航至 BugBox，篩選專案為 P2P計數器，列出相關問題]
```

### 批次更新狀態

```
用戶: 將 #1 和 #2 標記為已解決
助手: [逐一找到問題，更新狀態為已解決]
```

## 問題資料結構

```typescript
interface Issue {
  id: string;           // 唯一識別碼
  title: string;        // 問題標題
  description: string;  // 詳細描述
  system: string;       // 所屬系統
  category: string;     // 問題分類
  priority: string;     // 優先程度
  status: string;       // 處理狀態
  reporter: string;     // 回報者
  contact: string;      // 聯絡方式
  device: string;       // 裝置資訊
  timestamp: number;    // 建立時間
  screenshots: string[]; // 截圖附件
}
```

## 最佳實踐

1. **先列出再操作**: 查詢問題列表後再決定處理順序
2. **依優先級處理**: 緊急 > 高 > 中 > 低
3. **批次處理同類問題**: 相關問題一起修復
4. **更新狀態追蹤進度**: 修復後立即更新為「已解決」
5. **非程式問題標記**: 如環境問題、設定問題也應標記已解決並說明原因

## 常見問題

### Q: 無法連接 BugBox？

1. 確認 BugBox 服務已啟動（`http://localhost:8080`）
2. 檢查 Gun.js Relay Server 連線狀態
3. 嘗試重新整理頁面

### Q: 問題狀態更新失敗？

1. 確認已正確選擇問題
2. 等待 Gun.js 同步完成
3. 檢查網路連線

## 相關 Skills

- **systematic-debugging**: 系統化除錯流程
- **code-reviewer**: 程式碼審查
- **git-workflow**: Git 工作流程管理
