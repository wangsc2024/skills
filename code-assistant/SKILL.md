# Code Assistant Skill

基於個人知識庫提供程式開發輔助。**Claude Code 調用搜索 API 取得相關資料，再由 Claude 自己整合並回答。**

## 快速查詢（PowerShell 推薦）

```powershell
# 步驟 1: 搜索筆記標題
$notes = (Invoke-RestMethod "http://localhost:3000/api/notes?limit=100").notes
$notes | Where-Object { $_.title -match "關鍵字" } | Select-Object id, title

# 步驟 2: 獲取完整內容
$note = Invoke-RestMethod "http://localhost:3000/api/notes/{id}"
$note.contentText
```

## 搜索策略

| 場景 | 方法 | PowerShell 指令 |
|------|------|----------------|
| `useState`, `useEffect` | 關鍵字搜索 | `Invoke-RestMethod -Method POST .../keyword` |
| `NullPointerException` | 關鍵字搜索 | `Invoke-RestMethod -Method POST .../keyword` |
| 「React 狀態管理」 | 混合搜索 | `Invoke-RestMethod -Method POST .../hybrid` |
| 知道筆記名稱 | 標題搜索 | `$notes \| Where-Object { $_.title -match "..." }` |

## API 端點

Base URL: `http://localhost:3000`

### 筆記直接查詢（最快）

```powershell
# 列出所有筆記
$notes = (Invoke-RestMethod "http://localhost:3000/api/notes?limit=100").notes

# 搜索標題
$notes | Where-Object { $_.title -match "React|TypeScript" } | Select-Object id, title

# 獲取單筆
$note = Invoke-RestMethod "http://localhost:3000/api/notes/{id}"
$note.contentText
```

### 關鍵字搜索（BM25）

```powershell
$body = @{ query = "useCallback"; topK = 5 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://localhost:3000/api/search/keyword" -Method POST -Body $body -ContentType "application/json"
$result.results
```

### 混合搜索

```powershell
$body = @{ query = "搜索內容"; topK = 5 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://localhost:3000/api/search/hybrid" -Method POST -Body $body -ContentType "application/json"
$result.results
```

## 工作流程

```
使用者提問
    │
    ▼
提取關鍵字（函數名、錯誤代碼、概念）
    │
    ├─ 精確詞彙 → keyword 搜索
    ├─ 概念問題 → hybrid 搜索
    └─ 知道標題 → 標題搜索
    │
    ▼
取得 contentText
    │
    ▼
Claude 整合回答（附來源）
```

## 範例

**查詢 useCallback：**

```powershell
$body = @{ query = "useCallback"; topK = 5 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://localhost:3000/api/search/keyword" -Method POST -Body $body -ContentType "application/json"
$result.results | Select-Object noteId, title, score
```

**回答格式：**
```
根據您的知識庫：

**useCallback** 是 React Hook，用於記憶化回調函數...

**參考來源：**
- 《React Hooks 最佳實踐》
```

## 輔助 API

```powershell
# 健康檢查
Invoke-RestMethod "http://localhost:3000/api/health"

# 系統統計
Invoke-RestMethod "http://localhost:3000/api/stats"

# 列出標籤
(Invoke-RestMethod "http://localhost:3000/api/notes/tags").tags
```

## 回答規範

1. 開頭：「根據您的知識庫...」
2. 結構化：標題、列表、程式碼區塊
3. 標註引用：結尾列出參考來源
4. 無結果時：說明「知識庫中未找到」
