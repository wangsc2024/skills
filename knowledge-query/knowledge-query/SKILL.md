# Knowledge Query Skill

透過對話查詢和探索個人知識庫。**Claude 調用搜索 API 取得相關資料，再由 Claude 自己整合並回答。**

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
| 知道筆記名稱 | 標題搜索 | `$notes \| Where-Object { $_.title -match "關鍵字" }` |
| 模糊搜索 | 混合搜索 | `Invoke-RestMethod -Method POST -Uri ".../hybrid" -Body $json` |
| 精確詞彙 | 關鍵字搜索 | `Invoke-RestMethod -Method POST -Uri ".../keyword" -Body $json` |

## API 端點

Base URL: `http://localhost:3000`

### 1. 筆記直接查詢（最快）

```powershell
# 列出所有筆記
$notes = (Invoke-RestMethod "http://localhost:3000/api/notes?limit=100").notes

# 搜索標題含關鍵字的筆記
$notes | Where-Object { $_.title -match "四聖諦|八正道" } | Select-Object id, title

# 獲取單筆完整內容
$note = Invoke-RestMethod "http://localhost:3000/api/notes/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
$note.contentText

# 列出所有標籤
(Invoke-RestMethod "http://localhost:3000/api/notes/tags").tags
```

### 2. 混合搜索（語義+關鍵字）

```powershell
$body = @{ query = "搜索內容"; topK = 5 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://localhost:3000/api/search/hybrid" -Method POST -Body $body -ContentType "application/json"
$result.results
```

### 3. 關鍵字搜索（BM25）

```powershell
$body = @{ query = "useCallback"; topK = 10 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://localhost:3000/api/search/keyword" -Method POST -Body $body -ContentType "application/json"
$result.results
```

### 4. 取得格式化上下文

```powershell
$body = @{ query = "問題"; topK = 5 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://localhost:3000/api/search/retrieve" -Method POST -Body $body -ContentType "application/json"
$result.formattedContext
```

## 標準查詢流程

```
使用者提問
    │
    ▼
從問題提取關鍵字
    │
    ▼
$notes | Where-Object { $_.title -match "關鍵字" }
    │
    ▼
Invoke-RestMethod ".../notes/{id}"
    │
    ▼
取得 $note.contentText
    │
    ▼
Claude 整合回答（附來源）
```

## 範例：查詢「四聖諦」與「八正道」

```powershell
# 1. 列出筆記
$notes = (Invoke-RestMethod "http://localhost:3000/api/notes?limit=100").notes

# 2. 搜索相關標題
$found = $notes | Where-Object { $_.title -match "四聖諦|八正道" } | Select-Object id, title
$found

# 3. 獲取完整內容
foreach ($n in $found) {
    $note = Invoke-RestMethod "http://localhost:3000/api/notes/$($n.id)"
    Write-Host "=== $($note.title) ===" -ForegroundColor Green
    Write-Host $note.contentText
    Write-Host ""
}
```

## 輔助功能

### 系統狀態
```powershell
Invoke-RestMethod "http://localhost:3000/api/health"
Invoke-RestMethod "http://localhost:3000/api/stats"
```

### Web 研究（需啟用）
```powershell
$body = @{ topic = "主題"; maxResults = 5; autoCreateNote = $true; tags = @("標籤") } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:3000/api/research" -Method POST -Body $body -ContentType "application/json"
```

### 批量匯入
```powershell
$body = @{
    notes = @(
        @{ title = "標題"; contentText = "內容"; tags = @("標籤") }
    )
    autoSync = $true
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri "http://localhost:3000/api/import" -Method POST -Body $body -ContentType "application/json"
```

## 回答規範

1. **標明來源**：「根據您的知識庫...」
2. **結構化呈現**：標題、列表、表格
3. **引用筆記**：結尾列出參考來源
4. **誠實透明**：無結果時說明「知識庫中未找到」

## 注意事項

- **優先使用標題搜索**：比向量搜索更精確
- **PowerShell 原生方法**：避免 JSON 轉義問題
- **contentText 是純文字**：用於回答整合
- **content 是 Tiptap JSON**：用於 UI 顯示
