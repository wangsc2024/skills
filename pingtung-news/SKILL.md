---
name: pingtung-news
description: 屏東縣政府新聞查詢工具。查詢屏東縣最新新聞、依關鍵字搜尋新聞、依日期範圍篩選新聞稿。適用於查詢屏東新聞、屏東縣政、周春米縣長施政、屏東縣府公告等相關資訊。
---

# 屏東新聞通

查詢屏東縣政府新聞稿資料，支援多種環境。

## 資料來源

- **原始 API**：`https://www.pthg.gov.tw/OpenDataForm.aspx?s=05393D58A2AA3E5B`
- **MCP 服務**：`https://ptnews-mcp.pages.dev`
- **資料範圍**：2022年12月25日起（周春米縣長就職日）

---

## 查詢方式選擇

根據執行環境選擇適當的查詢方式：

| 環境 | 推薦方式 | 說明 |
|------|----------|------|
| **Claude Code** | MCP 服務 (Bash curl) | 支援搜尋、日期篩選 |
| **Claude Chat** | WebFetch 原始 API | 直接抓取 JSON 資料 |

---

## 方式一：REST API + WebFetch（Claude Chat 適用）

使用 WebFetch 工具抓取 MCP 服務的 REST API 端點：

### REST API 端點

| 端點 | 說明 | 參數 |
|------|------|------|
| `/api/latest` | 取得最新新聞 | count=1-50（預設 10） |
| `/api/search` | 關鍵字搜尋 | keyword=關鍵字, limit=1-100 |
| `/api/date` | 日期範圍查詢 | start=YYYY-MM-DD, end=YYYY-MM-DD |

### WebFetch 查詢範例

1. **最新 5 則新聞**
   ```
   URL: https://ptnews-mcp.pages.dev/api/latest?count=5
   Prompt: 列出新聞標題和發布日期
   ```

2. **關鍵字搜尋「長照」**
   ```
   URL: https://ptnews-mcp.pages.dev/api/search?keyword=長照&limit=10
   Prompt: 列出包含長照的新聞
   ```

3. **查詢 2026 年 2 月新聞**
   ```
   URL: https://ptnews-mcp.pages.dev/api/date?start=2026-02-01&end=2026-02-28
   Prompt: 列出這段期間的新聞
   ```

### 回傳格式

```json
[
  {
    "id": "5094",
    "title": "新聞標題",
    "department": "發布單位代碼",
    "publishedAt": "2026-02-03T12:02:00+08:00",
    "content": "新聞完整內容...",
    "summary": "前 200 字摘要..."
  }
]
```

---

## 方式二：MCP 服務（Claude Code 適用）

使用 Bash 工具執行 curl 指令呼叫 MCP 服務，支援進階查詢功能。

### 可用 Tools

| Tool 名稱 | 說明 | 參數 |
|-----------|------|------|
| `pingtung_news_latest` | 取得最新新聞 | count: 1-50（預設 10） |
| `pingtung_news_search` | 關鍵字搜尋 | keyword: 必填, limit: 1-100 |
| `pingtung_news_by_date` | 日期範圍查詢 | start, end: YYYY-MM-DD 格式 |

### 取得最新新聞

```bash
curl -s -X POST https://ptnews-mcp.pages.dev/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"pingtung_news_latest","arguments":{"count":10}}}'
```

### 關鍵字搜尋

```bash
curl -s -X POST https://ptnews-mcp.pages.dev/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"pingtung_news_search","arguments":{"keyword":"長照","limit":20}}}'
```

### 日期範圍查詢

```bash
curl -s -X POST https://ptnews-mcp.pages.dev/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"pingtung_news_by_date","arguments":{"start":"2024-01-01","end":"2024-01-31"}}}'
```

---

## 常見查詢對照

| 用戶說法 | Claude Chat (WebFetch) | Claude Code (MCP) |
|---------|------------------------|-------------------|
| 「屏東最新新聞」 | WebFetch + "列出最新 5 則" | pingtung_news_latest |
| 「屏東長照新聞」 | WebFetch + "找包含長照的" | pingtung_news_search keyword:"長照" |
| 「周春米相關新聞」 | WebFetch + "找包含周春米的" | pingtung_news_search keyword:"周春米" |
| 「2024年1月新聞」 | WebFetch + "列出 2024/01 的" | pingtung_news_by_date |

---

## 輸出格式建議

### 新聞列表格式

```markdown
| # | 日期 | 標題 |
|---|------|------|
| 1 | 2024-01-15 | 新聞標題... |
| 2 | 2024-01-14 | 新聞標題... |
```

### 新聞詳情格式

```markdown
## 新聞標題

**日期**：2024-01-15 14:30

**內容**：
新聞完整內容...
```

---

## 注意事項

1. **即時資料**：資料直接從縣府開放資料取得，確保最新
2. **5 分鐘快取**：MCP 服務會快取資料 5 分鐘
3. **日期格式**：MCP 查詢使用 `YYYY-MM-DD` 格式
4. **關鍵字搜尋**：會同時搜尋標題和內容
5. **中文編碼**：Windows curl 需使用 Unicode 轉義（如 `\u68d2\u7403` = 棒球）

## 服務狀態檢查

```bash
curl -s https://ptnews-mcp.pages.dev/health
# 回傳：{"status":"ok","timestamp":"..."}
```
