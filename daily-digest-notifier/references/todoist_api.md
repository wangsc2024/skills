# Todoist REST API v2 參考

## 認證

所有請求需要在 Header 中包含 API Token：

```
Authorization: Bearer YOUR_API_TOKEN
```

取得 Token: https://todoist.com/app/settings/integrations/developer

## 常用端點

### 取得任務

```
GET https://api.todoist.com/rest/v2/tasks
```

參數:
- `project_id` (可選): 專案 ID
- `section_id` (可選): 區段 ID
- `label` (可選): 標籤名稱
- `filter` (可選): 過濾條件
- `lang` (可選): 語言代碼
- `ids` (可選): 任務 ID 列表，逗號分隔

### 過濾器語法

| 過濾器 | 說明 |
|--------|------|
| `today` | 今日到期任務 |
| `tomorrow` | 明日到期任務 |
| `overdue` | 過期任務 |
| `7 days` | 未來 7 天 |
| `no date` | 無日期任務 |
| `p1` | 最高優先級 |
| `p2` | 高優先級 |
| `p3` | 中優先級 |
| `p4` | 低優先級 |
| `#專案名稱` | 特定專案 |
| `@標籤` | 特定標籤 |
| `/區段` | 特定區段 |
| `assigned to: me` | 指派給我 |
| `assigned by: me` | 我指派的 |

組合範例:
- `today | overdue` - 今日或過期
- `7 days & p1` - 未來 7 天的高優先級
- `#工作 & @重要` - 工作專案中帶重要標籤的

### 任務物件結構

```json
{
  "id": "2995104339",
  "project_id": "2203306141",
  "section_id": null,
  "content": "任務標題",
  "description": "任務描述",
  "is_completed": false,
  "labels": ["標籤1", "標籤2"],
  "priority": 4,
  "comment_count": 0,
  "creator_id": "2671355",
  "created_at": "2019-12-11T22:36:50.000000Z",
  "due": {
    "date": "2025-01-30",
    "string": "Jan 30",
    "lang": "en",
    "is_recurring": false,
    "datetime": "2025-01-30T12:00:00.000000Z"
  },
  "url": "https://todoist.com/showTask?id=2995104339"
}
```

### 優先級對應

| API 值 | 顯示 | 說明 |
|--------|------|------|
| 4 | p1 | 最高優先級（紅色） |
| 3 | p2 | 高優先級（橙色） |
| 2 | p3 | 中優先級（藍色） |
| 1 | p4 | 低優先級/無（無色） |

注意：API 中的 priority 值與 UI 顯示相反（4=p1 最高）

### 取得專案

```
GET https://api.todoist.com/rest/v2/projects
```

### 取得標籤

```
GET https://api.todoist.com/rest/v2/labels
```

## 速率限制

- 請求限制: 450 次/15 分鐘
- 超過會返回 429 狀態碼

## Python 範例

```python
import requests

API_TOKEN = "your_token"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# 取得今日任務
response = requests.get(
    "https://api.todoist.com/rest/v2/tasks",
    headers=headers,
    params={"filter": "today"}
)

tasks = response.json()
for task in tasks:
    print(f"[p{5-task['priority']}] {task['content']}")
```

## 同步 API (進階)

若需要批量操作或增量同步，使用 Sync API：

```
POST https://api.todoist.com/sync/v9/sync
```

詳見: https://developer.todoist.com/sync/v9/
