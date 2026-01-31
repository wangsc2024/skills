---
name: todoist
description: Todoist 待辦事項整合 - 查詢、新增、完成、刪除任務。當用戶說「我的待辦事項」、「todoist」、「新增任務」、「完成任務」、「今日任務」、「過期任務」、「查詢 todo」時觸發。支援專案、標籤、優先級、截止日期等完整功能。需要設定 TODOIST_API_TOKEN 環境變數。
---

# Todoist 待辦事項整合

透過 REST API v2 管理 Todoist 任務。

## 環境設定

```bash
export TODOIST_API_TOKEN="your_api_token"
```

Token 取得：https://todoist.com/app/settings/integrations/developer

## 快速使用

### 查詢任務

```bash
# 今日 + 過期任務
python scripts/todoist.py list

# 自訂過濾器
python scripts/todoist.py list --filter "p1 | p2"
python scripts/todoist.py list --filter "7 days"
python scripts/todoist.py list --filter "#工作 & @重要"
```

### 新增任務

```bash
python scripts/todoist.py add "完成報告" --due "tomorrow" --priority 4
```

### 完成任務

```bash
python scripts/todoist.py complete <task_id>
```

## API 使用（Python）

```python
import os
import requests

TOKEN = os.environ["TODOIST_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# 查詢任務
def get_tasks(filter_query="today | overdue"):
    response = requests.get(
        "https://api.todoist.com/rest/v2/tasks",
        headers=HEADERS,
        params={"filter": filter_query}
    )
    return response.json()

# 新增任務
def add_task(content, due_string=None, priority=1):
    data = {"content": content}
    if due_string:
        data["due_string"] = due_string
    if priority:
        data["priority"] = priority  # 4=p1最高, 1=p4最低
    
    response = requests.post(
        "https://api.todoist.com/rest/v2/tasks",
        headers=HEADERS,
        json=data
    )
    return response.json()

# 完成任務
def complete_task(task_id):
    requests.post(
        f"https://api.todoist.com/rest/v2/tasks/{task_id}/close",
        headers=HEADERS
    )
```

## 過濾器語法

| 過濾器 | 說明 |
|--------|------|
| `today` | 今日任務 |
| `tomorrow` | 明日任務 |
| `overdue` | 過期任務 |
| `7 days` | 未來 7 天 |
| `no date` | 無日期任務 |
| `p1`, `p2`, `p3`, `p4` | 按優先級 |
| `#專案名稱` | 特定專案 |
| `@標籤` | 特定標籤 |
| `assigned to: me` | 指派給我 |

組合：`today | overdue`（或）、`#工作 & p1`（且）

## 優先級對應

| API 值 | 顯示 | Emoji | 說明 |
|--------|------|-------|------|
| 4 | p1 | 🔴 | 最高優先級 |
| 3 | p2 | 🟡 | 高優先級 |
| 2 | p3 | 🔵 | 中優先級 |
| 1 | p4 | ⚪ | 低優先級 |

## 任務物件結構

```json
{
  "id": "2995104339",
  "content": "任務標題",
  "description": "任務描述",
  "project_id": "2203306141",
  "priority": 4,
  "due": {
    "date": "2025-01-30",
    "datetime": "2025-01-30T12:00:00Z",
    "is_recurring": false
  },
  "labels": ["工作", "重要"],
  "is_completed": false,
  "url": "https://todoist.com/showTask?id=2995104339"
}
```

## 格式化輸出

```python
def format_tasks(tasks):
    EMOJI = {4: "🔴", 3: "🟡", 2: "🔵", 1: "⚪"}
    lines = []
    
    for task in sorted(tasks, key=lambda x: x.get("priority", 1), reverse=True):
        emoji = EMOJI.get(task.get("priority", 1), "⚪")
        content = task.get("content", "")
        
        # 檢查過期
        due = task.get("due", {})
        overdue = ""
        if due and due.get("date"):
            from datetime import datetime
            due_date = datetime.strptime(due["date"][:10], "%Y-%m-%d").date()
            if due_date < datetime.now().date():
                overdue = " ⏰(過期!)"
        
        lines.append(f"{emoji} {content}{overdue}")
    
    return "\n".join(lines)
```

## 錯誤處理

| 狀態碼 | 原因 | 解決方案 |
|--------|------|---------|
| 401 | Token 無效 | 檢查 TODOIST_API_TOKEN |
| 403 | 權限不足 | 確認 Token 權限 |
| 404 | 任務不存在 | 確認 task_id |
| 429 | 請求過多 | 等待後重試（限制 450/15min） |

## 參考資料

- 完整 API 文件：`references/api_reference.md`
- 過濾器語法：`references/filter_syntax.md`
