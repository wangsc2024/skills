---
name: daily-digest-notifier
description: |
  每日摘要通知器 - 整合 Google Calendar 與 Todoist，透過 ntfy.sh 發送推播通知。
  Use when: 發送每日摘要、整合行事曆與待辦、自動通知今日行程，or when user mentions 今日摘要, daily digest, 行事曆通知.
  Triggers: "daily digest", "今日摘要", "每日摘要", "行事曆通知", "待辦提醒", "今天的行程", "today's schedule"
version: 1.0.0
compatibility:
  network: true
  endpoints: [api.todoist.com, ntfy.sh]
---

# Daily Digest Notifier

整合 Google Calendar 和 Todoist，透過 ntfy.sh 發送每日摘要通知。

## 依賴技能

- **todoist** - Todoist API 操作（查詢、新增、完成任務）
- **ntfy** - ntfy.sh 推播通知

## 工作流程

```
┌─────────────────┐     ┌─────────────────┐
│ Google Calendar │     │    Todoist      │
│   (gcal API)    │     │   (REST API)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  整合與格式化    │
            │  (digest.py)    │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │    ntfy.sh      │
            │   推播通知       │
            └─────────────────┘
```

## 快速開始

### 1. 設定環境變數

```bash
export TODOIST_API_TOKEN="your_todoist_api_token"
export NTFY_TOPIC="your_unique_topic_name"
```

### 2. 執行每日摘要

```bash
python scripts/digest.py
```

## 核心元件

### Google Calendar 查詢

使用 Claude 內建的 `list_gcal_events` 工具查詢行事曆：

```python
# 查詢今日事件
from datetime import datetime, timedelta

today_start = datetime.now().replace(hour=0, minute=0, second=0).isoformat() + "Z"
today_end = datetime.now().replace(hour=23, minute=59, second=59).isoformat() + "Z"

# 透過 Claude 工具呼叫
# list_gcal_events(time_min=today_start, time_max=today_end)
```

### Todoist API 查詢

```python
import requests

def get_todoist_tasks(api_token, filter_query="today | overdue"):
    """取得 Todoist 任務"""
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # 取得今日任務
    response = requests.get(
        "https://api.todoist.com/rest/v2/tasks",
        headers=headers,
        params={"filter": filter_query}
    )
    return response.json()
```

### ntfy.sh 通知

```python
import requests

def send_ntfy_notification(topic, title, message, priority=3, tags=None):
    """發送 ntfy 通知"""
    payload = {
        "topic": topic,
        "title": title,
        "message": message,
        "priority": priority,
        "tags": tags or ["calendar", "bell"],
        "markdown": True
    }
    
    response = requests.post("https://ntfy.sh", json=payload)
    return response.status_code == 200
```

## 通知格式範本

### 每日摘要格式

```
📅 今日摘要 - 2025/01/30

━━━ 📆 行事曆 (3 項) ━━━
• 09:00 團隊站立會議
• 14:00 客戶簡報
• 18:30 晚餐約會

━━━ ✅ 待辦事項 (5 項) ━━━
🔴 完成專案報告 (高優先)
🟡 回覆郵件 (中優先)
⚪ 整理文件 (低優先)
⏰ 繳費 (已過期!)

祝您有美好的一天！🌟
```

### 優先級對應

| Todoist 優先級 | 顯示 | ntfy 優先級 |
|---------------|------|------------|
| p1 (最高)     | 🔴   | 5 (urgent) |
| p2            | 🟡   | 4 (high)   |
| p3            | 🔵   | 3 (default)|
| p4 (最低)     | ⚪   | 2 (low)    |

## 進階功能

### 自訂查詢參數

```python
# 查詢特定時間範圍
digest_config = {
    "calendar": {
        "days_ahead": 1,        # 查詢未來幾天
        "include_all_day": True # 包含全天事件
    },
    "todoist": {
        "filter": "today | overdue | p1",  # Todoist 過濾器
        "include_completed": False
    },
    "notification": {
        "topic": "my-digest",
        "priority": 3,
        "quiet_hours": ["22:00", "07:00"]  # 勿擾時段
    }
}
```

### 排程執行 (Cron)

```bash
# 每天早上 7:00 發送
0 7 * * * cd /path/to/skill && python scripts/digest.py

# 每天下午 6:00 發送明日預覽
0 18 * * * cd /path/to/skill && python scripts/digest.py --tomorrow
```

## 錯誤處理

| 錯誤 | 原因 | 解決方案 |
|------|------|---------|
| `401 Unauthorized` | Todoist API Token 無效 | 檢查 `TODOIST_API_TOKEN` |
| `gcal 無回應` | Google Calendar 未授權 | 透過 Claude 重新授權 |
| `ntfy 發送失敗` | Topic 名稱無效或網路問題 | 檢查 `NTFY_TOPIC` 和網路 |

## 與 Claude 整合使用

當使用者說「通知我今天的行程」時：

1. 使用 `list_gcal_events` 取得今日行事曆
2. 使用 `scripts/todoist_client.py` 取得 Todoist 任務
3. 使用 `scripts/digest.py` 格式化並發送通知

完整腳本請參考 `scripts/` 目錄。

## 相關參考

- Todoist API 文件: `references/todoist_api.md`
- 進階通知設定: `references/notification_templates.md`
