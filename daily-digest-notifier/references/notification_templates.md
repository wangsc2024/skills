# 通知模板參考

## 預設模板

### 每日摘要模板

```
📅 今日摘要 - {date} {weekday}

━━━ 📆 行事曆 ({event_count} 項) ━━━
{calendar_events}

━━━ ✅ 待辦事項 ({task_count} 項) ━━━
{todoist_tasks}

祝您有美好的一天！🌟
```

### 緊急提醒模板

```
🚨 緊急提醒

{urgent_items}

請立即處理！
```

### 會議提醒模板

```
📢 即將開始的會議

🕐 {time}
📋 {meeting_title}
📍 {location}
👥 {attendees}

{meeting_link}
```

## 自訂模板設定

在 `scripts/digest.py` 中設定：

```python
TEMPLATES = {
    "daily": """
📅 {title}

{calendar_section}

{tasks_section}

{footer}
""",
    "urgent": """
🚨 緊急事項

{items}
""",
    "meeting": """
📢 {meeting_title}
🕐 {time} | 📍 {location}
""",
}
```

## 條件格式

### 空內容處理

```python
# 無事件時
if not events:
    calendar_section = "📆 今日無行事曆事件"
else:
    calendar_section = format_events(events)
```

### 優先級顏色

| 優先級 | Emoji | 說明 |
|--------|-------|------|
| p1 | 🔴 | 最高優先級，立即處理 |
| p2 | 🟡 | 高優先級，今日完成 |
| p3 | 🔵 | 中優先級，按計畫進行 |
| p4 | ⚪ | 低優先級，有空再做 |
| 過期 | ⏰ | 需要立即關注 |

### 時間格式

```python
# 24 小時制
time_str = datetime.strftime("%H:%M")  # 14:30

# 12 小時制
time_str = datetime.strftime("%I:%M %p")  # 02:30 PM

# 相對時間
def relative_time(dt):
    diff = dt - datetime.now()
    if diff.days == 0:
        return "今天"
    elif diff.days == 1:
        return "明天"
    elif diff.days == -1:
        return "昨天"
    else:
        return dt.strftime("%m/%d")
```

## 多語言支援

```python
LANG = {
    "zh-TW": {
        "today": "今日",
        "tomorrow": "明日",
        "calendar": "行事曆",
        "tasks": "待辦事項",
        "no_events": "無行事曆事件",
        "no_tasks": "無待辦事項",
        "overdue": "已過期",
        "greeting": "祝您有美好的一天！🌟",
        "weekdays": ["週一", "週二", "週三", "週四", "週五", "週六", "週日"],
    },
    "en": {
        "today": "Today",
        "tomorrow": "Tomorrow",
        "calendar": "Calendar",
        "tasks": "Tasks",
        "no_events": "No calendar events",
        "no_tasks": "No tasks",
        "overdue": "Overdue",
        "greeting": "Have a great day! 🌟",
        "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }
}
```

## ntfy 特殊功能

### Markdown 支援

啟用 `"markdown": true` 後支援：

```markdown
**粗體** _斜體_ `程式碼`
[連結](https://example.com)
```

### 動作按鈕範例

```json
{
  "actions": [
    {
      "action": "view",
      "label": "開啟 Todoist",
      "url": "https://todoist.com/app"
    },
    {
      "action": "view",
      "label": "開啟行事曆", 
      "url": "https://calendar.google.com"
    },
    {
      "action": "http",
      "label": "完成首要任務",
      "url": "https://api.todoist.com/rest/v2/tasks/{task_id}/close",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer {token}"
      },
      "clear": true
    }
  ]
}
```

### 延遲發送

```python
# 早上 7:00 發送
payload["delay"] = "tomorrow 7am"

# 30 分鐘後發送
payload["delay"] = "30m"

# 特定時間
payload["delay"] = "2025-01-30T07:00:00+08:00"
```

## 情境範本

### 工作日模板

```
📅 工作日摘要 - {date}

🎯 今日重點
{top_priority_tasks}

📆 會議安排
{meetings}

📋 待辦清單
{other_tasks}

💪 加油！
```

### 週末模板

```
🌴 週末提醒 - {date}

📋 未完成事項
{pending_tasks}

📅 週末活動
{weekend_events}

好好休息！😊
```

### 週一模板

```
🌟 新的一週開始了！

📊 本週重點 ({week_range})
{week_highlights}

📅 今日行程
{today_events}

✅ 待辦事項
{today_tasks}

開啟美好的一週！💪
```
