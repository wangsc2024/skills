#!/usr/bin/env python3
"""
Daily Digest Notifier - 整合 Google Calendar 和 Todoist，透過 ntfy.sh 發送通知
"""

import os
import sys
import json
import requests
import argparse
from datetime import datetime, timedelta
from typing import Optional


# ============ 設定 ============

class Config:
    TODOIST_API_TOKEN = os.environ.get("TODOIST_API_TOKEN", "")
    NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "daily-digest")
    NTFY_URL = "https://ntfy.sh"
    TODOIST_API_URL = "https://api.todoist.com/rest/v2"


# ============ Todoist 客戶端 ============

class TodoistClient:
    """Todoist API 客戶端"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def get_tasks(self, filter_query: str = "today | overdue") -> list:
        """取得任務列表
        
        常用過濾器:
        - "today" - 今日任務
        - "overdue" - 過期任務  
        - "today | overdue" - 今日 + 過期
        - "p1" - 最高優先級
        - "7 days" - 未來 7 天
        - "#專案名稱" - 特定專案
        - "@標籤" - 特定標籤
        """
        try:
            response = requests.get(
                f"{Config.TODOIST_API_URL}/tasks",
                headers=self.headers,
                params={"filter": filter_query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Todoist API 錯誤: {e}", file=sys.stderr)
            return []
    
    def get_projects(self) -> dict:
        """取得專案列表（用於顯示專案名稱）"""
        try:
            response = requests.get(
                f"{Config.TODOIST_API_URL}/projects",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return {p["id"]: p["name"] for p in response.json()}
        except requests.RequestException:
            return {}


# ============ ntfy 通知 ============

class NtfyNotifier:
    """ntfy.sh 通知發送器"""
    
    PRIORITY_MAP = {
        1: 5,  # p1 (最高) -> urgent
        2: 4,  # p2 -> high
        3: 3,  # p3 -> default
        4: 2,  # p4 (最低) -> low
    }
    
    PRIORITY_EMOJI = {
        1: "🔴",
        2: "🟡", 
        3: "🔵",
        4: "⚪",
    }
    
    def __init__(self, topic: str):
        self.topic = topic
    
    def send(
        self,
        message: str,
        title: Optional[str] = None,
        priority: int = 3,
        tags: Optional[list] = None,
        click_url: Optional[str] = None,
        actions: Optional[list] = None
    ) -> bool:
        """發送通知
        
        Args:
            message: 通知內容
            title: 通知標題
            priority: 優先級 1-5
            tags: emoji 標籤列表
            click_url: 點擊後開啟的 URL
            actions: 動作按鈕列表
        """
        payload = {
            "topic": self.topic,
            "message": message,
            "priority": priority,
            "markdown": True,
        }
        
        if title:
            payload["title"] = title
        if tags:
            payload["tags"] = tags
        if click_url:
            payload["click"] = click_url
        if actions:
            payload["actions"] = actions
        
        try:
            response = requests.post(
                Config.NTFY_URL,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            print(f"✅ 通知已發送至 {self.topic}")
            return True
        except requests.RequestException as e:
            print(f"❌ ntfy 發送失敗: {e}", file=sys.stderr)
            return False


# ============ 摘要生成器 ============

class DigestGenerator:
    """每日摘要生成器"""
    
    def __init__(self, todoist_token: str, ntfy_topic: str):
        self.todoist = TodoistClient(todoist_token) if todoist_token else None
        self.notifier = NtfyNotifier(ntfy_topic)
    
    def format_calendar_events(self, events: list) -> str:
        """格式化行事曆事件"""
        if not events:
            return "📆 今日無行事曆事件"
        
        lines = [f"━━━ 📆 行事曆 ({len(events)} 項) ━━━"]
        
        for event in events:
            # 處理時間
            start = event.get("start", {})
            if "dateTime" in start:
                time_str = datetime.fromisoformat(
                    start["dateTime"].replace("Z", "+00:00")
                ).strftime("%H:%M")
            else:
                time_str = "全天"
            
            title = event.get("summary", "無標題")
            location = event.get("location", "")
            
            line = f"• {time_str} {title}"
            if location:
                line += f" 📍{location}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def format_todoist_tasks(self, tasks: list) -> str:
        """格式化 Todoist 任務"""
        if not tasks:
            return "✅ 今日無待辦事項"
        
        lines = [f"━━━ ✅ 待辦事項 ({len(tasks)} 項) ━━━"]
        
        # 按優先級排序 (p1=1 最高)
        sorted_tasks = sorted(tasks, key=lambda x: x.get("priority", 4), reverse=True)
        
        for task in sorted_tasks:
            priority = task.get("priority", 4)
            # Todoist priority: 4=p1(最高), 3=p2, 2=p3, 1=p4(最低)
            # 轉換為我們的顯示: 1=最高, 4=最低
            display_priority = 5 - priority if priority > 0 else 4
            emoji = NtfyNotifier.PRIORITY_EMOJI.get(display_priority, "⚪")
            
            content = task.get("content", "")
            due = task.get("due", {})
            
            line = f"{emoji} {content}"
            
            # 檢查是否過期
            if due and due.get("date"):
                due_date = datetime.strptime(due["date"][:10], "%Y-%m-%d").date()
                if due_date < datetime.now().date():
                    line += " ⏰(已過期!)"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def generate_digest(
        self,
        calendar_events: list = None,
        todoist_filter: str = "today | overdue",
        include_tomorrow: bool = False
    ) -> tuple[str, str]:
        """生成每日摘要
        
        Args:
            calendar_events: 行事曆事件（從 Claude 工具取得）
            todoist_filter: Todoist 過濾條件
            include_tomorrow: 是否包含明日預覽
            
        Returns:
            (title, message) 元組
        """
        today = datetime.now()
        date_str = today.strftime("%Y/%m/%d")
        weekday = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"][today.weekday()]
        
        title = f"📅 {'明日' if include_tomorrow else '今日'}摘要 - {date_str} {weekday}"
        
        sections = []
        
        # 行事曆部分
        if calendar_events is not None:
            sections.append(self.format_calendar_events(calendar_events))
        
        # Todoist 部分
        if self.todoist:
            tasks = self.todoist.get_tasks(todoist_filter)
            sections.append(self.format_todoist_tasks(tasks))
        
        # 加入結尾
        sections.append("\n祝您有美好的一天！🌟")
        
        message = "\n\n".join(sections)
        
        return title, message
    
    def send_digest(
        self,
        calendar_events: list = None,
        todoist_filter: str = "today | overdue",
        priority: int = 3,
        tags: list = None
    ) -> bool:
        """生成並發送摘要通知"""
        title, message = self.generate_digest(calendar_events, todoist_filter)
        
        default_tags = ["calendar", "white_check_mark", "bell"]
        
        return self.notifier.send(
            message=message,
            title=title,
            priority=priority,
            tags=tags or default_tags,
            actions=[
                {
                    "action": "view",
                    "label": "開啟 Todoist",
                    "url": "https://todoist.com/app"
                },
                {
                    "action": "view", 
                    "label": "開啟行事曆",
                    "url": "https://calendar.google.com"
                }
            ]
        )


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="每日摘要通知器")
    parser.add_argument(
        "--tomorrow", 
        action="store_true",
        help="發送明日預覽"
    )
    parser.add_argument(
        "--filter",
        default="today | overdue",
        help="Todoist 過濾條件 (預設: 'today | overdue')"
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="通知優先級 1-5 (預設: 3)"
    )
    parser.add_argument(
        "--topic",
        default=Config.NTFY_TOPIC,
        help=f"ntfy topic (預設: {Config.NTFY_TOPIC})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示摘要，不發送通知"
    )
    
    args = parser.parse_args()
    
    # 檢查必要設定
    if not Config.TODOIST_API_TOKEN:
        print("⚠️  警告: 未設定 TODOIST_API_TOKEN，將跳過 Todoist 任務")
    
    # 建立生成器
    generator = DigestGenerator(
        todoist_token=Config.TODOIST_API_TOKEN,
        ntfy_topic=args.topic
    )
    
    # 注意：calendar_events 需要從 Claude 的 list_gcal_events 工具取得
    # 這裡設為 None，實際使用時由 Claude 注入
    calendar_events = None
    
    if args.dry_run:
        title, message = generator.generate_digest(
            calendar_events=calendar_events,
            todoist_filter=args.filter,
            include_tomorrow=args.tomorrow
        )
        print(f"\n{'='*50}")
        print(f"📬 {title}")
        print(f"{'='*50}")
        print(message)
        print(f"{'='*50}\n")
    else:
        success = generator.send_digest(
            calendar_events=calendar_events,
            todoist_filter=args.filter,
            priority=args.priority
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
