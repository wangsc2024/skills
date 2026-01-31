#!/usr/bin/env python3
"""
Claude 整合腳本 - 接收 Claude 工具的 Google Calendar 結果並整合發送通知

使用方式:
1. Claude 使用 list_gcal_events 取得行事曆
2. Claude 將結果以 JSON 傳入此腳本
3. 腳本整合 Todoist 並發送 ntfy 通知
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 導入同目錄的模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digest import DigestGenerator, Config


def parse_gcal_events(events_json: str) -> list:
    """解析 Google Calendar 事件 JSON
    
    支援格式:
    1. JSON 字串
    2. 從 stdin 讀取
    3. 從檔案讀取
    """
    if events_json == "-":
        # 從 stdin 讀取
        events_json = sys.stdin.read()
    elif os.path.isfile(events_json):
        # 從檔案讀取
        with open(events_json, "r", encoding="utf-8") as f:
            events_json = f.read()
    
    try:
        data = json.loads(events_json)
        
        # 處理不同格式
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 可能是 {items: [...]} 格式
            return data.get("items", data.get("events", [data]))
        else:
            return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析錯誤: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Claude 整合腳本 - 接收行事曆並發送通知",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用 JSON 字串
  python claude_integration.py --events '[{"summary": "會議", "start": {"dateTime": "2025-01-30T09:00:00Z"}}]'
  
  # 從 stdin 讀取
  echo '{"items": [...]}' | python claude_integration.py --events -
  
  # 從檔案讀取
  python claude_integration.py --events events.json
  
  # 不包含行事曆，只發送 Todoist
  python claude_integration.py --todoist-only
"""
    )
    
    parser.add_argument(
        "--events",
        help="Google Calendar 事件 JSON (字串、檔案路徑或 - 表示 stdin)"
    )
    parser.add_argument(
        "--todoist-filter",
        default="today | overdue",
        help="Todoist 過濾條件"
    )
    parser.add_argument(
        "--todoist-only",
        action="store_true",
        help="只發送 Todoist 任務，不包含行事曆"
    )
    parser.add_argument(
        "--topic",
        default=Config.NTFY_TOPIC,
        help="ntfy topic"
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="通知優先級"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示摘要，不發送"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="輸出 JSON 格式的通知內容"
    )
    
    args = parser.parse_args()
    
    # 解析行事曆事件
    calendar_events = None
    if args.events and not args.todoist_only:
        calendar_events = parse_gcal_events(args.events)
        print(f"📆 已載入 {len(calendar_events)} 個行事曆事件")
    
    # 建立生成器
    generator = DigestGenerator(
        todoist_token=Config.TODOIST_API_TOKEN,
        ntfy_topic=args.topic
    )
    
    # 生成摘要
    title, message = generator.generate_digest(
        calendar_events=calendar_events,
        todoist_filter=args.todoist_filter
    )
    
    if args.output_json:
        # 輸出 JSON 格式
        output = {
            "title": title,
            "message": message,
            "topic": args.topic,
            "priority": args.priority,
            "tags": ["calendar", "white_check_mark", "bell"]
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    
    if args.dry_run:
        # 預覽模式
        print(f"\n{'='*50}")
        print(f"📬 {title}")
        print(f"{'='*50}")
        print(message)
        print(f"{'='*50}")
        print(f"\nTopic: {args.topic}")
        print(f"Priority: {args.priority}")
        return
    
    # 發送通知
    success = generator.notifier.send(
        message=message,
        title=title,
        priority=args.priority,
        tags=["calendar", "white_check_mark", "bell"],
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
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
