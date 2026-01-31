#!/usr/bin/env python3
"""
Todoist 客戶端 - 獨立使用的 Todoist API 工具
"""

import os
import sys
import json
import requests
import argparse
from datetime import datetime
from typing import Optional


class TodoistClient:
    """Todoist REST API v2 客戶端"""
    
    API_BASE = "https://api.todoist.com/rest/v2"
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[dict]:
        """發送 API 請求"""
        url = f"{self.API_BASE}/{endpoint}"
        try:
            response = requests.request(
                method, url, 
                headers=self.headers,
                timeout=10,
                **kwargs
            )
            response.raise_for_status()
            if response.text:
                return response.json()
            return None
        except requests.RequestException as e:
            print(f"❌ API 錯誤: {e}", file=sys.stderr)
            return None
    
    # ===== 任務操作 =====
    
    def get_tasks(self, filter_query: str = None, project_id: str = None) -> list:
        """取得任務列表"""
        params = {}
        if filter_query:
            params["filter"] = filter_query
        if project_id:
            params["project_id"] = project_id
        
        return self._request("GET", "tasks", params=params) or []
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """取得單一任務"""
        return self._request("GET", f"tasks/{task_id}")
    
    def create_task(
        self, 
        content: str,
        description: str = None,
        project_id: str = None,
        due_string: str = None,
        priority: int = 1,
        labels: list = None
    ) -> Optional[dict]:
        """建立任務"""
        data = {"content": content}
        if description:
            data["description"] = description
        if project_id:
            data["project_id"] = project_id
        if due_string:
            data["due_string"] = due_string
        if priority:
            data["priority"] = priority  # 1=p4, 4=p1
        if labels:
            data["labels"] = labels
        
        return self._request("POST", "tasks", json=data)
    
    def complete_task(self, task_id: str) -> bool:
        """完成任務"""
        result = self._request("POST", f"tasks/{task_id}/close")
        return result is None  # 成功時返回空內容
    
    def reopen_task(self, task_id: str) -> bool:
        """重新開啟任務"""
        result = self._request("POST", f"tasks/{task_id}/reopen")
        return result is None
    
    def delete_task(self, task_id: str) -> bool:
        """刪除任務"""
        result = self._request("DELETE", f"tasks/{task_id}")
        return result is None
    
    # ===== 專案操作 =====
    
    def get_projects(self) -> list:
        """取得所有專案"""
        return self._request("GET", "projects") or []
    
    def get_project(self, project_id: str) -> Optional[dict]:
        """取得單一專案"""
        return self._request("GET", f"projects/{project_id}")
    
    # ===== 標籤操作 =====
    
    def get_labels(self) -> list:
        """取得所有標籤"""
        return self._request("GET", "labels") or []
    
    # ===== 格式化輸出 =====
    
    @staticmethod
    def format_task(task: dict, show_project: bool = False) -> str:
        """格式化單一任務"""
        priority = task.get("priority", 1)
        emoji = {4: "🔴", 3: "🟡", 2: "🔵", 1: "⚪"}.get(priority, "⚪")
        
        content = task.get("content", "")
        
        # 截止日期
        due_str = ""
        due = task.get("due")
        if due:
            due_date = due.get("date", "")[:10]
            if due_date:
                due_dt = datetime.strptime(due_date, "%Y-%m-%d").date()
                today = datetime.now().date()
                if due_dt < today:
                    due_str = " ⏰(過期!)"
                elif due_dt == today:
                    due_str = " 📅(今日)"
        
        # 標籤
        labels = task.get("labels", [])
        labels_str = " " + " ".join([f"@{l}" for l in labels]) if labels else ""
        
        return f"{emoji} {content}{due_str}{labels_str}"
    
    def print_tasks(self, tasks: list):
        """列印任務列表"""
        if not tasks:
            print("✅ 無任務")
            return
        
        # 按優先級排序
        sorted_tasks = sorted(tasks, key=lambda x: x.get("priority", 1), reverse=True)
        
        for task in sorted_tasks:
            print(self.format_task(task))


def main():
    parser = argparse.ArgumentParser(description="Todoist CLI 工具")
    parser.add_argument("--token", help="API Token (或設定 TODOIST_API_TOKEN 環境變數)")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出任務")
    list_parser.add_argument("-f", "--filter", default="today | overdue", help="過濾條件")
    list_parser.add_argument("--json", action="store_true", help="輸出 JSON 格式")
    
    # add 命令
    add_parser = subparsers.add_parser("add", help="新增任務")
    add_parser.add_argument("content", help="任務內容")
    add_parser.add_argument("-d", "--due", help="截止日期 (如: today, tomorrow, 2025-01-30)")
    add_parser.add_argument("-p", "--priority", type=int, choices=[1,2,3,4], default=1, help="優先級 (1=p4, 4=p1)")
    
    # complete 命令
    complete_parser = subparsers.add_parser("complete", help="完成任務")
    complete_parser.add_argument("task_id", help="任務 ID")
    
    # projects 命令
    subparsers.add_parser("projects", help="列出專案")
    
    # labels 命令
    subparsers.add_parser("labels", help="列出標籤")
    
    args = parser.parse_args()
    
    # 取得 Token
    token = args.token or os.environ.get("TODOIST_API_TOKEN")
    if not token:
        print("❌ 請設定 TODOIST_API_TOKEN 環境變數或使用 --token 參數", file=sys.stderr)
        sys.exit(1)
    
    client = TodoistClient(token)
    
    if args.command == "list":
        tasks = client.get_tasks(filter_query=args.filter)
        if args.json:
            print(json.dumps(tasks, indent=2, ensure_ascii=False))
        else:
            print(f"📋 任務列表 (filter: {args.filter})\n")
            client.print_tasks(tasks)
    
    elif args.command == "add":
        task = client.create_task(
            content=args.content,
            due_string=args.due,
            priority=args.priority
        )
        if task:
            print(f"✅ 已建立任務: {task.get('content')}")
            print(f"   ID: {task.get('id')}")
        else:
            print("❌ 建立任務失敗")
            sys.exit(1)
    
    elif args.command == "complete":
        if client.complete_task(args.task_id):
            print(f"✅ 已完成任務 {args.task_id}")
        else:
            print(f"❌ 完成任務失敗")
            sys.exit(1)
    
    elif args.command == "projects":
        projects = client.get_projects()
        print("📁 專案列表\n")
        for p in projects:
            print(f"• {p.get('name')} (ID: {p.get('id')})")
    
    elif args.command == "labels":
        labels = client.get_labels()
        print("🏷️  標籤列表\n")
        for l in labels:
            print(f"• @{l.get('name')} (ID: {l.get('id')})")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
