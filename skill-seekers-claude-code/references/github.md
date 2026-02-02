# GitHub 倉庫抓取指南

## 概述

GitHubScraper 負責從 GitHub 倉庫提取文件、程式碼結構和相關資訊。

## 基本用法

```python
import httpx
import base64
import os

async def scrape_github_repo(repo: str):
    """
    抓取 GitHub 倉庫
    
    Args:
        repo: 倉庫路徑，格式為 owner/repo
    """
    
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    api_base = f"https://api.github.com/repos/{repo}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 取得倉庫資訊
        repo_resp = await client.get(api_base, headers=headers)
        repo_data = repo_resp.json()
        
        return {
            "repo": repo,
            "description": repo_data.get("description"),
            "stars": repo_data.get("stargazers_count"),
            "language": repo_data.get("language"),
            "topics": repo_data.get("topics", [])
        }
```

## 抓取內容類型

### README

```python
async def get_readme(client, api_base, headers):
    """取得 README 內容"""
    resp = await client.get(f"{api_base}/readme", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data.get("content", ""))
        return content.decode("utf-8")
    return ""
```

### 目錄結構

```python
async def get_structure(client, api_base, headers):
    """取得目錄結構"""
    resp = await client.get(f"{api_base}/contents", headers=headers)
    if resp.status_code == 200:
        return [
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "path": item.get("path"),
                "size": item.get("size")
            }
            for item in resp.json()
        ]
    return []
```

### 文件目錄

```python
async def get_docs(client, api_base, headers):
    """取得 docs 目錄內容"""
    resp = await client.get(f"{api_base}/contents/docs", headers=headers)
    if resp.status_code == 200:
        return [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "url": item.get("html_url")
            }
            for item in resp.json()
            if item.get("name", "").endswith(".md")
        ]
    return []
```

### 檔案內容

```python
async def get_file_content(client, api_base, path, headers):
    """取得特定檔案內容"""
    resp = await client.get(f"{api_base}/contents/{path}", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("encoding") == "base64":
            content = base64.b64decode(data.get("content", ""))
            return content.decode("utf-8")
    return ""
```

## API 速率限制

GitHub API 有嚴格的速率限制：

| 認證狀態 | 限制 |
|----------|------|
| 未認證 | 60 次/小時 |
| 已認證 | 5,000 次/小時 |

### 設定 Token

```bash
# 環境變數
export GITHUB_TOKEN=ghp_your_token_here
```

```python
# 程式碼中使用
headers = {"Authorization": f"token {os.environ.get('GITHUB_TOKEN')}"}
```

### 檢查剩餘配額

```python
async def check_rate_limit(client, headers):
    """檢查 API 配額"""
    resp = await client.get(
        "https://api.github.com/rate_limit",
        headers=headers
    )
    data = resp.json()
    core = data.get("resources", {}).get("core", {})
    return {
        "remaining": core.get("remaining"),
        "limit": core.get("limit"),
        "reset": core.get("reset")  # Unix timestamp
    }
```

## 程式碼分析

### 表面分析 (Surface)

只分析導出的公開 API：

```python
def analyze_python_exports(content: str):
    """分析 Python 檔案的導出"""
    import ast
    
    tree = ast.parse(content)
    exports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'):
                exports.append({
                    "type": "function",
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args]
                })
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith('_'):
                exports.append({
                    "type": "class",
                    "name": node.name
                })
    
    return exports
```

### 深度分析 (Deep)

包含內部函數、型別定義、依賴關係：

```python
def deep_analyze_python(content: str):
    """深度分析 Python 檔案"""
    import ast
    
    tree = ast.parse(content)
    
    analysis = {
        "imports": [],
        "functions": [],
        "classes": [],
        "constants": []
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                analysis["imports"].append(f"{module}.{alias.name}")
        elif isinstance(node, ast.FunctionDef):
            analysis["functions"].append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "decorators": [d.id for d in node.decorator_list if hasattr(d, 'id')],
                "docstring": ast.get_docstring(node)
            })
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, ast.FunctionDef)
            ]
            analysis["classes"].append({
                "name": node.name,
                "methods": methods,
                "docstring": ast.get_docstring(node)
            })
    
    return analysis
```

## 輸出格式

```json
{
  "repo": "facebook/react",
  "description": "A declarative, efficient, and flexible JavaScript library for building user interfaces.",
  "stars": 220000,
  "forks": 45000,
  "language": "JavaScript",
  "topics": ["javascript", "frontend", "ui", "react"],
  "readme": "# React\n\n...",
  "structure": [
    {"name": "src", "type": "dir", "path": "src"},
    {"name": "packages", "type": "dir", "path": "packages"},
    {"name": "README.md", "type": "file", "path": "README.md"}
  ],
  "docs": [
    {"name": "getting-started.md", "path": "docs/getting-started.md"},
    {"name": "api.md", "path": "docs/api.md"}
  ],
  "code_analysis": {
    "functions": [...],
    "classes": [...],
    "exports": [...]
  }
}
```

## 錯誤處理

```python
async def safe_github_scrape(repo: str):
    """安全的 GitHub 抓取"""
    try:
        result = await scrape_github_repo(repo)
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"倉庫不存在: {repo}")
        elif e.response.status_code == 403:
            print("API 配額用盡，請設定 GITHUB_TOKEN")
        elif e.response.status_code == 401:
            print("Token 無效")
        return None
    except Exception as e:
        print(f"未預期錯誤: {e}")
        return None
```
