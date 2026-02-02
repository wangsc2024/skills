# GitHub 倉庫抓取指南

## 概述

GitHubScraper 負責從 GitHub 倉庫提取文件、程式碼結構和相關資訊。

## 基本用法

```python
from scripts.github_scraper import GitHubScraper

scraper = GitHubScraper(repo="facebook/react")
result = await scraper.scrape()
```

## 進階設定

```python
scraper = GitHubScraper(
    repo="facebook/react",
    
    # 內容選項
    include_readme=True,      # README.md
    include_docs=True,        # /docs 目錄
    include_code=True,        # 原始碼分析
    include_issues=False,     # GitHub Issues
    include_releases=True,    # Release 資訊
    include_wiki=False,       # Wiki 頁面
    
    # 程式碼分析
    code_analysis_depth="surface",  # surface, deep
    max_files=100,
    
    # 檔案過濾
    file_patterns={
        "include": ["*.md", "*.py", "*.ts", "*.js"],
        "exclude": ["*.test.*", "*.spec.*", "node_modules/*"]
    },
    
    # 認證（提高 API 限制）
    github_token="ghp_xxx"  # 或使用環境變數 GITHUB_TOKEN
)
```

## 抓取內容

### README.md

```python
result = await scraper.scrape()
readme = result["readme"]  # Markdown 內容
```

### 倉庫結構

```python
structure = result["structure"]
# [
#   {"name": "src", "type": "dir", "path": "src"},
#   {"name": "README.md", "type": "file", "path": "README.md"},
#   ...
# ]
```

### 程式碼分析

```python
# 啟用程式碼分析
scraper = GitHubScraper(repo="...", include_code=True)
result = await scraper.scrape()

code_analysis = result["code_analysis"]
# {
#   "functions": [
#     {"name": "useState", "file": "src/hooks.ts", "params": ["initialState"]},
#     ...
#   ],
#   "classes": [...],
#   "exports": [...],
#   "dependencies": ["react", "react-dom", ...]
# }
```

### Issues 和 PRs

```python
scraper = GitHubScraper(
    repo="...",
    include_issues=True,
    max_issues=50
)
result = await scraper.scrape()

issues = result["issues"]
# [
#   {
#     "number": 1,
#     "title": "Bug: ...",
#     "labels": ["bug", "help wanted"],
#     "state": "open",
#     "body": "..."
#   },
#   ...
# ]
```

### Releases

```python
scraper = GitHubScraper(repo="...", include_releases=True)
result = await scraper.scrape()

releases = result["releases"]
# [
#   {
#     "tag": "v18.2.0",
#     "name": "React 18.2.0",
#     "date": "2023-06-14",
#     "notes": "## What's Changed\n..."
#   },
#   ...
# ]
```

## API 速率限制

GitHub API 有速率限制：
- 未認證：60 次/小時
- 已認證：5000 次/小時

```python
# 使用 token 提高限制
import os
os.environ["GITHUB_TOKEN"] = "ghp_your_token"

# 或直接傳入
scraper = GitHubScraper(
    repo="...",
    github_token="ghp_your_token"
)

# 檢查剩餘配額
remaining = await scraper.check_rate_limit()
print(f"剩餘請求數：{remaining}")
```

### 取得 GitHub Token

1. 前往 GitHub Settings → Developer settings → Personal access tokens
2. 點擊 "Generate new token (classic)"
3. 選擇權限：`repo`（讀取公開/私有倉庫）
4. 複製 token

## 程式碼分析深度

### surface（表面分析）

```python
scraper = GitHubScraper(
    repo="...",
    code_analysis_depth="surface"
)
# 結果：
# - 公開函數、類別、常數
# - 導出的 API
# - 檔案層級的結構
```

### deep（深度分析）

```python
scraper = GitHubScraper(
    repo="...",
    code_analysis_depth="deep"
)
# 結果：
# - 包含內部函數
# - 完整型別定義
# - 依賴關係圖
# - 程式碼複雜度指標
```

## 檔案過濾

```python
scraper = GitHubScraper(
    repo="...",
    file_patterns={
        "include": [
            "*.md",           # Markdown 文件
            "*.py",           # Python 檔案
            "*.ts", "*.tsx",  # TypeScript
            "*.js", "*.jsx",  # JavaScript
            "src/**/*",       # src 目錄下所有檔案
        ],
        "exclude": [
            "*.test.*",       # 測試檔案
            "*.spec.*",       # 規格檔案
            "node_modules/*", # 依賴目錄
            "dist/*",         # 建置輸出
            "*.min.js",       # 壓縮檔案
            "__pycache__/*",  # Python 快取
        ]
    }
)
```

## 輸出範例

```json
{
  "repo": "facebook/react",
  "description": "A declarative, efficient, and flexible JavaScript library for building user interfaces.",
  "stars": 220000,
  "forks": 45000,
  "language": "JavaScript",
  "topics": ["javascript", "frontend", "ui", "react"],
  "license": "MIT",
  "readme": "# React\n\nReact is a JavaScript library for building user interfaces.\n\n...",
  "structure": [
    {"name": "packages", "type": "dir", "path": "packages"},
    {"name": "scripts", "type": "dir", "path": "scripts"},
    {"name": "README.md", "type": "file", "path": "README.md"}
  ],
  "docs": [
    {"name": "getting-started.md", "path": "docs/getting-started.md", "content": "..."},
    {"name": "hooks.md", "path": "docs/hooks.md", "content": "..."}
  ],
  "code_analysis": {
    "functions": [
      {"name": "useState", "file": "packages/react/src/ReactHooks.js", "exported": true},
      {"name": "useEffect", "file": "packages/react/src/ReactHooks.js", "exported": true}
    ],
    "classes": [],
    "exports": ["useState", "useEffect", "useContext", "useReducer"],
    "dependencies": []
  },
  "releases": [
    {"tag": "v18.2.0", "date": "2023-06-14", "notes": "..."}
  ],
  "statistics": {
    "total_files": 150,
    "total_lines": 50000,
    "languages": {"JavaScript": 80, "TypeScript": 15, "Markdown": 5}
  }
}
```

## 與文件合併

```python
from scripts.github_scraper import GitHubScraper
from scripts.scraper import DocumentationScraper
from scripts.utils import merge_content, detect_conflicts

async def create_unified_skill(doc_url: str, repo: str, name: str):
    # 抓取文件
    doc_scraper = DocumentationScraper(base_url=doc_url)
    doc_content = await doc_scraper.scrape()
    
    # 抓取 GitHub
    github_scraper = GitHubScraper(repo=repo, include_code=True)
    github_content = await github_scraper.scrape()
    
    # 偵測衝突
    conflicts = detect_conflicts(doc_content, github_content)
    
    # 合併
    merged = merge_content([doc_content, github_content])
    merged["conflicts"] = conflicts
    
    return merged
```

## 錯誤處理

```python
from scripts.github_scraper import GitHubScraper, GitHubError

try:
    result = await scraper.scrape()
except GitHubError as e:
    match e.code:
        case "NOT_FOUND":
            print("倉庫不存在或為私有")
        case "RATE_LIMITED":
            print("API 配額用盡，請設定 GITHUB_TOKEN")
        case "AUTH_FAILED":
            print("Token 無效或已過期")
        case "FORBIDDEN":
            print("沒有權限存取此倉庫")
```
