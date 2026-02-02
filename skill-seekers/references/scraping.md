# 網頁抓取指南

## 概述

DocumentationScraper 是 Skill Seekers 的核心元件，負責從技術文件網站抓取內容。

## 抓取流程

```
1. 初始化
   ├── 解析 base_url
   ├── 載入設定
   └── 建立 HTTP client

2. 來源偵測
   ├── 檢查 /llms.txt (優先)
   ├── 檢查 /sitemap.xml
   └── 分析導航結構

3. 頁面探索
   ├── 從起始頁開始
   ├── 提取所有內部連結
   ├── 過濾排除路徑
   └── 建立抓取佇列

4. 內容抓取
   ├── 平行抓取頁面
   ├── 提取標題、內容、程式碼
   ├── 偵測程式語言
   └── 自動分類

5. 後處理
   ├── 移除重複內容
   ├── 整理程式碼範例
   └── 生成統計資訊
```

## 基本用法

```python
from scripts.scraper import DocumentationScraper

# 最簡單的用法
scraper = DocumentationScraper(base_url="https://docs.python.org/3/")
result = await scraper.scrape()
```

## 進階設定

```python
scraper = DocumentationScraper(
    base_url="https://docs.python.org/3/",
    
    # 抓取設定
    max_pages=100,           # 最大頁數
    mode="full",             # quick, full, custom
    rate_limit=0.5,          # 請求間隔（秒）
    timeout=30,              # 請求超時（秒）
    
    # 內容選擇器
    selectors={
        "main_content": "article, main, .content",
        "title": "h1",
        "code_blocks": "pre code",
        "navigation": "nav, .sidebar"
    },
    
    # URL 過濾
    include_patterns=["/tutorial", "/library", "/reference"],
    exclude_patterns=["/whatsnew", "/bugs", "/license"],
    
    # 其他選項
    extract_code=True,       # 提取程式碼區塊
    detect_language=True,    # 偵測程式語言
    follow_redirects=True    # 跟隨重定向
)
```

## llms.txt 支援

llms.txt 是專為 LLM 設計的文件格式，Skill Seekers 會優先使用：

```python
# 自動檢查並使用 llms.txt
scraper = DocumentationScraper(base_url="https://example.com/docs")

# 手動檢查
llms_content = await scraper.check_llms_txt()
if llms_content:
    print("找到 llms.txt！")
    # 直接使用 llms.txt 內容，跳過爬蟲
```

檢查順序：
1. `/llms-full.txt` - 完整版
2. `/llms.txt` - 標準版
3. `/llms-small.txt` - 精簡版

## 選擇器指南

### 常見文件框架選擇器

| 框架 | main_content | title | code_blocks |
|------|-------------|-------|-------------|
| MkDocs | `.md-content` | `h1` | `pre code` |
| Docusaurus | `article` | `h1` | `pre code` |
| Sphinx | `.body` | `h1` | `.highlight pre` |
| GitBook | `.page-inner` | `h1` | `pre code` |
| VuePress | `.theme-default-content` | `h1` | `div[class*="language-"] pre` |
| Nextra | `main` | `h1` | `pre code` |
| ReadTheDocs | `.document` | `h1` | `.highlight pre` |
| Docsify | `article` | `h1` | `pre code` |

### 自動偵測

如果不提供選擇器，會嘗試以下順序：

```python
DEFAULT_SELECTORS = {
    "main_content": [
        "article",
        "main",
        '[role="main"]',
        ".content",
        ".main-content",
        ".documentation",
        "#content"
    ],
    "title": ["h1", "title"],
    "code_blocks": ["pre code", "pre", ".highlight code"]
}
```

## 程式碼語言偵測

自動偵測程式碼區塊的語言：

```python
LANGUAGE_PATTERNS = {
    "python": [r"\bdef\s+\w+\(", r"\bimport\s+\w+", r"\bclass\s+\w+:"],
    "javascript": [r"\bconst\s+\w+\s*=", r"\blet\s+\w+", r"=>\s*{"],
    "typescript": [r":\s*(string|number|boolean)", r"\binterface\s+\w+"],
    "jsx": [r"<\w+[^>]*/>", r"useState\s*\(", r"useEffect\s*\("],
    "java": [r"\bpublic\s+class\s+\w+", r"System\.out\.println"],
    "go": [r"\bfunc\s+\w+\s*\(", r"\bpackage\s+\w+"],
    "rust": [r"\bfn\s+\w+\s*\(", r"\blet\s+mut\s+"],
    "bash": [r"#!/bin/bash", r"\$\{?\w+\}?"],
    "sql": [r"\bSELECT\s+", r"\bFROM\s+", r"\bWHERE\s+"],
    "html": [r"<!DOCTYPE", r"<html", r"<div"],
    "css": [r"\{[^}]*:\s*[^;]+;", r"@media"],
    "json": [r'^\s*\{[\s\S]*"[\w]+"']
}
```

支援的語言：
- Python, JavaScript, TypeScript, JSX/TSX
- Java, Go, Rust, C, C++, C#
- HTML, CSS, SCSS, LESS
- SQL, GraphQL
- Bash, Shell, PowerShell
- JSON, YAML, TOML
- Markdown, XML

## 模式說明

### Quick 模式（預設）

```python
scraper = DocumentationScraper(base_url=url, mode="quick")
# - 最多 20 頁
# - 優先抓取入門和 API 頁面
# - 適合快速了解框架
```

### Full 模式

```python
scraper = DocumentationScraper(base_url=url, mode="full")
# - 抓取所有頁面（最多 max_pages）
# - 完整分類
# - 適合建立完整知識庫
```

### Custom 模式

```python
scraper = DocumentationScraper(
    base_url=url,
    mode="custom",
    include_patterns=["/api", "/hooks"]  # 只抓取指定路徑
)
# - 只抓取符合 include_patterns 的頁面
# - 適合只需要特定章節
```

## 錯誤處理

```python
from scripts.scraper import DocumentationScraper, ScraperError

try:
    result = await scraper.scrape()
except ScraperError as e:
    match e.code:
        case "RATE_LIMITED":
            # 被速率限制
            await asyncio.sleep(60)
            result = await scraper.scrape()
        case "AUTH_REQUIRED":
            # 需要登入
            print("此網站需要登入")
        case "TIMEOUT":
            # 請求超時
            scraper.timeout = 60
            result = await scraper.scrape()
        case "NOT_FOUND":
            # 頁面不存在
            print("找不到頁面")
```

## 效能優化

### 平行抓取

```python
# 預設使用 asyncio 平行抓取
scraper = DocumentationScraper(
    base_url=url,
    max_concurrent=10  # 同時最多 10 個請求
)
```

### 快取

```python
# 啟用快取避免重複抓取
scraper = DocumentationScraper(
    base_url=url,
    cache_dir=".cache/",
    cache_ttl=3600  # 快取 1 小時
)
```

### 增量更新

```python
# 只抓取新增或修改的頁面
scraper = DocumentationScraper(
    base_url=url,
    incremental=True,
    last_run="2025-01-28T00:00:00Z"
)
```

## 輸出結構

```python
result = await scraper.scrape()

# result 結構
{
    "url": "https://example.com/docs",
    "pages": [
        {
            "url": "https://example.com/docs/intro",
            "title": "Introduction",
            "content": "...",
            "code_blocks": [
                {"language": "python", "code": "..."}
            ],
            "category": "getting_started"
        }
    ],
    "categories": {
        "getting_started": 10,
        "api_reference": 25,
        "guides": 15
    },
    "statistics": {
        "total_pages": 50,
        "total_code_blocks": 120,
        "languages_detected": ["python", "javascript", "bash"]
    }
}
```
