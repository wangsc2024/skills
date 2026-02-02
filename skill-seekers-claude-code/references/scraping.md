# 網頁抓取指南

## 概述

文件網站抓取是 Skill Seekers 的核心功能，負責從技術文件網站提取內容並整理成結構化格式。

## 抓取流程

```
1. 初始化
   ├── 解析 base_url
   ├── 載入設定
   └── 建立 HTTP client

2. 來源偵測（優先順序）
   ├── 檢查 /llms-full.txt
   ├── 檢查 /llms.txt
   ├── 檢查 /llms-small.txt
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

## llms.txt 支援

llms.txt 是專為 LLM 設計的文件格式，抓取器會優先使用：

```python
# 檢查順序
llms_urls = [
    f"{base_url}/llms-full.txt",   # 完整版
    f"{base_url}/llms.txt",         # 標準版
    f"{base_url}/llms-small.txt"    # 精簡版
]
```

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

### 預設選擇器

```python
DEFAULT_SELECTORS = {
    "main_content": "article, main, [role='main'], .content, .main-content",
    "title": "h1, title",
    "code_blocks": "pre code, pre, .highlight code"
}
```

## 程式碼語言偵測

### 偵測邏輯

1. 優先使用 HTML class 提示（如 `language-python`）
2. 根據語法特徵自動偵測

### 支援的語言

| 語言 | 識別特徵 |
|------|----------|
| Python | `def`, `import`, `from ... import` |
| JavaScript | `const`, `let`, `=>`, `function` |
| TypeScript | `: string`, `: number`, `interface` |
| JSX/TSX | `useState`, `useEffect`, `<Component />` |
| Java | `public class`, `System.out.println` |
| Go | `func`, `package` |
| Rust | `fn`, `let mut` |
| Bash | `#!/bin/bash`, `$VAR` |
| SQL | `SELECT`, `FROM`, `WHERE` |
| HTML | `<!DOCTYPE`, `<html>`, `<div>` |
| CSS | `@media`, `{ property: value; }` |
| JSON | `{ "key": value }` |

## URL 過濾

### 預設排除路徑

```python
EXCLUDE_PATTERNS = [
    '/blog',
    '/changelog',
    '/about',
    '/community',
    '/twitter',
    '/github',
    '/discord',
    '/newsletter'
]
```

### 自訂過濾

```python
scraper = DocumentationScraper(
    base_url="https://docs.example.com",
    include_patterns=["/api", "/guide"],
    exclude_patterns=["/legacy", "/deprecated"]
)
```

## 速率限制

為避免對目標網站造成負擔，預設每次請求間隔 0.5 秒：

```python
await asyncio.sleep(0.5)  # 速率限制
```

可透過設定調整：

```python
scraper = DocumentationScraper(
    base_url=url,
    rate_limit=1.0  # 每秒 1 個請求
)
```

## 錯誤處理

### 常見錯誤

| 錯誤 | 原因 | 解決方案 |
|------|------|----------|
| 403 Forbidden | 被網站封鎖 | 增加延遲、使用 llms.txt |
| 404 Not Found | 頁面不存在 | 檢查 URL 是否正確 |
| Timeout | 連線超時 | 增加 timeout 設定 |
| 空內容 | 選擇器不匹配 | 調整 selectors 設定 |

### 錯誤處理範例

```python
try:
    result = await scraper.scrape()
except httpx.HTTPStatusError as e:
    if e.response.status_code == 403:
        print("被速率限制，請稍後再試")
    elif e.response.status_code == 404:
        print("頁面不存在")
except httpx.TimeoutException:
    print("連線超時，嘗試增加 timeout")
```

## 效能優化

### 平行抓取

使用 asyncio 進行平行抓取：

```python
tasks = [fetch_page(url) for url in urls]
pages = await asyncio.gather(*tasks)
```

### 快取

建議對已抓取的頁面進行快取，避免重複抓取：

```python
import hashlib
import os

def get_cache_path(url: str) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return f".cache/{url_hash}.json"

def load_from_cache(url: str):
    path = get_cache_path(url)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
```
