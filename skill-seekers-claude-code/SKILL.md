---
name: skill-seekers
description: "自動化工具，將技術文件網站、GitHub 倉庫、PDF 檔案轉換為結構化 AI 知識庫。當使用者要求抓取文件、建立 skill、分析 GitHub 倉庫、或提到「知識庫」「文件轉換」等關鍵字時使用此技能。"
---

# Skill Seekers

## Overview

Skill Seekers 是一個自動化工具，能將技術文件網站、GitHub 倉庫、PDF 檔案轉換為結構化的 AI 知識庫。它會自動抓取內容、智慧分類、提取程式碼範例，並生成標準化的 Skill 格式輸出。

## Quick Reference

| 任務 | 方法 |
|------|------|
| 抓取文件網站 | 使用 `scripts/scraper.py` |
| 抓取 GitHub 倉庫 | 使用 `scripts/github_scraper.py` |
| 處理 PDF | 使用 `scripts/pdf_extractor.py` |
| 生成 Skill | 使用 `scripts/skill_generator.py` |

### 核心工作流程

```
1. 分析來源 → 檢查 llms.txt、sitemap、導航結構
2. 抓取內容 → 快速(20頁) / 完整(100+頁) / 自訂
3. 內容分類 → 自動分類為 8 大類別
4. 生成輸出 → SKILL.md + references/*.md + metadata.json
```

### 分類關鍵字

| 類別 | 關鍵字 |
|------|--------|
| `getting_started` | intro, quickstart, installation, setup |
| `core_concepts` | concepts, fundamentals, basics, overview |
| `api_reference` | api, reference, methods, functions |
| `guides` | guide, tutorial, how-to, walkthrough |
| `examples` | example, sample, demo, cookbook |
| `configuration` | config, settings, options |
| `troubleshooting` | error, debug, faq, troubleshoot |
| `advanced` | advanced, deep-dive, internals |

---

## 抓取文件網站

### 基本用法

```python
import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

async def scrape_documentation(base_url: str, max_pages: int = 50):
    """抓取文件網站"""
    
    visited = set()
    pages = []
    to_visit = [base_url]
    
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    
    # 排除路徑
    exclude_patterns = ['/blog', '/changelog', '/about', '/community']
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 先檢查 llms.txt
        llms_content = await check_llms_txt(client, base_url)
        if llms_content:
            print("✅ 發現 llms.txt，使用優化版本")
            return {"source": "llms.txt", "content": llms_content}
        
        while to_visit and len(pages) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                response = await client.get(url, follow_redirects=True)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取頁面內容
                page_data = extract_page_content(soup, url)
                pages.append(page_data)
                
                # 探索連結
                for a in soup.select('a[href]'):
                    href = a.get('href', '')
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)
                    
                    if parsed.netloc == base_domain:
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if not any(p in clean_url.lower() for p in exclude_patterns):
                            if clean_url not in visited:
                                to_visit.append(clean_url)
                
                await asyncio.sleep(0.5)  # 速率限制
                
            except Exception as e:
                print(f"錯誤: {url} - {e}")
    
    return categorize_and_build_result(pages)

async def check_llms_txt(client, base_url: str):
    """檢查 llms.txt"""
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    
    for path in ['/llms-full.txt', '/llms.txt', '/llms-small.txt']:
        try:
            response = await client.get(f"{base}{path}")
            if response.status_code == 200:
                return response.text
        except:
            pass
    return None

def extract_page_content(soup, url: str):
    """提取頁面內容"""
    # 移除不需要的元素
    for tag in soup.select('nav, footer, header, script, style, .sidebar'):
        tag.decompose()
    
    # 標題
    title = ""
    if soup.h1:
        title = soup.h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.string or ""
    
    # 主要內容
    main = soup.select_one('article, main, .content, [role="main"]')
    content = main.get_text(separator='\n', strip=True) if main else ""
    
    # 程式碼區塊
    code_blocks = []
    for code in soup.select('pre code, pre'):
        code_text = code.get_text(strip=True)
        if len(code_text) > 20:
            lang = detect_language(code_text, code.get('class', []))
            code_blocks.append({"language": lang, "code": code_text[:2000]})
    
    # 分類
    category = categorize_page(url, title)
    
    return {
        "url": url,
        "title": title,
        "content": content[:10000],
        "code_blocks": code_blocks[:10],
        "category": category
    }

def detect_language(code: str, classes: list) -> str:
    """偵測程式碼語言"""
    for cls in classes:
        if isinstance(cls, str) and 'language-' in cls:
            return cls.replace('language-', '')
    
    patterns = {
        "python": [r"\bdef\s+\w+\(", r"\bimport\s+\w+"],
        "javascript": [r"\bconst\s+\w+\s*=", r"=>\s*{"],
        "typescript": [r":\s*(string|number|boolean)"],
        "jsx": [r"useState\s*\(", r"useEffect\s*\("],
        "bash": [r"#!/bin/bash", r"\$\w+"],
    }
    
    for lang, pats in patterns.items():
        for p in pats:
            if re.search(p, code):
                return lang
    return "text"

def categorize_page(url: str, title: str) -> str:
    """分類頁面"""
    keywords = {
        "getting_started": ["intro", "quickstart", "installation", "setup"],
        "core_concepts": ["concepts", "fundamentals", "basics", "overview"],
        "api_reference": ["api", "reference", "methods", "functions"],
        "guides": ["guide", "tutorial", "how-to"],
        "examples": ["example", "sample", "demo"],
        "configuration": ["config", "settings", "options"],
        "troubleshooting": ["error", "debug", "faq"],
        "advanced": ["advanced", "deep-dive", "internals"],
    }
    
    text = (url + title).lower()
    for cat, kws in keywords.items():
        if any(kw in text for kw in kws):
            return cat
    return "general"

def categorize_and_build_result(pages: list) -> dict:
    """整理結果"""
    categories = {}
    languages = set()
    code_count = 0
    
    for page in pages:
        cat = page.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
        for block in page.get("code_blocks", []):
            code_count += 1
            if block.get("language"):
                languages.add(block["language"])
    
    return {
        "pages": pages,
        "categories": categories,
        "statistics": {
            "total_pages": len(pages),
            "total_code_blocks": code_count,
            "languages_detected": list(languages)
        }
    }
```

### 執行範例

```python
# 抓取 React 文件
result = asyncio.run(scrape_documentation("https://react.dev/learn", max_pages=30))
print(f"抓取完成: {result['statistics']['total_pages']} 頁")
```

---

## 抓取 GitHub 倉庫

```python
import httpx
import base64

async def scrape_github_repo(repo: str, include_code: bool = False):
    """抓取 GitHub 倉庫"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {}
        # 如有 token: headers["Authorization"] = f"token {os.environ.get('GITHUB_TOKEN')}"
        
        api_base = f"https://api.github.com/repos/{repo}"
        
        # 倉庫資訊
        repo_resp = await client.get(api_base, headers=headers)
        repo_data = repo_resp.json()
        
        # README
        readme_content = ""
        readme_resp = await client.get(f"{api_base}/readme", headers=headers)
        if readme_resp.status_code == 200:
            readme_data = readme_resp.json()
            readme_content = base64.b64decode(readme_data.get("content", "")).decode("utf-8")
        
        # 目錄結構
        structure = []
        contents_resp = await client.get(f"{api_base}/contents", headers=headers)
        if contents_resp.status_code == 200:
            for item in contents_resp.json():
                structure.append({
                    "name": item.get("name"),
                    "type": item.get("type"),
                    "path": item.get("path")
                })
        
        # docs 目錄
        docs = []
        docs_resp = await client.get(f"{api_base}/contents/docs", headers=headers)
        if docs_resp.status_code == 200:
            for item in docs_resp.json()[:20]:
                if item.get("name", "").endswith(".md"):
                    docs.append({
                        "name": item.get("name"),
                        "path": item.get("path")
                    })
        
        return {
            "repo": repo,
            "description": repo_data.get("description", ""),
            "stars": repo_data.get("stargazers_count", 0),
            "language": repo_data.get("language", ""),
            "topics": repo_data.get("topics", []),
            "readme": readme_content[:10000],
            "structure": structure,
            "docs": docs
        }

# 執行
result = asyncio.run(scrape_github_repo("facebook/react"))
print(f"⭐ Stars: {result['stars']}")
```

---

## 生成 Skill

```python
import json
import os
from datetime import datetime

def generate_skill(content: dict, name: str, output_dir: str):
    """生成完整 Skill"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/references", exist_ok=True)
    
    pages = content.get("pages", [])
    stats = content.get("statistics", {})
    categories = content.get("categories", {})
    
    # 收集程式碼範例
    examples = []
    for page in pages:
        for block in page.get("code_blocks", [])[:2]:
            if len(block.get("code", "")) > 50:
                examples.append({
                    "source": page.get("title", ""),
                    "language": block.get("language", ""),
                    "code": block.get("code", "")
                })
    
    # 生成 SKILL.md
    skill_md = f"""# {name} Skill

## Description

{name} 技術文件知識庫，包含核心概念、API 參考與實用程式碼範例。

## When to Use

- 需要查詢 {name} 相關技術問題
- 尋找程式碼範例與最佳實踐
- 了解 API 用法與設定方式

## Statistics

| 項目 | 數值 |
|------|------|
| 總頁數 | {stats.get('total_pages', 0)} |
| 程式碼區塊 | {stats.get('total_code_blocks', 0)} |
| 偵測語言 | {', '.join(stats.get('languages_detected', []))} |

## Categories

"""
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        skill_md += f"- **{cat.replace('_', ' ').title()}**: {count} 頁\n"
    
    skill_md += "\n## Core Concepts\n\n"
    
    concept_pages = [p for p in pages if p.get("category") in ["core_concepts", "getting_started"]][:5]
    for page in concept_pages:
        skill_md += f"### {page.get('title', 'Untitled')}\n\n"
        skill_md += f"{page.get('content', '')[:500]}...\n\n"
    
    skill_md += "## Code Examples\n\n"
    
    for i, ex in enumerate(examples[:5], 1):
        skill_md += f"### Example {i}: {ex.get('source', '')}\n\n"
        skill_md += f"```{ex.get('language', '')}\n{ex.get('code', '')}\n```\n\n"
    
    skill_md += """## Related Resources

- 原始文件來源
- GitHub 倉庫（如適用）
"""
    
    # 儲存 SKILL.md
    with open(f"{output_dir}/SKILL.md", "w", encoding="utf-8") as f:
        f.write(skill_md)
    
    # 儲存分類文件
    for cat in categories.keys():
        cat_pages = [p for p in pages if p.get("category") == cat]
        cat_md = f"# {cat.replace('_', ' ').title()}\n\n"
        
        for page in cat_pages[:20]:
            cat_md += f"## {page.get('title', 'Untitled')}\n\n"
            cat_md += f"{page.get('content', '')[:2000]}\n\n"
        
        with open(f"{output_dir}/references/{cat}.md", "w", encoding="utf-8") as f:
            f.write(cat_md)
    
    # 儲存 metadata.json
    metadata = {
        "skill_name": name.lower().replace(" ", "-"),
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "statistics": stats,
        "categories": list(categories.keys())
    }
    
    with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Skill 已儲存至 {output_dir}/")
    return skill_md
```

---

## 完整工作流程範例

```python
import asyncio

async def create_skill_from_url(url: str, name: str):
    """完整工作流程：從 URL 建立 Skill"""
    
    print(f"🔄 開始抓取 {url}...")
    
    # 1. 抓取內容
    content = await scrape_documentation(url, max_pages=50)
    
    print(f"✅ 抓取完成: {content['statistics']['total_pages']} 頁")
    print(f"📁 分類: {list(content['categories'].keys())}")
    
    # 2. 生成 Skill
    output_dir = f"output/{name.lower()}"
    generate_skill(content, name, output_dir)
    
    print(f"🎉 完成！Skill 位於 {output_dir}/")
    return output_dir

# 執行
asyncio.run(create_skill_from_url(
    url="https://fastapi.tiangolo.com/",
    name="FastAPI"
))
```

---

## 依賴安裝

```bash
pip install httpx beautifulsoup4
```

## 限制說明

1. **需要登入的頁面** - 無法抓取，建議使用者貼上內容
2. **JavaScript 動態載入** - 可能抓取不完整，建議使用 llms.txt
3. **大型網站 (500+ 頁)** - 建議分批處理
4. **速率限制** - 內建 0.5 秒延遲

## 相關參考

- `references/scraping.md` - 網頁抓取詳細指南
- `references/github.md` - GitHub 抓取指南
- `references/categorization.md` - 分類邏輯說明
- `references/output-formats.md` - 輸出格式規範
