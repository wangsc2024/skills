"""
DocumentationScraper - 網頁文件抓取器
Skill Seekers 核心元件
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime

try:
    import httpx
    from bs4 import BeautifulSoup
except ImportError:
    print("請安裝依賴：pip install httpx beautifulsoup4")
    raise


class ScraperError(Exception):
    """抓取器錯誤"""
    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code
        super().__init__(message)


@dataclass
class DocumentationScraper:
    """文件網站抓取器"""
    
    base_url: str
    max_pages: int = 50
    mode: str = "quick"  # quick, full, custom
    rate_limit: float = 0.5
    timeout: float = 30.0
    max_concurrent: int = 10
    
    selectors: Dict[str, str] = field(default_factory=lambda: {
        "main_content": "article, main, .content, [role='main'], .documentation",
        "title": "h1",
        "code_blocks": "pre code, pre"
    })
    
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "/blog", "/changelog", "/about", "/community", 
        "/twitter", "/github", "/discord", "/forum",
        "/newsletter", "/sponsor", "/donate"
    ])
    
    # 私有屬性
    _visited: Set[str] = field(default_factory=set, repr=False)
    _pages: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _client: Optional[Any] = field(default=None, repr=False)
    
    # 分類關鍵字
    CATEGORY_KEYWORDS = {
        "getting_started": [
            "intro", "quickstart", "installation", "setup", 
            "start", "begin", "first", "hello", "tutorial"
        ],
        "core_concepts": [
            "concepts", "fundamentals", "basics", "overview", 
            "introduction", "understand", "learn", "essential"
        ],
        "api_reference": [
            "api", "reference", "methods", "functions", "classes", 
            "modules", "interface", "types", "sdk"
        ],
        "guides": [
            "guide", "tutorial", "how-to", "walkthrough", 
            "learn", "step-by-step", "recipe"
        ],
        "examples": [
            "example", "sample", "demo", "cookbook", 
            "recipes", "patterns", "showcase"
        ],
        "configuration": [
            "config", "settings", "options", "parameters", 
            "environment", "setup", "customize"
        ],
        "troubleshooting": [
            "error", "debug", "faq", "troubleshoot", 
            "issue", "problem", "fix", "solution"
        ],
        "advanced": [
            "advanced", "deep-dive", "internals", "architecture", 
            "performance", "optimization", "scaling"
        ]
    }
    
    # 語言偵測模式
    LANGUAGE_PATTERNS = {
        "python": [
            r"\bdef\s+\w+\s*\(", 
            r"\bimport\s+\w+", 
            r"\bfrom\s+\w+\s+import",
            r"\bclass\s+\w+.*:",
            r"if\s+__name__\s*==\s*['\"]__main__['\"]"
        ],
        "javascript": [
            r"\bconst\s+\w+\s*=", 
            r"\blet\s+\w+\s*=", 
            r"=>\s*[{(]",
            r"\bfunction\s+\w+\s*\(",
            r"\bexport\s+(default\s+)?(function|class|const)"
        ],
        "typescript": [
            r":\s*(string|number|boolean|any|void)\b", 
            r"\binterface\s+\w+",
            r"\btype\s+\w+\s*=",
            r"<\w+(\s*,\s*\w+)*>"
        ],
        "jsx": [
            r"<\w+[^>]*/>", 
            r"useState\s*[<(]",
            r"useEffect\s*\(",
            r"className\s*=",
            r"import.*from\s+['\"]react['\"]"
        ],
        "java": [
            r"\bpublic\s+(static\s+)?(class|void|int|String)", 
            r"System\.out\.println",
            r"\bpackage\s+[\w.]+;",
            r"@\w+(\s*\([^)]*\))?"
        ],
        "go": [
            r"\bfunc\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\(", 
            r"\bpackage\s+\w+",
            r":=",
            r"\btype\s+\w+\s+struct"
        ],
        "rust": [
            r"\bfn\s+\w+\s*[<(]", 
            r"\blet\s+(mut\s+)?\w+",
            r"\bimpl\s+\w+",
            r"\buse\s+\w+::"
        ],
        "bash": [
            r"#!/bin/(bash|sh)", 
            r"\$\{?\w+\}?",
            r"\becho\s+",
            r"\bif\s+\[\["
        ],
        "sql": [
            r"\bSELECT\s+", 
            r"\bFROM\s+", 
            r"\bWHERE\s+",
            r"\bINSERT\s+INTO",
            r"\bCREATE\s+(TABLE|DATABASE)"
        ],
        "html": [
            r"<!DOCTYPE", 
            r"<html", 
            r"<div\b",
            r"<head>",
            r"<body>"
        ],
        "css": [
            r"\{[^}]*:\s*[^;]+;[^}]*\}", 
            r"@media",
            r"\.([\w-]+)\s*\{",
            r"#([\w-]+)\s*\{"
        ],
        "json": [
            r'^\s*\{[\s\S]*"[\w]+"',
            r'^\s*\[[\s\S]*\]'
        ],
        "yaml": [
            r"^\w+:\s*$",
            r"^\s*-\s+\w+:",
            r"^\s{2,}\w+:"
        ]
    }

    async def check_llms_txt(self) -> Optional[str]:
        """檢查 llms.txt 檔案"""
        parsed = urlparse(self.base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        
        llms_urls = [
            f"{base}/llms-full.txt",
            f"{base}/llms.txt",
            f"{base}/llms-small.txt"
        ]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in llms_urls:
                try:
                    response = await client.get(url, follow_redirects=True)
                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "")
                        if "text" in content_type:
                            print(f"✅ 發現 {url}")
                            return response.text
                except Exception:
                    continue
        
        return None

    async def check_sitemap(self) -> List[str]:
        """檢查 sitemap.xml"""
        parsed = urlparse(self.base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        
        sitemap_urls = [
            f"{base}/sitemap.xml",
            f"{base}/sitemap_index.xml",
            f"{base}/sitemap/sitemap.xml"
        ]
        
        urls = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for sitemap_url in sitemap_urls:
                try:
                    response = await client.get(sitemap_url, follow_redirects=True)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'xml')
                        for loc in soup.find_all('loc'):
                            url = loc.text.strip()
                            if self._should_include(url):
                                urls.append(url)
                        if urls:
                            print(f"✅ 從 sitemap 發現 {len(urls)} 個 URL")
                            break
                except Exception:
                    continue
        
        return urls

    async def scrape(self) -> Dict[str, Any]:
        """執行抓取"""
        print(f"🔄 開始抓取: {self.base_url}")
        print(f"   模式: {self.mode}, 最大頁數: {self.max_pages}")
        
        # 先檢查 llms.txt
        llms_content = await self.check_llms_txt()
        if llms_content:
            return self._process_llms_txt(llms_content)
        
        # 決定最大頁數
        max_pages = self.max_pages
        if self.mode == "quick":
            max_pages = min(max_pages, 20)
        
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=self.max_concurrent)
        ) as self._client:
            
            # 先嘗試 sitemap
            sitemap_urls = await self.check_sitemap()
            
            if sitemap_urls:
                links = sitemap_urls[:max_pages]
            else:
                # 手動探索連結
                links = await self._discover_links(max_pages)
            
            print(f"📋 準備抓取 {len(links)} 頁")
            
            # 平行抓取（使用信號量限制並行數）
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def fetch_with_semaphore(url):
                async with semaphore:
                    return await self._fetch_page(url)
            
            tasks = [fetch_with_semaphore(url) for url in links]
            self._pages = await asyncio.gather(*tasks)
            
            # 過濾失敗的頁面
            self._pages = [p for p in self._pages if p.get("content")]
            
            print(f"✅ 成功抓取 {len(self._pages)} 頁")
        
        return self._build_result()

    async def _discover_links(self, max_pages: int) -> List[str]:
        """探索網站連結"""
        to_visit = [self.base_url]
        discovered = []
        
        parsed_base = urlparse(self.base_url)
        base_domain = parsed_base.netloc
        
        while to_visit and len(discovered) < max_pages:
            url = to_visit.pop(0)
            
            # 正規化 URL
            url = self._normalize_url(url)
            
            if url in self._visited:
                continue
            
            self._visited.add(url)
            
            try:
                response = await self._client.get(url)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                discovered.append(url)
                
                # 提取連結
                for a in soup.select('a[href]'):
                    href = a.get('href', '')
                    if not href or href.startswith('#') or href.startswith('mailto:'):
                        continue
                    
                    full_url = urljoin(url, href)
                    full_url = self._normalize_url(full_url)
                    parsed = urlparse(full_url)
                    
                    # 只處理同網域的連結
                    if parsed.netloc == base_domain:
                        if self._should_include(full_url):
                            if full_url not in self._visited and full_url not in to_visit:
                                to_visit.append(full_url)
                
                # 速率限制
                await asyncio.sleep(self.rate_limit)
                
            except Exception as e:
                print(f"   ⚠️ 探索失敗: {url} ({str(e)[:50]})")
                continue
        
        return discovered

    def _normalize_url(self, url: str) -> str:
        """正規化 URL"""
        parsed = urlparse(url)
        # 移除 fragment 和多餘的斜線
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _should_include(self, url: str) -> bool:
        """檢查 URL 是否應該包含"""
        url_lower = url.lower()
        
        # 檢查排除模式
        for pattern in self.exclude_patterns:
            if pattern.lower() in url_lower:
                return False
        
        # 排除非文件類型
        excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', 
                              '.pdf', '.zip', '.tar', '.gz', '.mp4', '.mp3']
        for ext in excluded_extensions:
            if url_lower.endswith(ext):
                return False
        
        # 檢查包含模式（如果有設定）
        if self.include_patterns:
            return any(p.lower() in url_lower for p in self.include_patterns)
        
        return True

    async def _fetch_page(self, url: str) -> Dict[str, Any]:
        """抓取單一頁面"""
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除不需要的元素
            for tag in soup.select('nav, footer, header, script, style, .sidebar, '
                                   '.navigation, .toc, .breadcrumb, .edit-link, '
                                   '.prev-next, .footer, .header'):
                tag.decompose()
            
            # 提取標題
            title = self._extract_title(soup)
            
            # 提取主要內容
            content = self._extract_content(soup)
            
            # 提取程式碼區塊
            code_blocks = self._extract_code_blocks(soup)
            
            # 分類
            category = self._categorize(url, title)
            
            return {
                "url": url,
                "title": title,
                "content": content[:15000],  # 限制內容長度
                "code_blocks": code_blocks[:15],  # 限制程式碼區塊數量
                "category": category
            }
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise ScraperError("Rate limited", "RATE_LIMITED")
            elif e.response.status_code == 401:
                raise ScraperError("Authentication required", "AUTH_REQUIRED")
            elif e.response.status_code == 404:
                return {"url": url, "error": "Not found", "code": "NOT_FOUND"}
            else:
                return {"url": url, "error": str(e)}
        except Exception as e:
            return {"url": url, "error": str(e)}

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取標題"""
        # 嘗試多種選擇器
        selectors = [
            self.selectors.get("title", "h1"),
            "h1",
            "title",
            ".page-title",
            ".article-title"
        ]
        
        for selector in selectors:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                if text and len(text) < 200:
                    return text
        
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """提取主要內容"""
        # 嘗試多種選擇器
        for selector in self.selectors.get("main_content", "").split(", "):
            main = soup.select_one(selector.strip())
            if main:
                return main.get_text(separator='\n', strip=True)
        
        # Fallback to body
        if soup.body:
            return soup.body.get_text(separator='\n', strip=True)
        
        return ""

    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取程式碼區塊"""
        code_blocks = []
        
        for code in soup.select(self.selectors.get("code_blocks", "pre code, pre")):
            code_text = code.get_text(strip=True)
            
            # 過濾太短的程式碼
            if len(code_text) < 20:
                continue
            
            # 偵測語言
            classes = code.get('class', [])
            if isinstance(classes, str):
                classes = [classes]
            
            lang = self._detect_language(code_text, classes)
            
            code_blocks.append({
                "language": lang,
                "code": code_text[:3000]  # 限制單個程式碼區塊長度
            })
        
        return code_blocks

    def _detect_language(self, code: str, classes: List[str]) -> str:
        """偵測程式碼語言"""
        # 先檢查 class
        for cls in classes:
            if isinstance(cls, str):
                if 'language-' in cls:
                    return cls.replace('language-', '').lower()
                if 'lang-' in cls:
                    return cls.replace('lang-', '').lower()
                # 直接的語言名稱
                lang_names = ['python', 'javascript', 'typescript', 'java', 
                             'go', 'rust', 'bash', 'shell', 'sql', 'html', 
                             'css', 'json', 'yaml', 'jsx', 'tsx']
                for lang in lang_names:
                    if lang in cls.lower():
                        return lang
        
        # 根據語法偵測
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    return lang
        
        return "text"

    def _categorize(self, url: str, title: str) -> str:
        """分類頁面"""
        url_lower = url.lower()
        title_lower = title.lower() if title else ""
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in url_lower or keyword in title_lower:
                    return category
        
        return "general"

    def _process_llms_txt(self, content: str) -> Dict[str, Any]:
        """處理 llms.txt 內容"""
        return {
            "source": "llms.txt",
            "url": self.base_url,
            "pages": [{
                "url": self.base_url,
                "title": "LLMs.txt Content",
                "content": content,
                "code_blocks": [],
                "category": "llms_txt"
            }],
            "categories": {"llms_txt": 1},
            "statistics": {
                "total_pages": 1,
                "total_code_blocks": 0,
                "languages_detected": []
            }
        }

    def _build_result(self) -> Dict[str, Any]:
        """建立結果"""
        categories = {}
        languages = set()
        total_code_blocks = 0
        
        for page in self._pages:
            cat = page.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1
            
            for block in page.get("code_blocks", []):
                total_code_blocks += 1
                if block.get("language") and block["language"] != "text":
                    languages.add(block["language"])
        
        return {
            "url": self.base_url,
            "scraped_at": datetime.now().isoformat(),
            "pages": self._pages,
            "categories": categories,
            "statistics": {
                "total_pages": len(self._pages),
                "total_code_blocks": total_code_blocks,
                "languages_detected": sorted(list(languages))
            }
        }


# CLI 入口點
async def main():
    """命令列入口"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方式: python scraper.py <url> [--mode quick|full] [--max-pages N]")
        sys.exit(1)
    
    url = sys.argv[1]
    mode = "quick"
    max_pages = 50
    
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--max-pages" and i + 1 < len(sys.argv):
            max_pages = int(sys.argv[i + 1])
    
    scraper = DocumentationScraper(
        base_url=url,
        mode=mode,
        max_pages=max_pages
    )
    
    result = await scraper.scrape()
    
    print(f"\n📊 抓取結果:")
    print(f"   總頁數: {result['statistics']['total_pages']}")
    print(f"   程式碼區塊: {result['statistics']['total_code_blocks']}")
    print(f"   偵測語言: {', '.join(result['statistics']['languages_detected'])}")
    print(f"   分類: {result['categories']}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
