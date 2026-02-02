"""
工具函數
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """清理文字"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 1000) -> str:
    """截斷文字"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def merge_content(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合併多個來源的內容"""
    merged = {
        "pages": [],
        "categories": {},
        "statistics": {
            "total_pages": 0,
            "total_code_blocks": 0,
            "languages_detected": set()
        }
    }
    
    for source in sources:
        merged["pages"].extend(source.get("pages", []))
        
        for cat, count in source.get("categories", {}).items():
            merged["categories"][cat] = merged["categories"].get(cat, 0) + count
        
        stats = source.get("statistics", {})
        merged["statistics"]["total_pages"] += stats.get("total_pages", 0)
        merged["statistics"]["total_code_blocks"] += stats.get("total_code_blocks", 0)
        merged["statistics"]["languages_detected"].update(stats.get("languages_detected", []))
    
    merged["statistics"]["languages_detected"] = list(merged["statistics"]["languages_detected"])
    return merged


def detect_conflicts(doc_content: Dict[str, Any], code_content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """偵測文件與程式碼的衝突"""
    conflicts = []
    
    # 提取文件中的 API
    doc_apis = set()
    for page in doc_content.get("pages", []):
        for block in page.get("code_blocks", []):
            code = block.get("code", "")
            # 簡單提取函數名
            for match in re.finditer(r"(?:def|function|func)\s+(\w+)", code):
                doc_apis.add(match.group(1))
    
    # 提取程式碼中的 API
    code_apis = set()
    code_analysis = code_content.get("code_analysis", {})
    for func in code_analysis.get("functions", []):
        code_apis.add(func.get("name", ""))
    
    # 找出只在文件中的（可能過時）
    for api in doc_apis - code_apis:
        if api:
            conflicts.append({
                "item": api,
                "type": "missing_in_code",
                "description": "文件中提到但程式碼中找不到"
            })
    
    # 找出只在程式碼中的（可能缺少文件）
    for api in code_apis - doc_apis:
        if api and not api.startswith("_"):
            conflicts.append({
                "item": api,
                "type": "missing_in_docs",
                "description": "程式碼中存在但文件未提及"
            })
    
    return conflicts


def categorize_content(title: str, url: str, content: str) -> str:
    """自動分類內容"""
    text = f"{title} {url} {content[:500]}".lower()
    
    categories = {
        "getting_started": ["intro", "quickstart", "installation", "setup", "start"],
        "core_concepts": ["concepts", "fundamentals", "basics", "overview"],
        "api_reference": ["api", "reference", "methods", "functions"],
        "guides": ["guide", "tutorial", "how-to", "walkthrough"],
        "examples": ["example", "sample", "demo", "cookbook"],
        "configuration": ["config", "settings", "options"],
        "troubleshooting": ["error", "debug", "faq", "troubleshoot"],
        "advanced": ["advanced", "deep-dive", "internals", "performance"]
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category
    
    return "general"
