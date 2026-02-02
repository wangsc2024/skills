# 內容自動分類指南

## 概述

Skill Seekers 使用關鍵字比對來自動分類抓取的內容，將頁面歸類到預設的 8 大類別中。

## 預設分類

| 類別 | 說明 | 識別關鍵字 |
|------|------|-----------|
| `getting_started` | 入門指南 | intro, quickstart, installation, setup, start, begin, first-steps |
| `core_concepts` | 核心概念 | concepts, fundamentals, basics, overview, introduction, understand |
| `api_reference` | API 參考 | api, reference, methods, functions, classes, modules, interface |
| `guides` | 教學指南 | guide, tutorial, how-to, walkthrough, learn, step-by-step |
| `examples` | 程式範例 | example, sample, demo, cookbook, recipes, patterns, snippets |
| `configuration` | 設定說明 | config, settings, options, parameters, environment, customize |
| `troubleshooting` | 問題排解 | error, debug, faq, troubleshoot, issue, problem, fix, common-issues |
| `advanced` | 進階主題 | advanced, deep-dive, internals, architecture, performance, optimization |

## 分類邏輯

```python
def categorize_page(url: str, title: str, content: str = "") -> str:
    """
    分類頁面
    
    優先順序：
    1. URL 路徑比對（最高權重）
    2. 標題關鍵字比對
    3. 內容關鍵字比對（可選）
    4. 預設為 'general'
    """
    
    CATEGORY_KEYWORDS = {
        "getting_started": [
            "intro", "quickstart", "installation", "setup", 
            "start", "begin", "first-steps", "getting-started"
        ],
        "core_concepts": [
            "concepts", "fundamentals", "basics", "overview",
            "introduction", "understand", "principles"
        ],
        "api_reference": [
            "api", "reference", "methods", "functions",
            "classes", "modules", "interface", "endpoints"
        ],
        "guides": [
            "guide", "tutorial", "how-to", "walkthrough",
            "learn", "step-by-step", "building"
        ],
        "examples": [
            "example", "sample", "demo", "cookbook",
            "recipes", "patterns", "snippets", "showcase"
        ],
        "configuration": [
            "config", "settings", "options", "parameters",
            "environment", "customize", "preferences"
        ],
        "troubleshooting": [
            "error", "debug", "faq", "troubleshoot",
            "issue", "problem", "fix", "common-issues"
        ],
        "advanced": [
            "advanced", "deep-dive", "internals", "architecture",
            "performance", "optimization", "scaling"
        ]
    }
    
    # 合併 URL 和標題進行比對
    text = (url + " " + title).lower()
    
    # 計算每個類別的匹配分數
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[category] = score
    
    # 返回最高分的類別
    if scores:
        return max(scores, key=scores.get)
    
    return "general"
```

## 自訂分類

### 覆寫預設關鍵字

```python
custom_categories = {
    "hooks": ["hooks", "useState", "useEffect", "useContext", "useReducer"],
    "components": ["component", "props", "children", "render", "jsx"],
    "state": ["state", "redux", "context", "store", "zustand"],
    "routing": ["router", "route", "navigation", "link", "redirect"]
}

def categorize_with_custom(url: str, title: str, custom_keywords: dict) -> str:
    text = (url + " " + title).lower()
    
    for category, keywords in custom_keywords.items():
        if any(kw in text for kw in keywords):
            return category
    
    # 如果自訂分類沒有匹配，使用預設
    return categorize_page(url, title)
```

### 手動調整分類

```python
# 抓取後手動調整
result = await scraper.scrape()

for page in result["pages"]:
    # 根據特定條件重新分類
    if "migration" in page["url"].lower():
        page["category"] = "guides"
    elif "deprecated" in page["title"].lower():
        page["category"] = "advanced"
```

## 分類統計

```python
from collections import Counter

def get_category_stats(pages: list) -> dict:
    """統計各分類的頁數"""
    categories = Counter(p.get("category", "general") for p in pages)
    
    total = len(pages)
    stats = {
        "total_pages": total,
        "categories": {}
    }
    
    for cat, count in categories.most_common():
        stats["categories"][cat] = {
            "count": count,
            "percentage": round(count / total * 100, 1)
        }
    
    return stats
```

輸出範例：

```json
{
  "total_pages": 150,
  "categories": {
    "api_reference": {"count": 45, "percentage": 30.0},
    "guides": {"count": 35, "percentage": 23.3},
    "getting_started": {"count": 20, "percentage": 13.3},
    "examples": {"count": 18, "percentage": 12.0},
    "core_concepts": {"count": 15, "percentage": 10.0},
    "configuration": {"count": 10, "percentage": 6.7},
    "troubleshooting": {"count": 5, "percentage": 3.3},
    "advanced": {"count": 2, "percentage": 1.3}
  }
}
```

## 分類品質評估

### 檢查未分類頁面

```python
def find_uncategorized(pages: list) -> list:
    """找出未分類的頁面"""
    return [
        {"url": p["url"], "title": p["title"]}
        for p in pages
        if p.get("category") == "general"
    ]
```

### 檢查可能分類錯誤的頁面

```python
def find_suspicious_categorization(pages: list) -> list:
    """找出可能分類錯誤的頁面"""
    suspicious = []
    
    for page in pages:
        url = page.get("url", "").lower()
        title = page.get("title", "").lower()
        category = page.get("category", "")
        
        # 檢查是否有明顯的不匹配
        if category == "api_reference" and "tutorial" in url:
            suspicious.append({
                "page": page["url"],
                "current": category,
                "suggested": "guides"
            })
        elif category == "getting_started" and "advanced" in title:
            suspicious.append({
                "page": page["url"],
                "current": category,
                "suggested": "advanced"
            })
    
    return suspicious
```

## 最佳實踐

1. **先執行預設分類**，再根據結果微調
2. **檢查 `general` 類別**，手動分類或擴充關鍵字
3. **保持分類一致性**，同類型內容應歸入同一類別
4. **定期審查分類品質**，特別是大型文件網站
