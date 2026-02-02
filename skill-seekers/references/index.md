# Skill Seekers 參考文件索引

## 核心指南

| 文件 | 說明 |
|------|------|
| [scraping.md](./scraping.md) | 網頁抓取完整指南 |
| [github.md](./github.md) | GitHub 倉庫抓取指南 |
| [pdf.md](./pdf.md) | PDF 處理指南 |
| [categorization.md](./categorization.md) | 內容自動分類說明 |
| [output-formats.md](./output-formats.md) | 輸出格式規範 |
| [templates.md](./templates.md) | 設定模板與範例 |

## 快速導航

### 我想抓取文件網站
→ 查看 [scraping.md](./scraping.md)

### 我想分析 GitHub 倉庫
→ 查看 [github.md](./github.md)

### 我想處理 PDF 檔案
→ 查看 [pdf.md](./pdf.md)

### 我想了解分類邏輯
→ 查看 [categorization.md](./categorization.md)

### 我想自訂輸出格式
→ 查看 [output-formats.md](./output-formats.md)

## 常見任務

### 快速建立 Skill

```python
# 最簡單的方式
from scripts.scraper import DocumentationScraper
from scripts.skill_generator import SkillGenerator

async def quick_skill(url: str, name: str):
    scraper = DocumentationScraper(base_url=url, mode="quick")
    content = await scraper.scrape()
    
    generator = SkillGenerator(content, name=name)
    generator.save(f"output/{name.lower()}")
```

### 檢查 llms.txt

```python
scraper = DocumentationScraper(base_url="https://example.com")
llms = await scraper.check_llms_txt()
if llms:
    print("✅ 有 llms.txt，可直接使用")
```

### 合併多個來源

```python
from scripts.utils import merge_content

merged = merge_content([doc_content, github_content, pdf_content])
generator = SkillGenerator(merged, name="MyProject")
```
