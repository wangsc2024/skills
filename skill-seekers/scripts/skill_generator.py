"""
SkillGenerator - 生成 SKILL.md 和相關文件
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class SkillGenerator:
    """Skill 生成器"""
    
    content: Dict[str, Any]
    name: str
    language: str = "zh-TW"
    
    def generate(self, include_conflicts: bool = False) -> str:
        """生成 SKILL.md"""
        pages = self.content.get("pages", [])
        statistics = self.content.get("statistics", {})
        categories = self.content.get("categories", {})
        
        code_examples = self._collect_code_examples(pages)
        
        skill_md = self._build_skill_md(pages, statistics, categories, code_examples)
        
        if include_conflicts and "conflicts" in self.content:
            skill_md += self._build_conflict_report(self.content["conflicts"])
        
        return skill_md
    
    def _collect_code_examples(self, pages: list) -> list:
        examples = []
        for page in pages:
            for block in page.get("code_blocks", [])[:2]:
                if len(block.get("code", "")) > 50:
                    examples.append({
                        "source": page.get("title", ""),
                        "language": block.get("language", ""),
                        "code": block.get("code", "")
                    })
        return examples[:10]
    
    def _build_skill_md(self, pages, statistics, categories, code_examples) -> str:
        md = f"""# {self.name} Skill

## Description

{self.name} 技術文件知識庫，包含核心概念、API 參考與實用程式碼範例。

## When to Use

- 需要查詢 {self.name} 相關技術問題時
- 尋找程式碼範例與最佳實踐
- 了解 API 用法與設定方式
- 解決開發過程中遇到的問題

## Statistics

| 項目 | 數值 |
|------|------|
| 總頁數 | {statistics.get('total_pages', 0)} |
| 程式碼區塊 | {statistics.get('total_code_blocks', 0)} |
| 偵測語言 | {', '.join(statistics.get('languages_detected', []))} |

## Categories

"""
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            display_name = cat.replace('_', ' ').title()
            md += f"- **{display_name}**: {count} 頁\n"
        
        md += "\n## Core Concepts\n\n"
        
        concept_pages = [p for p in pages if p.get("category") in ["core_concepts", "getting_started"]][:5]
        for page in concept_pages:
            title = page.get("title", "Untitled")
            content = page.get("content", "")[:500]
            md += f"### {title}\n\n{content}...\n\n"
        
        md += "## Code Examples\n\n"
        
        for i, example in enumerate(code_examples[:5], 1):
            source = example.get("source", "")
            lang = example.get("language", "")
            code = example.get("code", "")
            md += f"### Example {i}: {source}\n\n```{lang}\n{code}\n```\n\n"
        
        md += """## Related Resources

- 原始文件來源
- GitHub 倉庫（如適用）
"""
        return md
    
    def _build_conflict_report(self, conflicts: list) -> str:
        md = "\n## ⚠️ 文件與程式碼衝突報告\n\n"
        for conflict in conflicts:
            md += f"### {conflict.get('item', 'Unknown')}\n\n"
            md += f"**類型**: {conflict.get('type', '')}\n\n"
        return md
    
    def save(self, output_dir: str):
        """儲存所有輸出檔案"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/references", exist_ok=True)
        
        skill_md = self.generate()
        with open(f"{output_dir}/SKILL.md", "w", encoding="utf-8") as f:
            f.write(skill_md)
        
        self._save_references(output_dir)
        self._save_metadata(output_dir)
        
        print(f"✅ Skill 已儲存至 {output_dir}/")
    
    def _save_references(self, output_dir: str):
        pages = self.content.get("pages", [])
        categories = self.content.get("categories", {})
        
        index_md = "# 參考文件索引\n\n"
        for cat in categories.keys():
            display_name = cat.replace('_', ' ').title()
            index_md += f"- [{display_name}](./{cat}.md)\n"
        
        with open(f"{output_dir}/references/index.md", "w", encoding="utf-8") as f:
            f.write(index_md)
        
        for cat in categories.keys():
            cat_pages = [p for p in pages if p.get("category") == cat]
            display_name = cat.replace('_', ' ').title()
            cat_md = f"# {display_name}\n\n"
            
            for page in cat_pages[:20]:
                cat_md += f"## {page.get('title', 'Untitled')}\n\n"
                cat_md += f"{page.get('content', '')[:2000]}\n\n"
            
            with open(f"{output_dir}/references/{cat}.md", "w", encoding="utf-8") as f:
                f.write(cat_md)
    
    def _save_metadata(self, output_dir: str):
        statistics = self.content.get("statistics", {})
        categories = self.content.get("categories", {})
        
        metadata = {
            "skill_name": self.name.lower().replace(" ", "-"),
            "display_name": self.name,
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "language": self.language,
            "statistics": statistics,
            "categories": list(categories.keys())
        }
        
        with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
