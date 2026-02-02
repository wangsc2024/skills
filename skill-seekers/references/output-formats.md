# 輸出格式規範

## 概述

Skill Seekers 生成的輸出遵循標準化格式，確保與 Claude Code 和其他 AI 工具的相容性。

## 目錄結構

```
output/
└── {skill-name}/
    ├── SKILL.md              # 主要技能說明文件
    ├── metadata.json         # 中繼資料
    └── references/           # 分類參考文件
        ├── index.md          # 參考文件索引
        ├── getting_started.md
        ├── core_concepts.md
        ├── api_reference.md
        ├── guides.md
        ├── examples.md
        ├── configuration.md
        ├── troubleshooting.md
        └── advanced.md
```

## SKILL.md 標準格式

```markdown
# {Framework Name} Skill

## Description
[一句話描述此 skill 的用途和適用場景]

## When to Use
- [使用場景 1]
- [使用場景 2]
- [使用場景 3]

## Core Concepts

### {概念 1}
[解釋文字]

```{language}
// 程式碼範例
```

### {概念 2}
[解釋文字]

## Quick Reference

### 常用 API
| API | 用途 | 範例 |
|-----|------|------|
| `method1()` | 描述 | `code` |
| `method2()` | 描述 | `code` |

### 常見模式

```{language}
// 模式名稱
{程式碼範例}
```

## Code Examples

### Example 1: {標題}
{說明}

```{language}
{完整可執行的程式碼}
```

### Example 2: {標題}
{說明}

```{language}
{完整可執行的程式碼}
```

## Common Pitfalls
- ❌ {錯誤做法}
  ```{language}
  // 錯誤範例
  ```
  ✅ {正確做法}
  ```{language}
  // 正確範例
  ```

## Related Resources
- [{資源名稱}]({URL})
- [{資源名稱}]({URL})
```

## metadata.json 格式

```json
{
  "skill_name": "react",
  "display_name": "React",
  "version": "1.0.0",
  "generated_at": "2025-01-29T12:00:00Z",
  "generator": "skill-seekers",
  "generator_version": "2.5.0",
  
  "source": {
    "type": "documentation",
    "url": "https://react.dev/",
    "llms_txt": false
  },
  
  "statistics": {
    "total_pages": 150,
    "total_code_blocks": 320,
    "total_words": 45000,
    "categories": 8
  },
  
  "languages_detected": [
    "javascript",
    "typescript",
    "jsx",
    "bash"
  ],
  
  "categories": [
    {
      "name": "getting_started",
      "display_name": "Getting Started",
      "file": "references/getting_started.md",
      "page_count": 15,
      "word_count": 5000
    },
    {
      "name": "api_reference",
      "display_name": "API Reference",
      "file": "references/api_reference.md",
      "page_count": 45,
      "word_count": 15000
    }
  ],
  
  "tags": [
    "frontend",
    "javascript",
    "ui",
    "react",
    "hooks"
  ],
  
  "compatibility": {
    "claude_code": true,
    "claude_ai": true,
    "chatgpt": true,
    "gemini": true
  }
}
```

## 分類參考文件格式

### index.md

```markdown
# {Skill Name} 參考文件

## 目錄

| 類別 | 說明 | 頁數 |
|------|------|------|
| [Getting Started](./getting_started.md) | 入門指南 | 15 |
| [Core Concepts](./core_concepts.md) | 核心概念 | 20 |
| [API Reference](./api_reference.md) | API 文件 | 45 |
| [Guides](./guides.md) | 教學指南 | 30 |
| [Examples](./examples.md) | 程式範例 | 25 |
| [Troubleshooting](./troubleshooting.md) | 問題排解 | 15 |

## 快速連結

- 安裝指南：[Getting Started](./getting_started.md#installation)
- API 總覽：[API Reference](./api_reference.md#overview)
- 常見問題：[Troubleshooting](./troubleshooting.md#faq)
```

### 分類文件格式

```markdown
# {Category Name}

## 概覽

[此分類的簡短說明]

---

## {Page Title 1}

**來源**: {original_url}

{頁面內容}

### 程式碼範例

```{language}
{code}
```

---

## {Page Title 2}

**來源**: {original_url}

{頁面內容}

---

[更多頁面...]
```

## 品質標準

### SKILL.md 檢查清單

- [ ] 所有程式碼區塊都有標記語言
- [ ] 至少 5 個實用程式碼範例
- [ ] 範例程式碼可直接執行
- [ ] 包含常見錯誤和解決方案
- [ ] 表格格式正確對齊
- [ ] 標題層級不超過 h3
- [ ] 無拼寫錯誤
- [ ] 版本資訊標註清楚
- [ ] 連結都可正常存取

### 程式碼範例標準

1. **完整性**：可直接複製執行
2. **註解**：關鍵步驟有說明
3. **變數命名**：有意義且一致
4. **錯誤處理**：包含基本的錯誤處理
5. **最佳實踐**：遵循該語言/框架的慣例

### 範例品質對照

**❌ 不好的範例：**
```python
def f(x):
    return x+1
```

**✅ 好的範例：**
```python
def increment(value: int) -> int:
    """將數值加 1
    
    Args:
        value: 要增加的數值
        
    Returns:
        增加後的數值
    """
    return value + 1

# 使用範例
result = increment(5)  # 結果: 6
```

## 輸出檔案大小限制

| 檔案類型 | 建議大小 | 最大大小 |
|----------|----------|----------|
| SKILL.md | < 50 KB | 100 KB |
| 單一參考文件 | < 100 KB | 200 KB |
| metadata.json | < 10 KB | 50 KB |
| 整個 skill 目錄 | < 1 MB | 5 MB |

超過限制時：
1. 分割大型參考文件
2. 移除重複內容
3. 壓縮程式碼範例
4. 考慮建立子 skill

## 多平台輸出

### Claude Code / Claude AI

預設格式，直接使用。

### ChatGPT (GPTs)

```python
generator = SkillGenerator(content, name=name)
generator.save(output_dir, format="chatgpt")

# 額外生成：
# - instructions.md (GPT Instructions)
# - knowledge/*.md (知識庫檔案，適合上傳)
```

### Gemini

```python
generator.save(output_dir, format="gemini")

# 生成 .tar.gz 格式
```

### 通用 Markdown

```python
generator.save(output_dir, format="markdown")

# 生成純 Markdown，無特定平台優化
```
