---
name: ai-tech-digest
description: |
  AI 技術日報追蹤與分析工具。涵蓋 Claude (Anthropic)、OpenAI、Gemini (Google) 等主要 AI 公司的最新動態。
  Use when: 查詢 AI 最新技術動態、整理 AI 公司技術報告、生成 AI 技術週報或日報，or when user mentions AI 新聞, 技術日報, AI digest.
  Triggers: "AI 技術日報", "ai tech digest", "AI 新聞", "技術追蹤", "Claude 新聞", "OpenAI 新聞", "Gemini 新聞", "AI 週報"
version: 1.0.0
---

# AI 技術日報 Skill (AI Tech Digest)

涵蓋 Claude (Anthropic)、OpenAI、Gemini (Google) 等主要 AI 公司的最新技術動態追蹤與分析。

## When to Use This Skill

- 查詢 AI 最新技術動態或新聞
- 整理 AI 公司技術報告
- 追蹤 Claude、OpenAI、Gemini 最新發展
- 生成 AI 技術週報或日報
- 比較不同 AI 模型的最新能力

## 資料來源配置

### 官方來源

```yaml
sources:
  anthropic:
    name: "Anthropic (Claude)"
    official:
      - url: "https://www.anthropic.com/news"
        type: "news"
        description: "官方新聞與公告"
      - url: "https://www.anthropic.com/research"
        type: "research"
        description: "研究論文與技術報告"
  openai:
    name: "OpenAI"
    official:
      - url: "https://openai.com/blog"
        type: "blog"
        description: "官方部落格"
  google:
    name: "Google (Gemini)"
    official:
      - url: "https://blog.google/technology/ai/"
        type: "blog"
        description: "Google AI 部落格"
```

## 輸出格式

### 日報格式

```markdown
# AI 技術日報 - YYYY-MM-DD

## 重點摘要
- [要點1]
- [要點2]

## Anthropic (Claude)
### 新功能/更新
- ...

## OpenAI
### 新功能/更新
- ...

## Google (Gemini)
### 新功能/更新
- ...
```

## 相關 Skills

- `hackernews-ai-digest` - Hacker News AI 新聞摘要
