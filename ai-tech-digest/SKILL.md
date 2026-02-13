---
name: ai-tech-digest
description: |
  AI 技術日報追蹤與分析工具。涵蓋 Claude、OpenAI、Gemini、Unsloth 等主要 AI 平台的最新動態。
  Use when: 查詢 AI 最新技術動態、整理 AI 公司技術報告、生成 AI 技術週報或日報，or when user mentions AI 新聞, 技術日報, AI digest.
  Triggers: "AI 技術日報", "ai tech digest", "AI 新聞", "技術追蹤", "Claude 新聞", "OpenAI 新聞", "Gemini 新聞", "AI 週報", "Unsloth"
version: 2.0.0
---

# AI 技術日報 Skill

涵蓋 Claude、OpenAI、Gemini、Unsloth 等主要 AI 平台的最新技術動態追蹤與分析。

## When to Use This Skill

- 查詢 AI 最新技術動態或新聞
- 整理 AI 公司技術報告
- 追蹤 Claude、OpenAI、Gemini、Unsloth 最新發展
- 生成 AI 技術週報或日報
- 比較不同 AI 模型的最新能力
- 製作團隊技術週報

---

## 監控目標

| 公司/專案 | 重點產品 | 關注領域 |
|-----------|----------|----------|
| **Anthropic** | Claude, Claude Code, MCP | 對話模型、Agent、安全性 |
| **OpenAI** | GPT-4, ChatGPT, Sora, o1 | 多模態、推理、API |
| **Google DeepMind** | Gemini, Gemma, AlphaFold | 多模態、科學研究 |
| **Unsloth** | Unsloth Fine-tuning | 高效微調、LoRA、量化 |

---

## 資料來源

### 官方來源 (最高優先)

```yaml
Anthropic:
  - https://www.anthropic.com/news
  - https://www.anthropic.com/research
  - https://docs.anthropic.com/en/release-notes
  - https://github.com/anthropics (releases)

OpenAI:
  - https://openai.com/blog
  - https://openai.com/research
  - https://platform.openai.com/docs/changelog
  - https://github.com/openai (releases)

Google DeepMind:
  - https://deepmind.google/discover/blog
  - https://blog.google/technology/ai
  - https://ai.google.dev/changelog
  - https://github.com/google-deepmind (releases)

Unsloth:
  - https://unsloth.ai/blog
  - https://github.com/unslothai/unsloth/releases
  - https://huggingface.co/unsloth (new models)
```

### 技術社群 (高優先)

```yaml
綜合新聞:
  - https://www.theverge.com/ai-artificial-intelligence
  - https://techcrunch.com/category/artificial-intelligence
  - https://venturebeat.com/ai
  - https://the-decoder.com

論文與研究:
  - https://arxiv.org/list/cs.AI/recent
  - https://arxiv.org/list/cs.CL/recent
  - https://huggingface.co/papers

開發者社群:
  - https://news.ycombinator.com
  - https://www.reddit.com/r/LocalLLaMA
  - https://www.reddit.com/r/MachineLearning
```

---

## 每日報告格式

```markdown
# AI 技術日報
日期: YYYY-MM-DD (星期X)

---

## 📌 今日重點摘要

> [用 2-3 句話總結今日最重要的 1-3 則新聞]

---

## 🔷 Anthropic / Claude

### [新聞標題]
- **來源**: [網站名稱](URL)
- **日期**: YYYY-MM-DD
- **摘要**: [2-3 句重點摘要]
- **影響**: [對開發者/使用者的影響]
- **關鍵字**: `Claude`, `MCP`, `Agent`

---

## 🟢 OpenAI

### [新聞標題]
- **來源**: [網站名稱](URL)
- **摘要**: [2-3 句重點摘要]
- **關鍵字**: `GPT-4`, `API`, `Sora`

---

## 🔵 Google / Gemini

### [新聞標題]
- **來源**: [網站名稱](URL)
- **摘要**: [2-3 句重點摘要]
- **關鍵字**: `Gemini`, `Gemma`, `AI Studio`

---

## 🦥 Unsloth

### [新聞標題]
- **來源**: [網站名稱](URL)
- **摘要**: [2-3 句重點摘要]
- **技術細節**: [效能提升、支援模型等]

---

## 📚 值得關注的論文

| 標題 | 機構 | 連結 | 重點 |
|------|------|------|------|
| [論文名] | [機構] | [arXiv](url) | [一句話重點] |

---

## 🔗 延伸閱讀

- [標題1](URL)
- [標題2](URL)
```

---

## 新聞重要性評分

```
⭐⭐⭐⭐⭐ 必報導:
  - 新模型發布 (Claude 4, GPT-5, Gemini 2.0)
  - 重大 API 變更
  - 重要安全事件
  - 定價變更

⭐⭐⭐⭐ 高優先:
  - 功能更新
  - 效能改進
  - 新工具發布

⭐⭐⭐ 中優先:
  - 小版本更新
  - 文件更新
  - 教學文章

⭐⭐ 低優先:
  - 評論文章
  - 比較分析

⭐ 可選:
  - 傳聞、預測
```

### 來源可信度

```
最高: 官方部落格、官方文件、GitHub Releases
高:   主流科技媒體 (Verge, TechCrunch)
中:   專業 AI 媒體 (The Decoder, VentureBeat AI)
低:   個人部落格、社群討論
最低: 未經驗證的推文、傳聞
```

---

## 搜尋關鍵字

### 英文

```
Anthropic: claude, anthropic, claude 3, mcp protocol, claude code
OpenAI: openai, gpt-4, gpt-4o, chatgpt, sora, o1, assistants api
Google: gemini, gemini pro, google deepmind, gemma, ai studio
Unsloth: unsloth, fine-tuning, lora, qlora, 4bit quantization
```

### 中文

```
Claude 相關: Claude 更新, Anthropic 發布
OpenAI 相關: ChatGPT 更新, GPT-4 新功能
Gemini 相關: Gemini 更新, Google AI
Unsloth 相關: Unsloth 微調, 高效訓練
```

---

## 每日流程 (建議 15-30 分鐘)

```
1. 檢查官方來源 (5-10 分鐘)
   □ Anthropic News/Blog
   □ OpenAI Blog
   □ Google AI Blog
   □ Unsloth GitHub Releases

2. 掃描技術新聞網站 (5-10 分鐘)
   □ The Verge AI
   □ TechCrunch AI

3. 檢查社群討論 (3-5 分鐘)
   □ Hacker News 首頁
   □ Reddit r/LocalLLaMA

4. 整理與撰寫 (5-10 分鐘)
   □ 篩選重要新聞
   □ 撰寫摘要
   □ 格式化輸出
```

---

## 自動化建議

### RSS 訂閱

```yaml
RSS Feeds:
  - https://www.anthropic.com/news/rss.xml
  - https://openai.com/blog/rss.xml
  - https://blog.google/technology/ai/rss
  - https://techcrunch.com/category/artificial-intelligence/feed
```

### GitHub Watch

```
Repositories to Watch:
  - anthropics/anthropic-cookbook
  - openai/openai-cookbook
  - google/generative-ai-docs
  - unslothai/unsloth
```

---

## 注意事項

1. **時效性**: AI 領域變化快，優先報導 24-48 小時內的新聞
2. **驗證**: 重大消息需確認官方來源
3. **版權**: 摘要而非複製全文，附上原始連結
4. **偏見**: 平衡報導各家公司，避免偏頗

---

## 相關 Skills

- `hackernews-ai-digest` - Hacker News AI 新聞摘要
