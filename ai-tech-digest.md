# AI 技術日報 Skill (AI Tech Digest)
# 版本: 1.0
# 適用專案: AI 技術追蹤與分析
# 涵蓋範圍: Claude (Anthropic)、OpenAI、Gemini (Google)

---

## Skill 觸發條件

當使用者請求以下類型任務時，自動啟用此 Skill：
- 查詢 AI 最新技術動態或新聞
- 整理 AI 公司技術報告
- 追蹤 Claude、OpenAI、Gemini 最新發展
- 生成 AI 技術週報或日報
- 比較不同 AI 模型的最新能力

---

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
      - url: "https://docs.anthropic.com/en/release-notes"
        type: "changelog"
        description: "API 更新日誌"
    social:
      - platform: "twitter"
        handle: "@AnthropicAI"
      - platform: "github"
        repo: "anthropics/anthropic-cookbook"
    keywords:
      - "Claude"
      - "Claude 3"
      - "Claude Opus"
      - "Claude Sonnet"
      - "Constitutional AI"
      - "RLHF"

  openai:
    name: "OpenAI"
    official:
      - url: "https://openai.com/blog"
        type: "blog"
        description: "官方部落格"
      - url: "https://openai.com/research"
        type: "research"
        description: "研究論文"
      - url: "https://platform.openai.com/docs/changelog"
        type: "changelog"
        description: "API 更新日誌"
    social:
      - platform: "twitter"
        handle: "@OpenAI"
      - platform: "github"
        repo: "openai/openai-cookbook"
    keywords:
      - "GPT-4"
      - "GPT-5"
      - "ChatGPT"
      - "DALL-E"
      - "Sora"
      - "OpenAI o1"
      - "OpenAI o3"

  google:
    name: "Google (Gemini)"
    official:
      - url: "https://blog.google/technology/ai/"
        type: "blog"
        description: "Google AI 部落格"
      - url: "https://deepmind.google/research/"
        type: "research"
        description: "DeepMind 研究"
      - url: "https://ai.google.dev/gemini-api/docs/changelog"
        type: "changelog"
        description: "Gemini API 更新日誌"
    social:
      - platform: "twitter"
        handle: "@GoogleAI"
      - platform: "github"
        repo: "google-gemini/cookbook"
    keywords:
      - "Gemini"
      - "Gemini Ultra"
      - "Gemini Pro"
      - "Gemini Flash"
      - "Gemini 2.0"
      - "Bard"
      - "PaLM"
```

### 第三方來源

```yaml
third_party:
  news_aggregators:
    - name: "Hacker News"
      url: "https://news.ycombinator.com"
      filter: ["AI", "LLM", "Claude", "GPT", "Gemini"]

    - name: "Reddit r/MachineLearning"
      url: "https://reddit.com/r/MachineLearning"

    - name: "Reddit r/LocalLLaMA"
      url: "https://reddit.com/r/LocalLLaMA"

    - name: "AI News"
      url: "https://www.artificialintelligence-news.com"

  research_platforms:
    - name: "arXiv"
      url: "https://arxiv.org/list/cs.AI/recent"
      categories: ["cs.AI", "cs.CL", "cs.LG"]

    - name: "Papers With Code"
      url: "https://paperswithcode.com"

  benchmarks:
    - name: "LMSYS Chatbot Arena"
      url: "https://chat.lmsys.org/?leaderboard"

    - name: "Open LLM Leaderboard"
      url: "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
```

---

## 執行指令集

### /ai-digest:fetch [date]
**抓取指定日期的 AI 新聞**

參數：
- `date`：目標日期（預設：today，格式：YYYY-MM-DD）

執行流程：
1. 從所有配置的來源抓取資料
2. 過濾指定日期範圍的內容
3. 初步分類與去重
4. 儲存原始資料

**輸出結構：**
```
data/
└── ai-digest/
    └── raw/
        └── 2025-12-30/
            ├── anthropic.json
            ├── openai.json
            ├── google.json
            └── third-party.json
```

**輸出範例：**
```json
{
  "fetch_date": "2025-12-30",
  "source": "anthropic",
  "items": [
    {
      "id": "anth-20251230-001",
      "title": "Claude 3.5 Opus 發布",
      "url": "https://www.anthropic.com/news/claude-3-5-opus",
      "published_at": "2025-12-30T08:00:00Z",
      "type": "announcement",
      "summary": "Anthropic 發布最新的 Claude 3.5 Opus 模型...",
      "tags": ["model-release", "claude", "opus"],
      "importance": "high"
    }
  ],
  "metadata": {
    "fetched_at": "2025-12-30T10:30:00Z",
    "total_items": 5,
    "new_items": 3
  }
}
```

---

### /ai-digest:analyze [date]
**分析並分類當日新聞**

參數：
- `date`：目標日期（預設：today）

執行流程：
1. 讀取原始抓取資料
2. 使用 AI 進行內容分析
3. 提取關鍵資訊
4. 評估重要性等級
5. 生成標籤與分類

**分類類型：**
```yaml
categories:
  model_release:
    name: "模型發布"
    description: "新模型版本發布或重大更新"
    priority: 1

  api_update:
    name: "API 更新"
    description: "API 功能更新、新端點、價格調整"
    priority: 2

  research_paper:
    name: "研究論文"
    description: "學術論文、技術報告"
    priority: 3

  product_feature:
    name: "產品功能"
    description: "消費端產品新功能"
    priority: 4

  partnership:
    name: "合作夥伴"
    description: "企業合作、投資消息"
    priority: 5

  policy_safety:
    name: "政策與安全"
    description: "AI 安全、使用政策、監管相關"
    priority: 6

  benchmark:
    name: "評測排名"
    description: "效能評測、排行榜更新"
    priority: 7
```

**重要性評估：**
```yaml
importance_levels:
  critical:
    score: 5
    criteria:
      - "重大模型發布 (如 GPT-5, Claude 4)"
      - "突破性技術公告"
      - "重大安全事件"
    notification: "immediate"

  high:
    score: 4
    criteria:
      - "模型版本更新"
      - "重要 API 變更"
      - "影響廣泛的功能更新"
    notification: "daily_highlight"

  medium:
    score: 3
    criteria:
      - "一般功能更新"
      - "研究論文發布"
      - "效能改進"
    notification: "daily_digest"

  low:
    score: 2
    criteria:
      - "小型更新"
      - "Bug 修復"
      - "文件更新"
    notification: "weekly_summary"

  info:
    score: 1
    criteria:
      - "一般新聞報導"
      - "社群討論"
    notification: "archive_only"
```

---

### /ai-digest:report [format] [date]
**生成技術報告**

參數：
- `format`：報告格式（daily | weekly | monthly | custom）
- `date`：目標日期或日期範圍

執行流程：
1. 彙整分析後的資料
2. 按重要性排序
3. 生成結構化報告
4. 輸出指定格式

**輸出範例（daily）：**

```markdown
# 🤖 AI 技術日報
## 2025年12月30日

---

### 📊 今日摘要

| 公司 | 更新數 | 重要更新 |
|------|--------|----------|
| Anthropic (Claude) | 3 | 1 |
| OpenAI | 2 | 1 |
| Google (Gemini) | 4 | 2 |

---

### 🔥 重要更新

#### [Critical] Claude 3.5 Opus 正式發布
**來源：** Anthropic 官方公告
**時間：** 2025-12-30 08:00 UTC

Anthropic 今日正式發布 Claude 3.5 Opus，這是目前最強大的 Claude 模型...

**關鍵亮點：**
- 推理能力提升 40%
- 程式碼生成準確度達 95%
- 支援 200K context window
- 新增視覺理解能力

**API 變更：**
```python
# 新模型名稱
model = "claude-3.5-opus-20251230"

# 價格調整
# Input: $15 / 1M tokens
# Output: $75 / 1M tokens
```

**相關連結：**
- [官方公告](https://anthropic.com/news/...)
- [API 文件](https://docs.anthropic.com/...)
- [遷移指南](https://docs.anthropic.com/migration/...)

---

#### [High] OpenAI 推出 GPT-4 Turbo 視覺增強版
**來源：** OpenAI Blog
**時間：** 2025-12-30 06:00 UTC

OpenAI 發布 GPT-4 Turbo 的視覺能力更新...

---

### 📚 研究論文

#### Constitutional AI: A Practical Guide
**作者：** Anthropic Research Team
**發布：** arXiv 2025.12.30

摘要：本文詳細介紹 Constitutional AI 的實作方法...

**論文連結：** [arXiv:2512.xxxxx](https://arxiv.org/abs/...)

---

### 🔧 API 更新

| 平台 | 更新內容 | 影響範圍 |
|------|----------|----------|
| Claude API | 新增 batch processing 端點 | 高吞吐量應用 |
| OpenAI API | 調整 rate limit 策略 | 所有用戶 |
| Gemini API | 支援 Gemini 2.0 Flash | 開發者預覽 |

---

### 📈 排行榜變動

#### LMSYS Chatbot Arena (更新於 2025-12-30)

| 排名 | 模型 | ELO | 變動 |
|------|------|-----|------|
| 1 | Claude 3.5 Opus | 1350 | 🆕 |
| 2 | GPT-4 Turbo | 1320 | ↓1 |
| 3 | Gemini Ultra | 1305 | ↓1 |
| 4 | Claude 3 Opus | 1280 | ↓1 |
| 5 | GPT-4 | 1260 | - |

---

### 🔮 值得關注

- **傳聞：** OpenAI 可能在 Q1 2026 發布 GPT-5
- **動態：** Google 正在測試 Gemini 2.0 多模態能力
- **社群：** Reddit 討論 Claude 3.5 Opus 的程式碼能力

---

### 📅 即將到來

| 日期 | 事件 | 公司 |
|------|------|------|
| 2026-01-15 | OpenAI DevDay | OpenAI |
| 2026-01-20 | Google I/O AI 專場 | Google |
| 2026-02-01 | Anthropic 年度報告 | Anthropic |

---

*報告生成時間：2025-12-30 23:00 UTC*
*資料來源：官方公告、arXiv、社群討論*
```

---

### /ai-digest:compare [models]
**比較不同模型的最新能力**

參數：
- `models`：要比較的模型（以逗號分隔，預設：claude,gpt,gemini）

執行流程：
1. 收集各模型最新規格
2. 整理評測數據
3. 生成比較表格
4. 分析優劣勢

**輸出範例：**

```markdown
# 🔍 AI 模型能力比較
## 更新日期：2025-12-30

### 最新模型版本

| 特性 | Claude 3.5 Opus | GPT-4 Turbo | Gemini 2.0 Ultra |
|------|-----------------|-------------|------------------|
| 發布日期 | 2025-12-30 | 2025-11-15 | 2025-12-01 |
| Context Window | 200K | 128K | 1M |
| 多模態 | 文字+圖片 | 文字+圖片+音訊 | 文字+圖片+影片+音訊 |
| 輸入價格 | $15/1M | $10/1M | $12.50/1M |
| 輸出價格 | $75/1M | $30/1M | $37.50/1M |

### 評測分數

| 評測 | Claude 3.5 Opus | GPT-4 Turbo | Gemini 2.0 Ultra |
|------|-----------------|-------------|------------------|
| MMLU | 92.3% | 90.1% | 91.5% |
| HumanEval | 95.2% | 91.0% | 88.5% |
| MATH | 78.5% | 72.3% | 75.0% |
| Arena ELO | 1350 | 1320 | 1305 |

### 最佳使用場景

**Claude 3.5 Opus：**
- ✅ 程式碼生成與分析
- ✅ 長文件處理
- ✅ 複雜推理任務

**GPT-4 Turbo：**
- ✅ 多模態應用
- ✅ 創意寫作
- ✅ 廣泛的外掛生態

**Gemini 2.0 Ultra：**
- ✅ 超長上下文
- ✅ 影片理解
- ✅ Google 生態整合
```

---

### /ai-digest:subscribe [topics]
**設定追蹤訂閱**

參數：
- `topics`：訂閱主題（可選多個）

可訂閱主題：
```yaml
topics:
  all:
    description: "所有更新"

  model_releases:
    description: "模型發布"
    includes: ["claude", "gpt", "gemini", "open-source"]

  api_changes:
    description: "API 變更"
    includes: ["pricing", "endpoints", "rate-limits"]

  research:
    description: "研究論文"
    includes: ["arxiv", "papers", "benchmarks"]

  claude_only:
    description: "僅 Claude 相關"
    source: "anthropic"

  openai_only:
    description: "僅 OpenAI 相關"
    source: "openai"

  gemini_only:
    description: "僅 Gemini 相關"
    source: "google"
```

**輸出範例：**
```yaml
# .ai-digest/subscriptions.yaml

user: "developer"
created: "2025-12-30"

subscriptions:
  - topic: "model_releases"
    priority: "high"
    notification: "immediate"

  - topic: "api_changes"
    priority: "medium"
    notification: "daily"

  - topic: "research"
    priority: "low"
    notification: "weekly"

filters:
  exclude_sources: []
  min_importance: "medium"

delivery:
  format: "markdown"
  channel: "file"  # file | email | slack | discord
  output_path: "./reports/"
```

---

### /ai-digest:archive [query]
**搜尋歷史報告**

參數：
- `query`：搜尋關鍵字或日期範圍

執行流程：
1. 搜尋本地存檔
2. 支援關鍵字與日期過濾
3. 返回相關報告列表

**輸出範例：**
```markdown
## 搜尋結果：「Claude API」

找到 15 筆相關記錄

### 最近更新

1. **2025-12-30** - Claude 3.5 Opus 發布
   - 重要性：Critical
   - 標籤：model-release, api-update

2. **2025-12-15** - Claude API rate limit 調整
   - 重要性：High
   - 標籤：api-update, pricing

3. **2025-12-01** - Claude 3 Sonnet 效能優化
   - 重要性：Medium
   - 標籤：performance, api-update

[查看更多...]
```

---

### /ai-digest:schedule [cron]
**設定自動排程**

參數：
- `cron`：Cron 表達式（預設：每日早上 8 點）

**排程配置：**
```yaml
# .ai-digest/schedule.yaml

schedules:
  daily_digest:
    cron: "0 8 * * *"  # 每天早上 8:00
    tasks:
      - fetch
      - analyze
      - report:daily
    output: "./reports/daily/"

  weekly_summary:
    cron: "0 10 * * 0"  # 每週日早上 10:00
    tasks:
      - report:weekly
    output: "./reports/weekly/"

  breaking_news:
    trigger: "importance >= critical"
    tasks:
      - notify
    channels: ["slack", "email"]
```

---

## 輸出目錄結構

```
ai-digest/
├── config/
│   ├── sources.yaml         # 資料來源配置
│   ├── subscriptions.yaml   # 訂閱設定
│   └── schedule.yaml        # 排程設定
├── data/
│   └── raw/
│       └── YYYY-MM-DD/      # 按日期存放原始資料
├── reports/
│   ├── daily/
│   │   └── YYYY-MM-DD.md    # 日報
│   ├── weekly/
│   │   └── YYYY-WXX.md      # 週報
│   └── monthly/
│       └── YYYY-MM.md       # 月報
├── archive/
│   └── index.json           # 搜尋索引
└── templates/
    ├── daily.md             # 日報模板
    ├── weekly.md            # 週報模板
    └── comparison.md        # 比較報告模板
```

---

## 報告模板

### 日報模板

```markdown
# 🤖 AI 技術日報
## {{date}}

---

### 📊 今日摘要
{{summary_table}}

---

### 🔥 重要更新
{{#each critical_updates}}
#### [{{importance}}] {{title}}
**來源：** {{source}}
**時間：** {{published_at}}

{{content}}

{{#if code_changes}}
**程式碼變更：**
\`\`\`{{language}}
{{code_changes}}
\`\`\`
{{/if}}

**相關連結：**
{{#each links}}
- [{{name}}]({{url}})
{{/each}}

---
{{/each}}

### 📚 研究論文
{{#each papers}}
#### {{title}}
**作者：** {{authors}}
**摘要：** {{abstract}}

[論文連結]({{url}})
{{/each}}

---

### 📈 排行榜變動
{{leaderboard_table}}

---

*報告生成時間：{{generated_at}}*
```

---

## 開發檢查清單

### Phase 1: 基礎設施
- [ ] 設定資料來源配置
- [ ] 建立抓取排程
- [ ] 建立資料儲存結構

### Phase 2: 資料處理
- [ ] 實作網頁抓取器
- [ ] 實作內容分析器
- [ ] 實作去重機制

### Phase 3: 報告生成
- [ ] 實作日報生成
- [ ] 實作週報生成
- [ ] 實作比較報告

### Phase 4: 通知系統
- [ ] 實作即時通知
- [ ] 整合通知管道
- [ ] 建立訂閱管理

---

## 使用範例

### 快速開始

```bash
# 抓取今日新聞
/ai-digest:fetch

# 生成今日報告
/ai-digest:report daily

# 比較最新模型
/ai-digest:compare claude,gpt,gemini

# 搜尋歷史
/ai-digest:archive "Claude API 更新"
```

### 自動化設定

```bash
# 設定每日自動抓取與報告
/ai-digest:schedule "0 8 * * *"

# 訂閱關注主題
/ai-digest:subscribe model_releases,api_changes
```

---

## 版本資訊

```yaml
version: "1.0"
created: "2025-12-30"
author: "AI Tech Digest Team"

changelog:
  - version: "1.0"
    date: "2025-12-30"
    changes:
      - "初始版本建立"
      - "支援 Claude、OpenAI、Gemini 三大平台"
      - "實作日報、週報、月報生成"
      - "實作模型比較功能"
      - "實作訂閱與排程系統"
```
