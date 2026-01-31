# Agentic RAG 技術資訊研究報告

> 研究目標：全面蒐集 Agentic RAG 相關技術資訊，建立完整的 Agentic RAG Skill 基礎
> 
> 執行方式：Ralph Loop 20 回合自主研究
> 
> 研究日期：2026-01-13

---

## 目錄

1. [Agentic RAG 概述](#1-agentic-rag-概述)
2. [黃金數據集建立方法](#2-黃金數據集建立方法)
3. [評估體系架構](#3-評估體系架構)
4. [答案精準度評估框架](#4-答案精準度評估框架)
5. [Agent 介入機制](#5-agent-介入機制)
6. [工具使用方案](#6-工具使用方案)
7. [實作指南](#7-實作指南)
8. [最佳實踐](#8-最佳實踐)

---

## 研究進度

| 回合 | 主題 | 狀態 |
|------|------|------|
| 1 | Agentic RAG 基礎概念與架構 | ✅ 完成 |
| 2 | 黃金數據集建立方法 | ✅ 完成 |
| 3 | RAGAS 評估框架 | ✅ 完成 |
| 4 | 其他評估框架比較 | ✅ 完成 |
| 5 | Agent 路由與決策機制 | ✅ 完成 |
| 6 | 工具選擇與調用策略 | ✅ 完成 |
| 7 | 多步驟推理與規劃 | ✅ 完成 |
| 8 | 檢索策略與重排序 | ✅ 完成 |
| 9 | 上下文壓縮與優化 | ✅ 完成 |
| 10 | 迭代式檢索機制 | ✅ 完成 |
| 11 | 合成數據生成 | ✅ 完成 |
| 12 | 評估指標深入分析 | ✅ 完成 |
| 13 | 企業級部署考量 | ✅ 完成 |
| 14 | 錯誤處理與回退策略 | ✅ 完成 |
| 15 | 快取與效能優化 | ✅ 完成 |
| 16 | 可觀測性與監控 | ✅ 完成 |
| 17 | 安全性考量 | ✅ 完成 |
| 18 | 開源框架比較 | ✅ 完成 |
| 19 | 案例研究 | ✅ 完成 |
| 20 | 綜合整理與 Skill 規劃 | ✅ 完成 |

---

## 1. Agentic RAG 概述

### 1.1 定義與演進

**Agentic RAG** 是將自主 AI 代理嵌入 RAG 管道的進階架構。相較於傳統 RAG 的靜態工作流程，Agentic RAG 透過代理的自主決策能力，實現動態檢索策略、迭代優化與複雜任務管理。

#### 傳統 RAG vs Agentic RAG

| 特性 | 傳統 RAG | Agentic RAG |
|------|----------|-------------|
| 工作流程 | 靜態、單次檢索 | 動態、迭代式 |
| 決策能力 | 預定義規則 | 自主推理與決策 |
| 資料來源 | 單一知識庫 | 多源整合 |
| 錯誤處理 | 有限 | 自我反思與修正 |
| 複雜查詢 | 受限 | 多步驟推理 |

### 1.2 核心架構類型

#### A. 單代理架構 (Single-Agent)
- 一個集中式代理管理檢索、路由與資訊整合
- 適合工具或資料源數量有限的場景
- 簡單易實作，但擴展性有限

#### B. 多代理系統 (Multi-Agent)
- 多個專門代理協作處理複雜任務
- 任務分工：檢索代理、推理代理、合成代理
- 提高可擴展性與適應性

#### C. 層級代理架構 (Hierarchical)
- 主代理協調多個子代理
- 適合企業級複雜工作流程
- 支援跨域知識整合

### 1.3 四大核心設計模式

#### 1. Reflection（反思）
```
生成初始輸出 → 自我評估 → 識別錯誤 → 迭代改進
```
- 代理評估自身決策與輸出
- 識別錯誤並進行迭代優化
- 提升多步驟推理任務的準確性

#### 2. Planning（規劃）
```
理解任務 → 分解子任務 → 優先排序 → 逐步執行
```
- 將複雜任務分解為結構化子任務
- 創建任務執行路線圖
- 減少計算開銷，提高效率

#### 3. Tool Use（工具使用）
```
識別需求 → 選擇工具 → 執行調用 → 整合結果
```
- 動態調用外部工具與 API
- 支援向量搜索、網頁搜索、SQL 查詢等
- 擴展 LLM 能力邊界

#### 4. Multi-Agent Collaboration（多代理協作）
```
任務分配 → 專業化處理 → 結果聚合 → 協調輸出
```
- 任務專門化與平行處理
- 代理間通訊與結果共享
- 提高工作流程的可擴展性

### 1.4 典型工作流程

```
用戶查詢
    ↓
[路由代理] ─→ 決定使用哪些知識源/工具
    ↓
[檢索代理] ─→ 從多個來源檢索相關資訊
    ↓
[驗證代理] ─→ 評估檢索結果品質
    ↓
[推理代理] ─→ 多步驟推理與分析
    ↓
[生成代理] ─→ 合成最終回應
    ↓
[反思迴圈] ─→ 必要時重新檢索或優化
    ↓
最終輸出
```

### 1.5 主要框架與工具

| 框架 | 特點 | 適用場景 |
|------|------|----------|
| **LangGraph** | 狀態圖管理、靈活工作流程 | 複雜對話與多步驟推理 |
| **LlamaIndex** | 強大索引與查詢引擎 | 文件處理與知識管理 |
| **CrewAI** | 多代理協作框架 | 團隊式任務分工 |
| **AutoGen** | 微軟多代理框架 | 對話式代理開發 |

### 1.6 關鍵能力

1. **多步驟推理**：將複雜查詢分解為可執行步驟
2. **動態工具使用**：根據需求選擇適當工具
3. **增強可靠性**：主動驗證檢索資訊品質
4. **上下文感知**：維護對話狀態與歷史
5. **自適應學習**：根據反饋改進策略

---

## 2. 黃金數據集建立方法

### 2.1 數據集層級

```
┌─────────────────────────────────────────────────────────┐
│                    黃金數據集 (Gold)                     │
│         經 SME 審核驗證的高品質評估數據                  │
├─────────────────────────────────────────────────────────┤
│                    白銀數據集 (Silver)                   │
│         LLM 自動生成，經初步驗證的合成數據               │
├─────────────────────────────────────────────────────────┤
│                    銅質數據集 (Bronze)                   │
│         原始生成數據，未經驗證                           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 黃金數據集核心組成

| 欄位 | 說明 | 用途 |
|------|------|------|
| **question** | 基於來源文件的問題 | 模擬用戶查詢 |
| **ground_truth** | 預期的準確答案 | 評估答案正確性 |
| **context** | RAG 檢索的上下文 | 評估檢索品質 |
| **answer** | RAG 生成的答案 | 端到端評估 |
| **metadata** | 來源、類型、難度等 | 分析與除錯 |

### 2.3 合成數據生成策略

#### A. RAGAS TestsetGenerator

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings
)

# 從文件生成測試集
dataset = generator.generate_with_langchain_docs(
    docs, 
    testset_size=100
)

# 控制問題類型分佈
# - reasoning: 推理類問題
# - conditioning: 條件類問題  
# - multi-context: 多上下文問題
```

#### B. 問題類型分類

| 問題類型 | 說明 | 比例建議 |
|----------|------|----------|
| **fact_single** | 單一事實查詢 | 30% |
| **fact_multi** | 多文件事實整合 | 20% |
| **reasoning** | 邏輯推理問題 | 25% |
| **comparison** | 比較分析問題 | 15% |
| **conditional** | 條件判斷問題 | 10% |

#### C. 語句提取策略

```
原始文件
    ↓
提取關鍵語句/事實
    ↓
轉換為問題形式
    ↓
生成對應答案
    ↓
標注問題類型
```

### 2.4 白銀轉黃金流程

```
合成數據生成（Bronze）
        ↓
    LLM 品質篩選
        ↓
    去除低品質問題
        ↓
白銀數據集（Silver）
        ↓
    SME 領域專家審核
        ↓
    標注者一致性檢查
        ↓
    偏見審計
        ↓
黃金數據集（Gold）
```

### 2.5 SME 審核流程

#### 審核維度

1. **事實正確性 (Veracity)**
   - 答案是否與來源文件一致
   - 是否有過時或錯誤資訊

2. **問題品質**
   - 問題是否清晰明確
   - 是否反映真實用戶場景

3. **答案完整性**
   - 是否完整回答問題
   - 是否遺漏重要資訊

4. **上下文相關性**
   - 標注的上下文是否充足
   - 是否包含所需資訊

#### 標注者一致性

```python
# 計算 Inter-Annotator Agreement
# Cohen's Kappa 或 Fleiss' Kappa

# 建議流程：
# 1. 試標註回合
# 2. 校準會議
# 3. 正式標註
# 4. 定期抽查
```

### 2.6 資料集設計最佳實踐

#### 覆蓋率考量

- **主題覆蓋**：涵蓋所有重要業務主題
- **難度分佈**：簡單、中等、困難問題混合
- **邊緣案例**：包含異常情況和邊界條件
- **對抗樣本**：包含可能導致錯誤的輸入

#### 避免常見陷阱

| 陷阱 | 解決方案 |
|------|----------|
| 過度擬合合成數據 | 平衡合成與人工數據 |
| 資料污染 | 嚴格去重檢查 |
| 分佈偏差 | 確保類型多樣性 |
| 缺乏安全測試 | 加入對抗性樣本 |

### 2.7 資料集規模建議

| 用途 | 建議規模 | 說明 |
|------|----------|------|
| 概念驗證 | 50-100 | 快速迭代 |
| 開發測試 | 200-500 | 穩定評估 |
| 生產評估 | 500-1000+ | 全面覆蓋 |
| 持續監控 | 滾動更新 | 反映變化 |

### 2.8 工具與平台

| 工具 | 特點 | 適用場景 |
|------|------|----------|
| **RAGAS** | 自動生成 + 評估 | 快速啟動 |
| **DeepEval** | 合成器 + 風格控制 | 客製化需求 |
| **Langfuse** | 數據集管理 + 可觀測 | 生產環境 |
| **Kili Technology** | 協作標註平台 | 團隊合作 |
| **FMEval (AWS)** | 企業級評估套件 | AWS 生態 |

---

## 3. 評估體系架構

### 3.1 RAGAS 評估框架

RAGAS (Retrieval Augmented Generation Assessment) 是最廣泛使用的 RAG 評估框架，提供無需人工標註的自動化評估。

#### 核心指標

| 指標 | 計算公式 | 評估對象 |
|------|----------|----------|
| **Faithfulness** | 支持聲明數 / 總聲明數 | 生成器 |
| **Answer Relevancy** | 答案與問題的語義相似度 | 生成器 |
| **Context Precision** | 相關上下文排序品質 | 檢索器 |
| **Context Recall** | 涵蓋所需資訊的比例 | 檢索器 |

#### Faithfulness 計算流程

```
Step 1: 從回答中提取所有聲明 (claims)
Step 2: 對每個聲明，檢查是否可從上下文推導
Step 3: 計算比例

Faithfulness = 受支持的聲明數 / 總聲明數

範例：
- 回答包含 4 個聲明
- 其中 3 個有上下文支持
- Faithfulness = 3/4 = 0.75
```

#### Answer Relevancy 計算

```
Step 1: 從答案生成 N 個可能的問題
Step 2: 計算生成問題與原問題的語義相似度
Step 3: 取平均值

高相關性：答案直接回應問題
低相關性：答案偏離問題主題
```

### 3.2 RAG Triad 模型

TruLens 提出的 RAG 三元組評估模型：

```
        ┌─────────────┐
        │   Query     │
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌───────┐  ┌───────┐  ┌───────┐
│Context│  │Answer │  │Ground │
│Relev. │  │Relev. │  │edness │
└───────┘  └───────┘  └───────┘
    │          │          │
    └──────────┴──────────┘
               │
        ┌──────┴──────┐
        │  RAG Score  │
        └─────────────┘
```

| 維度 | 評估內容 | 問題識別 |
|------|----------|----------|
| **Context Relevancy** | 檢索內容與查詢相關性 | 檢索失敗 |
| **Answer Relevancy** | 答案與查詢相關性 | 生成偏離 |
| **Groundedness** | 答案基於上下文程度 | 幻覺問題 |

---

## 4. 答案精準度評估框架

### 4.1 框架比較總覽

| 框架 | 類型 | 核心優勢 | 最佳場景 |
|------|------|----------|----------|
| **RAGAS** | 開源 | 無需標註、整合廣泛 | 快速迭代 |
| **DeepEval** | 開源 | 單元測試風格、CI/CD | 工程團隊 |
| **TruLens** | 商業 | 版本追蹤、反饋函數 | 企業監控 |
| **LlamaIndex Eval** | 開源 | 原生整合、檢索評估 | LlamaIndex 用戶 |
| **Arize Phoenix** | 開源 | OpenTelemetry、可觀測 | 生產除錯 |
| **Langfuse** | 混合 | 追蹤 + 評估 | 全生命週期 |
| **Braintrust** | 商業 | 快速上手、框架無關 | 快速驗證 |

### 4.2 主要框架詳解

#### A. DeepEval

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

# 建立測試案例
test_case = LLMTestCase(
    input="What is RAG?",
    actual_output=response,
    retrieval_context=retrieved_chunks,
    expected_output=ground_truth  # 可選
)

# 執行評估
assert_test(
    test_case=test_case, 
    metrics=[
        FaithfulnessMetric(),
        AnswerRelevancyMetric(),
        ContextualRecallMetric()
    ]
)
```

**特點：**
- Pytest 整合，適合 CI/CD
- 50+ 預建指標
- 合成數據生成
- 自解釋評分

#### B. TruLens

```python
from trulens_eval import Feedback, Tru

# 定義反饋函數
f_groundedness = Feedback(
    provider.groundedness_measure_with_cot_reasons
).on_context().on_output()

f_answer_relevance = Feedback(
    provider.relevance
).on_input().on_output()

# 建立記錄器
tru_recorder = TruChain(
    chain,
    feedbacks=[f_groundedness, f_answer_relevance]
)

# 評估
with tru_recorder as recording:
    response = chain.invoke(question)
```

**特點：**
- 版本比較儀表板
- 與 LangChain/LlamaIndex 深度整合
- 即時反饋函數

#### C. LlamaIndex Eval

```python
from llama_index.evaluation import (
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    CorrectnessEvaluator
)

# 評估回應
evaluator = FaithfulnessEvaluator()
result = evaluator.evaluate_response(
    query=query,
    response=response
)
```

**特點：**
- 原生 LlamaIndex 整合
- 檢索排名指標 (MRR, Hit Rate)
- 批次評估支援

### 4.3 指標分類

#### 檢索器指標

| 指標 | 說明 | 需要標註 |
|------|------|----------|
| Context Precision | 相關文件排序品質 | 是 |
| Context Recall | 涵蓋率 | 是 |
| Context Relevancy | 整體相關性 | 否 |
| Hit Rate@K | 前 K 結果命中率 | 是 |
| MRR | 平均倒數排名 | 是 |
| NDCG | 標準化折扣累積增益 | 是 |

#### 生成器指標

| 指標 | 說明 | 需要標註 |
|------|------|----------|
| Faithfulness | 忠實於上下文 | 否 |
| Answer Relevancy | 答案相關性 | 否 |
| Answer Correctness | 答案正確性 | 是 |
| Semantic Similarity | 語義相似度 | 是 |
| Completeness | 完整性 | 否 |
| Conciseness | 簡潔性 | 否 |

### 4.4 LLM-as-Judge 模式

```
評估流程：
1. 準備評估 Prompt（包含評分標準）
2. 將 Query + Context + Response 送入 Judge LLM
3. LLM 輸出分數與理由
4. 解析並記錄結果

常用 Judge 模型：
- GPT-4o（最佳準確度）
- Claude 3.5 Sonnet
- 本地模型（成本考量）
```

### 4.5 評估最佳實踐

```
┌─────────────────────────────────────────────────────────┐
│                    評估策略矩陣                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  開發階段：                                             │
│  • 使用合成數據快速迭代                                 │
│  • 專注於組件級評估（檢索 vs 生成）                     │
│  • 小規模測試集（50-100 題）                           │
│                                                         │
│  預生產階段：                                           │
│  • 引入 SME 標註的黃金數據集                           │
│  • 端到端評估                                          │
│  • 對抗性測試（邊緣案例）                              │
│                                                         │
│  生產環境：                                             │
│  • 持續監控（無標註評估）                              │
│  • 用戶反饋整合                                        │
│  • 定期回歸測試                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Agent 介入機制

### 5.1 路由代理 (Routing Agent)

路由代理是 Agentic RAG 最基本的形式，負責決定查詢應該發送到哪個資料來源或工具。

#### 路由決策流程

```
用戶查詢
    ↓
┌─────────────────────────────────────────────────────────┐
│                    路由代理 (Router)                     │
│                                                         │
│   分析查詢意圖 → 評估可用工具 → 選擇最佳路徑            │
│                                                         │
└─────────────────────────────────────────────────────────┘
    ↓
    ├─→ Vector Store（語義搜索）
    ├─→ Web Search（即時資訊）
    ├─→ SQL Database（結構化查詢）
    ├─→ API 調用（外部服務）
    └─→ 直接回應（通用知識）
```

#### 路由策略類型

| 策略 | 說明 | 適用場景 |
|------|------|----------|
| **單一路由** | 選擇一個工具執行 | 簡單查詢 |
| **並行路由** | 多工具同時執行 | 複雜查詢 |
| **串行路由** | 工具依序執行 | 依賴性任務 |
| **條件路由** | 根據中間結果決定 | 動態工作流程 |

### 5.2 查詢規劃代理 (Query Planning Agent)

負責將複雜查詢分解為可執行的子任務。

```python
# 查詢規劃範例
class QueryPlanningAgent:
    def plan(self, query: str) -> List[SubQuery]:
        """
        將複雜查詢分解為子查詢
        
        輸入: "Compare Q3 2023 and Q1 2024 financial performance"
        
        輸出:
        [
            SubQuery(query="Q3 2023 financial data", source="sql_db"),
            SubQuery(query="Q1 2024 financial data", source="sql_db"),
            SubQuery(query="comparison analysis", source="llm")
        ]
        """
        pass
```

### 5.3 文件評分代理 (Document Grading Agent)

在生成回應前評估檢索文件的品質。

```
檢索結果
    ↓
┌─────────────────────────────────────────────────────────┐
│                   文件評分代理                           │
│                                                         │
│   評估維度：                                            │
│   • 相關性：文件是否與查詢相關？                        │
│   • 完整性：是否包含足夠資訊？                          │
│   • 矛盾性：是否與其他文件矛盾？                        │
│   • 時效性：資訊是否過時？                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
    ↓
    ├─→ 通過 → 進入生成階段
    └─→ 不通過 → 觸發重新檢索或查詢改寫
```

### 5.4 自我修正代理 (Self-Correction Agent)

實現 Corrective RAG (CRAG) 模式的核心組件。

#### CRAG 工作流程

```
初始檢索
    ↓
評估文件品質
    ↓
┌─────────────────────────────────────────────────────────┐
│                     決策節點                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   高品質 ─→ 直接生成回應                                │
│   中等品質 ─→ 知識精煉後生成                            │
│   低品質 ─→ 觸發網頁搜索補充                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.5 ReAct 代理框架

結合推理 (Reasoning) 與行動 (Acting) 的代理模式。

```
ReAct 循環：

Thought: 分析當前狀態，決定下一步行動
    ↓
Action: 選擇並執行工具
    ↓
Observation: 觀察工具執行結果
    ↓
Thought: 評估結果，決定是否需要更多步驟
    ↓
... (迭代直到完成)
    ↓
Final Answer: 生成最終回應
```

---

## 6. 工具使用方案

### 6.1 工具分類框架

```
┌─────────────────────────────────────────────────────────┐
│                      工具三支柱                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 資料存取工具                                        │
│     • Vector Store (語義搜索)                           │
│     • SQL/NoSQL Database                               │
│     • Graph Database                                   │
│     • Web Search API                                   │
│     • File System                                      │
│                                                         │
│  2. 計算工具                                            │
│     • Code Interpreter (Python)                        │
│     • Calculator / Math Engine                         │
│     • Data Transformation (ETL)                        │
│     • ML Model Inference                               │
│                                                         │
│  3. 行動工具                                            │
│     • API 調用 (外部服務)                               │
│     • Email / Notification                             │
│     • Database Write                                   │
│     • File Creation                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 6.2 工具定義最佳實踐

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchTool(BaseModel):
    """向量資料庫搜索工具
    
    當需要從內部文件中檢索相關資訊時使用此工具。
    適用於：產品文件、政策文件、技術規格等。
    """
    
    query: str = Field(
        description="搜索查詢，應為具體明確的問題或關鍵詞"
    )
    top_k: int = Field(
        default=5,
        description="返回的文件數量，1-10"
    )
    filter_category: Literal["all", "policy", "technical", "product"] = Field(
        default="all",
        description="文件類別過濾器"
    )
```

**關鍵原則：**
- 清晰的工具描述（LLM 依賴此做決策）
- 明確的參數類型與說明
- 合理的預設值
- 限制範圍以防止濫用

### 6.3 工具選擇策略

| 策略 | 描述 | 優點 | 缺點 |
|------|------|------|------|
| **預定義選擇** | 開發者預先決定工具 | 可預測、可控 | 缺乏彈性 |
| **LLM 自主選擇** | LLM 根據查詢選擇 | 靈活、智能 | 可能錯誤選擇 |
| **混合策略** | 規則 + LLM 結合 | 平衡可控與靈活 | 複雜度較高 |

### 6.4 防止工具濫用

```python
# 安全控制措施
class ToolGuardrails:
    MAX_TOOL_CALLS = 10          # 單次對話最大調用次數
    MAX_SEQUENTIAL_SAME = 3       # 同一工具連續最大次數
    TIMEOUT_SECONDS = 30          # 單次調用超時
    
    def validate_call(self, tool_name: str, call_count: int) -> bool:
        """驗證工具調用是否允許"""
        if call_count > self.MAX_TOOL_CALLS:
            return False
        return True
```

### 6.5 工具調用模式

#### A. 順序調用
```
Query → Tool A → Result A → Tool B → Result B → Response
```

#### B. 並行調用
```
Query ┬→ Tool A → Result A ─┐
      ├→ Tool B → Result B ─┼→ Aggregate → Response
      └→ Tool C → Result C ─┘
```

#### C. 條件調用
```
Query → Evaluate → [Condition A] → Tool A
                   [Condition B] → Tool B
                   [Else]        → Direct Response
```

---

## 7. 進階 RAG 模式

### 7.1 Self-RAG (自反思 RAG)

自適應檢索與自我批判的 RAG 框架。

```
┌─────────────────────────────────────────────────────────┐
│                     Self-RAG 流程                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 決定是否需要檢索（Retrieve Token）                  │
│     ↓                                                   │
│  2. 執行檢索（如需要）                                  │
│     ↓                                                   │
│  3. 評估檢索相關性（IsRel Token）                       │
│     ↓                                                   │
│  4. 生成回應                                            │
│     ↓                                                   │
│  5. 評估回應是否受支持（IsSup Token）                   │
│     ↓                                                   │
│  6. 評估整體有用性（IsUse Token）                       │
│     ↓                                                   │
│  7. 選擇最佳回應                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**特點：**
- 自適應檢索（可跳過、單次或多次）
- 自我批判機制
- 樹狀解碼選擇最佳路徑

### 7.2 Corrective RAG (CRAG)

引入修正機制的 RAG 框架。

```
檢索文件
    ↓
┌─────────────────────────────────────────────────────────┐
│                   檢索評估器                             │
│                                                         │
│   分類結果：                                            │
│   • Correct（正確）→ 直接使用                           │
│   • Incorrect（錯誤）→ 網頁搜索補充                     │
│   • Ambiguous（模糊）→ 精煉後使用                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
    ↓
知識精煉（分解-重組算法）
    ↓
生成回應
```

### 7.3 Adaptive-RAG

根據查詢複雜度動態調整檢索策略。

| 查詢類型 | 策略 |
|----------|------|
| 簡單查詢 | 直接 LLM 回答 |
| 中等查詢 | 單步檢索 |
| 複雜查詢 | 多步檢索 + 推理 |

### 7.4 Multi-Hop RAG

處理需要跨多個文件推理的複雜查詢。

```
複雜查詢
    ↓
分解為子問題
    ↓
┌──────────────┐
│ Sub-Q1 → Doc1 │
│ Sub-Q2 → Doc2 │
│ Sub-Q3 → Doc3 │
└──────────────┘
    ↓
整合推理
    ↓
最終答案
```

---

## 8. 檢索策略與重排序

### 8.1 混合搜索 (Hybrid Search)

結合稀疏與密集檢索的優勢。

```
┌─────────────────────────────────────────────────────────┐
│                     混合搜索架構                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  查詢 ─┬─→ BM25（稀疏）─→ 候選集 A ─┐                   │
│        │                            ├─→ 融合 → 重排序   │
│        └─→ 向量（密集）─→ 候選集 B ─┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 融合策略

| 方法 | 說明 |
|------|------|
| **RRF** | 倒數排名融合，k=60 常用 |
| **加權線性** | 手動調整各來源權重 |
| **分數標準化** | 標準化後加權平均 |

### 8.2 重排序模型

| 模型類型 | 優點 | 缺點 |
|----------|------|------|
| **Cross-Encoder** | 最高精度 | 計算成本高 |
| **ColBERT** | 延遲互動、效率佳 | 需要特殊索引 |
| **LLM Reranker** | 語義理解強 | API 成本 |

### 8.3 三階段檢索管道

```
Stage 1: BM25
  ↓ (200 候選)
Stage 2: Dense Retrieval
  ↓ (50 候選)  
Stage 3: Cross-Encoder Reranking
  ↓ (10 結果)
Final Results
```

---

## 9. 開源框架比較

### 9.1 主要框架

| 框架 | 編排風格 | 最佳場景 | 學習曲線 |
|------|----------|----------|----------|
| **LangGraph** | 聲明式圖 | 複雜分支、循環 | 中等 |
| **LangChain** | 命令式 + 組合 | 快速原型 | 低 |
| **LlamaIndex** | 命令式 | 索引與查詢 | 低 |
| **Haystack** | 模組化 | 企業搜索 | 中等 |
| **DSPy** | 簽名優先 | 自動優化 | 高 |

### 9.2 LangGraph 實作範例

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 定義工作流程
workflow = StateGraph(MessagesState)

# 添加節點
workflow.add_node("generate_query_or_respond", generate_node)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_node)
workflow.add_node("generate_answer", generate_answer_node)

# 定義邊
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END}
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {"relevant": "generate_answer", "irrelevant": "rewrite_question"}
)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)

# 編譯
graph = workflow.compile()
```

### 9.3 LlamaIndex Agentic RAG

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# 為每個文件建立查詢引擎
doc_engines = {}
for doc in documents:
    index = VectorStoreIndex.from_documents([doc])
    doc_engines[doc.id] = index.as_query_engine()

# 建立工具
tools = [
    QueryEngineTool.from_defaults(
        query_engine=engine,
        name=f"doc_{doc_id}",
        description=f"Query tool for document {doc_id}"
    )
    for doc_id, engine in doc_engines.items()
]

# 建立代理
agent = ReActAgent.from_tools(tools, llm=llm)
response = agent.chat("Your question here")
```

---

## 10. 最佳實踐與注意事項

### 10.1 設計原則

1. **簡單開始**：先實作基礎 RAG，再逐步添加代理能力
2. **組件解耦**：檢索與生成分開評估
3. **失敗處理**：設計回退機制和錯誤恢復
4. **成本控制**：監控 API 調用和 Token 使用

### 10.2 效能優化

| 策略 | 說明 |
|------|------|
| 語義快取 | 緩存相似查詢結果 |
| 分層處理 | 簡單查詢走快速路徑 |
| 早停機制 | 高信度時跳過額外檢索 |
| 輕量路由 | 用小模型做路由決策 |

### 10.3 監控指標

```
┌─────────────────────────────────────────────────────────┐
│                    關鍵監控指標                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  品質指標：                                             │
│  • Faithfulness（忠實度）                              │
│  • Answer Relevancy（答案相關性）                      │
│  • Context Precision/Recall                            │
│                                                         │
│  效能指標：                                             │
│  • 端到端延遲                                          │
│  • 檢索延遲 vs 生成延遲                                │
│  • Token 消耗                                          │
│                                                         │
│  可靠性指標：                                           │
│  • 幻覺率                                              │
│  • 工具調用成功率                                      │
│  • 回退觸發率                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Skill 建構建議

基於以上研究，建議 Agentic RAG Skill 應包含以下模組：

### 11.1 核心模組

1. **概念理論**：定義、架構類型、設計模式
2. **評估體系**：黃金數據集建立、RAGAS 指標、框架比較
3. **代理機制**：路由、規劃、工具使用、自我修正
4. **檢索策略**：混合搜索、重排序、多跳推理
5. **實作指南**：框架選擇、代碼範例、部署考量

### 11.2 實作模板

```
Agentic RAG Skill/
├── SKILL.md                 # 主要說明文件
├── references/
│   ├── concepts.md          # 核心概念
│   ├── evaluation.md        # 評估體系
│   ├── agent-patterns.md    # 代理模式
│   ├── retrieval.md         # 檢索策略
│   └── frameworks.md        # 框架比較
├── templates/
│   ├── langgraph/           # LangGraph 範例
│   ├── llamaindex/          # LlamaIndex 範例
│   └── evaluation/          # 評估腳本
└── examples/
    ├── simple-rag.py
    ├── corrective-rag.py
    └── multi-agent-rag.py
```

---

## 12. 參考資源

### 論文
- [Agentic RAG Survey (arXiv:2501.09136)](https://arxiv.org/abs/2501.09136)
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217)
- [Self-RAG: Learning to Retrieve, Generate and Critique](https://arxiv.org/abs/2310.11511)
- [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)

### 框架文檔
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [DeepEval Documentation](https://docs.confident-ai.com/)

### 教程
- [DeepLearning.AI: Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)
- [LangGraph Agentic RAG Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)

---

**研究完成日期：2026-01-13**

**研究成果：本報告涵蓋 Agentic RAG 的完整技術棧，包括基礎概念、評估體系、代理機制、工具使用、檢索策略、開源框架等，足以作為建立完整 Agentic RAG Skill 的基礎資料。**
