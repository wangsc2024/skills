# LlamaIndex 最新功能研究報告

> **資料來源**: Context7 API (developers.llamaindex.ai)
> **研究日期**: 2026-02-07
> **資料等級**: L1 (公開)
> **版本**: v0.14.6

---

## 核心新功能概覽

### 1. AgentWorkflow（多代理工作流）

LlamaIndex 最重要的新功能是 **AgentWorkflow** 系統，支援多代理協作：

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

# 定義多個專業代理
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="搜索網頁並記錄筆記",
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],  # 可移交給寫作代理
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="撰寫 Markdown 報告",
    tools=[write_report],
    can_handoff_to=["ReviewAgent"],
)

# 組合成工作流
workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent="ResearchAgent",
    initial_state={"notes": {}, "report": ""},
)

response = await workflow.run(user_msg="撰寫一份 AI 趨勢報告")
```

**關鍵特性**：
- **代理移交 (Handoff)**：代理間可自動轉移控制權
- **共享狀態**：所有代理共用 `state` 字典
- **記憶管理**：內建 `ChatMemoryBuffer` 支援對話歷史

---

### 2. Structured Output（結構化輸出）

支援 Pydantic 模型定義輸出格式：

```python
from pydantic import BaseModel, Field

class Weather(BaseModel):
    location: str = Field(description="地點")
    weather: str = Field(description="天氣狀況")

workflow = AgentWorkflow(
    agents=[weather_agent],
    output_cls=Weather,  # 強制結構化輸出
)

response = await workflow.run("東京天氣如何？")
print(response.get_pydantic_model(Weather))
# Weather(location='Tokyo', weather='Sunny, 15°C')
```

---

### 3. Streaming Events（串流事件）

即時串流代理的輸出：

```python
from llama_index.core.agent.workflow import AgentStream

handler = workflow.run(user_msg="舊金山天氣？")

async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)  # 即時輸出
```

**可用事件類型**：

| 事件 | 說明 |
|------|------|
| `AgentInput` | 代理接收到的輸入 |
| `AgentOutput` | 代理的最終輸出 |
| `AgentStream` | 串流中的文字片段 |
| `ToolCall` | 工具呼叫事件 |
| `ToolCallResult` | 工具執行結果 |

---

### 4. Context & State Management（上下文與狀態管理）

```python
from llama_index.core.workflow import Context

@step
async def my_step(self, ctx: Context, ev: InputEvent):
    # 讀取狀態
    state = await ctx.store.get("state")

    # 更新狀態
    state["notes"].append("新筆記")
    await ctx.store.set("state", state)

    # 寫入串流事件
    ctx.write_event_to_stream(StreamEvent(delta="處理中..."))
```

---

### 5. FunctionAgent vs ReActAgent

LlamaIndex 會自動選擇代理類型：

```python
# 自動選擇（推薦）
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[my_tool],
    llm=llm,  # 若 LLM 支援 function calling → FunctionAgent
              # 否則 → ReActAgent
)
```

| 代理類型 | 適用場景 |
|----------|----------|
| **FunctionAgent** | 支援 function calling 的 LLM（GPT-4、Claude） |
| **ReActAgent** | 不支援 function calling 的 LLM |

---

### 6. PlannerWorkflow（規劃器工作流）

進階功能：讓 LLM 自動規劃多步驟任務：

```python
class PlannerWorkflow(Workflow):
    @step
    async def plan(self, ctx: Context, ev: InputEvent):
        # LLM 生成執行計畫
        response = await self.llm.astream_chat(messages)

        # 解析 XML 格式的計畫
        plan = parse_xml_plan(response)
        return ExecuteEvent(plan=plan)

    @step
    async def execute(self, ctx: Context, ev: ExecuteEvent):
        for step in ev.plan.steps:
            agent = self.agents[step.agent_name]
            await call_agent(ctx, agent, step.input)
```

---

### 7. Memory Integration（記憶體整合）

```python
from llama_index.core.memory import ChatSummaryMemoryBuffer

# 使用摘要記憶體（長對話自動摘要）
memory = ChatSummaryMemoryBuffer.from_defaults(
    llm=llm,
    context_window=4096,
)

response = await agent.run("Hello!", memory=memory)
```

---

## 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentWorkflow                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ ResearchAgent│───▶│ WriteAgent  │───▶│ ReviewAgent │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Shared State                        │   │
│  │  { "notes": [...], "report": "...", "review": "..." } │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     Memory                            │   │
│  │              ChatMemoryBuffer                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速入門範例

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")

# 定義工具
def search_web(query: str) -> str:
    """搜索網頁"""
    return f"搜索結果：{query}"

def write_report(content: str) -> str:
    """撰寫報告"""
    return f"報告已完成：{content}"

# 快速建立工作流
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[search_web, write_report],
    llm=llm,
    system_prompt="你是一個研究助手",
)

# 執行
import asyncio

async def main():
    response = await workflow.run(user_msg="研究 AI 趨勢並撰寫報告")
    print(response)

asyncio.run(main())
```

---

## 相關連結

- [LlamaIndex 官方文檔](https://developers.llamaindex.ai/python/framework)
- [AgentWorkflow API](https://developers.llamaindex.ai/python/framework/-api-reference/agent)

---

## 標籤

`#LlamaIndex` `#AgentWorkflow` `#MultiAgent` `#RAG` `#Python` `#AI框架`
