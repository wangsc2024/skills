---
name: autogen
description: AutoGen is Microsoft's open-source framework for building AI agents and multi-agent applications. Use this skill when working with conversational agents, multi-agent orchestration, AgentChat, group chats, human-in-the-loop workflows, and AI agent collaboration in Python.
---

# AutoGen Skill

Microsoft AutoGen 是建構 AI Agent 和多代理應用程式的開源框架。

## When to Use This Skill

- 建構 AI Agent 或多代理系統
- 實作 AgentChat 對話代理
- 設計 Group Chat 多代理協作
- 需要 Human-in-the-loop 工作流程
- 使用 AutoGen Studio 無程式碼原型

## 核心架構

```
AutoGen 架構層次
├── AutoGen Studio     # 無程式碼 UI（原型設計）
├── AgentChat          # 高階 API（推薦入門）
│   ├── Agents         # 預設行為代理
│   ├── Teams          # 多代理設計模式
│   └── Tools          # 工具整合
├── Core               # 底層事件驅動框架
│   ├── Actor Model    # 非同步訊息傳遞
│   └── Runtime        # 分散式執行環境
└── Extensions         # 社群擴充套件
```

## 安裝

```bash
# AgentChat（推薦入門）
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# AutoGen Studio（無程式碼 UI）
pip install -U autogenstudio
autogenstudio ui --port 8080

# Core（進階用途）
pip install -U "autogen-core"
```

## Quick Start

### 基本代理

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o")
    agent = AssistantAgent("assistant", model)
    result = await agent.run(task="Say 'Hello World!'")
    print(result)

asyncio.run(main())
```

### 雙代理對話

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o")

    # 定義代理
    coder = AssistantAgent(
        "coder",
        model,
        system_message="你是一位 Python 程式設計師，撰寫簡潔的程式碼。"
    )
    reviewer = AssistantAgent(
        "reviewer",
        model,
        system_message="你是程式碼審查員，提供建設性反饋。完成時說 APPROVE。"
    )

    # 建立團隊
    termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat([coder, reviewer], termination_condition=termination)

    # 執行任務
    result = await team.run(task="寫一個計算費氏數列的函數")
    print(result)

asyncio.run(main())
```

### 帶工具的代理

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 定義工具
def get_weather(city: str) -> str:
    """取得指定城市的天氣"""
    return f"{city} 今天晴朗，氣溫 25°C"

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o")
    agent = AssistantAgent(
        "weather_agent",
        model,
        tools=[get_weather],
        system_message="你是天氣助手，使用工具回答天氣問題。"
    )
    result = await agent.run(task="台北今天天氣如何？")
    print(result)

asyncio.run(main())
```

## 核心元件

### Agents 代理類型

| 類型 | 用途 | 說明 |
|------|------|------|
| `AssistantAgent` | 通用助手 | 最常用，支援工具呼叫 |
| `UserProxyAgent` | 人機互動 | 代表使用者，可執行程式碼 |
| `CodeExecutorAgent` | 程式執行 | 在沙盒中執行程式碼 |
| `MultimodalWebSurfer` | 網頁瀏覽 | 可瀏覽網頁的代理 |

### Teams 團隊模式

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| `RoundRobinGroupChat` | 輪流發言 | 簡單協作流程 |
| `SelectorGroupChat` | 智慧選擇 | 動態決定發言者 |
| `Swarm` | 群體智慧 | 工具導向的代理切換 |
| `MagenticOneGroupChat` | Magentic-One | 複雜任務分解 |
| `GraphFlow` | 圖形工作流 | 有向圖定義流程 |

### Termination 終止條件

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,      # 最大訊息數
    TextMentionTermination,     # 文字觸發（如 "DONE"）
    TokenUsageTermination,      # Token 上限
    TimeoutTermination,         # 時間限制
    HandoffTermination,         # 交接終止
)

# 組合條件
from autogen_agentchat.conditions import ExternalTermination
termination = MaxMessageTermination(10) | TextMentionTermination("APPROVE")
```

## 進階模式

### 記憶系統

```python
from autogen_agentchat.memory import ListMemory

memory = ListMemory()
agent = AssistantAgent(
    "assistant",
    model,
    memory=memory  # 跨對話記憶
)
```

### 程式碼執行

```python
from autogen_ext.code_executors import DockerCommandLineCodeExecutor

executor = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",
    timeout=60,
    work_dir="./workspace"
)

agent = CodeExecutorAgent("executor", code_executor=executor)
```

### MCP 整合

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# 連接 MCP 伺服器
mcp = McpWorkbench(
    server=StdioServerParams(command="uvx", args=["mcp-server-fetch"])
)
tools = await mcp.list_tools()

agent = AssistantAgent("mcp_agent", model, tools=tools)
```

## 最佳實踐

### 1. 選擇正確的 API 層級

- **初學者**：從 AgentChat 開始
- **原型設計**：使用 AutoGen Studio
- **生產環境**：使用 Core 獲得更多控制

### 2. 設計有效的系統提示

```python
system_message = """
你是一位專業的資料分析師。
- 任務完成後說 "DONE"
- 遇到錯誤時說明原因
- 使用提供的工具完成任務
"""
```

### 3. 合理設定終止條件

```python
# 避免無限迴圈
termination = (
    MaxMessageTermination(20) |      # 最多 20 則訊息
    TextMentionTermination("DONE") | # 或出現 DONE
    TimeoutTermination(300)          # 或 5 分鐘超時
)
```

### 4. 安全考量

- 使用 Docker 執行不信任的程式碼
- 限制代理可存取的工具和資源
- 實作適當的認證和授權

## 常見問題

### Q: AgentChat vs Core 該選哪個？

- **AgentChat**：快速開發、預設行為、適合大多數場景
- **Core**：需要自訂事件處理、分散式系統、多語言支援

### Q: 如何處理長時間運行的任務？

使用串流模式：

```python
async for message in team.run_stream(task="複雜任務"):
    print(message)
```

### Q: 如何除錯代理行為？

```python
from autogen_agentchat.logging import ConsoleLogHandler
import logging

logging.basicConfig(level=logging.DEBUG)
handler = ConsoleLogHandler()
```

## Reference Files

詳細 API 參考請查閱 references/ 目錄：

| 檔案 | 內容 |
|------|------|
| `getting_started.md` | 入門指南和教學 |
| `agentchat.md` | AgentChat API 完整參考 |
| `agents.md` | 所有代理類型詳細說明 |
| `teams.md` | 團隊模式完整文件 |
| `tools.md` | 工具定義和整合 |
| `memory.md` | 記憶系統 API |
| `models.md` | 模型配置參考 |
| `core.md` | Core API 底層框架 |

## Resources

- [官方文件](https://microsoft.github.io/autogen/)
- [GitHub](https://github.com/microsoft/autogen)
- [AutoGen Studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/)
- [從 0.2 遷移指南](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html)
