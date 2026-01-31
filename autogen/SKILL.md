---
name: autogen
description: |
  Microsoft AutoGen 多代理對話框架。支援建立可協作的 AI 代理系統，實現複雜任務的自動化分解與執行。
  Use when: 建立多代理系統、設計代理協作流程、實作對話式 AI、自動化複雜任務，or when user mentions AutoGen, multi-agent, 多代理, 協作代理.
  Triggers: "AutoGen", "autogen", "multi-agent", "多代理", "agent chat", "協作代理", "ConversableAgent", "AssistantAgent", "UserProxyAgent", "GroupChat"
version: 1.0.0
---

# AutoGen Skill

Microsoft AutoGen 是用於建立多代理對話系統的框架，讓多個 AI 代理能夠協作完成複雜任務。

## When to Use This Skill

- 建立多代理協作系統
- 設計代理間對話流程
- 實作程式碼執行代理
- 建立 GroupChat 多代理討論
- 整合外部工具與 API
- 自動化複雜工作流程

## Quick Reference

### 安裝

```bash
pip install pyautogen
```

### 基本雙代理對話

```python
from autogen import ConversableAgent

# 建立助手代理
assistant = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    system_message="你是一個有幫助的 AI 助手。"
)

# 建立使用者代理
user_proxy = ConversableAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# 啟動對話
user_proxy.initiate_chat(
    assistant,
    message="寫一個計算費氏數列的 Python 函數"
)
```

### 程式碼執行代理

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    name="coder",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={
        "executor": "local",
        "work_dir": "workspace"
    }
)

user_proxy.initiate_chat(
    assistant,
    message="建立一個爬取網頁標題的程式"
)
```

### GroupChat 多代理協作

```python
from autogen import GroupChat, GroupChatManager

# 建立多個代理
planner = AssistantAgent(name="planner", ...)
coder = AssistantAgent(name="coder", ...)
reviewer = AssistantAgent(name="reviewer", ...)

# 建立群組對話
groupchat = GroupChat(
    agents=[planner, coder, reviewer],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat)

# 啟動群組討論
planner.initiate_chat(
    manager,
    message="設計並實作一個 TODO 應用"
)
```

### 工具整合

```python
from autogen import register_function

@register_function()
def search_web(query: str) -> str:
    """搜尋網頁"""
    # 實作搜尋邏輯
    return results

assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "model": "gpt-4",
        "functions": [search_web]
    }
)
```

### LLM 配置

```python
llm_config = {
    "model": "gpt-4",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "timeout": 120,
    "cache_seed": 42  # 啟用快取
}
```

## Reference Files

詳細文檔請參考 `references/` 目錄：

| 檔案 | 內容 |
|------|------|
| `getting_started.md` | 快速入門指南 |
| `agents.md` | 代理類型與配置 |
| `agentchat.md` | 代理對話機制 |
| `tools.md` | 工具整合指南 |
| `models.md` | 模型配置與支援 |
| `memory.md` | 記憶與狀態管理 |
| `core.md` | 核心 API 參考 |

## Resources

- 官方文檔: https://microsoft.github.io/autogen/
- GitHub: https://github.com/microsoft/autogen
