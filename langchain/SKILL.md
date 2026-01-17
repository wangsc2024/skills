---
name: langchain
description: |
  LangChain is an open source framework for building AI agents with pre-built agent architecture and integrations for any model or tool. Use this skill when working with LLM applications, chains, agents, RAG pipelines, memory systems, tool integrations, and AI orchestration in Python.
  Use when: building LLM applications, creating AI agents, implementing RAG systems, managing conversation memory, tool calling, or when user mentions LangChain, LCEL, agent, chain, 代理, 鏈, RAG, prompt, LangGraph.
  Triggers: "LangChain", "LCEL", "agent", "chain", "代理", "鏈", "RAG", "prompt template", "LangGraph", "memory", "tool calling", "AI orchestration"
---

# LangChain Skill

LangChain 是一個開源框架，用於建構由大型語言模型驅動的應用程式。提供預建的 Agent 架構、工具整合、RAG 管線和記憶系統。

## When to Use This Skill

This skill should be triggered when:
- Building AI agents with tool-calling capabilities
- Implementing RAG (Retrieval-Augmented Generation) pipelines
- Creating conversational AI with memory systems
- Orchestrating multi-agent workflows
- Integrating LLMs with external tools and APIs
- Building production-grade LLM applications

## Quick Reference

### Installation

```bash
# Using pip
pip install langchain langchain-text-splitters langchain-community bs4

# Using uv
uv add langchain langchain-text-splitters langchain-community bs4
```

### Basic Agent Creation

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}
)
```

## Agents

### Create Agent with Tools

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    system_prompt="You are a helpful assistant."
)
```

### Streaming Agent Responses

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news"}]},
    stream_mode="values",
):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

### Agent with Memory (Checkpointer)

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)

# Use thread_id for conversation continuity
config = {"configurable": {"thread_id": "user-123"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    config=config
)
```

## RAG (Retrieval-Augmented Generation)

### Complete RAG Pipeline

```python
import bs4
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool

# 1. Load documents
loader = WebBaseLoader(
    web_paths=("https://example.com/article",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="post-content"))
)
docs = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

# 3. Create vector store and index
vector_store = InMemoryVectorStore(embeddings)
document_ids = vector_store.add_documents(documents=all_splits)

# 4. Create retrieval tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# 5. Create RAG agent
agent = create_agent(
    model="gpt-4o",
    tools=[retrieve_context],
    system_prompt="Use retrieve_context to answer questions from the docs."
)
```

### Vector Stores

```python
# In-Memory (development)
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# Chroma (local persistence)
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# FAISS (high performance)
import faiss
from langchain_community.vectorstores import FAISS
vector_store = FAISS(
    embedding_function=embeddings,
    index=faiss.IndexFlatL2(embedding_dim),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# PostgreSQL with pgvector
from langchain_postgres import PGVector
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://...",
)
```

### Embeddings

```python
# OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Azure OpenAI
from langchain_openai import AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Voyage AI
from langchain_voyageai import VoyageAIEmbeddings
embeddings = VoyageAIEmbeddings(model="voyage-3")
```

## Middleware

### Human-in-the-Loop

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="gpt-4o",
    tools=[send_email, delete_database],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,      # Require approval
                "delete_database": True,  # Require approval
                "search": False,          # Auto-approve
            }
        ),
    ],
    checkpointer=InMemorySaver(),
)

# Resume after approval
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config={"configurable": {"thread_id": "123"}}
)
```

### Summarization Middleware

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),  # Summarize when > 4000 tokens
            keep=("messages", 20),      # Keep last 20 messages
        ),
    ],
)
```

### Tool Retry Middleware

```python
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            retry_on=(ConnectionError, TimeoutError),
        ),
    ],
)
```

### Model Fallback Middleware

```python
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4o-mini",              # First fallback
            "claude-3-5-sonnet",        # Second fallback
        ),
    ],
)
```

## Multi-Agent Systems

### Subagents as Tools

```python
from langchain.tools import tool
from langchain.agents import create_agent

# Create specialized subagent
research_agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    system_prompt="You are a research specialist."
)

# Wrap as tool for main agent
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# Main coordinator agent
main_agent = create_agent(
    model="gpt-4o",
    tools=[call_research_agent],
    system_prompt="You coordinate research tasks."
)
```

### Agent Handoffs

```python
from langgraph.graph import StateGraph, START, END

def route_to_agent(state):
    """Route to appropriate agent based on state."""
    return state.get("active_agent") or "sales_agent"

builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

builder.add_conditional_edges(START, route_to_agent, ["sales_agent", "support_agent"])
builder.add_conditional_edges("sales_agent", route_after_agent, ["support_agent", END])

graph = builder.compile()
```

## Structured Output

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract: John, john@example.com, 555-1234"}]
})

print(result["structured_response"])
# ContactInfo(name='John', email='john@example.com', phone='555-1234')
```

## SQL Agent

```python
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Connect to database
db = SQLDatabase.from_uri("sqlite:///my_database.db")

# Create SQL tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# Create SQL agent
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQL query.
Always limit to 5 results unless specified otherwise.
DO NOT make DML statements (INSERT, UPDATE, DELETE, DROP).
"""

agent = create_agent(model, tools, system_prompt=system_prompt)

# Query the database
for step in agent.stream(
    {"messages": [{"role": "user", "content": "What are the top 5 customers?"}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

## Observability with LangSmith

```bash
# Set environment variables
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_api_key
```

```python
# Traces are automatically captured
agent = create_agent(
    model="gpt-4o",
    tools=[send_email, search_web],
    system_prompt="You are a helpful assistant."
)

# All steps are traced automatically
response = agent.invoke({
    "messages": [{"role": "user", "content": "Search for AI news"}]
})
```

## Best Practices

### 1. Use Specific Tool Descriptions
```python
@tool
def search_documents(query: str) -> str:
    """Search internal documents for information.

    Use this when the user asks about company policies,
    procedures, or internal knowledge base content.
    """
    return search_index(query)
```

### 2. Handle Errors Gracefully
```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}. Please try again.",
            tool_call_id=request.tool_call["id"]
        )
```

### 3. Use Thread IDs for Conversations
```python
config = {"configurable": {"thread_id": f"user-{user_id}"}}
response = agent.invoke(messages, config=config)
```

## Reference Files

This skill includes comprehensive documentation in `references/`:
- **agents.md** - Agent creation and configuration
- **getting_started.md** - Quick start guide
- **models.md** - Model integrations

## Resources

- **Official Docs**: https://python.langchain.com
- **LangGraph**: https://langchain-ai.github.io/langgraph
- **LangSmith**: https://smith.langchain.com
- **GitHub**: https://github.com/langchain-ai/langchain

## Notes

- LangChain v1.0+ uses `create_agent` as the primary agent builder
- Use `InMemorySaver` for development, persistent checkpointers for production
- Middleware can be stacked for complex agent behaviors
- LangGraph provides lower-level control for multi-agent orchestration
