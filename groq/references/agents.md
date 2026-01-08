# Groq - Agents

**Pages:** 8

---

## Configure the UserProxyAgent with code execution

**URL:** llms-txt#configure-the-userproxyagent-with-code-execution

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"executor": code_executor}
)
python
from typing import Annotated

def get_current_weather(location, unit="fahrenheit"):
    """Get the weather for some location"""
    weather_data = {
        "berlin": {"temperature": "13"},
        "istanbul": {"temperature": "40"},
        "san francisco": {"temperature": "55"}
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        return json.dumps({
            "location": location.title(),
            "temperature": weather_data[location_lower]["temperature"],
            "unit": unit
        })
    return json.dumps({"location": location, "temperature": "unknown"})

**Examples:**

Example 1 (unknown):
```unknown
#### Tool Integration
You can add tools for your agents to use by creating a function and registering it with the assistant. Here's an example of a weather forecast tool:
```

---

## Create agent

**URL:** llms-txt#create-agent

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

---

## Create a user proxy agent (no code execution in this example)

**URL:** llms-txt#create-a-user-proxy-agent-(no-code-execution-in-this-example)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False
)

---

## Create a user proxy agent that only handles code execution

**URL:** llms-txt#create-a-user-proxy-agent-that-only-handles-code-execution

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"executor": code_executor}
)

---

## Create your CrewAI agents with role, main goal/objective, and backstory/personality

**URL:** llms-txt#create-your-crewai-agents-with-role,-main-goal/objective,-and-backstory/personality

summarizer = Agent(
    role='Documentation Summarizer', # Agent's job title/function
    goal='Create concise summaries of technical documentation', # Agent's main objective
    backstory='Technical writer who excels at simplifying complex concepts', # Agent's background/expertise
    llm=llm, # LLM that powers your agent
    verbose=True # Show agent's thought process as it completes its task
)

translator = Agent(
    role='Technical Translator',
    goal='Translate technical documentation to other languages',
    backstory='Technical translator specializing in software documentation',
    llm=llm,
    verbose=True
)

---

## Define your agents' tasks

**URL:** llms-txt#define-your-agents'-tasks

summary_task = Task(
    description='Summarize this React hook documentation:\n\nuseFetch(url) is a custom hook for making HTTP requests. It returns { data, loading, error } and automatically handles loading states.',
    expected_output="A clear, concise summary of the hook's functionality",
    agent=summarizer # Agent assigned to task
)

translation_task = Task(
    description='Translate the summary to Turkish',
    expected_output="Turkish translation of the hook documentation",
    agent=translator,
    dependencies=[summary_task] # Must run after the summary task
)

---

## Start a conversation between the agents

**URL:** llms-txt#start-a-conversation-between-the-agents

**Contents:**
  - Advanced Features

user_proxy.initiate_chat(
    assistant,
    message="What are the key benefits of using Groq for AI apps?"
)
python
from pathlib import Path
from autogen.coding import LocalCommandLineCodeExecutor

**Examples:**

Example 1 (unknown):
```unknown
### Advanced Features

#### Code Generation and Execution
You can enable secure code execution by configuring the `UserProxyAgent` that allows your agents to write and execute Python code in a controlled environment:
```

---

## Use Cases

**URL:** llms-txt#use-cases

**Contents:**
- Real-time Fact Checker and News Agent
  - Solution with Compound
  - Why It's Great
  - Code Example
  - Why It's Great
- Chart Generation
  - Solution with Compound
  - Why It's Great
  - Usage and Results
- Natural Language Calculator and Code Extractor

Groq's compound systems excel at a wide range of use cases, particularly when real-time information is required.

## Real-time Fact Checker and News Agent

Your application needs to answer questions or provide information that requires up-to-the-minute knowledge, such as:
- Latest news
- Current stock prices
- Recent events
- Weather updates

Building and maintaining your own web scraping or search API integration is complex and time-consuming.

### Solution with Compound
Simply send the user's query to `groq/compound`. If the query requires current information beyond its training data, it will automatically trigger its built-in web search tool to fetch relevant, live data before formulating the answer.

### Why It's Great
- Get access to real-time information instantly without writing any extra code for search integration
- Leverage Groq's speed for a real-time, responsive experience

### Why It's Great
- Provides a unified interface for getting code help
- Potentially draws on live web data for new errors
- Executes code directly for validation
- Speeds up the debugging process

**Note**: `groq/compound-mini` uses one tool per turn, so it might search OR execute, not both simultaneously in one response.

Need to quickly create data visualizations from natural language descriptions? Compound's code execution capabilities can help generate charts without writing visualization code directly.

### Solution with Compound
Describe the chart you want in natural language, and Compound will generate and execute the appropriate Python visualization code. The model automatically parses your request, generates the visualization code using libraries like matplotlib or seaborn, and returns the chart.

### Why It's Great
- Generate charts from simple natural language descriptions
- Supports common chart types (scatter, line, bar, etc.)
- Handles all visualization code generation and execution
- Customize data points, labels, colors, and layouts as needed

### Usage and Results

## Natural Language Calculator and Code Extractor

You want users to perform calculations, run simple data manipulations, or execute small code snippets using natural language commands within your application, without building a dedicated parser or execution environment.

### Solution with Compound

Frame the user's request as a task involving computation or code. `groq/compound-mini` can recognize these requests and use its secure code execution tool to compute the result.

### Why It's Great
 - Effortlessly add computational capabilities
 - Users can ask things like:
   - "What's 15% of $540?"
   - "Calculate the standard deviation of [10, 12, 11, 15, 13]"
   - "Run this python code: print('Hello from Compound!')"

## Code Debugging Assistant

Developers often need quick help understanding error messages or testing small code fixes. Searching documentation or running snippets requires switching contexts.

### Solution with Compound
Users can paste an error message and ask for explanations or potential causes. Compound Mini might use web search to find recent discussions or documentation about that specific error. Alternatively, users can provide a code snippet and ask "What's wrong with this code?" or "Will this Python code run: ...?". It can use code execution to test simple, self-contained snippets.

### Why It's Great
- Provides a unified interface for getting code help
- Potentially draws on live web data for new errors
- Executes code directly for validation
- Speeds up the debugging process

**Note**: `groq/compound-mini` uses one tool per turn, so it might search OR execute, not both simultaneously in one response.

## Search Settings: Page (mdx)

URL: https://console.groq.com/docs/compound/search-settings

No content to display.

## Compound Beta: Page (mdx)

URL: https://console.groq.com/docs/compound/systems/compound-beta

No content to display.

URL: https://console.groq.com/docs/compound/systems

---
