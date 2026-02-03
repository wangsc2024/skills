# Groq - Tool Use

**Pages:** 10

---

## Define comprehensive tool set

**URL:** llms-txt#define-comprehensive-tool-set

**Contents:**
- Prompt Caching

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a specific timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone identifier, e.g. 'America/New_York' or 'UTC'"
                    }
                },
                "required": ["timezone"]
            }
        }
    }
]

def use_tools_with_caching():
    # First request - creates cache for all tool definitions
    first_request = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with access to various tools. Use the appropriate tools to answer user questions accurately."
            },
            {
                "role": "user",
                "content": "What's the weather like in New York City?"
            }
        ],
        model="moonshotai/kimi-k2-instruct-0905",
        tools=tools
    )

print("First request response:", first_request.choices[0].message)
    print("Usage:", first_request.usage)

# Check if the model wants to use tools
    if first_request.choices[0].message.tool_calls:
        print("Tool calls requested:", first_request.choices[0].message.tool_calls)

# Second request - tool definitions will be cached
    second_request = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with access to various tools. Use the appropriate tools to answer user questions accurately."
            },
            {
                "role": "user",
                "content": "Can you calculate the square root of 144 and tell me what time it is in Tokyo?"
            }
        ],
        model="moonshotai/kimi-k2-instruct-0905",
        tools=tools
    )

print("Second request response:", second_request.choices[0].message)
    print("Usage:", second_request.usage)

if second_request.choices[0].message.tool_calls:
        print("Tool calls requested:", second_request.choices[0].message.tool_calls)

# Third request - same tool definitions cached
    third_request = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with access to various tools. Use the appropriate tools to answer user questions accurately."
            },
            {
                "role": "user",
                "content": "Search for recent news about artificial intelligence developments."
            }
        ],
        model="moonshotai/kimi-k2-instruct-0905",
        tools=tools
    )

print("Third request response:", third_request.choices[0].message)
    print("Usage:", third_request.usage)

if third_request.choices[0].message.tool_calls:
        print("Tool calls requested:", third_request.choices[0].message.tool_calls)

if __name__ == "__main__":
    use_tools_with_caching()

URL: https://console.groq.com/docs/prompt-caching

---

## Define system messages and tools

**URL:** llms-txt#define-system-messages-and-tools

messages = [
    {"role": "system", "content": "You are a helpful weather assistant."},
    {"role": "user", "content": "What's the weather and temperature like in New York and London? Respond with one sentence for each city. Use tools to get the information."},
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get the temperature for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_condition",
            "description": "Get the weather condition for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

---

## Define weather tools

**URL:** llms-txt#define-weather-tools

def get_temperature(location: str):
    # This is a mock tool/function. In a real scenario, you would call a weather API.
    temperatures = {"New York": "22°C", "London": "18°C", "Tokyo": "26°C", "Sydney": "20°C"}
    return temperatures.get(location, "Temperature data not available")

def get_weather_condition(location: str):
    # This is a mock tool/function. In a real scenario, you would call a weather API.
    conditions = {"New York": "Sunny", "London": "Rainy", "Tokyo": "Cloudy", "Sydney": "Clear"}
    return conditions.get(location, "Weather condition data not available")

---

## Get Composio tools (GitHub in this example)

**URL:** llms-txt#get-composio-tools-(github-in-this-example)

composio_toolset = ComposioToolSet()
tools = composio_toolset.get_tools(apps=[App.GITHUB])

---

## Get the live view url to browse the browser in action (it's interactive!).

**URL:** llms-txt#get-the-live-view-url-to-browse-the-browser-in-action-(it's-interactive!).

**Contents:**
- Next Steps
- 🎨 Gradio + Groq: Easily Build Web Interfaces
- 🎨 Gradio + Groq: Easily Build Web Interfaces
  - Quick Start (2 minutes to hello world)

live_view_url = configured_session.data.live_view_url

print('session_id:', session_id, '\nlive_view_url:', live_view_url)
bash
pip install groq-gradio
bash
export GROQ_API_KEY="your-groq-api-key"
python
import gradio as gr
import groq_gradio
import os

**Examples:**

Example 1 (unknown):
```unknown
## Next Steps

- Explore the [API Reference](https://docs.anchorbrowser.io/api-reference?utm_source=groq) for detailed documentation
- Learn about [Authentication and Identity management](https://docs.anchorbrowser.io/api-reference/authentication?utm_source=groq)
- Check out [Advanced Proxy Configuration](https://docs.anchorbrowser.io/api-reference/proxies?utm_source=groq) for location-specific browsing
- Use more [Agentic tools](https://docs.anchorbrowser.io/agentic-browser-control?utm_source=groq)

---

## 🎨 Gradio + Groq: Easily Build Web Interfaces

URL: https://console.groq.com/docs/gradio

## 🎨 Gradio + Groq: Easily Build Web Interfaces

[Gradio](https://www.gradio.app/) is a powerful library for creating web interfaces for your applications that enables you to quickly build 
interactive demos for your fast Groq apps with features such as:

- **Interface Builder:** Create polished UIs with just a few lines of code, supporting text, images, audio, and more
- **Interactive Demos:** Build demos that showcase your LLM applications with multiple input/output components
- **Shareable Apps:** Deploy and share your Groq-powered applications with a single click

### Quick Start (2 minutes to hello world)

#### 1. Install the packages:
```

Example 2 (unknown):
```unknown
#### 2. Set up your API key:
```

Example 3 (unknown):
```unknown
#### 3. Create your first Gradio chat interface:
The following code creates a simple chat interface with `llama-3.3-70b-versatile` that includes a clean UI.
```

---

## imports calculate function from step 1

**URL:** llms-txt#imports-calculate-function-from-step-1

def run_conversation(user_prompt):
    # Initialize the conversation with system and user messages
    messages=[
        {
            "role": "system",
            "content": "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    # Define the available tools (i.e. functions) for our model to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    ]
    # Make the initial API call to Groq
    response = client.chat.completions.create(
        model=MODEL, # LLM to use
        messages=messages, # Conversation history
        stream=False,
        tools=tools, # Available tools (i.e. functions) for our LLM to use
        tool_choice="auto", # Let our LLM decide when to use tools
        max_completion_tokens=4096 # Maximum number of tokens to allow in our response
    )
    # Extract the response and any tool calls responses
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Define the available tools that can be called by the LLM
        available_functions = {
            "calculate": calculate,
        }
        # Add the LLM's response to the conversation
        messages.append(response_message)

# Process each tool calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # Call the tool and get the response
            function_response = function_to_call(
                expression=function_args.get("expression")
            )
            # Add the tool response to the conversation
            messages.append(
                {
                    "tool_calls_id": tool_call.id, 
                    "role": "tool", # Indicates this message is from tool use
                    "name": function_name,
                    "content": function_response,
                }
            )
        # Make a second API call with the updated conversation
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        # Return the final response
        return second_response.choices[0].message.content

---

## Initialize the Anchor Browser Client

**URL:** llms-txt#initialize-the-anchor-browser-client

client = Anchorbrowser(api_key=os.getenv("ANCHOR_API_KEY"))

---

## Print executed tools

**URL:** llms-txt#print-executed-tools

**Contents:**
- Browser Automation

if message.executed_tools:
    print(message.executed_tools[0])
```

## Browser Automation

URL: https://console.groq.com/docs/browser-automation

---

## Search results from the tool calls

**URL:** llms-txt#search-results-from-the-tool-calls

**Contents:**
- Web Search

if response.choices[0].message.executed_tools:
    print(response.choices[0].message.executed_tools[0].search_results)

URL: https://console.groq.com/docs/web-search

---

## View all available tools

**URL:** llms-txt#view-all-available-tools

composio apps
python
from langchain.agents import AgentType, initialize_agent
from langchain_groq import ChatGroq
from composio_langchain import ComposioToolSet, App

**Examples:**

Example 1 (unknown):
```unknown
#### 4. Create your first Composio-enabled Groq agent:

Running this code will create an agent that can interact with GitHub through natural language in mere seconds! Your agent will be able to:
- Perform GitHub operations like starring repos and creating issues for you
- Securely manage your OAuth flows and API keys
- Process natural language to convert your requests into specific tool actions 
- Provide feedback to let you know about the success or failure of operations
```

---
