---
name: groq
description: |
  Groq API documentation for ultra-fast LLM inference. Use for chat completions, speech-to-text (Whisper), text-to-speech, vision, tool use, and agentic AI with Groq Compound.
  Use when: need ultra-low latency inference, implementing speech-to-text, building fast AI applications, using LPU hardware, or when user mentions Groq, Whisper, ultra-fast, LPU, 超快推理, low latency, Compound.
  Triggers: "Groq", "Whisper", "ultra-fast", "LPU", "超快推理", "low latency", "Compound", "speech-to-text", "fast inference", "high TPS"
---

# Groq AI Skill

Comprehensive assistance with Groq API - the fastest LLM inference platform. Covers chat completions, speech-to-text, vision, tool use, code execution, web search, and agentic AI systems.

## When to Use This Skill

This skill should be triggered when:
- Building applications requiring ultra-fast LLM inference
- Using Groq's API for chat completions (Llama, GPT-OSS models)
- Implementing speech-to-text with Whisper models
- Working with vision/multimodal capabilities
- Using tool calling and function execution
- Building agentic AI with Groq Compound
- Implementing web search and browser automation
- Optimizing for low-latency AI applications

## Quick Reference

### Installation

```bash
# Python
pip install groq

# JavaScript/TypeScript
npm install groq-sdk
```

### Authentication

```python
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```

```javascript
import Groq from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
```

```bash
export GROQ_API_KEY=<your-api-key-here>
```

### Available Models

| Model ID | Description | Speed |
|----------|-------------|-------|
| `llama-3.3-70b-versatile` | Best for complex tasks | 280 tps |
| `llama-3.1-8b-instant` | Fast, cost-effective | 560 tps |
| `openai/gpt-oss-120b` | Flagship open model, 128K context | ~500 tps |
| `openai/gpt-oss-20b` | High-throughput, cost-sensitive | 1000 tps |
| `groq/compound` | Agentic AI with tools | ~450 tps |
| `groq/compound-mini` | Lighter agentic system | Fast |
| `whisper-large-v3` | Best accuracy STT | 189x realtime |
| `whisper-large-v3-turbo` | Fastest STT | 216x realtime |
| `llama-guard-4-12b` | Content moderation | 1200 tps |

## Common Patterns

### Chat Completion

```python
from groq import Groq

client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
```

```javascript
import Groq from "groq-sdk";

const groq = new Groq();

const chatCompletion = await groq.chat.completions.create({
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Explain quantum computing in simple terms" }
  ],
  model: "llama-3.3-70b-versatile",
});

console.log(chatCompletion.choices[0].message.content);
```

```bash
curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Response

```python
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a short poem"}],
    model="llama-3.3-70b-versatile",
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Speech-to-Text (Whisper)

```python
from groq import Groq

client = Groq()

with open("audio.mp3", "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )

print(transcription.text)
```

```javascript
import Groq from "groq-sdk";
import fs from "fs";

const groq = new Groq();

const transcription = await groq.audio.transcriptions.create({
  file: fs.createReadStream("audio.mp3"),
  model: "whisper-large-v3-turbo",
  response_format: "verbose_json",
});

console.log(transcription.text);
```

**Audio File Limits:**
- Max: 25 MB (free), 100 MB (dev tier)
- Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
- Minimum billed: 10 seconds

### Tool Use / Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Code Execution (GPT-OSS)

```javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
  messages: [
    { role: "user", content: "Calculate the square root of 12345." }
  ],
  model: "openai/gpt-oss-20b",
  tool_choice: "required",
  tools: [{ type: "code_interpreter" }],
});

// Final output
console.log(response.choices[0].message.content);

// Reasoning + internal tool calls
console.log(response.choices[0].message.reasoning);

// Code execution details
console.log(response.choices[0].message.executed_tools?.[0]);
```

### Web Search (Compound)

```python
response = client.chat.completions.create(
    model="groq/compound",
    messages=[{"role": "user", "content": "What are today's top tech news?"}],
    tools=[{"type": "web_search"}],
    tool_choice="auto"
)

print(response.choices[0].message.content)
```

### Browser Automation (Compound)

```python
# Enable both browser_automation AND web_search
response = client.chat.completions.create(
    model="groq/compound",
    messages=[{"role": "user", "content": "Research the latest AI developments"}],
    tools=[
        {"type": "web_search"},
        {"type": "browser_automation"}
    ],
    tool_choice="auto"
)

# Response includes:
# - content: Final synthesized response
# - reasoning: Internal decision-making process
# - executed_tools: Browser session details
```

### Structured Outputs (JSON Schema)

```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Extract structured data."},
        {"role": "user", "content": "John Doe, 30 years old, engineer at Google"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"},
                    "company": {"type": "string"}
                },
                "required": ["name", "age", "occupation"]
            }
        }
    }
)

import json
result = json.loads(response.choices[0].message.content)
```

### Vision (Image Understanding)

```python
response = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
```

## Reference Files

| File | Description |
|------|-------------|
| **getting_started.md** | Overview, quickstart, API keys |
| **models.md** | All models, capabilities, pricing |
| **text_generation.md** | Chat completions, streaming, structured outputs |
| **speech.md** | Whisper transcription and translation |
| **vision.md** | Image understanding |
| **reasoning.md** | Chain-of-thought, thinking models |
| **tool_use.md** | Function calling, web search, browser |
| **agents.md** | Agentic AI, Compound systems |
| **api_reference.md** | Complete API endpoints |
| **rate_limits.md** | Quotas and tiers |
| **production.md** | Latency optimization, best practices |

## Key Concepts

### Why Groq is Fast
- **LPU (Language Processing Unit)**: Custom hardware designed for LLM inference
- **Day-zero inference**: Near real-time responses even for 120B parameter models
- **128K token context**: Extended context for long documents/conversations

### Service Tiers
| Tier | Best For |
|------|----------|
| **Performance** | Production workloads, guaranteed uptime |
| **Flex** | Variable traffic, cost optimization |
| **Batch** | Bulk processing, lower priority |

### Rate Limits
- Free tier: Limited requests/minute
- Dev tier: Higher limits, 100MB audio files
- Production: Custom limits based on plan

### Pricing (Example)
- `llama-3.1-8b-instant`: $0.05/$0.08 per 1M tokens (input/output)
- `llama-3.3-70b-versatile`: $0.59/$0.79 per 1M tokens
- `openai/gpt-oss-120b`: $0.15/$0.75 per 1M tokens
- `whisper-large-v3`: $0.111/hour
- `whisper-large-v3-turbo`: $0.04/hour

## Best Practices

1. **Use streaming** for real-time applications
2. **Choose the right model**: 8B for speed, 70B/120B for quality
3. **Set temperature=0** for consistent outputs
4. **Use structured outputs** for reliable JSON parsing
5. **Implement retry logic** with exponential backoff
6. **Preprocess audio** to 16KHz mono for best STT results

## Resources

- **API Console**: https://console.groq.com/
- **Playground**: https://console.groq.com/playground
- **API Keys**: https://console.groq.com/keys
- **Cookbook**: https://github.com/groq/groq-api-cookbook
- **Community**: https://community.groq.com/
- **Pricing**: https://groq.com/pricing

## Working with This Skill

### For Beginners
Start with `getting_started.md` for quickstart, then `text_generation.md` for chat basics.

### For Specific Features
- Chat/streaming: `text_generation.md`
- Speech-to-text: `speech.md`
- Tool calling: `tool_use.md`
- Agentic AI: `agents.md`
- Image understanding: `vision.md`

### For Production
Review `production.md` for latency optimization and `rate_limits.md` for quota planning.

## OpenAI Compatibility

Groq API is OpenAI-compatible. You can use OpenAI SDKs with Groq:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Use as you would OpenAI
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Notes

- This skill was generated from official Groq documentation (llms.txt)
- Code examples include Python, JavaScript, and cURL variants
- API endpoints follow OpenAI-compatible format at `https://api.groq.com/openai/v1/`

## Updating

```bash
skill-seekers scrape --config configs/groq.json
```
