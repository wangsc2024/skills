---
name: mistral
description: |
  Mistral AI API and SDK documentation. Use for chat completions, vision, audio transcription, function calling, embeddings, fine-tuning, agents, and deploying Mistral models.
  Use when: working with Mistral AI models, implementing function calling, generating embeddings, vision tasks with Pixtral, or when user mentions Mistral, Pixtral, embeddings, function calling, 函數呼叫, agent API.
  Triggers: "Mistral", "Pixtral", "embeddings", "function calling", "函數呼叫", "agent API", "Mistral Large", "fine-tuning", "vision model"
version: 1.0.0
---

# Mistral AI Skill

Comprehensive assistance with Mistral AI API development, covering chat completions, vision, audio, function calling, embeddings, fine-tuning, agents, and deployment.

## When to Use This Skill

This skill should be triggered when:
- Building applications with Mistral AI models (mistral-large, mistral-small, mistral-embed, etc.)
- Implementing chat completions or streaming responses
- Working with Mistral vision capabilities (image understanding)
- Using audio transcription features
- Implementing function calling / tool use
- Creating text embeddings for RAG or semantic search
- Fine-tuning Mistral models
- Building multi-agent systems with handoffs
- Deploying Mistral on cloud or self-hosted environments

## Quick Reference

### Installation

```bash
# Python
pip install mistralai

# TypeScript/JavaScript
npm install @mistralai/mistralai
```

### Authentication

```python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
```

```typescript
import Mistral from '@mistralai/mistralai';

const apiKey = process.env.MISTRAL_API_KEY;
const client = new Mistral({apiKey: apiKey});
```

### Available Models

| Model | Use Case |
|-------|----------|
| `mistral-large-latest` | Complex reasoning, coding, multi-lingual |
| `mistral-medium-latest` | Balanced performance |
| `mistral-small-latest` | Fast, cost-effective |
| `mistral-embed` | Text embeddings |
| `pixtral-large-latest` | Vision + text (multimodal) |

## Common Patterns

### Chat Completion

```python
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

chat_response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "user", "content": "What is the best French cheese?"}
    ]
)
print(chat_response.choices[0].message.content)
```

```typescript
const chatResponse = await client.chat.complete({
  model: 'mistral-large-latest',
  messages: [{role: 'user', content: 'What is the best French cheese?'}],
});
console.log(chatResponse.choices[0].message.content);
```

```bash
curl --location "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Response

```python
stream_response = client.chat.stream(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Tell me a story"}]
)

for chunk in stream_response:
    if chunk.data.choices[0].delta.content:
        print(chunk.data.choices[0].delta.content, end="")
```

### Embeddings

```python
embeddings_response = client.embeddings.create(
    model="mistral-embed",
    inputs=["Embed this sentence.", "As well as this one."]
)
print(embeddings_response)
```

```typescript
const embeddingsResponse = await client.embeddings.create({
  model: 'mistral-embed',
  inputs: ["Embed this sentence.", "As well as this one."],
});
console.log(embeddingsResponse);
```

### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
```

### Vision (Multimodal)

```python
from mistralai import ImageURLChunk, TextChunk

response = client.chat.complete(
    model="pixtral-large-latest",
    messages=[
        {
            "role": "user",
            "content": [
                ImageURLChunk(image_url="https://example.com/image.jpg"),
                TextChunk(text="Describe this image")
            ]
        }
    ]
)
```

### Agents with Handoffs

```python
# Create agents
finance_agent = client.beta.agents.create(
    name="finance-agent",
    model="mistral-medium-latest",
    instructions="You are a financial advisor."
)

# Allow handoffs between agents
finance_agent = client.beta.agents.update(
    agent_id=finance_agent.id,
    handoffs=[other_agent.id]
)

# Start conversation
response = client.beta.conversations.start(
    agent_id=finance_agent.id,
    inputs="Calculate compound interest for $1000 at 5% for 10 years"
)
```

### Document OCR

```python
from mistralai import DocumentURLChunk

response = client.chat.complete(
    model="mistral-ocr-latest",
    messages=[
        {
            "role": "user",
            "content": [
                DocumentURLChunk(document_url="https://example.com/doc.pdf"),
                TextChunk(text="Extract the key information from this document")
            ]
        }
    ]
)
```

### Fine-Tuning

```python
# Upload training data
training_file = client.files.upload(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    model="mistral-small-latest",
    training_files=[training_file.id],
    hyperparameters={
        "training_steps": 100,
        "learning_rate": 1e-5
    }
)

# Check status
job_status = client.fine_tuning.jobs.get(job.id)
```

## Reference Files

This skill includes comprehensive documentation in `references/`:

| File | Description |
|------|-------------|
| **agents.md** | Multi-agent systems, handoffs, conversations |
| **api_reference.md** | Complete API endpoint reference |
| **audio.md** | Audio transcription and speech |
| **chat_completion.md** | Chat completions, streaming, conversations |
| **embeddings.md** | Text embeddings for RAG/search |
| **fine_tuning.md** | Model fine-tuning and customization |
| **function_calling.md** | Tool use and function calling |
| **getting_started.md** | Quick start and model overview |
| **moderation.md** | Content moderation and safety |
| **vision.md** | Image understanding (Pixtral) |
| **other.md** | OCR, document AI, deployment |

Use `view` to read specific reference files when detailed information is needed.

## Key Concepts

### Models
- **Premier models**: mistral-large, pixtral-large (best performance)
- **Open models**: mistral-small, mistral-embed (cost-effective)
- **Specialized**: mistral-ocr (document processing)

### Rate Limits
- Default: varies by tier
- Use streaming for long responses
- Implement exponential backoff for retries

### Best Practices
1. Always set `MISTRAL_API_KEY` as environment variable
2. Use streaming for real-time applications
3. Implement proper error handling
4. Cache embeddings for repeated queries
5. Use function calling for structured outputs

## Resources

- **Official Docs**: https://docs.mistral.ai/
- **API Reference**: https://docs.mistral.ai/api/
- **Discord**: https://discord.gg/mistralai
- **Cookbooks**: https://docs.mistral.ai/guides/

## Working with This Skill

### For Beginners
Start with the getting_started reference file, then try chat_completion for basic usage.

### For Specific Features
- Chat/streaming: `chat_completion.md`
- Tool use: `function_calling.md`
- RAG/search: `embeddings.md`
- Custom models: `fine_tuning.md`
- Multi-agent: `agents.md`
- Document processing: `other.md` (OCR section)

### For Code Examples
The quick reference section above contains common patterns. Each reference file includes language-specific examples (Python, TypeScript, cURL).

## Notes

- This skill was automatically generated from official Mistral AI documentation
- The documentation was sourced from llms.txt for optimal coverage
- Code examples include Python, TypeScript, and cURL variants
- API endpoints follow RESTful conventions at `https://api.mistral.ai/v1/`

## Updating

To refresh this skill with updated documentation:
```bash
skill-seekers scrape --config configs/mistral.json
```
