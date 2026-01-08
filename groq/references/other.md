# Groq - Other

**Pages:** 74

---

## Assistant Message Prefilling

**URL:** llms-txt#assistant-message-prefilling

**Contents:**
- How to Prefill Assistant Messages
- Example Usage
- OpenAI Compatibility

When using Groq API, you can have more control over your model output by prefilling `assistant` messages. This technique gives you the ability to direct any text-to-text model powered by Groq to:
- Skip unnecessary introductions or preambles
- Enforce specific output formats (e.g., JSON, XML)
- Maintain consistency in conversations

## How to Prefill Assistant Messages
To prefill, simply include your desired starting text in the `assistant` message and the model will generate a response starting with the `assistant` message. 
<br />
**Note:** For some models, adding a newline after the prefill `assistant` message leads to better results.  
<br />
**💡 Tip:** Use the stop sequence (`stop`) parameter in combination with prefilling for even more concise results. We recommend using this for generating code snippets.

## Example Usage
**Example 1: Controlling output format for concise code snippets**
<br />
When trying the below code, first try a request without the prefill and then follow up by trying another request with the prefill included to see the difference!

**Example 2: Extracting structured data from unstructured input**

## OpenAI Compatibility

URL: https://console.groq.com/docs/openai

---

## Billing FAQs

**URL:** llms-txt#billing-faqs

**Contents:**
- Upgrading to Developer Tier
  - What happens when I upgrade to the Developer tier?
  - What are the benefits of upgrading?
  - Can I downgrade back to the Free tier after I upgrade?
- Understanding Groq's Billing Model
  - How does Groq's billing cycle work?
  - How does progressive billing work?
  - What if I don't reach the next threshold?
  - When is payment withdrawn from my account?
- Monitoring Your Spending & Usage

## Upgrading to Developer Tier

### What happens when I upgrade to the Developer tier?

When you upgrade, **there's no immediate charge** - you'll be billed for tokens at month-end or when you reach progressive billing thresholds (see below for details).

To upgrade from the Free tier to the Developer tier, you'll need to provide a valid payment method (credit card, US bank account, or SEPA debit account).

Your upgrade takes effect immediately, but billing only occurs at the end of your monthly billing cycle or when you cross progressive thresholds ($1, $10, $100, $500, $1,000; see below for details).

### What are the benefits of upgrading?

The Developer tier is designed for developers and companies who want increased capacity and more features with pay-as-you-go pricing. Immediately after upgrading, you unlock several benefits:

**Core Features:**
- **Higher Token Limits:** Significantly increased rate limits for production workloads
- **Chat Support:** Direct access to our support team via chat
- **[Flex Service Tier](/docs/flex-processing):** Flexible processing options for your workloads
- **[Batch Processing](/docs/batch):** Submit and process large batches of requests efficiently
- **[Spend Limits](/docs/spend-limits):** Set automated spending limits and receive budget alerts

### Can I downgrade back to the Free tier after I upgrade?

Yes, you can downgrade to the Free tier at any time from your account Settings under [**Billing**](/settings/billing).

> **Note:** When you downgrade, we will issue a final invoice for any outstanding usage that has not yet been billed. You'll need to pay this final invoice before the downgrade is complete.

After downgrading:
- Your account returns to Free tier rate limits and restrictions
- You'll lose access to Developer tier benefits (priority support, unlimited requests, etc.)
- Any usage-based charges stop immediately
- You can upgrade again at any time if you need more capacity

## Understanding Groq's Billing Model

### How does Groq's billing cycle work?

Groq uses a monthly billing cycle, where you receive an invoice in arrears for usage. However, for new users, we also apply progressive billing thresholds to help ease you into pay-as-you-go usage.

### How does progressive billing work?

When you first start using Groq on the Developer plan, your billing follows a progressive billing model. In this model, an invoice is automatically triggered and payment is deducted when your cumulative usage reaches specific thresholds: $1, $10, $100, $500, and $1,000.

**Special billing for customers in India:** Customers with a billing address in India have different progressive billing thresholds. For India customers, the thresholds are only $1, $10, and then $100 recurring. The $500 and $1,000 thresholds do not apply to India customers. Instead, after reaching the initial $1 and $10 thresholds, billing will continue to trigger every time usage reaches another $100 increment.

This helps you monitor early usage and ensures you're not surprised by a large first bill. These are one-time thresholds for most customers. Once you cross the $1,000 lifetime usage threshold, only monthly billing continues (this does not apply to India customers who continue with recurring $100 billing).

### What if I don't reach the next threshold?

If you don't reach the next threshold, your usage will be billed on your regular end-of-month invoice.

**Example:**
- You cross $1 → you're charged immediately.
- You then use $2 more for the entire month (lifetime usage = $3, still below $10).
- That $2 will be invoiced at the end of your monthly billing cycle, not immediately.

This ensures you're not repeatedly charged for small amounts and are charged only when hitting a lifetime cumulative threshold or when your billing period ends.

Once your lifetime usage crosses the $1,000 threshold, the progressive thresholds no longer apply. From this point forward, your account is billed solely on a monthly cycle. All future usage is accrued and billed once per month, with payment automatically deducted when the invoice is issued.

### When is payment withdrawn from my account?

Payment is withdrawn automatically from your connected payment method each time an invoice is issued. This can happen in two cases:

- **Progressive billing phase:** When your usage first crosses the $1, $10, $100, $500, or $1,000 thresholds. For customers in India, payment is withdrawn at $1, $10, and then every $100 thereafter (the $500 and $1,000 thresholds do not apply).
- **Monthly billing phase:** At the end of each monthly billing cycle.

> **Note:** We only bill you once your usage has reached at least $0.50. If you see a total charge of < $0.50 or you get an invoice for < $0.50, there is no action required on your end.

## Monitoring Your Spending & Usage

### How can I view my current usage and spending in real time?

You can monitor your usage and charges in near real-time directly within your Groq Cloud dashboard. Simply navigate to [**Dashboard** → **Usage**](/dashboard/usage)

This dashboard allows you to:
- Track your current usage across models
- Understand how your consumption aligns with pricing per model

### Can I set spending limits or receive budget alerts?

Yes, Groq provides Spend Limits to help you control your API costs. You can set automated spending limits and receive proactive usage alerts as you approach your defined budget thresholds. [**More details here**](/docs/spend-limits)

## Invoices, Billing Info & Credits

### Where can I find my past invoices and payment history?

You can view and download all your invoices and receipts in the Groq Console:
[**Settings** → **Billing** → **Manage Billing**](/settings/billing/manage)

### Can I change my billing info and payment method?

You can update your billing details anytime from the Groq Console:
[**Settings** → **Billing** → **Manage Billing**](/settings/billing/manage)

### What payment methods do you accept?

Groq accepts credit cards (Visa, MasterCard, American Express, Discover), United States bank accounts, and SEPA debit accounts as payment methods.

### Are there promotional credits, or trial offers?

Yes! We occasionally offer promotional credits, such as during hackathons and special events. We encourage you to visit our [**Groq Community**](https://community.groq.com/) page to learn more and stay updated on announcements.

If you're building a startup, you may be eligible for the [**Groq for Startups**](https://groq.com/groq-for-startups) program, which unlocks $10,000 in credits to help you scale faster.

## Common Billing Questions & Troubleshooting

### How are refunds handled, if applicable?

Refunds are handled on a case-by-case basis. Due to the specific circumstances involved in each situation, we recommend reaching out directly to our customer support team at **support@groq.com** for assistance. They will review your case and provide guidance.

### What if a user believes there's an error in their bill?

Check your console's Usage and Billing tab first. If you still believe there's an issue:

Please contact our customer support team immediately at **support@groq.com**. They will investigate the specific circumstances of your billing dispute and guide you through the resolution process.

### Under what conditions can my account be suspended due to billing issues?

Account suspension or restriction due to billing issues typically occurs when there's a prolonged period of non-payment or consistently failed payment attempts. However, the exact conditions and resolution process are handled on a case-by-case basis. If your account is impacted, or if you have concerns, please reach out to our customer support team directly at **support@groq.com** for specific guidance regarding your account status.

### What happens if my payment fails? Why did my payment fail?

You may attempt to retry the payment up to two times. Before doing so, we recommend updating your payment method to ensure successful processing. If the issue persists, please contact our support team at support@groq.com for further assistance. Failed payments may result in service suspension. We will email you to remind you of your unpaid invoice.

### What should I do if my billing question isn't answered in the FAQ?

Feel free to contact **support@groq.com**
<br />

Need help? Contact our support team at **support@groq.com** with details about your billing questions.

## Models: Get Models (js)

URL: https://console.groq.com/docs/models/scripts/get-models

import Groq from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const getModels = async () => {
  return await groq.models.list();
};

getModels().then((models) => {
  // console.log(models);
});

## Models: Get Models (py)

URL: https://console.groq.com/docs/models/scripts/get-models.py

URL: https://console.groq.com/docs/models

**Examples:**

Example 1 (python):
```python
import requests
import os

api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())
```

---

## Build query parameters using requests params

**URL:** llms-txt#build-query-parameters-using-requests-params

url = "https://api.groq.com/openai/v1/batches"
params = [("id", batch_id) for batch_id in batch_ids]

---

## Collect the newest from AI News website

**URL:** llms-txt#collect-the-newest-from-ai-news-website

**Contents:**
- Advanced Session Configuration

task_result = client.agent.task(
    "Extract the latest news title from this AI News website",
    task_options={
        "url": "https://www.artificialintelligence-news.com/",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
    }
)

print("Latest news title:", task_result)

python
import os
from anchorbrowser import Anchorbrowser

**Examples:**

Example 1 (unknown):
```unknown
## Advanced Session Configuration
Create a session using advanced configuration (see Anchor [API reference](https://docs.anchorbrowser.io/api-reference/sessions/create-session?utm_source=groq)).
```

---

## Command to install Groq SDK package

**URL:** llms-txt#command-to-install-groq-sdk-package

javascript
// JavaScript code snippet for translation
shell

**Examples:**

Example 1 (unknown):
```unknown
The following code snippet demonstrates how to use Groq API to translate an audio file in JavaScript:
```

Example 2 (unknown):
```unknown
The following is an example cURL request:
```

---

## Compare the results of both prompts

**URL:** llms-txt#compare-the-results-of-both-prompts

**Contents:**
- Text Chat: Prompt Engineering.doc (ts)
- pip install pydantic

get_movie_data(poor_prompt, "Poor Prompt Example")
get_movie_data(effective_prompt, "Effective Prompt Example")
javascript
import { Groq } from "groq-sdk";
import { z } from "zod";

const client = new Groq();

// Define a schema for validation
const MovieSchema = z.object({
  title: z.string(),
  year: z.number().int(),
  director: z.string(),
  genre: z.array(z.string()),
  runtime_minutes: z.number().int(),
  rating: z.number().min(1).max(10),
  box_office_millions: z.number(),
  cast: z.array(
    z.object({
      actor: z.string(),
      character: z.string()
    })
  )
});

type Movie = z.infer<typeof MovieSchema>;

// Example of a poorly designed prompt
const poorPrompt = `
Give me information about a movie in JSON format.
`;

// Example of a well-designed prompt
const effectivePrompt = `
You are a movie database API. Return information about a movie with the following 
JSON structure:

{
  "title": "string",
  "year": number,
  "director": "string",
  "genre": ["string"],
  "runtime_minutes": number,
  "rating": number (1-10 scale),
  "box_office_millions": number,
  "cast": [
    {
      "actor": "string",
      "character": "string"
    }
  ]
}

The response must:
1. Include ALL fields shown above
2. Use only the exact field names shown
3. Follow the exact data types specified
4. Contain ONLY the JSON object and nothing else

IMPORTANT: Do not include any explanatory text, markdown formatting, or code blocks.
`;

// Function to run the completion and display results
async function getMovieData(prompt: string, title = "Example"): Promise<Movie | null> {
  console.log(`\n--- ${title} ---`);
  
  try {
    const completion = await client.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: "Tell me about The Matrix" },
      ],
    });
    
    const responseContent = completion.choices[0].message.content;
    console.log("Raw response:");
    console.log(responseContent);
    
    // Try to parse as JSON
    try {
      const movieData = JSON.parse(responseContent || "");
      console.log("\nSuccessfully parsed as JSON!");
      
      // Validate against schema
      try {
        const validatedMovie = MovieSchema.parse(movieData);
        console.log("All expected fields present and valid!");
        return validatedMovie;
      } catch (validationError) {
        if (validationError instanceof z.ZodError) {
          console.log("Schema validation failed:");
          console.log(validationError.errors.map(e => `- ${e.path.join('.')}: ${e.message}`).join('\n'));
        }
        return null;
      }
    } catch (syntaxError) {
      console.log("\nFailed to parse as JSON. Response is not valid JSON.");
      return null;
    }
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

// Compare the results of both prompts
async function comparePrompts() {
  await getMovieData(poorPrompt, "Poor Prompt Example");
  await getMovieData(effectivePrompt, "Effective Prompt Example");
}

// Run the examples
comparePrompts();
python
import os
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field 
from groq import Groq
import instructor

**Examples:**

Example 1 (unknown):
```unknown
---

## Text Chat: Prompt Engineering.doc (ts)

URL: https://console.groq.com/docs/text-chat/scripts/prompt-engineering.doc
```

Example 2 (unknown):
```unknown
---

## pip install pydantic

URL: https://console.groq.com/docs/text-chat/scripts/complex-schema-example.py
```

---

## configuration example, can be ommited for default values.

**URL:** llms-txt#configuration-example,-can-be-ommited-for-default-values.

session_config = {
    "session": {
        "recording": False,  # Disable session recording
        "proxy": {
            "active": True,
            "type": "anchor_residential",
            "country_code": "us"
        },
        "max_duration": 5,  # 5 minutes
        "idle_timeout": 1    # 1 minute
    }
}

client = Anchorbrowser(api_key=os.getenv("ANCHOR_API_KEY"))
configured_session = client.sessions.create(browser=session_config)

---

## Configure Phoenix tracer

**URL:** llms-txt#configure-phoenix-tracer

tracer_provider = register(
    project_name="default",
    endpoint="https://app.phoenix.arize.com/v1/traces",
)

---

## Connect GitHub (you'll be guided through OAuth flow to get things going)

**URL:** llms-txt#connect-github-(you'll-be-guided-through-oauth-flow-to-get-things-going)

---

## Create an AI assistant

**URL:** llms-txt#create-an-ai-assistant

assistant = AssistantAgent(
    name="groq_assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"config_list": config_list}
)

---

## Create an AI assistant that uses the weather tool

**URL:** llms-txt#create-an-ai-assistant-that-uses-the-weather-tool

assistant = AssistantAgent(
    name="groq_assistant",
    system_message="""You are a helpful AI assistant who can:
    - Use weather information tools
    - Write Python code for data visualization
    - Analyze and explain results""",
    llm_config={"config_list": config_list}
)

---

## Create a directory to store code files

**URL:** llms-txt#create-a-directory-to-store-code-files

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

---

## Create a directory to store code files from code executor

**URL:** llms-txt#create-a-directory-to-store-code-files-from-code-executor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

---

## Create a simple prompt

**URL:** llms-txt#create-a-simple-prompt

prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract product details into JSON with this structure:
        {{
            "name": "product name here",
            "price": number_here_without_currency_symbol,
            "features": ["feature1", "feature2", "feature3"]
        }}"""),
    ("user", "{input}")
])

---

## Create Groq client

**URL:** llms-txt#create-groq-client

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

---

## Create the chain that guarantees JSON output

**URL:** llms-txt#create-the-chain-that-guarantees-json-output

chain = prompt | llm | parser

def parse_product(description: str) -> dict:
    result = chain.invoke({"input": description})
    print(json.dumps(result, indent=2))

---

## cURL request example

**URL:** llms-txt#curl-request-example

**Contents:**
- Understanding Metadata Fields
  - Using Metadata for Debugging
  - Quality Thresholds and Regular Monitoring
- Prompting Guidelines
- Agno + Groq: Fast Agents
- Agno + Groq: Fast Agents
  - Python Quick Start (2 minutes to hello world)

json
{
  "text": "Your translated text appears here...",
  "x_groq": {
    "id": "req_unique_id"
  }
}
json
{
  "id": 8,
  "seek": 3000,
  "start": 43.92,
  "end": 50.16,
  "text": " document that the functional specification that you started to read through that isn't just the",
  "tokens": [51061, 4166, 300, 264, 11745, 31256],
  "temperature": 0,
  "avg_logprob": -0.097569615,
  "compression_ratio": 1.6637554,
  "no_speech_prob": 0.012814695
}
python web_search_agent.py
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

**Examples:**

Example 1 (unknown):
```unknown
The following is an example response:
```

Example 2 (unknown):
```unknown
## Understanding Metadata Fields
When working with Groq API, setting `response_format` to `verbose_json` outputs each segment of transcribed text with valuable metadata that helps us understand the quality and characteristics of our 
transcription, including `avg_logprob`, `compression_ratio`, and `no_speech_prob`. 

This information can help us with debugging any transcription issues. Let's examine what this metadata tells us using a real 
example:
```

Example 3 (unknown):
```unknown
As shown in the above example, we receive timing information as well as quality indicators. Let's gain a better understanding of what each field means:
- `id:8`: The 9th segment in the transcription (counting begins at 0)
- `seek`: Indicates where in the audio file this segment begins (3000 in this case)
- `start` and `end` timestamps: Tell us exactly when this segment occurs in the audio (43.92 to 50.16 seconds in our example)
- `avg_logprob` (Average Log Probability): -0.097569615 in our example indicates very high confidence. Values closer to 0 suggest better confidence, while more negative values (like -0.5 or lower) might 
indicate transcription issues.
- `no_speech_prob` (No Speech Probability): 0.0.012814695 is very low, suggesting this is definitely speech. Higher values (closer to 1) would indicate potential silence or non-speech audio.
- `compression_ratio`: 1.6637554 is a healthy value, indicating normal speech patterns. Unusual values (very high or low) might suggest issues with speech clarity or word boundaries.

### Using Metadata for Debugging
When troubleshooting transcription issues, look for these patterns:
- Low Confidence Sections: If `avg_logprob` drops significantly (becomes more negative), check for background noise, multiple speakers talking simultaneously, unclear pronunciation, and strong accents. 
Consider cleaning up the audio in these sections or adjusting chunk sizes around problematic chunk boundaries.
- Non-Speech Detection: High `no_speech_prob` values might indicate silence periods that could be trimmed, background music or noise, or non-verbal sounds being misinterpreted as speech. Consider noise 
reduction when preprocessing.
- Unusual Speech Patterns: Unexpected `compression_ratio` values can reveal stuttering or word repetition, speaker talking unusually fast or slow, or audio quality issues affecting word separation.

### Quality Thresholds and Regular Monitoring
We recommend setting acceptable ranges for each metadata value we reviewed above and flagging segments that fall outside these ranges to be able to identify and adjust preprocessing or chunking strategies for 
flagged sections. 

By understanding and monitoring these metadata values, you can significantly improve your transcription quality and quickly identify potential issues in your audio processing pipeline. 


## Prompting Guidelines
  The prompt parameter (max 224 tokens) helps provide context and maintain a consistent output style.
  Unlike chat completion prompts, these prompts only guide style and context, not specific actions.

  ### Best Practices

  - Provide relevant context about the audio content, such as the type of conversation, topic, or 
    speakers involved.
  - Use the same language as the language of the audio file.
  - Steer the model's output by denoting proper spellings or emulate a specific writing style or tone.
  - Keep the prompt concise and focused on stylistic guidance.

We can't wait to see what you build!

---

## Agno + Groq: Fast Agents

URL: https://console.groq.com/docs/agno

## Agno + Groq: Fast Agents

[Agno](https://github.com/agno-agi/agno) is a lightweight framework for building multi-modal Agents. It's easy to use, extremely fast and supports multi-modal inputs and outputs.

With Groq & Agno, you can build:

- **Agentic RAG**: Agents that can search different knowledge stores for RAG or dynamic few-shot learning.
- **Image Agents**: Agents that can understand images and make tool calls accordingly.
- **Reasoning Agents**: Agents that can reason using a reasoning model, then generate a result using another model.
- **Structured Outputs**: Agents that can generate pydantic objects adhering to a schema.

### Python Quick Start (2 minutes to hello world)

Agents are autonomous programs that use language models to achieve tasks. They solve problems by running tools, accessing knowledge and memory to improve responses.

Let's build a simple web search agent, with a tool to search DuckDuckGo to get better results. 

#### 1. Create a file called `web_search_agent.py` and add the following code:
```

---

## Define a complex nested schema

**URL:** llms-txt#define-a-complex-nested-schema

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Address

class ProductVariant(BaseModel):
    id: str
    name: str
    price: float
    inventory_count: int
    attributes: Dict[str, str]

class ProductReview(BaseModel):
    user_id: str
    rating: float = Field(ge=1, le=5)
    comment: str
    date: str

class Product(BaseModel):
    id: str
    name: str
    description: str
    main_category: str
    subcategories: List[str]
    variants: List[ProductVariant]
    reviews: List[ProductReview]
    average_rating: float = Field(ge=1, le=5)
    manufacturer: Dict[str, Union[str, ContactInfo]]

---

## Define a schema with Pydantic (Python's equivalent to Zod)

**URL:** llms-txt#define-a-schema-with-pydantic-(python's-equivalent-to-zod)

class Product(BaseModel):
    id: str
    name: str
    price: float
    description: str
    in_stock: bool
    tags: List[str] = Field(default_factory=list)

---

## Define batch IDs to check

**URL:** llms-txt#define-batch-ids-to-check

batch_ids = [
    "batch_01jh6xa7reempvjyh6n3yst111",
    "batch_01jh6xa7reempvjyh6n3yst222",
    "batch_01jh6xa7reempvjyh6n3yst333",
]

---

## Define the expected JSON structure

**URL:** llms-txt#define-the-expected-json-structure

parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "features": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
})

---

## Define the tool schema

**URL:** llms-txt#define-the-tool-schema

tool_schema = {
    "name": "get_weather_info",
    "description": "Get the weather information for any location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location for which we want to get the weather information (e.g., New York)"
            }
        },
        "required": ["location"]
    }
}

---

## Define weather tool

**URL:** llms-txt#define-weather-tool

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

---

## Define your schema with Pydantic

**URL:** llms-txt#define-your-schema-with-pydantic

class RecipeIngredient(BaseModel):
    name: str
    quantity: str
    unit: str = Field(description="The unit of measurement, like cup, tablespoon, etc.")

class Recipe(BaseModel):
    title: str
    description: str
    prep_time_minutes: int
    cook_time_minutes: int
    ingredients: List[RecipeIngredient]
    instructions: List[str] = Field(description="Step by step cooking instructions")

---

## Deploy to AWS Lambda

**URL:** llms-txt#deploy-to-aws-lambda

**Contents:**
- Additional Resources
- CrewAI + Groq: High-Speed Agent Orchestration
- CrewAI + Groq: High-Speed Agent Orchestration
  - Python Quick Start (2 minutes to hello world)

npm run mastra deploy -- --platform aws-lambda
typescript server.ts
import { Mastra } from '@mastra/core';
import { agents } from './mastra/agents';
import { workflows } from './mastra/workflows';

const mastra = new Mastra({
  agents,
  workflows,
});

const server = mastra.getServer();

server.listen(3000, () => {
  console.log('Mastra server running on port 3000');
});
bash
pip install crewai groq
bash
export GROQ_API_KEY="your-api-key"
python
from crewai import Agent, Task, Crew, LLM

**Examples:**

Example 1 (unknown):
```unknown
Or use the built-in server:
```

Example 2 (unknown):
```unknown
**Challenge:** Build a multi-agent research system that uses Groq for fast inference, coordinates multiple specialized agents (researcher, analyst, writer), maintains conversation memory, and generates comprehensive reports with proper citations!

## Additional Resources

- [Mastra Documentation](https://mastra.ai/en/docs)
- [Mastra Examples](https://mastra.ai/en/examples)
- [Mastra API Reference](https://mastra.ai/en/reference)
- [Mastra MCP Server Guide](https://mastra.ai/en/reference/tools/mcp-server)
- [Mastra GitHub](https://github.com/mastra-ai/mastra)
- [Groq with Vercel AI SDK](https://sdk.vercel.ai/providers/ai-sdk-providers/groq)

---

## CrewAI + Groq: High-Speed Agent Orchestration

URL: https://console.groq.com/docs/crewai

## CrewAI + Groq: High-Speed Agent Orchestration

CrewAI is a framework that enables the orchestration of multiple AI agents with specific roles, tools, and goals as a cohesive team to accomplish complex tasks and create sophisticated workflows.

Agentic workflows require fast inference due to their complexity. Groq's fast inference optimizes response times for CrewAI agent teams, enabling rapid autonomous decision-making and collaboration for:

- **Fast Agent Interactions:** Leverage Groq's fast inference speeds via Groq API for efficient agent communication
- **Reliable Performance:** Consistent response times across agent operations
- **Scalable Multi-Agent Systems:** Run multiple agents in parallel without performance degradation
- **Simple Integration:** Get started with just a few lines of code

### Python Quick Start (2 minutes to hello world)
#### 1. Install the required packages:
```

Example 3 (unknown):
```unknown
#### 2. Configure your Groq API key:
```

Example 4 (unknown):
```unknown
#### 3. Create your first Groq-powered CrewAI agent:

In CrewAI, **agents** are autonomous entities you can design to perform specific roles and achieve particular goals while **tasks** are specific assignments given to agents that detail the actions they
need to perform to achieve a particular goal. Tools can be assigned as tasks.
```

---

## Deploy to Cloudflare Workers

**URL:** llms-txt#deploy-to-cloudflare-workers

npm run mastra deploy -- --platform cloudflare

---

## Deploy to Vercel

**URL:** llms-txt#deploy-to-vercel

npm run mastra deploy -- --platform vercel

---

## Example 1: Calculation

**URL:** llms-txt#example-1:-calculation

computation_query = "Calculate the monthly payment for a $30,000 loan over 5 years at 6% annual interest."

---

## Example 1: Error Explanation (might trigger search)

**URL:** llms-txt#example-1:-error-explanation-(might-trigger-search)

debug_query_search = "I'm getting a 'Kubernetes CrashLoopBackOff' error on my pod. What are the common causes based on recent discussions?"

---

## Example 2: Code Check (might trigger code execution)

**URL:** llms-txt#example-2:-code-check-(might-trigger-code-execution)

debug_query_exec = "Will this Python code raise an error? `import numpy as np; a = np.array([1,2]); b = np.array([3,4,5]); print(a+b)`"

---

## Example 2: Simple code execution

**URL:** llms-txt#example-2:-simple-code-execution

code_query = "What is the output of this Python code snippet: `data = {'a': 1, 'b': 2}; print(data.keys())`"

---

## Example of a poorly designed prompt

**URL:** llms-txt#example-of-a-poorly-designed-prompt

poor_prompt = """
Give me information about a movie in JSON format.
"""

---

## Example of a well-designed prompt

**URL:** llms-txt#example-of-a-well-designed-prompt

effective_prompt = """
You are a movie database API. Return information about a movie with the following 
JSON structure:

{
  "title": "string",
  "year": number,
  "director": "string",
  "genre": ["string"],
  "runtime_minutes": number,
  "rating": number (1-10 scale),
  "box_office_millions": number,
  "cast": [
    {
      "actor": "string",
      "character": "string"
    }
  ]
}

The response must:
1. Include ALL fields shown above
2. Use only the exact field names shown
3. Follow the exact data types specified
4. Contain ONLY the JSON object and nothing else

IMPORTANT: Do not include any explanatory text, markdown formatting, or code blocks.
"""

---

## (example shell script)

**URL:** llms-txt#(example-shell-script)

**Contents:**
  - Performing a Chat Completion:

**Examples:**

Example 1 (unknown):
```unknown
### Performing a Chat Completion:
```

---

## Example usage

**URL:** llms-txt#example-usage

**Contents:**
- xRx + Groq: Easily Build Rich Multi-Modal Experiences
- xRx + Groq: Easily Build Rich Multi-Modal Experiences
  - Quick Start Guide (2 minutes + build time)
  - Option 1: Sample Apps Collection
  - Option 2: AI Voice Tutor
- 🗂️ LlamaIndex 🦙
- 🗂️ LlamaIndex 🦙
- Mastra + Groq: Build Production AI Agents & Workflows
- Mastra + Groq: Build Production AI Agents & Workflows
- Quick Start

description = """The Kees Van Der Westen Speedster is a high-end, single-group espresso machine known for its precision, performance, 
and industrial design. Handcrafted in the Netherlands, it features dual boilers for brewing and steaming, PID temperature control for 
consistency, and a unique pre-infusion system to enhance flavor extraction. Designed for enthusiasts and professionals, it offers 
customizable aesthetics, exceptional thermal stability, and intuitive operation via a lever system. The pricing is approximatelyt $14,499 
depending on the retailer and customization options."""

parse_product(description)
bash
git clone --recursive https://github.com/8090-inc/xrx-sample-apps.git
bash
cd xrx-sample-apps
bash
   cp env-example.txt .env
   bash
docker-compose up --build
bash
git clone --recursive https://github.com/bklieger-groq/mathtutor-on-groq.git
bash
cp env-example.txt .env
bash
LLM_API_KEY="your_groq_api_key_here"
GROQ_STT_API_KEY="your_groq_api_key_here"
ELEVENLABS_API_KEY="your_elevenlabs_api_key"  # For text-to-speech
bash
docker-compose up --build
bash
npx create-mastra@latest my-app
cd my-app
bash
npm install @ai-sdk/groq
bash
export GROQ_API_KEY="your-groq-api-key"
typescript src/mastra/agents/index.ts
const groq = createGroq({
  apiKey: process.env.GROQ_API_KEY,
});

const researchAgent = new Agent({
  name: 'Research Assistant',
  instructions: 'You are a helpful research assistant that provides accurate, well-sourced information.',
  model: {
    provider: groq,
    name: 'llama-3.3-70b-versatile',
    toolChoice: 'auto',
  },
});
typescript src/index.ts
import { Mastra } from '@mastra/core';
import { researchAgent } from './mastra/agents';

const mastra = new Mastra({
  agents: { researchAgent },
});

const result = await mastra
  .getAgent('researchAgent')
  .generate('What are the latest developments in AI inference optimization?');

console.log(result.text);
typescript agents/tool-agent.ts
const groq = createGroq({ apiKey: process.env.GROQ_API_KEY });

const weatherTool = createTool({
  id: 'get_weather',
  description: 'Get current weather for a location',
  inputSchema: z.object({
    location: z.string().describe('City name'),
  }),
  execute: async ({ context }) => {
    // API call to weather service
    return `Weather in ${context.location}: 72°F, sunny`;
  },
});

const weatherAgent = new Agent({
  name: 'Weather Assistant',
  instructions: 'You help users get weather information.',
  model: {
    provider: groq,
    name: 'llama-3.3-70b-versatile',
  },
  tools: { weatherTool },
});
typescript workflows/research-workflow.ts
const searchStep = new Step({
  id: 'search',
  execute: async ({ context }) => {
    // Search for information
    return { results: ['result1', 'result2', 'result3'] };
  },
});

const analyzeStep = new Step({
  id: 'analyze',
  execute: async ({ context, mastra }) => {
    const agent = mastra.getAgent('researchAgent');
    const analysis = await agent.generate(
      `Analyze these search results: ${context.results.join(', ')}`
    );
    return { analysis: analysis.text };
  },
});

const summarizeStep = new Step({
  id: 'summarize',
  execute: async ({ context, mastra }) => {
    const agent = mastra.getAgent('researchAgent');
    const summary = await agent.generate(
      `Summarize this analysis: ${context.analysis}`
    );
    return { summary: summary.text };
  },
});

const researchWorkflow = new Workflow({
  name: 'research-workflow',
  triggerSchema: z.object({
    query: z.string(),
  }),
});

researchWorkflow
  .step(searchStep)
  .then(analyzeStep)
  .then(summarizeStep)
  .commit();
typescript agents/memory-agent.ts
const groq = createGroq({ apiKey: process.env.GROQ_API_KEY });

const chatAgent = new Agent({
  name: 'Chat Assistant',
  instructions: 'You are a helpful assistant that remembers context.',
  model: {
    provider: groq,
    name: 'llama-3.3-70b-versatile',
  },
  enableMemory: true,
});

// Use with thread-based memory
const result = await chatAgent.generate(
  'What did we discuss earlier?',
  {
    threadId: 'user-123',
    resourceId: 'conversation-1',
  }
);
typescript mcp/server.ts
const notesTool = createTool({
  id: 'create_note',
  description: 'Create a new note',
  inputSchema: z.object({
    title: z.string(),
    content: z.string(),
  }),
  execute: async ({ context }) => {
    // Save note to database
    return `Note created: ${context.title}`;
  },
});

const mcpServer = new MCPServer({
  name: 'Notes Server',
  version: '1.0.0',
  tools: { notesTool },
});

// Start the server
await mcpServer.start();
typescript agents/mcp-agent.ts
const exaClient = new MCPClient({
  name: 'exa',
  serverUrl: `https://mcp.exa.ai/mcp?exaApiKey=${process.env.EXA_API_KEY}`,
});

const exaTools = await exaClient.getTools();

const searchAgent = new Agent({
  name: 'Search Agent',
  instructions: 'You help users search the web for information.',
  model: {
    provider: groq,
    name: 'llama-3.3-70b-versatile',
  },
  tools: exaTools,
});
typescript
const stream = await researchAgent.stream(
  'Explain quantum computing',
  { threadId: 'user-123' }
);

for await (const chunk of stream) {
  if (chunk.type === 'text-delta') {
    process.stdout.write(chunk.textDelta);
  }
}
typescript
const researcher = new Agent({ /* ... */ });
const writer = new Agent({ /* ... */ });
const editor = new Agent({ /* ... */ });

const supervisor = new Agent({
  name: 'Supervisor',
  model: { provider: groq, name: 'llama-3.3-70b-versatile' },
});

const result = await supervisor.network({
  agents: [researcher, writer, editor],
  prompt: 'Write a research article about AI',
  maxTurns: 5,
});
typescript
const safeAgent = new Agent({
  name: 'Safe Agent',
  model: { provider: groq, name: 'llama-3.3-70b-versatile' },
  guardrails: {
    input: [
      {
        check: (input: string) => !input.includes('harmful'),
        message: 'Input contains harmful content',
      },
    ],
    output: [
      {
        check: (output: string) => output.length < 1000,
        message: 'Output too long',
      },
    ],
  },
});
typescript
workflow
  .parallel([step1, step2, step3])
  .then(combineResults)
  .commit();
typescript
workflow
  .step(checkCondition)
  .branch({
    when: (context) => context.needsApproval,
    then: [requestApproval, processApproval],
    else: [autoProcess],
  })
  .commit();
typescript
const step = new Step({
  id: 'risky-operation',
  execute: async ({ context }) => {
    // Operation that might fail
  },
  retryConfig: {
    maxRetries: 3,
    delayMs: 1000,
  },
});
bash

**Examples:**

Example 1 (unknown):
```unknown
**Challenge:** Make the above code your own! Try extending it to include memory with conversation history handling via LangChain to enable
users to ask follow-up questions.

For more information on how to build robust, realtime applications with LangChain and Groq, see:
- [Official Documentation: LangChain](https://python.langchain.com/docs/integrations/chat/groq)
- [Groq API Cookbook: Benchmarking a RAG Pipeline with LangChain and LLama](https://github.com/groq/groq-api-cookbook/blob/main/tutorials/benchmarking-rag-langchain/benchmarking_rag.ipynb)
- [Webinar: Build Blazing-Fast LLM Apps with Groq, Langflow, & LangChain](https://youtu.be/4ukqsKajWnk?si=ebbbnFH0DySdoWbX)

---

## xRx + Groq: Easily Build Rich Multi-Modal Experiences

URL: https://console.groq.com/docs/xrx

## xRx + Groq: Easily Build Rich Multi-Modal Experiences

[xRx](https://github.com/8090-inc/xrx-core) is an open-source framework for building AI-powered applications that interact with users across multiple modalities — multimodality input (x), 
reasoning (R), and multimodality output (x). It allows developers to create sophisticated AI systems that seamlessly integrate text, voice, and 
other interaction forms, providing users with truly immersive experiences.

**Key Features:**
- **Multimodal Interaction**: Effortlessly integrate audio, text, widgets and other modalities for both input and output.
- **Advanced Reasoning**: Utilize comprehensive reasoning systems to enhance user interactions with intelligent and context-aware responses.
- **Modular Architecture**: Easily extend and customize components with a modular system of reusable building blocks.
- **Observability and Guardrails**: Built-in support for LLM observability and guardrails, allowing developers to monitor, debug, and optimize 
reasoning agents effectively.

### Quick Start Guide (2 minutes + build time)

The easiest way to use xRx is to start with an example app and customize it. You can either explore the sample apps collection or try our AI voice tutor for calculus that includes a whiteboard and internal math engine.

### Option 1: Sample Apps Collection

#### 1. Clone the Repository
```

Example 2 (unknown):
```unknown
Note: The `--recursive` flag is required as each app uses the xrx-core submodule.

#### 2. Navigate to Sample Apps
```

Example 3 (unknown):
```unknown
#### 3. Choose and Configure an Application
1. Navigate to your chosen app's directory
2. Copy the environment template:
```

Example 4 (unknown):
```unknown
3. Configure the required environment variables:
   - Each application has its own set of required variables
   - Check the `.env.example` file in the app's directory
   - Set all required API keys and configuration

> **Tip**: We recommend opening only the specific app folder in your IDE for a cleaner workspace.

#### 4. Follow App-Specific Setup
- Each application has its own README with specific instructions
- Complete any additional setup steps outlined in the app's README
- Ensure all dependencies are properly configured

#### 5. Launch the Application
```

---

## Extract and validate the response

**URL:** llms-txt#extract-and-validate-the-response

**Contents:**
- Text Chat: Basic Validation Zod.doc (ts)
- Text Chat: Complex Schema Example (js)
- Text Chat: Basic Validation Zod (js)
- Text Chat: Streaming Chat Completion (js)
- Text Chat: Streaming Chat Completion With Stop (js)
- pip install pydantic

try:
    response_content = completion.choices[0].message.content
    # Parse JSON
    json_data = json.loads(response_content)
    # Validate against schema
    product = Product(**json_data)
    print("Validation successful! Structured data:")
    print(product.model_dump_json(indent=2))
except json.JSONDecodeError:
    print("Error: The model did not return valid JSON")
except ValidationError as e:
    print(f"Error: The JSON did not match the expected schema: {e}")
javascript
import { Groq } from "groq-sdk";
import { z } from "zod"; // npm install zod

const client = new Groq();

// Define a schema with Zod
const ProductSchema = z.object({
  id: z.string(),
  name: z.string(),
  price: z.number().positive(),
  description: z.string(),
  in_stock: z.boolean(),
  tags: z.array(z.string()).default([]),
});

// Infer the TypeScript type from the Zod schema
type Product = z.infer<typeof ProductSchema>;

// Create a prompt that clearly defines the expected structure
const systemPrompt = `
You are a product catalog assistant. When asked about products,
always respond with valid JSON objects that match this structure:
{
  "id": "string",
  "name": "string",
  "price": number,
  "description": "string",
  "in_stock": boolean,
  "tags": ["string"]
}
Your response should ONLY contain the JSON object and nothing else.
`;

async function getStructuredResponse(): Promise<Product | undefined> {
  try {
    // Request structured data from the model
    const completion = await client.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "Tell me about a popular smartphone product" },
      ],
    });

// Extract the response
    const responseContent = completion.choices[0].message.content;
    
    // Parse and validate JSON
    const jsonData = JSON.parse(responseContent || "");
    const validatedData = ProductSchema.parse(jsonData);
    
    console.log("Validation successful! Structured data:");
    console.log(JSON.stringify(validatedData, null, 2));
    
    return validatedData;
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error("Schema validation failed:", error.errors);
    } else if (error instanceof SyntaxError) {
      console.error("JSON parsing failed: The model did not return valid JSON");
    } else {
      console.error("Error:", error);
    }
    return undefined;
  }
}

getStructuredResponse();
javascript
import Instructor from "@instructor-ai/instructor"; // npm install @instructor-ai/instructor
import { Groq } from "groq-sdk";
import { z } from "zod"; // npm install zod

// Set up the client with Instructor
const groq = new Groq();
const instructor = Instructor({
  client: groq,
  mode: "TOOLS"
})

// Define a complex nested schema
const AddressSchema = z.object({
  street: z.string(),
  city: z.string(),
  state: z.string(),
  zip_code: z.string(),
  country: z.string(),
});

const ContactInfoSchema = z.object({
  email: z.string().email(),
  phone: z.string().optional(),
  address: AddressSchema,
});

const ProductVariantSchema = z.object({
  id: z.string(),
  name: z.string(),
  price: z.number().positive(),
  inventory_count: z.number().int().nonnegative(),
  attributes: z.record(z.string()),
});

const ProductReviewSchema = z.object({
  user_id: z.string(),
  rating: z.number().min(1).max(5),
  comment: z.string(),
  date: z.string(),
});

const ManufacturerSchema = z.object({
  name: z.string(),
  founded: z.string(),
  contact_info: ContactInfoSchema,
});

const ProductSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  main_category: z.string(),
  subcategories: z.array(z.string()),
  variants: z.array(ProductVariantSchema),
  reviews: z.array(ProductReviewSchema),
  average_rating: z.number().min(1).max(5),
  manufacturer: ManufacturerSchema,
});

// System prompt with clear instructions about the complex structure
const systemPrompt = `
You are a product catalog API. Generate a detailed product with ALL required fields.
Your response must be a valid JSON object matching the schema I will use to validate it.
`;

async function getComplexProduct() {
  try {
    // Use instructor to create and validate in one step
    const product = await instructor.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_model: {
        name: "Product",
        schema: ProductSchema,
      },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "Give me details about a high-end camera product" },
      ],
      max_retries: 3,
    });

// Print the validated complex object
    console.log(`Product: ${product.name}`);
    console.log(`Description: ${product.description.substring(0, 100)}...`);
    console.log(`Variants: ${product.variants.length}`);
    console.log(`Reviews: ${product.reviews.length}`);
    console.log(`Manufacturer: ${product.manufacturer.name}`);
    console.log(`\nManufacturer Contact:`);
    console.log(`  Email: ${product.manufacturer.contact_info.email}`);
    console.log(`  Address: ${product.manufacturer.contact_info.address.city}, ${product.manufacturer.contact_info.address.country}`);

return product;
  } catch (error) {
    console.error("Error:", error);
  }
}

// Run the example
getComplexProduct();
javascript
import { Groq } from "groq-sdk";
import { z } from "zod"; // npm install zod

const client = new Groq();

// Define a schema with Zod
const ProductSchema = z.object({
  id: z.string(),
  name: z.string(),
  price: z.number().positive(),
  description: z.string(),
  in_stock: z.boolean(),
  tags: z.array(z.string()).default([]),
});

// Create a prompt that clearly defines the expected structure
const systemPrompt = `
You are a product catalog assistant. When asked about products,
always respond with valid JSON objects that match this structure:
{
  "id": "string",
  "name": "string",
  "price": number,
  "description": "string",
  "in_stock": boolean,
  "tags": ["string"]
}
Your response should ONLY contain the JSON object and nothing else.
`;

async function getStructuredResponse() {
  try {
    // Request structured data from the model
    const completion = await client.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "Tell me about a popular smartphone product" },
      ],
    });

// Extract the response
    const responseContent = completion.choices[0].message.content;
    
    // Parse and validate JSON
    const jsonData = JSON.parse(responseContent || "");
    const validatedData = ProductSchema.parse(jsonData);
    
    console.log("Validation successful! Structured data:");
    console.log(JSON.stringify(validatedData, null, 2));
    
    return validatedData;
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error("Schema validation failed:", error.errors);
    } else if (error instanceof SyntaxError) {
      console.error("JSON parsing failed: The model did not return valid JSON");
    } else {
      console.error("Error:", error);
    }
  }
}

// Run the example
getStructuredResponse();
javascript
import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const stream = await getGroqChatStream();
  for await (const chunk of stream) {
    // Print the completion returned by the LLM.
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
}

export async function getGroqChatStream() {
  return groq.chat.completions.create({
    //
    // Required parameters
    //
    messages: [
      // Set an optional system message. This sets the behavior of the
      // assistant and can be used to provide specific instructions for
      // how it should behave throughout the conversation.
      {
        role: "system",
        content: "You are a helpful assistant.",
      },
      // Set a user message for the assistant to respond to.
      {
        role: "user",
        content: "Explain the importance of fast language models",
      },
    ],

// The language model which will generate the completion.
    model: "openai/gpt-oss-20b",

//
    // Optional parameters
    //

// Controls randomness: lowering results in less random completions.
    // As the temperature approaches zero, the model will become deterministic
    // and repetitive.
    temperature: 0.5,

// The maximum number of tokens to generate. Requests can use up to
    // 2048 tokens shared between prompt and completion.
    max_completion_tokens: 1024,

// Controls diversity via nucleus sampling: 0.5 means half of all
    // likelihood-weighted options are considered.
    top_p: 1,

// A stop sequence is a predefined or user-specified text string that
    // signals an AI to stop generating content, ensuring its responses
    // remain focused and concise. Examples include punctuation marks and
    // markers like "[end]".
    stop: null,

// If set, partial message deltas will be sent.
    stream: true,
  });
}

main();
javascript
import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const stream = await getGroqChatStream();
  for await (const chunk of stream) {
    // Print the completion returned by the LLM.
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
}

export async function getGroqChatStream() {
  return groq.chat.completions.create({
    //
    // Required parameters
    //
    messages: [
      // Set an optional system message. This sets the behavior of the
      // assistant and can be used to provide specific instructions for
      // how it should behave throughout the conversation.
      {
        role: "system",
        content: "You are a helpful assistant.",
      },
      // Set a user message for the assistant to respond to.
      {
        role: "user",
        content:
          "Start at 1 and count to 10.  Separate each number with a comma and a space",
      },
    ],

// The language model which will generate the completion.
    model: "llama-3.3-70b-versatile",

//
    // Optional parameters
    //

// Controls randomness: lowering results in less random completions.
    // As the temperature approaches zero, the model will become deterministic
    // and repetitive.
    temperature: 0.5,

// The maximum number of tokens to generate. Requests can use up to
    // 2048 tokens shared between prompt and completion.
    max_completion_tokens: 1024,

// Controls diversity via nucleus sampling: 0.5 means half of all
    // likelihood-weighted options are considered.
    top_p: 1,

// A stop sequence is a predefined or user-specified text string that
    // signals an AI to stop generating content, ensuring its responses
    // remain focused and concise. Examples include punctuation marks and
    // markers like "[end]".
    //
    // For this example, we will use ", 6" so that the llm stops counting at 5.
    // If multiple stop values are needed, an array of string may be passed,
    // stop: [", 6", ", six", ", Six"]
    stop: ", 6",

// If set, partial message deltas will be sent.
    stream: true,
  });
}

main();
python
import os
from typing import List
from pydantic import BaseModel, Field 
import instructor 
from groq import Groq

**Examples:**

Example 1 (unknown):
```unknown
---

## Text Chat: Basic Validation Zod.doc (ts)

URL: https://console.groq.com/docs/text-chat/scripts/basic-validation-zod.doc
```

Example 2 (unknown):
```unknown
---

## Text Chat: Complex Schema Example (js)

URL: https://console.groq.com/docs/text-chat/scripts/complex-schema-example
```

Example 3 (unknown):
```unknown
---

## Text Chat: Basic Validation Zod (js)

URL: https://console.groq.com/docs/text-chat/scripts/basic-validation-zod
```

Example 4 (unknown):
```unknown
---

## Text Chat: Streaming Chat Completion (js)

URL: https://console.groq.com/docs/text-chat/scripts/streaming-chat-completion
```

---

## Final output

**URL:** llms-txt#final-output

print(response.choices[0].message.content)

---

## Get the session_id to run automation workflows to the same running session.

**URL:** llms-txt#get-the-session_id-to-run-automation-workflows-to-the-same-running-session.

session_id = configured_session.data.id

---

## Groq Client Libraries

**URL:** llms-txt#groq-client-libraries

**Contents:**
- Groq Python Library
  - Installation
- Groq JavaScript Library
  - Installation
  - Usage
- Groq Community Libraries
  - C#
  - Dart/Flutter
  - PHP
  - Ruby

Groq provides both a Python and JavaScript/Typescript client library.

## Groq Python Library

The [Groq Python library](https://pypi.org/project/groq/) provides convenient access to the Groq REST API from any Python 3.7+ application. The library includes type definitions for all request params and response fields, and offers both synchronous and asynchronous clients.

Use the library and your secret key to run:

While you can provide an `api_key` keyword argument, we recommend using [python-dotenv](https://github.com/theskumar/python-dotenv) to add `GROQ_API_KEY="My API Key"` to your `.env` file so that your API Key is not stored in source control.

The following response is generated:

## Groq JavaScript Library

The [Groq JavaScript library](https://www.npmjs.com/package/groq-sdk) provides convenient access to the Groq REST API from server-side TypeScript or JavaScript. The library includes type definitions for all request params and response fields, and offers both synchronous and asynchronous clients.

Use the library and your secret key to run:

The following response is generated:

## Groq Community Libraries

Groq encourages our developer community to build on our SDK. If you would like your library added, please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfkg3rPUnmZcTwRAS-MsmVHULMtD2I8LwsKPEasuqSsLlF0yA/viewform?usp=sf_link).

Please note that Groq does not verify the security of these projects. **Use at your own risk.**

- [jgravelle.GroqAPILibrary](https://github.com/jgravelle/GroqApiLibrary) by [jgravelle](https://github.com/jgravelle)

- [TAGonSoft.groq-dart](https://github.com/TAGonSoft/groq-dart) by [TAGonSoft](https://github.com/TAGonSoft)

- [lucianotonet.groq-php](https://github.com/lucianotonet/groq-php) by [lucianotonet](https://github.com/lucianotonet)

- [drnic.groq-ruby](https://github.com/drnic/groq-ruby) by [drnic](https://github.com/drnic)

---

## If there's a next cursor, use it to get the next page

**URL:** llms-txt#if-there's-a-next-cursor,-use-it-to-get-the-next-page

**Contents:**
- Batch: Retrieve (js)
- Batch: Retrieve (py)
- Batch: List Batches (js)
- Batch: Create Batch Job (py)
- Batch: Status (py)
- Batch: Upload File (js)
- Batch: Status (js)
- Batch: Multi Batch Status (js)
- Set up headers

if response.paging and response.paging.get("next_cursor"):
    next_response = client.batches.list(
        extra_query={
            "cursor": response.paging.get("next_cursor")
        }  # Use the next_cursor for next page
    )
    print("Next page:", next_response)
python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.files.content("file_01jh6xa97be52b7pg88czwrrwb")
response.write_to_file("batch_results.jsonl")
print("Batch file saved to batch_results.jsonl")
javascript
import Groq from 'groq-sdk';

const groq = new Groq();

async function main() {
  // Initial request - gets first page of batches
  const response = await groq.batches.list();
  console.log('First page:', response);

// If there's a next cursor, use it to get the next page
  if (response.paging && response.paging.next_cursor) {
    const nextResponse = await groq.batches.list({
      query: {
        cursor: response.paging.next_cursor, // Use the next_cursor for next page
      },
    });
    console.log('Next page:', nextResponse);
  }
}

main();
python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.batches.create(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id="file_01jh6x76wtemjr74t1fh0faj5t",
)
print(response.to_json())
python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.batches.retrieve("batch_01jh6xa7reempvjyh6n3yst2zw")

print(response.to_json())
javascript
import fs from 'fs';
import Groq from 'groq-sdk';

const groq = new Groq();

async function main() {
  const filePath = 'batch_file.jsonl'; // Path to your JSONL file

const response = await groq.files.create({
    purpose: 'batch',
    file: fs.createReadStream(filePath)
  });

console.log(response);
}

main();
javascript
import Groq from 'groq-sdk';

const groq = new Groq();

async function main() {
  const response = await groq.batches.retrieve("batch_01jh6xa7reempvjyh6n3yst2zw");
  console.log(response);
}

main();
javascript
async function main() {
  const batchIds = [
    "batch_01jh6xa7reempvjyh6n3yst111",
    "batch_01jh6xa7reempvjyh6n3yst222",
    "batch_01jh6xa7reempvjyh6n3yst333"
  ];

// Build query parameters using URLSearchParams
  const url = new URL('https://api.groq.com/openai/v1/batches');
  batchIds.forEach(id => url.searchParams.append('id', id));

try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      }
    });
    
    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error('Error:', error);
  }
}

main();
python
import os
import requests

**Examples:**

Example 1 (unknown):
```unknown
---

## Batch: Retrieve (js)

URL: https://console.groq.com/docs/batch/scripts/retrieve

import fs from 'fs';
import Groq from 'groq-sdk';

const groq = new Groq();

async function main() {
  const response = await groq.files.content("file_01jh6xa97be52b7pg88czwrrwb");
  fs.writeFileSync("batch_results.jsonl", await response.text());
  console.log("Batch file saved to batch_results.jsonl");
}

main();

---

## Batch: Retrieve (py)

URL: https://console.groq.com/docs/batch/scripts/retrieve.py
```

Example 2 (unknown):
```unknown
---

## Batch: List Batches (js)

URL: https://console.groq.com/docs/batch/scripts/list_batches
```

Example 3 (unknown):
```unknown
---

## Batch: Create Batch Job (py)

URL: https://console.groq.com/docs/batch/scripts/create_batch_job.py
```

Example 4 (unknown):
```unknown
---

## Batch: Status (py)

URL: https://console.groq.com/docs/batch/scripts/status.py
```

---

## Initialize Groq instrumentation

**URL:** llms-txt#initialize-groq-instrumentation

GroqInstrumentor().instrument(tracer_provider=tracer_provider)

---

## Initialize the Groq client

**URL:** llms-txt#initialize-the-groq-client

---

## Initial request - gets first page of batches

**URL:** llms-txt#initial-request---gets-first-page-of-batches

response = client.batches.list()
print("First page:", response)

---

## LoRA Inference on Groq

**URL:** llms-txt#lora-inference-on-groq

**Contents:**
- Enterprise Feature
- Why LoRA vs. Base Model?
  - Why LoRA vs. Traditional Fine-tuning?
- LoRA Options on GroqCloud
  - Two Hosting Modalities
  - LoRAs (Public Cloud)
  - LoRAs (Dedicated Instance)
  - Supported Models
- LoRA Pricing
- Getting Started

Groq provides inference services for pre-made Low-Rank Adaptation (LoRA) adapters. LoRA is a Parameter-efficient Fine-tuning (PEFT) technique that customizes model behavior without altering base model weights. Upload your existing LoRA adapters to run specialized inference while maintaining the performance and efficiency of Groq's infrastructure.

**Note**: Groq offers LoRA inference services only. We do not provide LoRA fine-tuning services - you must create your LoRA adapters externally using other providers or tools.

With LoRA inference on Groq, you can:
- **Run inference** with your pre-made LoRA adapters
- **Deploy multiple specialized adapters** alongside a single base model
- **Maintain high performance** without compromising inference speed
- **Leverage existing fine-tuned models** created with external tools

## Enterprise Feature

LoRA is available exclusively to enterprise-tier customers. To get started with LoRA on GroqCloud, please reach out to [our enterprise team](https://groq.com/enterprise-access).

## Why LoRA vs. Base Model?

Compared to using just the base model, LoRA adapters offer significant advantages:

- **Task-Specific Optimization**: Tune model outputs to your particular use case, enabling increased accuracy and quality of responses
- **Domain Expertise**: Adapt models to understand industry-specific terminology, context, and requirements
- **Consistent Behavior**: Ensure predictable outputs that align with your business needs and brand voice
- **Performance Maintenance**: Achieve customization without compromising the high-speed inference that Groq is known for

### Why LoRA vs. Traditional Fine-tuning?

LoRA provides several key advantages over traditional fine-tuning approaches:

**Lower Total Cost of Ownership**

LoRA reduces fine-tuning costs significantly by avoiding full base model fine-tuning. This efficiency makes it cost-effective to customize models at scale.

**Rapid Deployment with High Performance**

Smaller, task-specific LoRA adapters can match or exceed the performance of fully fine-tuned models while delivering faster inference. This translates to quicker experimentation, iteration, and real-world impact.

**Non-Invasive Model Adaptation**

Since LoRA adapters don't require changes to the base model, you avoid the complexity and liability of managing and validating a fully retrained system. Adapters are modular, independently versioned, and easily replaceable as your data evolves—simplifying governance and compliance.

**Full Control, Less Risk**

Customers keep control of how and when updates happen—no retraining, no surprise behavior changes. Just lightweight, swappable adapters that fit into existing systems with minimal disruption. And with self-service APIs, updating adapters is quick, intuitive, and doesn't require heavy engineering lift.

## LoRA Options on GroqCloud

### Two Hosting Modalities

Groq supports LoRAs through two deployment options: 
1. [LoRAs in our public cloud](#loras-public-cloud)
2. [LoRAs on a dedicated instance](#loras-dedicated-instance)

### LoRAs (Public Cloud)

Pay-per-token usage model with no dedicated hardware requirements, ideal for customers with a small number of LoRA adapters across different tasks like customer support, document summarization, and translation.

- No dedicated hardware requirements - pay per token usage
- Shared instance capabilities across customers with potential rate limiting
- Less consistent latency/throughput compared to dedicated instances
- Gradual rollout to enterprise customers only via [enterprise access form](https://groq.com/enterprise-access/)

### LoRAs (Dedicated Instance)

Deployed on dedicated Groq hardware instances purchased by the customer, providing optimized performance for multiple LoRA adapters and consistent inference speeds, best suited for high-traffic scenarios or customers serving personalized adapters to many end users.

- Dedicated hardware instances optimized for LoRA performance
- More consistent performance and lower average latency
- No LoRA-specific rate limiting
- Ideal for SaaS platforms with dozens of internal use cases or hundreds of customer-specific adapters

LoRA support is currently available for the following models:

| Model ID                        | Model                          | Base Model |
|---------------------------------|--------------------------------|------------|
| llama-3.1-8b-instant                  | Llama 3.1 8B | meta-llama/Llama-3.1-8B-Instruct |

Please reach out to our [enterprise support team](https://groq.com/enterprise-access) for additional model support.

Please reach out to our [enterprise support team](https://groq.com/enterprise-access) for pricing.

To begin using LoRA on GroqCloud:

1. **Contact Enterprise Sales**: [Reach out](https://groq.com/enterprise-access) to become an enterprise-tier customer
2. **Request LoRA Access**: Inform the team that you would like access to LoRA support
3. **Create Your LoRA Adapters**: Use external providers or tools to fine-tune Groq-supported base models (exact model versions required)
4. **Upload Adapters**: Use the self-serve portal to upload your LoRA adapters to GroqCloud
5. **Deploy**: Call the unique model ID created for your specific LoRA adapter(s)

**Important**: You must fine-tune the exact base model versions that Groq supports for your LoRA adapters to work properly.

## Using the Fine-Tuning API

Once you have access to LoRA, you can upload and deploy your adapters using Groq's Fine-Tuning API. This process involves two API calls: one to upload your LoRA adapter files and another to register them as a fine-tuned model. When you upload your LoRA adapters, Groq will store and process your files to provide this service. LoRA adapters are your Customer Data and will only be available for your organization's use.

- **Supported models**: Text generation models only
- **Supported ranks**: 8, 16, 32, and 64 only
- **File format**: ZIP file containing exactly 2 files

**Note**: Cold start times are proportional to the LoRA rank. Higher ranks (32, 64) will take longer to load initially but have no impact on inference performance once loaded.

### Step 1: Prepare Your LoRA Adapter Files

Create a ZIP file containing exactly these 2 files:

1. **`adapter_model.safetensors`** - A safetensors file containing your LoRA weights in float16 format
2. **`adapter_config.json`** - A JSON configuration file with required fields:
   - `"lora_alpha"`: (integer or float) The LoRA alpha parameter
   - `"r"`: (integer) The rank of your LoRA adapter (must be 8, 16, 32, or 64)

### Step 2: Upload the LoRA Adapter Files

Upload your ZIP file to the `/files` endpoint with `purpose="fine_tuning"`:

This returns a file ID that you'll use in the next step:

### Step 3: Register as Fine-Tuned Model

Use the file ID to register your LoRA adapter as a fine-tuned model:

This returns your unique model ID:

### Step 4: Use Your LoRA Model

Use the returned `fine_tuned_model` ID in your inference requests just like any other model:

## Frequently Asked Questions
### Does Groq offer LoRA fine-tuning services?

No. Groq provides LoRA inference services only. Customers must create their LoRA adapters externally using fine-tuning providers or tools (e.g., Hugging Face PEFT, Unsloth, or custom solutions) and then upload their pre-made adapters to Groq for inference. You must fine-tune the exact base model versions that Groq supports.

### Will LoRA support be available to Developer tier customers?

Not at this time. LoRA support is currently exclusive to enterprise tier customers. Stay tuned for updates.

### Does Groq have recommended fine-tuning providers?

Stay tuned for further updates on recommended fine-tuning providers.

### How do I get access to LoRA on GroqCloud?

[Contact our enterprise team](https://groq.com/enterprise-access) to discuss your LoRA requirements and get started.

### How long are LoRA adapter files retained for?

Your uploaded LoRA adapter files are stored and accessible solely to your organization for the entire time you use the LoRAs service.

- **Keep LoRA rank low (8 or 16)** to minimize cold start times - higher ranks increase loading latency
- **Use float16 precision** when loading the base model during fine-tuning to maintain optimal inference accuracy
- **Avoid 4-bit quantization** during LoRA training as it may cause small accuracy drops during inference
- **Save LoRA weights in float16 format** in your `adapter_model.safetensors` file
- **Test different ranks** to find the optimal balance between adaptation quality and cold start performance

## Libraries: Library Usage Response (json)

URL: https://console.groq.com/docs/libraries/scripts/library-usage-response.json

## Libraries: Library Usage (js)

URL: https://console.groq.com/docs/libraries/scripts/library-usage

## This is the default and can be omitted

URL: https://console.groq.com/docs/libraries/scripts/library-usage.py

## Groq Client Libraries

URL: https://console.groq.com/docs/libraries

**Examples:**

Example 1 (bash):
```bash
curl --location 'https://api.groq.com/openai/v1/files' \
--header "Authorization: Bearer ${TOKEN}" \
--form "file=@<file-name>.zip" \
--form 'purpose="fine_tuning"'
```

Example 2 (json):
```json
{
  "id": "file_01jxnqc8hqebx343rnkyxw47e",
  "object": "file",
  "bytes": 155220077,
  "created_at": 1749854594,
  "filename": "<file-name>.zip",
  "purpose": "fine_tuning"
}
```

Example 3 (bash):
```bash
curl --location 'https://api.groq.com/v1/fine_tunings' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer ${TOKEN}" \
--data '{
    "input_file_id": "<file-id>",
    "name": "my-lora-adapter",
    "type": "lora",
    "base_model": "llama-3.1-8b-instant"
}'
```

Example 4 (json):
```json
{
  "id": "ft_01jxx7abvdf6pafdthfbfmb9gy",
  "object": "fine_tuning",
  "data": {
    "name": "my-lora-adapter",
    "base_model": "llama-3.1-8b-instant",
    "type": "lora",
    "fine_tuned_model": "ft:llama-3.1-8b-instant:org_01hqed9y3fexcrngzqm9qh6ya9/my-lora-adapter-ef36419a0010"
  }
}
```

---

## Make the request

**URL:** llms-txt#make-the-request

**Contents:**
- Batch: Upload File (py)
- Groq Batch API

response = requests.get(url, headers=headers, params=params)
print(response.json())
python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

file_path = "batch_file.jsonl"
response = client.files.create(file=open(file_path, "rb"), purpose="batch")

URL: https://console.groq.com/docs/batch

**Examples:**

Example 1 (unknown):
```unknown
---

## Batch: Upload File (py)

URL: https://console.groq.com/docs/batch/scripts/upload_file.py
```

---

## No need for try/except or manual validation - instructor handles it!

**URL:** llms-txt#no-need-for-try/except-or-manual-validation---instructor-handles-it!

**Contents:**
- Set your API key

print(f"Recipe: {recipe.title}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Cook time: {recipe.cook_time_minutes} minutes")
print("\nIngredients:")
for ingredient in recipe.ingredients:
    print(f"- {ingredient.quantity} {ingredient.unit} {ingredient.name}")
print("\nInstructions:")
for i, step in enumerate(recipe.instructions, 1):
    print(f"{i}. {step}") 
python
import os
import json
from groq import Groq

**Examples:**

Example 1 (unknown):
```unknown
---

## Set your API key

URL: https://console.groq.com/docs/text-chat/scripts/prompt-engineering.py
```

---

## OpenAI Compatibility

**URL:** llms-txt#openai-compatibility

**Contents:**
- Configuring OpenAI to Use Groq API
- Currently Unsupported OpenAI Features
  - Text Completions
  - Temperature
  - Audio Transcription and Translation
- Responses API
  - Feedback
- Next Steps
- Prompt Caching: Multi Turn Conversations (js)
- Prompt Caching: Tool Definitions And Use (js)

We designed Groq API to be mostly compatible with OpenAI's client libraries, making it easy to 
configure your existing applications to run on Groq and try our inference speed.
<br />
We also have our own [Groq Python and Groq TypeScript libraries](/docs/libraries) that we encourage you to use.

## Configuring OpenAI to Use Groq API
To start using Groq with OpenAI's client libraries, pass your Groq API key to the `api_key` parameter
and change the `base_url` to `https://api.groq.com/openai/v1`:

You can find your API key [here](/keys).

## Currently Unsupported OpenAI Features

Note that although Groq API is mostly OpenAI compatible, there are a few features we don't support just yet:

### Text Completions
The following fields are currently not supported and will result in a 400 error (yikes) if they are supplied:
- `logprobs`

- If `N` is supplied, it must be equal to 1.

### Temperature
If you set a `temperature` value of 0, it will be converted to `1e-8`. If you run into any issues, please try setting the value to a float32 `> 0` and `<= 2`.

### Audio Transcription and Translation
The following values are not supported:
- `vtt`
- `srt`

Groq also supports the [Responses API](/docs/responses-api), which is a more advanced interface for generating model responses that supports both text and image inputs while producing text outputs. You can build stateful conversations by using previous responses as context, and extend your model's capabilities through function calling to connect with external systems and data sources.

### Feedback
If you'd like to see support for such features as the above on Groq API, please reach out to us and let us know by submitting a "Feature Request" via "Chat with us" in the menu after clicking your organization in the top right. We really value your feedback and would love to hear from you!

Migrate your prompts to open-source models using our [model migration guide](/docs/prompting/model-migration), or learn more about prompting in our [prompting guide](/docs/prompting).

## Prompt Caching: Multi Turn Conversations (js)

URL: https://console.groq.com/docs/prompt-caching/scripts/multi-turn-conversations

## Prompt Caching: Tool Definitions And Use (js)

URL: https://console.groq.com/docs/prompt-caching/scripts/tool-definitions-and-use

## Initial conversation with system message and first user input

URL: https://console.groq.com/docs/prompt-caching/scripts/multi-turn-conversations.py

## First request - creates cache for the large legal document

URL: https://console.groq.com/docs/prompt-caching/scripts/large-prompts-and-context.py

## Prompt Caching: Large Prompts And Context (js)

URL: https://console.groq.com/docs/prompt-caching/scripts/large-prompts-and-context

## Define comprehensive tool set

URL: https://console.groq.com/docs/prompt-caching/scripts/tool-definitions-and-use.py

from groq import Groq

**Examples:**

Example 1 (javascript):
```javascript
import Groq from "groq-sdk";

const groq = new Groq();

async function multiTurnConversation() {
  // Initial conversation with system message and first user input
  const initialMessages = [
    {
      role: "system",
      content: "You are a helpful AI assistant that provides detailed explanations about complex topics. Always provide comprehensive answers with examples and context."
    },
    {
      role: "user",
      content: "What is quantum computing?"
    }
  ];

  // First request - creates cache for system message
  const firstResponse = await groq.chat.completions.create({
    messages: initialMessages,
    model: "moonshotai/kimi-k2-instruct-0905"
  });

  console.log("First response:", firstResponse.choices[0].message.content);
  console.log("Usage:", firstResponse.usage);

  // Continue conversation - system message and previous context will be cached
  const conversationMessages = [
    ...initialMessages,
    firstResponse.choices[0].message,
    {
      role: "user",
      content: "Can you give me a simple example of how quantum superposition works?"
    }
  ];

  const secondResponse = await groq.chat.completions.create({
    messages: conversationMessages,
    model: "moonshotai/kimi-k2-instruct-0905"
  });

  console.log("Second response:", secondResponse.choices[0].message.content);
  console.log("Usage:", secondResponse.usage);

  // Continue with third turn
  const thirdTurnMessages = [
    ...conversationMessages,
    secondResponse.choices[0].message,
    {
      role: "user",
      content: "How does this relate to quantum entanglement?"
    }
  ];

  const thirdResponse = await groq.chat.completions.create({
    messages: thirdTurnMessages,
    model: "moonshotai/kimi-k2-instruct-0905"
  });

  console.log("Third response:", thirdResponse.choices[0].message.content);
  console.log("Usage:", thirdResponse.usage);
}

multiTurnConversation().catch(console.error);
```

Example 2 (javascript):
```javascript
import Groq from "groq-sdk";

const groq = new Groq();

// Define comprehensive tool set
const tools = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA"
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
            description: "The unit of temperature"
          }
        },
        required: ["location"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "calculate_math",
      description: "Perform mathematical calculations",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "Mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'"
          }
        },
        required: ["expression"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "search_web",
      description: "Search the web for current information",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query"
          },
          num_results: {
            type: "integer",
            description: "Number of results to return",
            minimum: 1,
            maximum: 10,
            default: 5
          }
        },
        required: ["query"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_time",
      description: "Get the current time in a specific timezone",
      parameters: {
        type: "object",
        properties: {
          timezone: {
            type: "string",
            description: "Timezone identifier, e.g. 'America/New_York' or 'UTC'"
          }
        },
        required: ["timezone"]
      }
    }
  }
];

async function useToolsWithCaching() {
  // First request - creates cache for all tool definitions
  const systemPrompt = "You are a helpful assistant with access to various tools. Use the appropriate tools to answer user questions accurately.";
  const firstRequest = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "user",
        content: "What's the weather like in New York City?"
      }
    ],
    model: "moonshotai/kimi-k2-instruct-0905",
    tools: tools
  });

  console.log("First request response:", firstRequest.choices[0].message);
  console.log("Usage:", firstRequest.usage);

  // Check if the model wants to use tools
  if (firstRequest.choices[0].message.tool_calls) {
    console.log("Tool calls requested:", firstRequest.choices[0].message.tool_calls);
  }

  // Second request - tool definitions will be cached
  const secondRequest = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "user",
        content: "Can you calculate the square root of 144 and tell me what time it is in Tokyo?"
      }
    ],
    model: "moonshotai/kimi-k2-instruct-0905",
    tools: tools
  });

  console.log("Second request response:", secondRequest.choices[0].message);
  console.log("Usage:", secondRequest.usage);

  if (secondRequest.choices[0].message.tool_calls) {
    console.log("Tool calls requested:", secondRequest.choices[0].message.tool_calls);
  }

  // Third request - same tool definitions cached
  const thirdRequest = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "user",
        content: "Search for recent news about artificial intelligence developments."
      }
    ],
    model: "moonshotai/kimi-k2-instruct-0905",
    tools: tools
  });

  console.log("Third request response:", thirdRequest.choices[0].message);
  console.log("Usage:", thirdRequest.usage);

  if (thirdRequest.choices[0].message.tool_calls) {
    console.log("Tool calls requested:", thirdRequest.choices[0].message.tool_calls);
  }
}

useToolsWithCaching().catch(console.error);
```

Example 3 (python):
```python
import os
from groq import Groq

client = Groq()

def multi_turn_conversation():
    # Initial conversation with system message and first user input
    initial_messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that provides detailed explanations about complex topics. Always provide comprehensive answers with examples and context."
        },
        {
            "role": "user",
            "content": "What is quantum computing?"
        }
    ]

    # First request - creates cache for system message
    first_response = client.chat.completions.create(
        messages=initial_messages,
        model="moonshotai/kimi-k2-instruct-0905"
    )

    print("First response:", first_response.choices[0].message.content)
    print("Usage:", first_response.usage)

    # Continue conversation - system message and previous context will be cached
    conversation_messages = [
        *initial_messages,
        first_response.choices[0].message,
        {
            "role": "user",
            "content": "Can you give me a simple example of how quantum superposition works?"
        }
    ]

    second_response = client.chat.completions.create(
        messages=conversation_messages,
        model="moonshotai/kimi-k2-instruct-0905"
    )

    print("Second response:", second_response.choices[0].message.content)
    print("Usage:", second_response.usage)

    # Continue with third turn
    third_turn_messages = [
        *conversation_messages,
        second_response.choices[0].message,
        {
            "role": "user",
            "content": "How does this relate to quantum entanglement?"
        }
    ]

    third_response = client.chat.completions.create(
        messages=third_turn_messages,
        model="moonshotai/kimi-k2-instruct-0905"
    )

    print("Third response:", third_response.choices[0].message.content)
    print("Usage:", third_response.usage)

if __name__ == "__main__":
    multi_turn_conversation()
```

Example 4 (python):
```python
from groq import Groq

client = Groq()

def analyze_legal_document():
    # First request - creates cache for the large legal document
    system_prompt = """
    You are a legal expert AI assistant. Analyze the following legal document and provide detailed insights.\n\nLEGAL DOCUMENT:
<entire contents of large legal document>
    """

    first_analysis = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "What are the key provisions regarding user account termination in this agreement?"
            }
        ],
        model="moonshotai/kimi-k2-instruct-0905"
    )

    print("First analysis:", first_analysis.choices[0].message.content)
    print("Usage:", first_analysis.usage)

    # Second request - legal document will be cached, only new question processed
    second_analysis = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "What are the intellectual property rights implications for users who submit content?"
            }
        ],
        model="moonshotai/kimi-k2-instruct-0905"
    )

    print("Second analysis:", second_analysis.choices[0].message.content)
    print("Usage:", second_analysis.usage)

    # Third request - same large context, different question
    third_analysis = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "Are there any concerning limitations of liability clauses that users should be aware of?"
            }
        ],
        model="moonshotai/kimi-k2-instruct-0905"
    )

    print("Third analysis:", third_analysis.choices[0].message.content)
    print("Usage:", third_analysis.usage)

if __name__ == "__main__":
    analyze_legal_document()
```

---

## Optional: Set a tracking URI and an experiment name if you have a tracking server

**URL:** llms-txt#optional:-set-a-tracking-uri-and-an-experiment-name-if-you-have-a-tracking-server

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Groq")

---

## Or: "What's the current weather in San Francisco?"

**URL:** llms-txt#or:-"what's-the-current-weather-in-san-francisco?"

---

## Patch the client with instructor

**URL:** llms-txt#patch-the-client-with-instructor

instructor_client = instructor.patch(client)

---

## Print all tool calls

**URL:** llms-txt#print-all-tool-calls

---

## Print the final content

**URL:** llms-txt#print-the-final-content

print(message.content)

---

## Print the validated complex object

**URL:** llms-txt#print-the-validated-complex-object

**Contents:**
- Text Chat: Complex Schema Example.doc (ts)
- Text Chat: Basic Chat Completion (js)
- Text Chat: Instructor Example.doc (ts)
- Text Chat: Instructor Example (js)
- Text Chat: Json Mode (js)
- Required parameters

print(f"Product: {product.name}")
print(f"Description: {product.description[:100]}...")
print(f"Variants: {len(product.variants)}")
print(f"Reviews: {len(product.reviews)}")
print(f"Manufacturer: {product.manufacturer.get('name')}")
print("\nManufacturer Contact:")
contact_info = product.manufacturer.get('contact_info')
if isinstance(contact_info, ContactInfo):
    print(f"  Email: {contact_info.email}")
    print(f"  Address: {contact_info.address.city}, {contact_info.address.country}") 
javascript
import Instructor from "@instructor-ai/instructor"; // npm install @instructor-ai/instructor
import { Groq } from "groq-sdk";
import { z } from "zod"; // npm install zod

// Set up the client with Instructor
const groq = new Groq();
const instructor = Instructor({
  client: groq,
  mode: "TOOLS"
})

// Define a complex nested schema
const AddressSchema = z.object({
  street: z.string(),
  city: z.string(),
  state: z.string(),
  zip_code: z.string(),
  country: z.string(),
});

const ContactInfoSchema = z.object({
  email: z.string().email(),
  phone: z.string().optional(),
  address: AddressSchema,
});

const ProductVariantSchema = z.object({
  id: z.string(),
  name: z.string(),
  price: z.number().positive(),
  inventory_count: z.number().int().nonnegative(),
  attributes: z.record(z.string()),
});

const ProductReviewSchema = z.object({
  user_id: z.string(),
  rating: z.number().min(1).max(5),
  comment: z.string(),
  date: z.string(),
});

const ManufacturerSchema = z.object({
  name: z.string(),
  founded: z.string(),
  contact_info: ContactInfoSchema,
});

const ProductSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  main_category: z.string(),
  subcategories: z.array(z.string()),
  variants: z.array(ProductVariantSchema),
  reviews: z.array(ProductReviewSchema),
  average_rating: z.number().min(1).max(5),
  manufacturer: ManufacturerSchema,
});

// Infer TypeScript types from Zod schemas
type Product = z.infer<typeof ProductSchema>;

// System prompt with clear instructions about the complex structure
const systemPrompt = `
You are a product catalog API. Generate a detailed product with ALL required fields.
Your response must be a valid JSON object matching the schema I will use to validate it.
`;

async function getComplexProduct(): Promise<Product | undefined> {
  try {
    // Use instructor to create and validate in one step
    const product = await instructor.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_model: {
        name: "Product",
        schema: ProductSchema,
      },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "Give me details about a high-end camera product" },
      ],
      max_retries: 3,
    });

// Print the validated complex object
    console.log(`Product: ${product.name}`);
    console.log(`Description: ${product.description.substring(0, 100)}...`);
    console.log(`Variants: ${product.variants.length}`);
    console.log(`Reviews: ${product.reviews.length}`);
    console.log(`Manufacturer: ${product.manufacturer.name}`);
    console.log(`\nManufacturer Contact:`);
    console.log(`  Email: ${product.manufacturer.contact_info.email}`);
    console.log(`  Address: ${product.manufacturer.contact_info.address.city}, ${product.manufacturer.contact_info.address.country}`);

return product;
  } catch (error) {
    console.error("Error:", error);
    return undefined;
  }
}

// Run the example
getComplexProduct();
javascript
import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const completion = await getGroqChatCompletion();
  console.log(completion.choices[0]?.message?.content || "");
}

export const getGroqChatCompletion = async () => {
  return groq.chat.completions.create({
    messages: [
      // Set an optional system message. This sets the behavior of the
      // assistant and can be used to provide specific instructions for
      // how it should behave throughout the conversation.
      {
        role: "system",
        content: "You are a helpful assistant.",
      },
      // Set a user message for the assistant to respond to.
      {
        role: "user",
        content: "Explain the importance of fast language models",
      },
    ],
    model: "openai/gpt-oss-20b",
  });
};

main();
javascript
import Instructor from "@instructor-ai/instructor"; // npm install @instructor-ai/instructor
import { Groq } from "groq-sdk";
import { z } from "zod"; // npm install zod

// Set up the Groq client with Instructor
const client = new Groq();
const instructor = Instructor({
  client,
  mode: "TOOLS"
});

// Define your schema with Zod
const RecipeIngredientSchema = z.object({
  name: z.string(),
  quantity: z.string(),
  unit: z.string().describe("The unit of measurement, like cup, tablespoon, etc."),
});

const RecipeSchema = z.object({
  title: z.string(),
  description: z.string(),
  prep_time_minutes: z.number().int().positive(),
  cook_time_minutes: z.number().int().positive(),
  ingredients: z.array(RecipeIngredientSchema),
  instructions: z.array(z.string()).describe("Step by step cooking instructions"),
});

// Infer TypeScript types from Zod schemas
type Recipe = z.infer<typeof RecipeSchema>;

async function getRecipe(): Promise<Recipe | undefined> {
  try {
    // Request structured data with automatic validation
    const recipe = await instructor.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_model: {
        name: "Recipe",
        schema: RecipeSchema,
      },
      messages: [
        { role: "user", content: "Give me a recipe for chocolate chip cookies" },
      ],
      max_retries: 2, // Instructor will retry if validation fails
    });

// No need for try/catch or manual validation - instructor handles it!
    console.log(`Recipe: ${recipe.title}`);
    console.log(`Prep time: ${recipe.prep_time_minutes} minutes`);
    console.log(`Cook time: ${recipe.cook_time_minutes} minutes`);
    console.log("\nIngredients:");
    recipe.ingredients.forEach((ingredient) => {
      console.log(`- ${ingredient.quantity} ${ingredient.unit} ${ingredient.name}`);
    });
    console.log("\nInstructions:");
    recipe.instructions.forEach((step, index) => {
      console.log(`${index + 1}. ${step}`);
    });

return recipe;
  } catch (error) {
    console.error("Error:", error);
    return undefined;
  }
}

// Run the example
getRecipe();
javascript
import Instructor from "@instructor-ai/instructor"; // npm install @instructor-ai/instructor
import { Groq } from "groq-sdk";
import { z } from "zod"; // npm install zod

// Set up the Groq client with Instructor
const client = new Groq();
const instructor = Instructor({
  client,
  mode: "TOOLS"
});

// Define your schema with Zod
const RecipeIngredientSchema = z.object({
  name: z.string(),
  quantity: z.string(),
  unit: z.string().describe("The unit of measurement, like cup, tablespoon, etc."),
});

const RecipeSchema = z.object({
  title: z.string(),
  description: z.string(),
  prep_time_minutes: z.number().int().positive(),
  cook_time_minutes: z.number().int().positive(),
  ingredients: z.array(RecipeIngredientSchema),
  instructions: z.array(z.string()).describe("Step by step cooking instructions"),
});

async function getRecipe() {
  try {
    // Request structured data with automatic validation
    const recipe = await instructor.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_model: {
        name: "Recipe",
        schema: RecipeSchema,
      },
      messages: [
        { role: "user", content: "Give me a recipe for chocolate chip cookies" },
      ],
      max_retries: 2, // Instructor will retry if validation fails
    });

// No need for try/catch or manual validation - instructor handles it!
    console.log(`Recipe: ${recipe.title}`);
    console.log(`Prep time: ${recipe.prep_time_minutes} minutes`);
    console.log(`Cook time: ${recipe.cook_time_minutes} minutes`);
    console.log("\nIngredients:");
    recipe.ingredients.forEach((ingredient) => {
      console.log(`- ${ingredient.quantity} ${ingredient.unit} ${ingredient.name}`);
    });
    console.log("\nInstructions:");
    recipe.instructions.forEach((step, index) => {
      console.log(`${index + 1}. ${step}`);
    });

return recipe;
  } catch (error) {
    console.error("Error:", error);
  }
}

// Run the example
getRecipe();
javascript
import Groq from "groq-sdk";
const groq = new Groq();

// Define the JSON schema for recipe objects
// This is the schema that the model will use to generate the JSON object, 
// which will be parsed into the Recipe class.
const schema = {
  $defs: {
    Ingredient: {
      properties: {
        name: { title: "Name", type: "string" },
        quantity: { title: "Quantity", type: "string" },
        quantity_unit: {
          anyOf: [{ type: "string" }, { type: "null" }],
          title: "Quantity Unit",
        },
      },
      required: ["name", "quantity", "quantity_unit"],
      title: "Ingredient",
      type: "object",
    },
  },
  properties: {
    recipe_name: { title: "Recipe Name", type: "string" },
    ingredients: {
      items: { $ref: "#/$defs/Ingredient" },
      title: "Ingredients",
      type: "array",
    },
    directions: {
      items: { type: "string" },
      title: "Directions",
      type: "array",
    },
  },
  required: ["recipe_name", "ingredients", "directions"],
  title: "Recipe",
  type: "object",
};

// Ingredient class representing a single recipe ingredient
class Ingredient {
  constructor(name, quantity, quantity_unit) {
    this.name = name;
    this.quantity = quantity;
    this.quantity_unit = quantity_unit || null;
  }
}

// Recipe class representing a complete recipe
class Recipe {
  constructor(recipe_name, ingredients, directions) {
    this.recipe_name = recipe_name;
    this.ingredients = ingredients;
    this.directions = directions;
  }
}

// Generates a recipe based on the recipe name
export async function getRecipe(recipe_name) {
  // Pretty printing improves completion results
  const jsonSchema = JSON.stringify(schema, null, 4);
  const chat_completion = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: `You are a recipe database that outputs recipes in JSON.\n'The JSON object must use the schema: ${jsonSchema}`,
      },
      {
        role: "user",
        content: `Fetch a recipe for ${recipe_name}`,
      },
    ],
    model: "openai/gpt-oss-20b",
    temperature: 0,
    stream: false,
    response_format: { type: "json_object" },
  });

const recipeJson = JSON.parse(chat_completion.choices[0].message.content);

// Map the JSON ingredients to the Ingredient class
  const ingredients = recipeJson.ingredients.map((ingredient) => {
    return new Ingredient(ingredient.name, ingredient.quantity, ingredient.quantity_unit);
  });

// Return the recipe object
  return new Recipe(recipeJson.recipe_name, ingredients, recipeJson.directions);
}

// Prints a recipe to the console with nice formatting
function printRecipe(recipe) {
  console.log("Recipe:", recipe.recipe_name);
  console.log();

console.log("Ingredients:");
  recipe.ingredients.forEach((ingredient) => {
    console.log(
      `- ${ingredient.name}: ${ingredient.quantity} ${
        ingredient.quantity_unit || ""
      }`,
    );
  });
  console.log();

console.log("Directions:");
  recipe.directions.forEach((direction, step) => {
    console.log(`${step + 1}. ${direction}`);
  });
}

// Main function that generates and prints a recipe
export async function main() {
  const recipe = await getRecipe("apple pie");
  printRecipe(recipe);
}

main();
python
from groq import Groq

stream = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],

# The language model which will generate the completion.
    model="llama-3.3-70b-versatile",

#
    # Optional parameters
    #

# Controls randomness: lowering results in less random completions.
    # As the temperature approaches zero, the model will become deterministic
    # and repetitive.
    temperature=0.5,

# The maximum number of tokens to generate. Requests can use up to
    # 2048 tokens shared between prompt and completion.
    max_completion_tokens=1024,

# Controls diversity via nucleus sampling: 0.5 means half of all
    # likelihood-weighted options are considered.
    top_p=1,

# A stop sequence is a predefined or user-specified text string that
    # signals an AI to stop generating content, ensuring its responses
    # remain focused and concise. Examples include punctuation marks and
    # markers like "[end]".
    stop=None,

# If set, partial message deltas will be sent.
    stream=True,
)

**Examples:**

Example 1 (unknown):
```unknown
---

## Text Chat: Complex Schema Example.doc (ts)

URL: https://console.groq.com/docs/text-chat/scripts/complex-schema-example.doc
```

Example 2 (unknown):
```unknown
---

## Text Chat: Basic Chat Completion (js)

URL: https://console.groq.com/docs/text-chat/scripts/basic-chat-completion
```

Example 3 (unknown):
```unknown
---

## Text Chat: Instructor Example.doc (ts)

URL: https://console.groq.com/docs/text-chat/scripts/instructor-example.doc
```

Example 4 (unknown):
```unknown
---

## Text Chat: Instructor Example (js)

URL: https://console.groq.com/docs/text-chat/scripts/instructor-example
```

---

## Process tool calls

**URL:** llms-txt#process-tool-calls

messages.append(response_message)

available_functions = {
    "get_temperature": get_temperature,
    "get_weather_condition": get_weather_condition,
}

for tool_call in tool_calls:
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(**function_args)

messages.append(
        {
            "role": "tool",
            "content": str(function_response),
            "tool_calls_id": tool_call.id,
        }
    )

---

## Projects

**URL:** llms-txt#projects

**Contents:**
- Why Use Projects?
- Project Structure
- Getting Started
  - Creating Your First Project
  - Project Selection
- Rate Limit Management
  - Understanding Rate Limits
  - Configuring Rate Limits
  - Example: Rate Limits Across Projects
- Usage Tracking

Projects provide organizations with a powerful framework for managing multiple applications, environments, and teams within a single Groq account. By organizing your work into projects, you can isolate workloads to gain granular control over resources, costs, access permissions, and usage tracking on a per-project basis.

## Why Use Projects?
- **Isolation and Organization:** Projects create logical boundaries between different applications, environments (development, staging, production), and use cases. This prevents resource conflicts and enables clear separation of concerns across your organization.
- **Cost Control and Visibility:** Track spending, usage patterns, and resource consumption at the project level. This granular visibility enables accurate cost allocation, budget management, and ROI analysis for specific initiatives.
- **Team Collaboration:** Control who can access what resources through project-based permissions. Teams can work independently within their projects while maintaining organizational oversight and governance.
- **Operational Excellence:** Configure rate limits, monitor performance, and debug issues at the project level. This enables optimized resource allocation and simplified troubleshooting workflows.

## Project Structure
Projects inherit settings and permissions from your organization while allowing project-specific customization. Your organization-level role determines your maximum permissions within any project.

Each project acts as an isolated workspace containing:

- **API Keys:** Project-specific credentials for secure access
- **Rate Limits:** Customizable quotas for each available model
- **Usage Data:** Consumption metrics, costs, and request logs
- **Team Access:** Role-based permissions for project members

The following are the roles that are inherited from your organization along with their permissions within a project:

- **Owner:** Full access to creating, updating, and deleting projects, modifying limits for models within projects, managing API keys, viewing usage and spending data across all projects, and managing project access.
- **Developer:** Currently same as Owner.
- **Reader:** Read-only access to projects and usage metrics, logs, and spending data.

## Getting Started
### Creating Your First Project
**1. Access Projects**: Navigate to the **Projects** section at the top lefthand side of the Console. You will see a dropdown that looks like **Organization** / **Projects**.
<br />
**2. Create Project:** Click the rightside **Projects** dropdown and click **Create Project** to create a new project by inputting a project name. You will also notice that there is an option to **Manage Projects** that will be useful later.
  > 
  > **Note:** Create separate projects for development, staging, and production environments, and use descriptive, consistent naming conventions (e.g. "myapp-dev", "myapp-staging", "myapp-prod") to avoid conflicts and maintain clear project boundaries.
  > 
<br />
**3. Configure Settings**: Once you create a project, you will be able to see it in the dropdown and under **Manage Projects**. Click **Manage Projects** and click **View** to customize project rate limits.
>
> **Note:** Start with conservative limits for new projects, increase limits based on actual usage patterns and needs, and monitor usage regularly to adjust as needed.
>
<br />
**4. Generate API Keys:** Once you've configured your project and selected it in the dropdown, it will persist across the console. Any API keys generated will be specific to the project you have selected. Any logs will also be project-specific.
<br />
**5. Start Building:** Begin making API calls using your project-specific API credentials

### Project Selection
Use the project selector in the top navigation to switch between projects. All Console sections automatically filter to show data for the selected project:
- API Keys
- Batch Jobs
- Logs and Usage Analytics

## Rate Limit Management
### Understanding Rate Limits
Rate limits control the maximum number of requests your project can make to models within a specific time window. Rate limits are applied per project, meaning each project has its own separate quota that doesn't interfere with other projects in your organization.
Each project can be configured to have custom rate limits for every available model, which allows you to:

- Allocate higher limits to production projects
- Set conservative limits for experimental or development projects
- Customize limits based on specific use case requirements

Custom project rate limits can only be set to values equal to or lower than your organization's limits. Setting a custom rate limit for a project does not increase your organization's overall limits, it only allows you to set more restrictive limits for that specific project. Organization limits always take precedence and act as a ceiling for all project limits.

### Configuring Rate Limits
To configure rate limits for a project:
1. Navigate to **Projects** in your settings
2. Select the project you want to configure
3. Adjust the limits for each model as needed

### Example: Rate Limits Across Projects

Let's say you've created three projects for your application:
- myapp-prod for production
- myapp-staging for testing
- myapp-dev for development

- Organization Limit: 100 requests per minute
- myapp-prod: 80 requests per minute
- myapp-staging: 30 requests per minute
- myapp-dev: Using default organization limits

**Here's how the rate limits work in practice:**

1. myapp-prod
   - Can make up to 80 requests per minute (custom project limit)
   - Even if other projects are idle, cannot exceed 80 requests per minute
   - Contributing to the organization's total limit of 100 requests per minute

2. myapp-staging
   - Limited to 30 requests per minute (custom project limit)
   - Cannot exceed this limit even if organization has capacity
   - Contributing to the organization's total limit of 100 requests per minute

3. myapp-dev
   - Inherits the organization limit of 100 requests per minute
   - Actual available capacity depends on usage from other projects
   - If myapp-prod is using 80 requests/min and myapp-staging is using 15 requests/min, myapp-dev can only use 5 requests/min

**What happens during high concurrent usage:**

If both myapp-prod and myapp-staging try to use their maximum configured limits simultaneously:
- myapp-prod attempts to use 80 requests/min
- myapp-staging attempts to use 30 requests/min
- Total attempted usage: 110 requests/min
- Organization limit: 100 requests/min

In this case, some requests will fail with rate limit errors because the combined usage exceeds the organization's limit. Even though each project is within its configured limits, the organization limit of 100 requests/min acts as a hard ceiling.

## Usage Tracking
Projects provide comprehensive usage tracking including:

- Monthly spend tracking: Monitor costs and spending patterns for each project
- Usage metrics: Track API calls, token usage, and request patterns
- Request logs: Access detailed logs for debugging and monitoring

Dashboard pages will automatically be filtered by your selected project. Access these insights by:

1. Selecting your project in the top left of the navigation bar
2. Navigate to the **Dashboard** to see your project-specific **Usage**, **Metrics**, and **Logs** pages

- **Explore** the [Rate Limits](/docs/rate-limits) documentation for detailed rate limit configuration
- **Learn** about [Groq Libraries](/docs/libraries) to integrate Projects into your applications
- **Join** our [developer community](https://community.groq.com) for Projects tips and best practices

Ready to get started? Create your first project in the [Projects dashboard](https://console.groq.com/settings/projects) and begin organizing your Groq applications today.

## Qwen3 32b: Page (mdx)

URL: https://console.groq.com/docs/model/qwen3-32b

No content to display.

## Deepseek R1 Distill Qwen 32b: Model (tsx)

URL: https://console.groq.com/docs/model/deepseek-r1-distill-qwen-32b

---

## Prometheus Metrics

**URL:** llms-txt#prometheus-metrics

**Contents:**
- Enterprise Feature
- APIs
- MetricsQL
- Querying
- Grafana
- Available Metrics
  - Request Metrics
  - Latency Metrics
  - Token Metrics
- Batch: Create Batch Job (js)

[Prometheus](https://prometheus.io/) is an open-source monitoring system that collects and stores metrics as time series data.
Its [stable API](https://prometheus.io/docs/prometheus/latest/querying/api/) is compatible with a range of systems and tools like [Grafana](https://grafana.com/oss/grafana).

## Enterprise Feature

This feature is only available to our Enterprise tier customers. To get started, please reach out to [our Enterprise team](https://groq.com/enterprise-access).

Groq exposes Prometheus metrics about your organization's usage through [VictoriaMetrics](https://victoriametrics.com/). It [supports](https://docs.victoriametrics.com/victoriametrics/#prometheus-querying-api-usage) most Prometheus querying API paths:

* `/api/v1/query`
* `/api/v1/query_range`
* `/api/v1/series`
* `/api/v1/labels`
* `/api/v1/label/<label_name>/values`
* `/api/v1/status/tsdb`

Prometheus queries against Groq endpoints use [MetricsQL](https://docs.victoriametrics.com/MetricsQL.html), a query language that extends Prometheus's native [PromQL](https://prometheus.io/docs/prometheus/latest/querying/basics/) query language.

Queries can be sent to the following endpoint:

To Authenticate, you will need to provide your Groq API key as a header in the `Authorization: Bearer <your-api-key>` format.

If you run Grafana, you can add Groq metrics as a Prometheus datasource:

1. Add a new Prometheus datasource in Grafana by navigating to Settings -> Data Sources -> Add data source -> Prometheus.
2. Enter the following URL under HTTP -> URL: `https://api.groq.com/v1/metrics/prometheus`
3. Set the `Authorization` header to your Groq API key:
  * Go to Custom HTTP Headers -> Add Header
    * Header: `Authorization`
    * Value: `Bearer <your-api-key>`
4. Save & Test.

Groq provides the following metrics:

### Request Metrics
* `requests:increase1m`
  * The number of requests made within a minute
* `requests:rate1m`
  * The average number of requests per second over a given minute

Broken out by `model` and `status_code`

### Latency Metrics
* `e2e_latency_seconds:{percentile}:rate5m`
  * Percentile end-to-end latency average over a 5 minute window for P99, P95, and P50
* `ttft_latency_seconds:{percentile}:rate5m`
  * Percentile time to first token latency average over a 5 minute window for P99, P95, and P50
* `queue_latency_seconds:{percentile}:rate5m`
  * Percentile queue latency (time request spends in queue before being processed) average over a 5 minute window for P99, P95, and P50

Broken out by `model`.

### Token Metrics
* `tokens_in:{percentile}:rate5m`
  * Percentile number of input tokens average over a 5 minute window for P99, P95, and P50
* `tokens_out:{percentile}:rate5m`
  * Percentile number of output tokens average over a 5 minute window for P99, P95, and P50

Broken out by `model`.

In addition to using the APIs directly, you can see a handful of curated charts directly in our console at [Metrics](/metrics/prometheus)

## Batch: Create Batch Job (js)

URL: https://console.groq.com/docs/batch/scripts/create_batch_job

## Initial request - gets first page of batches

URL: https://console.groq.com/docs/batch/scripts/list_batches.py

```python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

**Examples:**

Example 1 (unknown):
```unknown
https://api.groq.com/v1/metrics/prometheus
```

Example 2 (javascript):
```javascript
import Groq from 'groq-sdk';

const groq = new Groq();

async function main() {
  const response = await groq.batches.create({
    completion_window: "24h",
    endpoint: "/v1/chat/completions",
    input_file_id: "file_01jh6x76wtemjr74t1fh0faj5t",
  });
  console.log(response);
}

main();
```

---

## Prompt Caching

**URL:** llms-txt#prompt-caching

**Contents:**
- How It Works
- Supported Models
- Pricing
- Structuring Prompts for Optimal Caching
  - Optimal Prompt Structure
  - Example Structure
- Prompt Caching Examples
  - How Prompt Caching Works in Multi-Turn Conversations
  - How Prompt Caching Works with Large Context
  - How Prompt Caching Works with Tool Definitions

Model prompts often contain repetitive content, such as system prompts and tool definitions.
Prompt caching automatically reuses computation from recent requests when they share a common prefix, delivering significant cost savings and improved response times while maintaining data privacy through volatile-only storage that expires automatically.

Prompt caching works automatically on all your API requests with no code changes required and no additional fees.

1. **Prefix Matching**: When you send a request, the system examines and identifies matching prefixes from recently processed requests stored temporarily in volatile memory. Prefixes can include system prompts, tool definitions, few-shot examples, and more.

2. **Cache Hit**: If a matching prefix is found, cached computation is reused, dramatically reducing latency and token costs by 50% for cached portions.

3. **Cache Miss**: If no match exists, your prompt is processed normally, with the prefix temporarily cached for potential future matches.

4. **Automatic Expiration**: All cached data automatically expires within a few hours, which helps ensure privacy while maintaining the benefits.

Prompt caching works automatically on all your API requests to supported models with no code changes required and no additional fees. Groq tries to maximize cache hits, but this is not guaranteed. Pricing discount will only apply on successful cache hits.

Cached tokens do not count towards your rate limits. However, cached tokens are subtracted from your limits after processing, so it's still possible to hit your limits if you are sending a large number of input tokens in parallel requests.

Prompt caching is currently only supported for the following models:

| Model ID                        | Model                          |
|---------------------------------|--------------------------------|
| moonshotai/kimi-k2-instruct-0905                  | Kimi K2 |
| openai/gpt-oss-20b                  | GPT-OSS 20B |
| openai/gpt-oss-120b                  | GPT-OSS 120B |
| openai/gpt-oss-safeguard-20b                  | GPT-OSS-Safeguard 20B |

We're starting with a limited selection of models and will roll out prompt caching to more models soon.

Prompt caching is provided at no additional cost. There is a 50% discount for cached input tokens.

## Structuring Prompts for Optimal Caching

Cache hits are only possible for exact prefix matches within a prompt. To realize caching benefits, you need to think strategically about prompt organization:

### Optimal Prompt Structure
Place static content like instructions and examples at the beginning of your prompt, and put variable content, such as user-specific information, at the end. This maximizes the length of the reusable prefix across different requests.

If you put variable information (like timestamps or user IDs) at the beginning, even identical system instructions later in the prompt won't benefit from caching because the prefixes won't match.

**Place static content first:**
- System prompts and instructions
- Few-shot examples
- Tool definitions
- Schema definitions
- Common context or background information

**Place dynamic content last:**
- User-specific queries
- Variable data
- Timestamps
- Session-specific information
- Unique identifiers

### Example Structure

This structure maximizes the likelihood that the static prefix portion will match across different requests, enabling cache hits while keeping user-specific content at the end.

## Prompt Caching Examples

### How Prompt Caching Works in Multi-Turn Conversations

In this example, we demonstrate how to use prompt caching in a multi-turn conversation.

During each turn, the system automatically caches the longest matching prefix from previous requests. The system message and conversation history that remain unchanged between requests will be cached, while only new user messages and assistant responses need fresh processing.

This approach is useful for maintaining context in ongoing conversations without repeatedly processing the same information.

**For the first request:**
- `prompt_tokens`: Number of tokens in the system message and first user message
- `cached_tokens`: 0 (no cache hit on first request)

**For subsequent requests within the cache lifetime:**
- `prompt_tokens`: Total number of tokens in the entire conversation (system message + conversation history + new user message)
- `cached_tokens`: Number of tokens in the system message and previous conversation history that were served from cache

When set up properly, you should see increasing cache efficiency as the conversation grows, with the system message and earlier conversation turns being served from cache while only new content requires processing.

### How Prompt Caching Works with Large Context

In this example, we demonstrate caching large static content like legal documents, research papers, or extensive context that remains constant across multiple queries.

The large legal document in the system message represents static content that benefits significantly from caching. Once cached, subsequent requests with different questions about the same document will reuse the cached computation for the document analysis, processing only the new user questions.

This approach is particularly effective for document analysis, research assistance, or any scenario where you need to ask multiple questions about the same large piece of content.

**For the first request:**
- `prompt_tokens`: Total number of tokens in the system message (including the large legal document) and user message
- `cached_tokens`: 0 (no cache hit on first request)

**For subsequent requests within the cache lifetime:**
- `prompt_tokens`: Total number of tokens in the system message (including the large legal document) and user message
- `cached_tokens`: Number of tokens in the entire cached system message (including the large legal document)

The caching efficiency is particularly high in this scenario since the large document (which may be thousands of tokens) is reused across multiple requests, while only small user queries (typically dozens of tokens) need fresh processing.

### How Prompt Caching Works with Tool Definitions

In this example, we demonstrate caching tool definitions.

All tool definitions, including their schemas, descriptions, and parameters, are cached as a single prefix when they remain consistent across requests. This is particularly valuable when you have a comprehensive set of tools that you want to reuse across multiple requests without re-processing them each time.

The system message and all tool definitions form the static prefix that gets cached, while user queries remain dynamic and are processed fresh for each request.

This approach is useful when you have a consistent set of tools that you want to reuse across multiple requests without re-processing them each time.

**For the first request:**
- `prompt_tokens`: Total number of tokens in the system message, tool definitions, and user message
- `cached_tokens`: 0 (no cache hit on first request)

**For subsequent requests within the cache lifetime:**
- `prompt_tokens`: Total number of tokens in the system message, tool definitions, and user message
- `cached_tokens`: Number of tokens in all cached tool definitions and system prompt

Tool definitions can be quite lengthy due to detailed parameter schemas and descriptions, making caching particularly beneficial for reducing both latency and costs when the same tool set is used repeatedly.

## Requirements and Limitations

### Caching Requirements
- **Exact Prefix Matching**: Cache hits require exact matches of the beginning of your prompt
- **Minimum Prompt Length**: The minimum cacheable prompt length varies by model, ranging from 128 to 1024 tokens depending on the specific model used

To check how much of your prompt was cached, see the response [usage fields](#response-usage-structure).

### What Can Be Cached

- **Complete message arrays** including system, user, and assistant messages
- **Tool definitions** and function schemas
- **System instructions** and prompt templates
- **One-shot** and **few-shot examples**
- **Structured output schemas**
- **Large static content** like legal documents, research papers, or extensive context that remains constant across multiple queries
- **Image inputs**, including image URLs and base64-encoded images

- **Exact Matching**: Even minor changes in cached portions prevent cache hits and force a new cache to be created
- **No Manual Control**: Cache clearing and management is automatic only

## Tracking Cache Usage

You can monitor how many tokens are being served from cache by examining the `usage` field in your API response. The response includes detailed token usage information, including how many tokens were cached.

### Response Usage Structure

### Understanding the Fields

- **`prompt_tokens`**: Total number of tokens in your input prompt
- **`cached_tokens`**: Number of input tokens that were served from cache (within `prompt_tokens_details`)
- **`completion_tokens`**: Number of tokens in the model's response
- **`total_tokens`**: Sum of prompt and completion tokens

In the example above, out of 4641 prompt tokens, 4608 tokens (99.3%) were served from cache, resulting in significant cost savings and improved response time.

### Calculating Cache Hit Rate

To calculate your cache hit rate:

For the example above: `4608 / 4641 × 100% = 99.3%`

A higher cache hit rate indicates better prompt structure optimization leading to lower latency and more cost savings.

## Troubleshooting
- Verify that sections that you want to cache are identical between requests
- Check that calls are made within the cache lifetime (a few hours). Calls that are too far apart will not benefit from caching.
- Ensure that `tool_choice`, tool usage, and image usage remain consistent between calls
- Validate that you are caching at least the [minimum number of tokens](#caching-requirements) through the [usage fields](#response-usage-structure).

Changes to cached sections, including `tool_choice` and image usage, will invalidate the cache and require a new cache to be created. Subsequent calls will use the new cache.

## Frequently Asked Questions

### How is data privacy maintained with caching?

All cached data exists only in volatile memory and automatically expires within a few hours. No prompt or response content is ever stored in persistent storage or shared between organizations.

### Does caching affect the quality or consistency of responses?

No. Prompt caching only affects the processing of the input prompt, not the generation of responses. The actual model inference and response generation occur normally, maintaining identical output quality whether caching is used or not.

### Can I disable prompt caching?

Prompt caching is automatically enabled and cannot be manually disabled. This helps customers benefit from reduced costs and latency. Prompts are not stored in persistent storage.

### How do I know if my requests are benefiting from caching?

You can track cache usage by examining the `usage` field in your API responses. Cache hits are not guaranteed, but Groq tries to maximize them. See the [Tracking Cache Usage](#tracking-cache-usage) section above for detailed information on how to monitor cached tokens and calculate your cache hit rate.

### Are there any additional costs for using prompt caching?

No. Prompt caching is provided at no additional cost and can help to reduce your costs by 50% for cached tokens while improving response times.

### Does caching affect rate limits?

Cached tokens do not count toward your rate limits.

### Can I manually clear or refresh caches?

No manual cache management is available. All cache expiration and cleanup happens automatically.

### Does the prompt caching discount work with batch requests?

Batch requests can still benefit from prompt caching, but the prompt caching discount does not stack with the batch discount. [Batch requests](/docs/batch) already receive a 50% discount on all tokens, and while caching functionality remains active, no additional discount is applied to cached tokens in batch requests.

## Firecrawl + Groq: AI-Powered Web Scraping & Data Extraction

URL: https://console.groq.com/docs/firecrawl

## Firecrawl + Groq: AI-Powered Web Scraping & Data Extraction

[Firecrawl](https://firecrawl.dev) is an enterprise-grade web scraping platform that turns any website into clean, AI-ready data. Combined with Groq's fast inference through MCP, you can build intelligent agents that scrape websites, extract structured data, and conduct deep research with natural language instructions.

**Key Features:**
- **Enterprise Web Scraping:** Handles JavaScript, authentication, and anti-bot detection automatically
- **Structured Extraction:** Define JSON schemas and get consistent data across sources
- **Deep Research:** Multi-hop reasoning that synthesizes information from multiple pages
- **Batch Processing:** Scrape multiple URLs efficiently with parallel processing
- **Fast Results:** Sub-10 second responses when combined with Groq's inference

#### 1. Install the required packages:

#### 2. Get your API keys:
- **Groq:** [console.groq.com/keys](https://console.groq.com/keys)
- **Firecrawl:** [firecrawl.dev/app/api-keys](https://firecrawl.dev/app/api-keys)

#### 3. Create your first web scraping agent:

### Structured Data Extraction

Extract data in specific JSON formats across multiple sources:

### Deep Research & Multi-Hop Analysis

Conduct comprehensive research across multiple sources:

### Batch Web Scraping

Scrape multiple URLs in parallel:

## Available Firecrawl MCP Tools

Firecrawl MCP provides several powerful tools for web scraping, data extraction, and research:

| Tool | Description |
|------|-------------|
| **`firecrawl_scrape`** | Scrape content from a single URL with advanced options and formatting |
| **`firecrawl_batch_scrape`** | Scrape multiple URLs efficiently with built-in rate limiting and parallel processing |
| **`firecrawl_check_batch_status`** | Check the status of a batch operation and retrieve results |
| **`firecrawl_search`** | Search the web and optionally extract content from search results |
| **`firecrawl_crawl`** | Start an asynchronous crawl with advanced options for depth and link following |
| **`firecrawl_extract`** | Extract structured information from web pages using LLM capabilities and JSON schemas |
| **`firecrawl_deep_research`** | Conduct comprehensive deep web research with intelligent crawling and LLM analysis |
| **`firecrawl_generate_llmstxt`** | Generate standardized llms.txt files that define how LLMs should interact with a site |

**Challenge:** Build an AI-powered competitive intelligence system that monitors competitor websites, extracts key business metrics, and generates automated reports using Firecrawl and Groq!

## Additional Resources

For more detailed documentation and resources on building web intelligence applications with Groq and Firecrawl, see:

- [Firecrawl Documentation](https://docs.firecrawl.dev)
- [Firecrawl API Reference](https://docs.firecrawl.dev/api-reference)
- [Firecrawl MCP Server](https://mcp.firecrawl.dev)
- [Groq API Cookbook: Firecrawl MCP Tutorial](https://github.com/groq/groq-api-cookbook/blob/main/tutorials/03-mcp/mcp-firecrawl/mcp-firecrawl.ipynb)
- [Groq Responses API Documentation](https://console.groq.com/docs/api-reference#responses)

## Compound: Natural Language.doc (ts)

URL: https://console.groq.com/docs/compound/scripts/natural-language.doc

## Log the tools that were used to generate the response

URL: https://console.groq.com/docs/compound/scripts/executed_tools.py

```python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="groq/compound",
    messages=[
        {"role": "user", "content": "What did Groq release last week?"}
    ]
)

**Examples:**

Example 1 (text):
```text
[SYSTEM PROMPT - Static]
[TOOL DEFINITIONS - Static]  
[FEW-SHOT EXAMPLES - Static]
[COMMON INSTRUCTIONS - Static]
[USER QUERY - Dynamic]
[SESSION DATA - Dynamic]
```

Example 2 (json):
```json
{
  "id": "chatcmpl-...",
  "model": "moonshotai/kimi-k2-instruct",
  "usage": {
      "queue_time": 0.026959759,
      "prompt_tokens": 4641,
      "prompt_time": 0.009995497,
      "completion_tokens": 1817,
      "completion_time": 5.57691751,
      "total_tokens": 6458,
      "total_time": 5.586913007,
      "prompt_tokens_details": {
          "cached_tokens": 4608
      }
  },
  ... other fields
}
```

Example 3 (unknown):
```unknown
Cache Hit Rate = cached_tokens / prompt_tokens × 100%
```

Example 4 (bash):
```bash
pip install openai python-dotenv
```

---

## Prompt design is critical for structured outputs

**URL:** llms-txt#prompt-design-is-critical-for-structured-outputs

system_prompt = """
You are a product catalog assistant. When asked about products,
always respond with valid JSON objects that match this structure:
{
  "id": "string",
  "name": "string",
  "price": number,
  "description": "string",
  "in_stock": boolean,
  "tags": ["string"]
}
Your response should ONLY contain the JSON object and nothing else.
"""

---

## Python code snippet for translation

**URL:** llms-txt#python-code-snippet-for-translation

**Examples:**

Example 1 (unknown):
```unknown
The Groq SDK package can be installed using the following command:
```

---

## Qwen-2.5-32B

**URL:** llms-txt#qwen-2.5-32b

**Contents:**
- Overview
- Key Features
- Additional Information
- Key Technical Specifications
  - Key Technical Specifications
  - Model Architecture
  - Training and Data
  - Key Use Cases
  - Best Practices
  - Quick Start

Qwen-2.5-32B is Alibaba's flagship model, delivering near-instant responses with GPT-4 level capabilities across a wide range of tasks. Built on 5.5 trillion tokens of diverse training data, it excels at everything from creative writing to complex reasoning.

The model can be accessed at [https://chat.groq.com/?model=qwen-2.5-32b](https://chat.groq.com/?model=qwen-2.5-32b).

* GPT-4 level capabilities 
* Near-instant responses 
* Excels in creative writing and complex reasoning 
* Built on 5.5 trillion tokens of diverse training data

## Additional Information

* The model is available for use on the Groq Hosted AI Models website. 
* It is suited for a wide range of tasks.

## Key Technical Specifications

URL: https://console.groq.com/docs/model/playai-tts

### Key Technical Specifications

### Model Architecture

PlayAI Dialog v1.0 is based on a transformer architecture optimized for high-quality speech output. The model supports a large variety of accents and styles, with specialized voice cloning capabilities and configurable parameters for tone, style, and narrative focus.

### Training and Data

The model was trained on millions of audio samples with diverse characteristics:

* Sources: Publicly available video and audio works, interactive dialogue datasets, and licensed creative content
* Volume: Millions of audio samples spanning diverse genres and conversational styles
* Processing: Standard audio normalization, tokenization, and quality filtering

* **Creative Content Generation**: Ideal for writers, game developers, and content creators who need to vocalize text for creative projects, interactive storytelling, and narrative development with human-like audio quality.
* **Voice Agentic Experiences**: Build conversational AI agents and interactive applications with natural-sounding speech output, supporting dynamic conversation flows and gaming scenarios.
* **Customer Support and Accessibility**: Create voice-enabled customer support systems and accessibility tools with customizable voices and multilingual support (English and Arabic).

* Use voice cloning and parameter customization to adjust tone, style, and narrative focus for your specific use case.
* Consider cultural sensitivity when selecting voices, as the model may reflect biases present in training data regarding pronunciations and accents.
* Provide user feedback on problematic outputs to help improve the model through iterative updates and bias mitigation.
* Ensure compliance with Play.ht's Terms of Service and avoid generating harmful, misleading, or plagiarized content.
* For best results, keep input text under 10K characters and experiment with different voices to find the best fit for your application.

To get started, please visit our [text to speech documentation page](/docs/text-to-speech) for usage and examples.

### Limitations and Bias Considerations

#### Known Limitations

* **Cultural Bias**: The model's outputs can reflect biases present in its training data. It might underrepresent certain pronunciations and accents.
* **Variability**: The inherently stochastic nature of creative generation means that outputs can be unpredictable and may require human curation.

#### Bias and Fairness Mitigation

* **Bias Audits**: Regular reviews and bias impact assessments are conducted to identify poor quality or unintended audio generations.
* **User Controls**: Users are encouraged to provide feedback on problematic outputs, which informs iterative updates and bias mitigation strategies.

### Ethical and Regulatory Considerations

* All training data has been processed and anonymized in accordance with GDPR and other relevant data protection laws.
* We do not train on any of our user data.

#### Responsible Use Guidelines

* This model should be used in accordance with [Play.ht's Terms of Service](https://play.ht/terms/#partner-hosted-deployment-terms)
* Users should ensure the model is applied responsibly, particularly in contexts where content sensitivity is important.
* The model should not be used to generate harmful, misleading, or plagiarized content.

### Maintenance and Updates

* PlayAI Dialog v1.0 is the inaugural release.
* Future versions will integrate more languages, emotional controllability, and custom voices.

#### Support and Feedback

* Users are invited to submit feedback and report issues via "Chat with us" on [Groq Console](https://console.groq.com).
* Regular updates and maintenance reviews are scheduled to ensure ongoing compliance with legal standards and to incorporate evolving best practices.

* **License**: PlayAI-Groq Commercial License

## Llama 3.1 8b Instant: Model (tsx)

URL: https://console.groq.com/docs/model/llama-3.1-8b-instant

## Groq Hosted Models: llama-3.1-8b-instant

llama-3.1-8b-instant on Groq offers rapid response times with production-grade reliability, suitable for latency-sensitive applications. The model balances efficiency and performance, providing quick responses for chat interfaces, content filtering systems, and large-scale data processing workloads.

### OpenGraph Metadata

* **Title**: Groq Hosted Models: llama-3.1-8b-instant
* **Description**: llama-3.1-8b-instant on Groq offers rapid response times with production-grade reliability, suitable for latency-sensitive applications. The model balances efficiency and performance, providing quick responses for chat interfaces, content filtering systems, and large-scale data processing workloads.
* **URL**: https://chat.groq.com/?model=llama-3.1-8b-instant
* **Site Name**: Groq Hosted AI Models
* **Locale**: en_US
* **Type**: website

* **Card**: summary_large_image
* **Title**: Groq Hosted Models: llama-3.1-8b-instant
* **Description**: llama-3.1-8b-instant on Groq offers rapid response times with production-grade reliability, suitable for latency-sensitive applications. The model balances efficiency and performance, providing quick responses for chat interfaces, content filtering systems, and large-scale data processing workloads.

* **Index**: true
* **Follow**: true

### Alternates Metadata

* **Canonical**: https://chat.groq.com/?model=llama-3.1-8b-instant

## Compound Beta: Page (mdx)

URL: https://console.groq.com/docs/agentic-tooling/compound-beta

No content to display.

## Agentic Tooling: Page (mdx)

URL: https://console.groq.com/docs/agentic-tooling

No content to display.

## Compound Beta Mini: Page (mdx)

URL: https://console.groq.com/docs/agentic-tooling/compound-beta-mini

No content to display.

## Compound: Page (mdx)

URL: https://console.groq.com/docs/agentic-tooling/groq/compound

No content to display.

## Compound Mini: Page (mdx)

URL: https://console.groq.com/docs/agentic-tooling/groq/compound-mini

No content to display.

## ✨ Vercel AI SDK + Groq: Rapid App Development

URL: https://console.groq.com/docs/ai-sdk

## ✨ Vercel AI SDK + Groq: Rapid App Development

Vercel's AI SDK enables seamless integration with Groq, providing developers with powerful tools to leverage language models hosted on Groq for a variety of applications. By combining Vercel's cutting-edge platform with Groq's advanced inference capabilities, developers can create scalable, high-speed applications with ease.

### Why Choose the Vercel AI SDK?
- A versatile toolkit for building applications powered by advanced language models like Llama 3.3 70B 
- Ideal for creating chat interfaces, document summarization, and natural language generation
- Simple setup and flexible provider configurations for diverse use cases
- Fully supports standalone usage and seamless deployment with Vercel
- Scalable and efficient for handling complex tasks with minimal configuration

### Quick Start Guide in JavaScript (5 minutes to deployment)

#### 1. Create a new Next.js project with the AI SDK template:

#### 2. Install the required packages:

#### 3. Create a `.env.local` file in your project root and configure your Groq API Key:

#### 4. Create a new directory structure for your Groq API endpoint:

#### 5. Initialize the AI SDK by creating an API route file called `route.ts` in `app/api/chat`:

**Challenge**: Now that you have your basic chat interface working, try enhancing it to create a specialized code explanation assistant!

#### 6. Create your front end interface by updating the `app/page.tsx` file:

#### 7. Run your development enviornment to test our application locally:

#### 8. Easily deploy your application using Vercel CLI by installing `vercel` and then running the `vercel` command:

The CLI will guide you through a few simple prompts:
- If this is your first time using Vercel CLI, you'll be asked to create an account or log in
- Choose to link to an existing Vercel project or create a new one
- Confirm your deployment settings

Once you've gone through the prompts, your app will be deployed instantly and you'll receive a production URL! 🚀

### Additional Resources

For more details on integrating Groq with the Vercel AI SDK, see the following:
- [Official Documentation: Vercel](https://sdk.vercel.ai/providers/ai-sdk-providers/groq)
- [Vercel Templates for Groq](https://sdk.vercel.ai/providers/ai-sdk-providers/groq)

## Parallel + Groq: Fast Web Search for Real-Time AI Research

URL: https://console.groq.com/docs/parallel

## Parallel + Groq: Fast Web Search for Real-Time AI Research

[Parallel](https://parallel.ai) provides a web search MCP server that gives AI models access to real-time web data. Combined with Groq's industry-leading inference speeds (1000+ tokens/second), you can build research agents that find and analyze current information in seconds, not minutes.

**Key Features:**
- **Real-Time Information:** Access current events, breaking news, and live data
- **Parallel Processing:** Search multiple sources simultaneously
- **Ultra-Fast:** Groq's inference makes tool calling nearly instant
- **Source Transparency:** See exactly which websites were searched
- **Accurate Results:** Fresh data means current answers, not outdated information

#### 1. Install the required packages:

#### 2. Get your API keys:
- **Groq:** [console.groq.com/keys](https://console.groq.com/keys)
- **Parallel:** [platform.parallel.ai](https://platform.parallel.ai)

#### 3. Create your first real-time research agent:

### Multi-Company Comparison

Compare multiple companies side-by-side:

### Real-Time Market Data

Get current financial information:

### Breaking News Monitoring

Track developing stories:

## Performance Comparison

Real comparison from testing:
- **Groq (openai/gpt-oss-120b):** 11.15s, 472 chars/sec
- **OpenAI (gpt-5):** 88.38s, 42 chars/sec

**Groq is 8x faster** due to LPU architecture, instant tool call decisions, and fast synthesis of search results.

**Challenge:** Build a real-time market intelligence platform that monitors news, tracks competitor activities, analyzes trends, compares products, and generates daily briefings!

## Additional Resources

- [Parallel Documentation](https://docs.parallel.ai)
- [Parallel Platform](https://platform.parallel.ai)
- [Groq Responses API](https://console.groq.com/docs/api-reference#responses)

## Tavily + Groq: Real-Time Search, Scraping & Crawling for AI

URL: https://console.groq.com/docs/tavily

## Tavily + Groq: Real-Time Search, Scraping & Crawling for AI

[Tavily](https://tavily.com) is a comprehensive web search, scraping, and crawling API designed specifically for AI agents. It provides real-time web access, content extraction, and advanced search capabilities. Combined with Groq's ultra-fast inference through MCP, you can build intelligent agents that research topics, monitor websites, and extract structured data in seconds.

**Key Features:**
- **Multi-Modal Search:** Web search, content extraction, and crawling in one API
- **AI-Optimized Results:** Clean, structured data designed for LLM consumption
- **Advanced Filtering:** Search by date range, domain, content type, and more
- **Content Extraction:** Pull complete article content from any URL
- **Search Depth Control:** Choose between basic and advanced search
- **Fast Execution:** Groq's inference makes synthesis nearly instant

#### 1. Install the required packages:

#### 2. Get your API keys:
- **Groq:** [console.groq.com/keys](https://console.groq.com/keys)
- **Tavily:** [app.tavily.com](https://app.tavily.com/home)

#### 3. Create your first research agent:

### Time-Filtered Research

Search within specific time ranges:

### Product Information Extraction

Extract structured product data:

### Multi-Source Content Extraction

Extract and compare content from multiple URLs:

## Available Tavily Tools

| Tool | Description |
|------|-------------|
| **`tavily_search`** | Search with advanced filters (time, depth, topic, max results) |
| **`tavily_extract`** | Extract full content from specific URLs |
| **`tavily_scrape`** | Scrape single pages with clean output |
| **`tavily_batch_scrape`** | Scrape multiple URLs in parallel |
| **`tavily_crawl`** | Crawl websites with depth and pattern controls |

### Search Parameters

**Search Depth:**
- `basic` - Fast, surface-level results (under 3 seconds)
- `advanced` - Comprehensive, deep results (5-10 seconds)

**Time Range:**
- `day`, `week`, `month`, `year`

**Topic:**
- `general`, `news`

**Challenge:** Build an automated content curation system that monitors news sources, filters by relevance, extracts key information, generates summaries, and publishes daily digests!

## Additional Resources

- [Tavily Documentation](https://docs.tavily.com)
- [Tavily API Reference](https://docs.tavily.com/api-reference)
- [Tavily App](https://app.tavily.com/home)
- [Groq Responses API](https://console.groq.com/docs/api-reference#responses)

## Script: Openai Compat (py)

URL: https://console.groq.com/docs/scripts/openai-compat.py

import os
import openai

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

## Script: Openai Compat (js)

URL: https://console.groq.com/docs/scripts/openai-compat

import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1"
});

## AutoGen + Groq: Building Multi-Agent AI Applications

URL: https://console.groq.com/docs/autogen

## AutoGen + Groq: Building Multi-Agent AI Applications

[AutoGen](https://microsoft.github.io/autogen/) developed by [Microsoft Research](https://www.microsoft.com/research/) is an open-source framework for building multi-agent AI applications. By powering the
AutoGen agentic framework with Groq's fast inference speed, you can create sophisticated AI agents that work together to solve complex tasks fast with features including:

- **Multi-Agent Orchestration:** Create and manage multiple agents that can collaborate in realtime
- **Tool Integration:** Easily connect agents with external tools and APIs
- **Flexible Workflows:** Support both autonomous and human-in-the-loop conversation patterns
- **Code Generation & Execution:** Enable agents to write, review, and execute code safely

### Python Quick Start (3 minutes to hello world)
#### 1. Install the required packages:

#### 2. Configure your Groq API key:

#### 3. Create your first multi-agent application with Groq:
In AutoGen, **agents** are autonomous entities that can engage in conversations and perform tasks. The example below shows how to create a simple two-agent system with `llama-3.3-70b-versatile` where
`UserProxyAgent` initiates the conversation with a question and `AssistantAgent` responds:

```python
import os
from autogen import AssistantAgent, UserProxyAgent

**Examples:**

Example 1 (bash):
```bash
npx create-next-app@latest my-groq-app --typescript --tailwind --src-dir
cd my-groq-app
```

Example 2 (bash):
```bash
npm install @ai-sdk/groq ai
npm install react-markdown
```

Example 3 (bash):
```bash
GROQ_API_KEY="your-api-key"
```

Example 4 (bash):
```bash
mkdir -p src/app/api/chat
```

---

## Register the tool with the assistant

**URL:** llms-txt#register-the-tool-with-the-assistant

@assistant.register_for_llm(description="Weather forecast for cities.")
def weather_forecast(
    location: Annotated[str, "City name"],
    unit: Annotated[str, "Temperature unit (fahrenheit/celsius)"] = "fahrenheit"
) -> str:
    weather_details = get_current_weather(location=location, unit=unit)
    weather = json.loads(weather_details)
    return f"{weather['location']} will be {weather['temperature']} degrees {weather['unit']}"
python
import os
import json
from pathlib import Path
from typing import Annotated
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

**Examples:**

Example 1 (unknown):
```unknown
#### Complete Code Example
Here is our quick start agent code example combined with code execution and tool use that you can play with:
```

---

## Register weather tool with the assistant

**URL:** llms-txt#register-weather-tool-with-the-assistant

@assistant.register_for_llm(description="Weather forecast for cities.")
def weather_forecast(
    location: Annotated[str, "City name"],
    unit: Annotated[str, "Temperature unit (fahrenheit/celsius)"] = "fahrenheit"
) -> str:
    weather_details = get_current_weather(location=location, unit=unit)
    weather = json.loads(weather_details)
    return f"{weather['location']} will be {weather['temperature']} degrees {weather['unit']}"

---

## Security Onboarding

**URL:** llms-txt#security-onboarding

**Contents:**
- Overview
- Secure API Key Management
- Key Rotation & Revocation
- Transport Security (TLS)
- Input and Prompt Safety
- Rate Limiting and Retry Logic
- Logging & Monitoring
- Secure Tool Use & Agent Integrations
- Incident Response
- Resources

Welcome to the **Groq Security Onboarding** guide.  
This page walks through best practices for protecting your API keys, securing client configurations, and hardening integrations before moving into production.

Security is a shared responsibility between Groq and our customers.  
While Groq ensures secure API transport and service isolation, customers are responsible for securing client-side configurations, keys, and data handling.

All Groq API traffic is encrypted in transit using TLS 1.2+ and authenticated via API keys.

## Secure API Key Management

Never expose or hardcode API keys directly into your source code.  
Use environment variables or a secret management system.

**Warning:** Never embed keys in frontend code or expose them in browser bundles. If you need client-side usage, route through a trusted backend proxy.

## Key Rotation & Revocation

* Rotate API keys periodically (e.g., quarterly).
* Revoke keys immediately if compromise is suspected.
* Use per-environment keys (dev / staging / prod).
* Log all API key creations and deletions.

## Transport Security (TLS)

Groq APIs enforce HTTPS (TLS 1.2 or higher).
You should **never** disable SSL verification.

## Input and Prompt Safety

When integrating Groq into user-facing systems, ensure that user inputs cannot trigger prompt injection or tool misuse.

* Sanitize user input before embedding in prompts.
* Avoid exposing internal system instructions or hidden context.
* Validate model outputs (especially JSON / code / commands).
* Limit model access to safe tools or actions only.

## Rate Limiting and Retry Logic

Implement client-side rate limiting and exponential backoff for 429 / 5xx responses.

## Logging & Monitoring

Maintain structured logs for all API interactions.

* Timestamp
* Endpoint
* Request latency
* Key / service ID (non-secret)
* Error codes

**Tip:** Avoid logging sensitive data or raw model responses containing user information.

## Secure Tool Use & Agent Integrations

When using Groq's **Tool Use** or external function execution features:

* Expose only vetted, sandboxed tools.
* Restrict external network calls.
* Audit all registered tools and permissions.
* Validate arguments and outputs.

If you suspect your API key is compromised:

1. Revoke the key immediately from the [Groq Console](https://console.groq.com/keys).
2. Rotate to a new key and redeploy secrets.
3. Review logs for suspicious activity.
4. Notify your security admin.

**Warning:** Never reuse compromised keys, even temporarily.

- [Groq API Documentation](/docs/api-reference)
- [Prompt Engineering Guide](/docs/prompting)
- [Understanding and Optimizing Latency](/docs/production-readiness/optimizing-latency)
- [Production-Ready Checklist](/docs/production-readiness/production-ready-checklist)
- [Groq Developer Community](https://community.groq.com)
- [OpenBench](https://openbench.dev)

*This security guide should be customized based on your specific application requirements and updated based on production learnings.*

## Production-Ready Checklist for Applications on GroqCloud

URL: https://console.groq.com/docs/production-readiness/production-ready-checklist

---

## Set up headers

**URL:** llms-txt#set-up-headers

headers = {
    "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
    "Content-Type": "application/json",
}

---

## Set up instructor with Groq

**URL:** llms-txt#set-up-instructor-with-groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

---

## Set up the client with instructor

**URL:** llms-txt#set-up-the-client-with-instructor

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
instructor_client = instructor.patch(client)

---

## Systems

**URL:** llms-txt#systems

**Contents:**
- Getting Started
- System Comparison
- Key Differences
  - Compound
  - Compound Mini
- Available Tools
- When to Choose Which System
  - Choose Compound When:
  - Choose Compound Mini When:
- Compound Beta Mini: Page (mdx)

Groq offers two compound AI systems that intelligently use external tools to provide more accurate, up-to-date, and capable responses than traditional LLMs alone. Both systems support web search and code execution, but differ in their approach to tool usage.

- **[Compound](/docs/compound/systems/compound)** (`groq/compound`) - Full-featured system with up to 10 tool calls per request
- **[Compound Mini](/docs/compound/systems/compound-mini)** (`groq/compound-mini`) - Streamlined system with up to 1 tool call and average 3x lower latency

Groq's compound AI systems should not be used by customers for processing protected health information as it is not a HIPAA Covered Cloud Service under Groq's Business Associate Addendum at this time.

Both systems use the same API interface - simply change the `model` parameter to `groq/compound` or `groq/compound-mini` to get started.

| Feature | Compound | Compound Mini |
|---------|---------------|-------------------|
| **Tool Calls per Request** | Up to 10 | Up to 1 |
| **Average Latency** | Standard | 3x Lower |
| **Token Speed** | ~350 tps | ~350 tps |
| **Best For** | Complex multi-step tasks | Quick single-step queries |

### Compound
- **Multiple Tool Calls**: Can perform up to **10 server-side tool calls** before returning an answer
- **Complex Workflows**: Ideal for tasks requiring multiple searches, code executions, or iterative problem-solving
- **Comprehensive Analysis**: Can gather information from multiple sources and perform multi-step reasoning
- **Use Cases**: Research tasks, complex data analysis, multi-part coding challenges

### Compound Mini
- **Single Tool Call**: Performs up to **1 server-side tool call** before returning an answer
- **Fast Response**: Average 3x lower latency compared to Compound
- **Direct Answers**: Perfect for straightforward queries that need one piece of current information
- **Use Cases**: Quick fact-checking, single calculations, simple web searches

Both systems support the same set of tools:

- **Web Search** - Access real-time information from the web
- **Code Execution** - Execute Python code automatically
- **Visit Website** - Access and analyze specific website content
- **Browser Automation** - Interact with web pages through automated browser actions
- **Wolfram Alpha** - Access computational knowledge and mathematical calculations

For more information about tool capabilities, see the [Built-in Tools](/docs/compound/built-in-tools) page.

## When to Choose Which System

### Choose Compound When:
- You need comprehensive research across multiple sources
- Your task requires iterative problem-solving
- You're building complex analytical workflows
- You need multi-step code generation and testing

### Choose Compound Mini When:
- You need quick answers to straightforward questions
- Latency is a critical factor for your application
- You're building real-time applications
- Your queries typically require only one tool call

## Compound Beta Mini: Page (mdx)

URL: https://console.groq.com/docs/compound/systems/compound-beta-mini

No content to display.

## Key Technical Specifications

URL: https://console.groq.com/docs/compound/systems/compound

### Key Technical Specifications

*   **Model Architecture**: Compound is powered by [Llama 4 Scout](/docs/model/meta-llama/llama-4-scout-17b-16e-instruct) and [GPT-OSS 120B](/docs/model/openai/gpt-oss-120b) for intelligent reasoning and tool use.

*   **Performance Metrics**: Groq developed a new evaluation benchmark for measuring search capabilities called [RealtimeEval](https://github.com/groq/realtime-eval). This benchmark is designed to evaluate tool-using systems on current events and live data. On the benchmark, Compound outperformed GPT-4o-search-preview and GPT-4o-mini-search-preview significantly.

## Learn More About Agentic Tooling
Discover how to build powerful applications with real-time web search and code execution

*   **Realtime Web Search**: Automatically access up-to-date information from the web using the built-in web search tool.

*   **Code Execution**: Execute Python code automatically using the code execution tool powered by [E2B](https://e2b.dev/).

*   **Code Generation and Technical Tasks**: Create AI tools for code generation, debugging, and technical problem-solving with high-quality multilingual support.

*   Use system prompts to improve steerability and reduce false refusals. Compound is designed to be highly steerable with appropriate system prompts.

*   Consider implementing system-level protections like Llama Guard for input filtering and response validation.

*   Deploy with appropriate safeguards when working in specialized domains or with critical content.

*   Compound should not be used by customers for processing protected health information. It is not a HIPAA Covered Cloud Service under Groq's Business Associate Addendum for customers at this time.

### Quick Start
Experience the capabilities of `groq/compound` on Groq:

## Key Technical Specifications

URL: https://console.groq.com/docs/compound/systems/compound-mini

### Key Technical Specifications

Compound mini is powered by Llama 3.3 70B and GPT-OSS 120B for intelligent reasoning and tool use. Unlike groq/compound, it can only use one tool per request, but has an average of 3x lower latency.

* **Model Architecture**: 
    * Compound mini is powered by [Llama 3.3 70B](Llama 3.3 70B) and [GPT-OSS 120B](GPT-OSS 120B) for intelligent reasoning and tool use. Unlike [groq/compound](groq/compound), it can only use one tool per request, but has an average of 3x lower latency.
* **Performance Metrics**: 
    * Groq developed a new evaluation benchmark for measuring search capabilities called [RealtimeEval](https://github.com/groq/realtime-eval). This benchmark is designed to evaluate tool-using systems on current events and live data. On the benchmark, Compound Mini outperformed GPT-4o-search-preview and GPT-4o-mini-search-preview significantly.

### Quick Start
Experience the capabilities of `groq/compound-mini` on Groq:

* **Realtime Web Search**: 
    * Automatically access up-to-date information from the web using the built-in web search tool.
* **Code Execution**: 
    * Execute Python code automatically using the code execution tool powered by [E2B](https://e2b.dev/).
* **Code Generation and Technical Tasks**: 
    * Create AI tools for code generation, debugging, and technical problem-solving with high-quality multilingual support.

### Model Best Practices

* Use system prompts to improve steerability and reduce false refusals. Compound mini is designed to be highly steerable with appropriate system prompts.
* Consider implementing system-level protections like Llama Guard for input filtering and response validation.
* Deploy with appropriate safeguards when working in specialized domains or with critical content.

### Learn More About Agentic Tooling
Discover how to build powerful applications with real-time web search and code execution

Rate limits for `groq/compound-mini` are determined by the rate limits of the individual models that comprise them.

The use of this tool with a supported model or system in GroqCloud is not a HIPAA Covered Cloud Service under Groq's Business Associate Addendum at this time. This tool is also not available currently for use with regional / sovereign endpoints.

## E2B + Groq: Open-Source Code Interpreter

URL: https://console.groq.com/docs/e2b

## E2B + Groq: Open-Source Code Interpreter

[E2B](https://e2b.dev/) Code Interpreter is an open-source SDK that provides secure, sandboxed environments for executing code generated by LLMs via Groq API. Built specifically for AI data analysts, 
coding applications, and reasoning-heavy agents, E2B enables you to both generate and execute code in a secure sandbox environment in real-time.

### Python Quick Start (3 minutes to hello world)

#### 1. Install the required packages:

#### 2. Configure your Groq and [E2B](https://e2b.dev/docs) API keys:

#### 3. Create your first simple and fast Code Interpreter application that generates and executes code to analyze data:

Running the below code will create a secure sandbox environment, generate Python code using `llama-3.3-70b-versatile` powered by Groq, execute the code, and display the results. When you go to your 
[E2B Dashboard](https://e2b.dev/dashboard), you'll see your sandbox's data.

```python
from e2b_code_interpreter import Sandbox
from groq import Groq
import os

e2b_api_key = os.environ.get('E2B_API_KEY')
groq_api_key = os.environ.get('GROQ_API_KEY')

**Examples:**

Example 1 (bash):
```bash
pip install groq e2b-code-interpreter python-dotenv
```

Example 2 (bash):
```bash
export GROQ_API_KEY="your-groq-api-key"
export E2B_API_KEY="your-e2b-api-key"
```

---

## System prompt with clear instructions about the complex structure

**URL:** llms-txt#system-prompt-with-clear-instructions-about-the-complex-structure

system_prompt = """
You are a product catalog API. Generate a detailed product with ALL required fields.
Your response must be a valid JSON object matching the following schema:

{
  "id": "string",
  "name": "string",
  "description": "string",
  "main_category": "string",
  "subcategories": ["string"],
  "variants": [
    {
      "id": "string",
      "name": "string",
      "price": number,
      "inventory_count": number,
      "attributes": {"key": "value"}
    }
  ],
  "reviews": [
    {
      "user_id": "string",
      "rating": number (1-5),
      "comment": "string",
      "date": "string (YYYY-MM-DD)"
    }
  ],
  "average_rating": number (1-5),
  "manufacturer": {
    "name": "string",
    "founded": "string",
    "contact_info": {
      "email": "string",
      "phone": "string (optional)",
      "address": {
        "street": "string",
        "city": "string", 
        "state": "string",
        "zip_code": "string",
        "country": "string"
      }
    }
  }
}
"""

---

## The Groq integration is available in mlflow >= 2.20.0

**URL:** llms-txt#the-groq-integration-is-available-in-mlflow->=-2.20.0

**Contents:**
  - 2. Configure your Groq API key:
  - 3. (Optional) Start your mlflow server

pip install mlflow groq
bash
export GROQ_API_KEY="your-api-key"
bash

**Examples:**

Example 1 (unknown):
```unknown
### 2. Configure your Groq API key:
```

Example 2 (unknown):
```unknown
### 3. (Optional) Start your mlflow server
```

---

## This process is optional, but it is recommended to use MLflow tracking server for better visualization and additional features

**URL:** llms-txt#this-process-is-optional,-but-it-is-recommended-to-use-mlflow-tracking-server-for-better-visualization-and-additional-features

**Contents:**
  - 4. Create your first traced Groq application:

mlflow server
python
import mlflow
import groq

**Examples:**

Example 1 (unknown):
```unknown
### 4. Create your first traced Groq application:

Let's enable MLflow auto-tracing with the Groq SDK. For more configurations, refer to the [documentation for `mlflow.groq`](https://mlflow.org/docs/latest/python_api/mlflow.groq.html).
```

---

## Turn on auto tracing for Groq by calling mlflow.groq.autolog()

**URL:** llms-txt#turn-on-auto-tracing-for-groq-by-calling-mlflow.groq.autolog()

---

## Using a custom stop sequence for structured, concise output.

**URL:** llms-txt#using-a-custom-stop-sequence-for-structured,-concise-output.

---

## What are integrations?

**URL:** llms-txt#what-are-integrations?

**Contents:**
- AI Agent Frameworks
- Browser Automation
- LLM App Development
- Observability and Monitoring
- LLM Code Execution and Sandboxing
- UI and UX
- Tool Management
- Real-time Voice
- MCP (Model Context Protocol) Integration
- Integrations: Integration Buttons (ts)

Integrations are a way to connect your application to external services and enhance your Groq-powered applications with additional capabilities.
Browse the categories below to find integrations that suit your needs.

<div className="grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-4 text-sm text-groq-orange">
    <div className="flex flex-col gap-2">
        <a href="#ai-agent-frameworks" className="w-fit">AI Agent Frameworks</a>
        <a href="#browser-automation" className="w-fit">Browser Automation</a>
        <a href="#llm-app-development" className="w-fit">LLM App Development</a>
        <a href="#observability-and-monitoring" className="w-fit">Observability and Monitoring</a>
        <a href="#llm-code-execution-and-sandboxing" className="w-fit">LLM Code Execution and Sandboxing</a>
    </div>
    <div className="flex flex-col gap-2">
        <a href="#ui-and-ux" className="w-fit">UI and UX</a>
        <a href="#tool-management" className="w-fit">Tool Management</a>
        <a href="#realtime-voice" className="w-fit">Real-time Voice</a>
        <a href="#mcp-model-context-protocol-integration" className="w-fit">MCP Integration</a>
    </div>
</div>

## AI Agent Frameworks

Create autonomous AI agents that can perform complex tasks, reason, and collaborate effectively using Groq's fast inference capabilities.

## Browser Automation

Automate browser interactions and perform complex tasks and transform any browser-based task in to an API endpoint instantly with models via Groq.

## LLM App Development

Build powerful LLM applications with these frameworks and libraries that provide essential tools for working with Groq models.

## Observability and Monitoring

Track, analyze, and optimize your LLM applications with these integrations that provide insights into model performance and behavior.

## LLM Code Execution and Sandboxing

Enable secure code execution in controlled environments for your AI applications with these integrations.

Create beautiful and responsive user interfaces for your Groq-powered applications with these UI frameworks and tools.

Manage and orchestrate tools for your AI agents, enabling them to interact with external services and perform complex tasks.

Build voice-enabled applications that leverage Groq's fast inference for natural and responsive conversations.

## MCP (Model Context Protocol) Integration

Connect AI applications to external systems using the Model Context Protocol (MCP). Enable AI agents to use tools like GitHub, databases, and web services.

## Integrations: Integration Buttons (ts)

URL: https://console.groq.com/docs/integrations/integration-buttons

import type { IntegrationButton } from "./button-group";

type IntegrationGroup =
  | "ai-agent-frameworks"
  | "browser-automation"
  | "llm-app-development"
  | "observability"
  | "llm-code-execution"
  | "ui-and-ux"
  | "tool-management"
  | "real-time-voice"
  | "mcp-integration";

export const integrationButtons: Record<IntegrationGroup, IntegrationButton[]> =
  {
    "ai-agent-frameworks": [
      {
        title: "Agno",
        description:
          "Agno is a lightweight library for building Agents with memory, knowledge, tools and reasoning.",
        href: "/docs/agno",
        iconSrc: "/integrations/agno_black.svg",
        iconDarkSrc: "/integrations/agno_white.svg",
        color: "gray",
      },
      {
        title: "AutoGen",
        description:
          "AutoGen is a framework for building conversational AI systems that can operate autonomously or collaborate with humans and other agents.",
        href: "/docs/autogen",
        iconSrc: "/integrations/autogen.svg",
        color: "gray",
      },
      {
        title: "CrewAI",
        description:
          "CrewAI is a framework for orchestrating role-playing AI agents that work together to accomplish complex tasks.",
        href: "/docs/crewai",
        iconSrc: "/integrations/crewai.png",
        color: "gray",
      },
      {
        title: "xRx",
        description:
          "xRx is a reactive AI agent framework for building reliable and observable LLM agents with real-time feedback.",
        href: "/docs/xrx",
        iconSrc: "/integrations/xrx.png",
        color: "gray",
      },
    ],
    "browser-automation": [
      {
        title: "Anchor Browser",
        description:
          "Anchor Browser is a browser automation platform that allows you to automate workflows for web applications that lack APIs or have limited API coverage.",
        href: "/docs/anchorbrowser",
        iconSrc: "/integrations/anchorbrowser.png",
        color: "gray",
      },
    ],
    "llm-app-development": [
      {
        title: "LangChain",
        description:
          "LangChain is a framework for developing applications powered by language models through composability.",
        href: "/docs/langchain",
        iconSrc: "/integrations/langchain_black.png",
        iconDarkSrc: "/integrations/langchain_white.png",
        color: "gray",
      },
      {
        title: "LlamaIndex",
        description:
          "LlamaIndex is a data framework for building LLM applications with context augmentation over external data.",
        href: "/docs/llama-index",
        iconSrc: "/integrations/llamaindex_black.png",
        iconDarkSrc: "/integrations/llamaindex_white.png",
        color: "gray",
      },
      {
        title: "LiteLLM",
        description:
          "LiteLLM is a library that standardizes LLM API calls and provides robust tracking, fallbacks, and observability for LLM applications.",
        href: "/docs/litellm",
        iconSrc: "/integrations/litellm.png",
        color: "gray",
      },
      {
        title: "Vercel AI SDK",
        description:
          "Vercel AI SDK is a typescript library for building AI-powered applications in modern frontend frameworks.",
        href: "/docs/ai-sdk",
        iconSrc: "/vercel-integration.png",
        color: "gray",
      },
    ],
    observability: [
      {
        title: "Arize",
        description:
          "Arize is an observability platform for monitoring, troubleshooting, and explaining LLM applications.",
        href: "/docs/arize",
        iconSrc: "/integrations/arize_phoenix.png",
        color: "gray",
      },
      {
        title: "MLflow",
        description:
          "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle, including experiment tracking and model deployment.",
        href: "/docs/mlflow",
        iconSrc: "/integrations/mlflow-white.svg",
        iconDarkSrc: "/integrations/mlflow-black.svg",
        color: "gray",
      },
    ],
    "llm-code-execution": [
      {
        title: "E2B",
        description:
          "E2B provides secure sandboxed environments for LLMs to execute code and use tools in a controlled manner.",
        href: "/docs/e2b",
        iconSrc: "/integrations/e2b_black.png",
        iconDarkSrc: "/integrations/e2b_white.png",
        color: "gray",
      },
    ],
    "ui-and-ux": [
      {
        title: "FlutterFlow",
        description:
          "FlutterFlow is a visual development platform for building high-quality, custom, cross-platform apps with AI capabilities.",
        href: "/docs/flutterflow",
        iconSrc: "/integrations/flutterflow_black.png",
        iconDarkSrc: "/integrations/flutterflow_white.png",
        color: "gray",
      },
      {
        title: "Gradio",
        description:
          "Gradio is a Python library for quickly creating customizable UI components for machine learning models and LLM applications.",
        href: "/docs/gradio",
        iconSrc: "/integrations/gradio.svg",
        color: "gray",
      },
    ],
    "tool-management": [
      {
        title: "Composio",
        description:
          "Composio is a platform for managing and integrating tools with LLMs and AI agents for seamless interaction with external applications.",
        href: "/docs/composio",
        iconSrc: "/integrations/composio_black.png",
        iconDarkSrc: "/integrations/composio_white.png",
        color: "gray",
      },
      {
        title: "JigsawStack",
        description:
          "JigsawStack is a powerful AI SDK that integrates into any backend, automating tasks using LLMs with features like Mixture-of-Agents approach.",
        href: "/docs/jigsawstack",
        iconSrc: "/integrations/jigsaw.svg",
        color: "gray",
      },
      {
        title: "Toolhouse",
        description:
          "Toolhouse is a tool management platform that helps developers organize, secure, and scale tool usage across AI agents.",
        href: "/docs/toolhouse",
        iconSrc: "/integrations/toolhouse.svg",
        color: "gray",
      },
    ],
    "real-time-voice": [
      {
        title: "LiveKit",
        description:
          "LiveKit provides text-to-speech and real-time communication features that complement Groq's speech recognition for end-to-end AI voice applications.",
        href: "/docs/livekit",
        iconSrc: "/integrations/livekit_white.svg",
        color: "gray",
      },
    ],
    "mcp-integration": [
      {
        title: "BrowserBase",
        description:
          "BrowserBase is a headless browser infrastructure that provides reliable, scalable browser automation for web scraping, testing, and AI applications.",
        href: "/docs/browserbase",
        iconSrc: "/browserbase.png",
        color: "gray",
      },
      {
        title: "BrowserUse",
        description:
          "BrowserUse is an open-source Python library for browser automation that enables AI agents to interact with web pages through natural language commands.",
        href: "/docs/browseruse",
        iconSrc: "/browseruse.svg",
        color: "gray",
      },
      {
        title: "Exa",
        description:
          "Exa is an AI-powered search API that provides high-quality, structured web data for LLMs and AI applications with semantic search capabilities.",
        href: "/docs/exa",
        iconSrc: "/exa-light.png",
        iconDarkSrc: "/exa-dark.png",
        color: "gray",
      },
      {
        title: "Firecrawl",
        description:
          "Firecrawl is a web scraping and crawling API that converts websites into clean, structured markdown or JSON data for LLM consumption.",
        href: "/docs/firecrawl",
        iconSrc: "/firecrawl.png",
        color: "gray",
      },
      {
        title: "HuggingFace",
        description:
          "HuggingFace is a leading AI platform providing access to pre-trained models, datasets, and tools for natural language processing and machine learning.",
        href: "/docs/huggingface",
        iconSrc: "/huggingface.png",
        color: "gray",
      },
      {
        title: "Parallel",
        description:
          "Parallel is an AI-powered tool for automating complex workflows and processes by executing multiple tasks simultaneously across different platforms.",
        href: "/docs/parallel",
        iconSrc: "/parallel.svg",
        color: "gray",
      },
      {
        title: "Tavily",
        description:
          "Tavily is a search API designed specifically for AI agents and LLMs, providing real-time web search capabilities with structured, accurate results.",
        href: "/docs/tavily",
        iconSrc: "/tavily.png",
        color: "gray",
      },
      {
        title: "Mastra",
        description:
          "Mastra is a comprehensive framework for building AI applications with integrated tools, workflows, and data processing capabilities.",
        href: "/docs/mastra",
        iconSrc: "/mastra.png",
        color: "gray",
      },
    ],
  };

## 🦜️🔗 LangChain + Groq

URL: https://console.groq.com/docs/langchain

## 🦜️🔗 LangChain + Groq

While you could use the Groq SDK directly, [LangChain](https://www.langchain.com/) is a framework that makes it easy to build sophisticated applications 
with LLMs. Combined with Groq API for fast inference speed, you can leverage LangChain components such as:

- **Chains:** Compose multiple operations into a single workflow, connecting LLM calls, prompts, and tools together seamlessly (e.g., prompt → LLM → output parser)
- **Prompt Templates:** Easily manage your prompts and templates with pre-built structures to consisently format queries that can be reused across different models
- **Memory:** Add state to your applications by storing and retrieving conversation history and context 
- **Tools:** Extend your LLM applications with external capabilities like calculations, external APIs, or data retrievals
- **Agents:** Create autonomous systems that can decide which tools to use and how to approach complex tasks

### Quick Start (3 minutes to hello world)

#### 1. Install the package:

#### 2. Set up your API key:

#### 3. Create your first LangChain assistant:

Running the below code will create a simple chain that calls a model to extract product information from text and output it
as structured JSON. The chain combines a prompt that tells the model what information to extract, a parser that ensures the output follows a 
specific JSON format, and `llama-3.3-70b-versitable` to do the actual text processing.

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

**Examples:**

Example 1 (bash):
```bash
pip install langchain-groq
```

Example 2 (bash):
```bash
export GROQ_API_KEY="your-groq-api-key"
```

---

## Your Data in GroqCloud

**URL:** llms-txt#your-data-in-groqcloud

**Contents:**
- What Data Groq Retains
- When Customer Data May Be Retained
  - 1. Application State
  - 2. System Reliability and Abuse Monitoring
- Summary Table
- Zero Data Retention
- Data Location
- Key Takeaways
- Browser Search: Quickstart (js)
- Browser Search: Quickstart (py)

Understand how Groq uses customer data and the controls you have.

## What Data Groq Retains

Groq handles two distinct types of information:

1. **Usage Metadata (always retained)**  
   * We collect usage metadata for all users to measure service activity and system performance.  
   * This metadata does **not** contain customer inputs or outputs.

2. **Customer Data (retained only in limited circumstances)**  
   * **By default, Groq does not retain customer data for inference requests.**  
   * Customer data (inputs, outputs, and related state) is only retained in two cases:  
     1. **If you use features that require data retention to function** (e.g., batch jobs, fine-tuning and LoRAs).  
     2. **If needed to protect platform reliability** (e.g., to troubleshoot system failures or investigate abuse).

You can control these settings yourself in the [Data Controls settings](https://console.groq.com/settings/data-controls).

## When Customer Data May Be Retained

Review the [Data Location](#data-location) section below to learn where data is retained.

### 1. Application State

Certain API features require data retention to function:

* **Batch Processing (`/openai/v1/batches`)**: Input and output files retained for 30 days unless deleted earlier by the customer.  
* **Fine-tuning (`/openai/v1/fine_tunings`)**: Model weights and training datasets retained until deleted by the customer.

To prevent data retention for application state, you can disable these features for all users in your organization in [Data Controls settings](https://console.groq.com/settings/data-controls).

### 2. System Reliability and Abuse Monitoring

As noted above, inference requests are not retained by default. We may temporarily log inputs and outputs **only when**:

* Troubleshooting errors that degrade platform reliability, or  
* Investigating suspected abuse (e.g. rate-limit circumvention).

These logs are retained for up to **30 days**, unless legally required to retain longer. You may opt out of this storage in [Data Controls settings](https://console.groq.com/settings/data-controls), but you remain responsible for ensuring safe, compliant usage of the services in accordance with [the terms](https://groq.com/terms-of-use) and [Acceptable Use & Responsible AI Policy](/docs/legal/ai-policy).

| Product | Endpoints | Data Retention Type | Retention Period | ZDR Eligible |
| ------- | ----- | ----- | ----- | ----- |
| Inference | `/openai/v1/chat/completions`<br/>`/openai/v1/responses`<br/>`/openai/v1/audio/transcriptions`<br/>`/openai/v1/audio/translations`<br/>`/openai/v1/audio/speech` | System reliability and abuse monitoring | Up to 30 days | Yes |
| Batch | `/openai/v1/batches`<br/>`/openai/v1/files` (purpose: `batch`) | Application state | Up to 30 days | Yes (feature disabled) |
| Fine-tuning | `/openai/v1/fine_tunings`<br/>`/openai/v1/files` (purpose: `fine_tuning`) | Application state | Until deleted | Yes (feature disabled) |

## Zero Data Retention

All customers may enable Zero Data Retention (ZDR) in [Data Controls settings](https://console.groq.com/settings/data-controls).
When ZDR is enabled, Groq will not retain customer data for system reliability and abuse monitoring. As noted above, this also means that features that rely on data retention to function will be disabled. Organization admins can decide to enable ZDR globally or on a per-feature basis at any time on the Data Controls page in [Data Controls settings](https://console.groq.com/settings/data-controls).

All customer data is retained in Google Cloud Platform (GCP) buckets located in the United States. Groq maintains strict access controls and security standards as detailed in the [Groq Trust Center](https://trust.groq.com/). Where applicable, Customers can rely on standard contractual clauses (SCCs) for transfers between third countries and the U.S.

* **Usage metadata**: always collected, never includes customer data.  
* **Customer data**: not retained by default. Only retained if you opt into persistence features, or in cases for system reliability and abuse monitoring.  
* **Controls**: You can manage data retention in [Data Controls settings](https://console.groq.com/settings/data-controls), including opting into **Zero Data Retention**.

## Browser Search: Quickstart (js)

URL: https://console.groq.com/docs/browser-search/scripts/quickstart

## Browser Search: Quickstart (py)

URL: https://console.groq.com/docs/browser-search/scripts/quickstart.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user", 
            "content": "What happened in AI last week? Give me a concise, one paragraph summary of the most important events."
        }
    ],
    model="openai/gpt-oss-20b",
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    stream=False,
    stop=None,
    tool_choice="required",
    tools=[
        {
            "type": "browser_search"
        }
    ]
)

print(chat_completion.choices[0].message.content)

URL: https://console.groq.com/docs/browser-search

**Examples:**

Example 1 (javascript):
```javascript
import { Groq } from 'groq-sdk';

const groq = new Groq();

const chatCompletion = await groq.chat.completions.create({
  "messages": [
    {
      "role": "user",
      "content": "What happened in AI last week? Give me a concise, one paragraph summary of the most important events."
    }
  ],
  "model": "openai/gpt-oss-20b",
  "temperature": 1,
  "max_completion_tokens": 2048,
  "top_p": 1,
  "stream": false,
  "reasoning_effort": "medium",
  "stop": null,
  "tool_choice": "required",
  "tools": [
    {
      "type": "browser_search"
    }
  ]
});

console.log(chatCompletion.choices[0].message.content);
```

---
