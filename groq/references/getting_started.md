# Groq - Getting Started

**Pages:** 3

---

## Introduction to Tool Use

**URL:** llms-txt#introduction-to-tool-use

**Contents:**
- Supported Models
- Agentic Tooling
- How Tool Use Works
- Tool Use with Groq
  - Tools Specifications
  - Tool Calls Structure
  - Tool Calls Response
- Setting Up Tools
- Parallel Tool Use
- Error Handling

Tool use is a powerful feature that allows Large Language Models (LLMs) to interact with external resources, such as APIs, databases, and the web, to gather dynamic data they wouldn't otherwise have access to in their pre-trained (or static) state and perform actions beyond simple text generation. 
<br />
Tool use bridges the gap between the data that the LLMs were trained on with dynamic data and real-world actions, which opens up a wide array of realtime use cases for us to build powerful applications with, especially with Groq's insanely fast inference speed. 🚀

## Supported Models
| Model ID                         | Tool Use Support? | Parallel Tool Use Support? | JSON Mode Support? |
|----------------------------------|-------------------|----------------------------|--------------------|
| moonshotai/kimi-k2-instruct-0905                   | Yes               | Yes                        | Yes                |
| openai/gpt-oss-20b                   | Yes               | No                         | Yes                |
| openai/gpt-oss-safeguard-20b                   | Yes               | No                         | Yes                |
| openai/gpt-oss-120b                   | Yes               | No                         | Yes                |
| qwen/qwen3-32b                   | Yes               | Yes                         | Yes                |
| meta-llama/llama-4-scout-17b-16e-instruct                   | Yes               | Yes                        | Yes                |
| meta-llama/llama-4-maverick-17b-128e-instruct                   | Yes               | Yes                        | Yes                |
| llama-3.3-70b-versatile        | Yes               | Yes                        | Yes                |
| llama-3.1-8b-instant           | Yes               | Yes                        | Yes                |

In addition to the models that support custom tools above, Groq also offers agentic tool systems.
These are AI systems with tools like web search and code execution built directly into the system.
You don't need to specify any tools yourself - the system will automatically use its built-in tools as needed.
<br/

## How Tool Use Works
Groq API tool use structure is compatible with OpenAI's tool use structure, which allows for easy integration. See the following cURL example of a tool use request:
<br />

<br />
To integrate tools with Groq API, follow these steps:
1. Provide tools (or predefined functions) to the LLM for performing actions and accessing external data in 
real-time in addition to your user prompt within your Groq API request
2. Define how the tools should be used to teach the LLM how to use them effectively (e.g. by defining input and 
output formats)
3. Let the LLM autonomously decide whether or not the provided tools are needed for a user query by evaluating the user 
query, determining whether the tools can enhance its response, and utilizing the tools accordingly
4. Extract tool input, execute the tool code, and return results
5. Let the LLM use the tool result to formulate a response to the original prompt

This process allows the LLM to perform tasks such as real-time data retrieval, complex calculations, and external API 
interaction, all while maintaining a natural conversation with our end user.

## Tool Use with Groq

Groq API endpoints support tool use to almost instantly deliver structured JSON output that can be used to directly invoke functions from 
desired external resources.

### Tools Specifications
Tool use is part of the [Groq API chat completion request payload](https://console.groq.com/docs/api-reference#chat-create).
Groq API tool calls are structured to be OpenAI-compatible.

### Tool Calls Structure
The following is an example tool calls structure:

### Tool Calls Response
The following is an example tool calls response based on the above:

<br />
When a model decides to use a tool, it returns a response with a `tool_calls` object containing:
- `id`: a unique identifier for the tool call
- `type`: the type of tool calls, i.e. function
- `name`: the name of the tool being used
- `parameters`: an object containing the input being passed to the tool

## Setting Up Tools
To get started, let's go through an example of tool use with Groq API that you can use as a base to build more tools on
your own.
<br />
#### Step 1: Create Tool
Let's install Groq SDK, set up our Groq client, and create a function called `calculate` to evaluate a mathematical 
expression that we will represent as a tool.
<br />
Note: In this example, we're defining a function as our tool, but your tool can be any function or an external
resource (e.g. dabatase, web search engine, external API).

#### Step 2: Pass Tool Definition and Messages to Model 
Next, we'll define our `calculate` tool within an array of available `tools` and call our Groq API chat completion. You 
can read more about tool schema and supported required and optional fields above in [Tool Specifications](#tool-call-and-tool-response-structure).
<br />
By defining our tool, we'll inform our model about what our tool does and have the model decide whether or not to use the
tool. We should be as descriptive and specific as possible for our model to be able to make the correct tool use decisions.
<br />
In addition to our `tools` array, we will provide our `messages` array (e.g. containing system prompt, assistant prompt, and/or
user prompt).

#### Step 3: Receive and Handle Tool Results
After executing our chat completion, we'll extract our model's response and check for tool calls.
<br />
If the model decides that no tools should be used and does not generate a tool or function call, then the response will 
be a normal chat completion (i.e. `response_message = response.choices[0].message`) with a direct model reply to the user query. 
<br />
If the model decides that tools should be used and generates a tool or function call, we will:
1. Define available tool or function
2. Add the model's response to the conversation by appending our message
3. Process the tool call and add the tool response to our message
4. Make a second Groq API call with the updated conversation
5. Return the final response

## Parallel Tool Use
We learned about tool use and built single-turn tool use examples above. Now let's take tool use a step further and imagine
a workflow where multiple tools can be called simultaneously, enabling more efficient and effective responses.
<br />
This concept is known as **parallel tool use** and is key for building agentic workflows that can deal with complex queries, 
which is a great example of where inference speed becomes increasingly important (and thankfully we can access fast inference
speed with Groq API). 
<br />
Here's an example of parallel tool use with a tool for getting the temperature and the tool for getting the weather condition
to show parallel tool use with Groq API in action:

## Error Handling
Groq API tool use is designed to verify whether a model generates a valid tool calls object. When a model fails to generate a valid tool calls object, 
Groq API will return a 400 error with an explanation in the "failed_generation" field of the JSON body that is returned.

### Next Steps
For more information and examples of working with multiple tools in parallel using Groq API and Instructor, see our Groq API Cookbook
tutorial [here](https://github.com/groq/groq-api-cookbook/blob/main/tutorials/parallel-tool-use/parallel-tool-use.ipynb).

## Tool Use with Structured Outputs (Python)
Groq API offers best-effort matching for parameters, which means the model could occasionally miss parameters or 
misinterpret types for more complex tool calls. We recommend the [Instuctor](https://python.useinstructor.com/hub/groq/)
library to simplify the process of working with structured data and to ensure that the model's output adheres to a predefined
schema.
<br />
Here's an example of how to implement tool use using the Instructor library with Groq API:
<br />

### Benefits of Using Structured Outputs
- Type Safety: Pydantic models ensure that output adheres to the expected structure, reducing the risk of errors.
- Automatic Validation: Instructor automatically validates the model's output against the defined schema.

### Next Steps
For more information and examples of working with structured outputs using Groq API and Instructor, see our Groq API Cookbook
tutorial [here](https://github.com/groq/groq-api-cookbook/blob/main/tutorials/structured-output-instructor/structured_output_instructor.ipynb).

## Streaming Tool Use
The Groq API also offers streaming tool use, where you can stream tool use results to the client as they are generated.

- Provide detailed tool descriptions for optimal performance.
- We recommend tool use with the Instructor library for structured outputs.
- Implement a routing system when using fine-tuned models in your workflow.
- Handle tool execution errors by returning error messages with `"is_error": true`.

## Google Cloud Private Service Connect

URL: https://console.groq.com/docs/security/gcp-private-service-connect

## Google Cloud Private Service Connect

Private Service Connect (PSC) enables you to access Groq's API services through private network connections, eliminating exposure to the public internet. This guide explains how to set up Private Service Connect for secure access to Groq services.

Groq exposes its API endpoints in Google Cloud Platform as PSC _published services_. By configuring PSC endpoints, you can:
- Access Groq services through private IP addresses within your VPC
- Eliminate public internet exposure
- Maintain strict network security controls
- Minimize latency
- Reduce data transfer costs

- A Google Cloud project with [Private Service Connect enabled](https://cloud.google.com/vpc/docs/configure-private-service-connect-consumer)
- VPC network where you want to create the PSC endpoint
- Appropriate IAM permissions to create PSC endpoints and DNS zones
- Enterprise plan with Groq
- Provided Groq with your GCP Project ID
- Groq has accepted your GCP Project ID to our Private Service Connect

The steps below use us as an example. Make sure you configure your system
according to the region(s) you want to use.

#### 1. Connect an endpoint

1. Navigate to **Network services** > **Private Service Connect** in your Google Cloud Console
2. Go to the **Endpoints** section and click **Connect endpoint**
   * Under **Target**, select _Published service_
   * For **Target service**, enter a [published service](#published-services) target name.
   * For **Endpoint name**, enter a descriptive name (e.g., `groq-api-psc`)
   * Select your desired **Network** and **Subnetwork**
   * For **IP address**, create and select an internal IP from your subnet
   * Enable **Global access** if you need to connect from multiple regions
3. Click **Add endpoint** and verify the status shows as _Accepted_

#### 2. Configure Private DNS

1. Go to **Network services** > **Cloud DNS** in your Google Cloud Console
2. Create the first zone for groq.com:
   * Click **Create zone**
   * Set **Zone type** to _Private_
   * Enter a descriptive **Zone name** (e.g., `groq-api-private`)
   * For **DNS name**, enter `groq.com.`
   * Create an `A` record:
     * **DNS name**: `api`
     * **Resource record type**: `A`
     * Enter your PSC endpoint IP address
   * Link the private zone to your VPC network

3. Create the second zone for groqcloud.com:
   * Click **Create zone**
   * Set **Zone type** to _Private_
   * Enter a descriptive **Zone name** (e.g., `groqcloud-api-private`)
   * For **DNS name**, enter `groqcloud.com.`
   * Create an `A` record:
     * **DNS name**: `api.us`
     * **Resource record type**: `A`
     * Enter your PSC endpoint IP address
   * Link the private zone to your VPC network

#### 3. Validate the Connection

To verify your setup:

1. SSH into a VM in your VPC network
2. Test DNS resolution for both endpoints:
   
   Both should return your PSC endpoint IP address

3. Test API connectivity (using either endpoint):
   
   Should return a successful response through your private connection

### Published Services

| Service | PSC Target Name | Private DNS Names |
|---------|----------------|-------------------|
| API     | projects/groq-pe/regions/me-central2/serviceAttachments/groqcloud | api.groq.com, api.me-central-1.groqcloud.com |
| API     | projects/groq-pe/regions/us-central1/serviceAttachments/groqcloud | api.groq.com, api.us.groqcloud.com |

If you encounter connectivity issues:

1. Verify DNS resolution is working correctly for both domains
2. Check that your security groups and firewall rules allow traffic to the PSC endpoint
3. Ensure your service account has the necessary permissions
4. Verify the PSC endpoint status is _Accepted_
5. Confirm the model you are requesting is operating in the target region

To monitor and alert on an unexpected change in connectivity status for the PSC endpoint, use a [Google Cloud log-based alerting policy](https://cloud.google.com/logging/docs/alerting/log-based-alerts).

Below is an example of an alert policy that will alert the given notification channel in the case of a connection being _Closed_. This will require manual intervention to reconnect the endpoint.

- [Google Cloud Private Service Connect Documentation](https://cloud.google.com/vpc/docs/private-service-connect)

## Reasoning: Reasoning Hidden (js)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_hidden

## Reasoning: Reasoning Raw (js)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_raw

## Reasoning: Reasoning Gpt Oss High (py)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_gpt-oss-high.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How do airplanes fly? Be concise."
        }
    ],
    model="openai/gpt-oss-20b",
    reasoning_effort="high",
    include_reasoning=True,
    stream=False
)

print(chat_completion.choices[0].message)

## Reasoning: Reasoning Hidden (py)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_hidden.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How do airplanes fly? Be concise."
        }
    ],
    model="qwen/qwen3-32b",
    stream=False,
    reasoning_format="hidden"
)

print(chat_completion.choices[0].message)

## Reasoning: Reasoning Gpt Oss Excl (py)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_gpt-oss-excl.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How do airplanes fly? Be concise."
        }
    ],
    model="openai/gpt-oss-20b",
    stream=False,
    include_reasoning=False
)

print(chat_completion.choices[0].message)

## Reasoning: Reasoning Gpt Oss (py)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_gpt-oss.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How do airplanes fly? Be concise."
        }
    ],
    model="openai/gpt-oss-20b",
    stream=False
)

print(chat_completion.choices[0].message)

## Reasoning: Reasoning Gpt Oss Excl (js)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_gpt-oss-excl

## Reasoning: Reasoning Raw (py)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_raw.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How do airplanes fly? Be concise."
        }
    ],
    model="qwen/qwen3-32b",
    stream=False,
    reasoning_format="raw"
)

print(chat_completion.choices[0].message)

## Reasoning: Reasoning Parsed (js)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_parsed

## Reasoning: Reasoning Gpt Oss High (js)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_gpt-oss-high

## Reasoning: Reasoning Parsed (py)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_parsed.py

from groq import Groq

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How do airplanes fly? Be concise."
        }
    ],
    model="qwen/qwen3-32b",
    stream=False,
    reasoning_format="parsed"
)

print(chat_completion.choices[0].message)

## Reasoning: R1 (py)

URL: https://console.groq.com/docs/reasoning/scripts/r1.py

from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "How many r's are in the word strawberry?"
        }
    ],
    temperature=0.6,
    max_completion_tokens=1024,
    top_p=0.95,
    stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

## Reasoning: R1 (js)

URL: https://console.groq.com/docs/reasoning/scripts/r1

## Reasoning: Reasoning Gpt Oss (js)

URL: https://console.groq.com/docs/reasoning/scripts/reasoning_gpt-oss

URL: https://console.groq.com/docs/reasoning

**Examples:**

Example 1 (bash):
```bash
curl https://api.groq.com/openai/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $GROQ_API_KEY" \
-d '{
  "model": "llama-3.3-70b-versatile",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
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
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}'
```

Example 2 (json):
```json
{
  "model": "llama-3.3-70b-versatile",
  "messages": [
    {
      "role": "system",
      "content": "You are a weather assistant. Use the get_weather function to retrieve weather information for a given location."
    },
    {
      "role": "user",
      "content": "What's the weather like in New York today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
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
              "description": "The unit of temperature to use. Defaults to fahrenheit."
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_completion_tokens": 4096
}'
```

Example 3 (json):
```json
"model": "llama-3.3-70b-versatile",
"choices": [{
    "index": 0,
    "message": {
        "role": "assistant",
        "tool_calls": [{
            "id": "call_d5wg",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"New York, NY\"}"
            }
        }]
    },
    "logprobs": null,
    "finish_reason": "tool_calls"
}],
```

Example 4 (python):
```python
from groq import Groq
import json

client = Groq()

async def main():
    stream = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                # We first ask it to write a Poem, to show the case where there's text output before function calls, since that is also supported
                "content": "What is the weather in San Francisco and in Tokyo? First write a short poem.",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
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
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        stream=True
    )

    async for chunk in stream:
        print(json.dumps(chunk.model_dump()) + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Quickstart

**URL:** llms-txt#quickstart

**Contents:**
- Create an API Key
- Set up your API Key (recommended)
  - In your terminal of choice:
- Requesting your first chat completion
  - Execute this curl command in the terminal of your choice:

Get up and running with the Groq API in a few minutes, with the steps below.

For additional support, catch our [onboarding video](/docs/overview).

Please visit [here](/keys) to create an API Key.

## Set up your API Key (recommended)

Configure your API key as an environment variable. This approach streamlines your API usage by eliminating the need to include your API key in each request. Moreover, it enhances security by minimizing the risk of inadvertently including your API key in your codebase.

### In your terminal of choice:

## Requesting your first chat completion

### Execute this curl command in the terminal of your choice:

**Examples:**

Example 1 (shell):
```shell
export GROQ_API_KEY=<your-api-key-here>
```

---

## Structured Outputs

**URL:** llms-txt#structured-outputs

**Contents:**
- Introduction
- Supported models
  - Getting a structured response from unstructured text
  - SQL Query Generation
  - Email Classification
  - API Response Validation
- Schema Validation Libraries
  - Support Ticket Classification
- Implementation Guide
  - Schema Definition

Guarantee model responses strictly conform to your JSON schema for reliable, type-safe data structures.

## Introduction
Structured Outputs is a feature that makes your model responses strictly conform to your provided [JSON Schema](https://json-schema.org/overview/what-is-jsonschema) or throws an error if the model cannot produce a compliant response. The endpoint provides customers with the ability to obtain reliable data structures.

This feature's performance is dependent on the model's ability to produce a valid answer that matches your schema. If the model fails to generate a conforming response, the endpoint will return an error rather than an invalid or incomplete result.

1. **Binary output:** Either returns valid JSON Schema-compliant output or throws an error
2. **Type-safe responses:** No need to validate or retry malformed outputs
3. **Programmatic refusal detection:** Detect safety-based model refusals programmatically
4. **Simplified prompting:** No complex prompts needed for consistent formatting

In addition to supporting Structured Outputs in our API, our SDKs also enable you to easily define your schemas with [Pydantic](https://docs.pydantic.dev/latest/) and [Zod](https://zod.dev/) to ensure further type safety. The examples below show how to extract structured information from unstructured text.

Structured Outputs is available with the following models:

| Model ID                        | Model                          |
|---------------------------------|--------------------------------|
| `openai/gpt-oss-20b`                  | [GPT-OSS 20B](/docs/model/openai/gpt-oss-20b)
| `openai/gpt-oss-120b`                  | [GPT-OSS 120B](/docs/model/openai/gpt-oss-120b)
| `openai/gpt-oss-safeguard-20b`                  | [Safety GPT OSS 20B](/docs/model/openai/gpt-oss-safeguard-20b)
| `moonshotai/kimi-k2-instruct-0905`                  | [Kimi K2 Instruct](/docs/model/moonshotai/kimi-k2-instruct-0905)
| `meta-llama/llama-4-maverick-17b-128e-instruct` | [Llama 4 Maverick](/docs/model/meta-llama/llama-4-maverick-17b-128e-instruct)
| `meta-llama/llama-4-scout-17b-16e-instruct` | [Llama 4 Scout](/docs/model/meta-llama/llama-4-scout-17b-16e-instruct)

For all other models, you can use [JSON Object Mode](#json-object-mode) to get a valid JSON object, though it may not match your schema.

**Note:** [streaming](/docs/text-chat#streaming-a-chat-completion) and [tool use](/docs/tool-use) are not currently supported with Structured Outputs.

### Getting a structured response from unstructured text

### SQL Query Generation

You can generate structured SQL queries from natural language descriptions, helping ensure proper syntax and including metadata about the query structure.

### Email Classification

You can classify emails into structured categories with confidence scores, priority levels, and suggested actions.

### API Response Validation

You can validate and structure API responses with error handling, status codes, and standardized data formats for reliable integration.

## Schema Validation Libraries

When working with Structured Outputs, you can use popular schema validation libraries like [Zod](https://zod.dev/) for TypeScript and [Pydantic](https://docs.pydantic.dev/latest/) for Python. These libraries provide type safety, runtime validation, and seamless integration with JSON Schema generation.

### Support Ticket Classification

This example demonstrates how to classify customer support tickets using structured schemas with both Zod and Pydantic, ensuring consistent categorization and routing.

## Implementation Guide

### Schema Definition

Design your JSON Schema to constrain model responses. Reference the [examples](#examples) above and see [supported schema features](#schema-requirements) for technical limitations.

**Schema optimization tips:**
- Use descriptive property names and clear descriptions for complex fields
- Create evaluation sets to test schema effectiveness
- Include titles for important structural elements

Include the schema in your API request using the `response_format` parameter:

Complete implementation example:

Schema validation failures return HTTP 400 errors with the message `Generated JSON does not match the expected schema. Please adjust your prompt.`

**Resolution strategies:**
- Retry requests for transient failures
- Refine prompts for recurring schema mismatches
- Simplify complex schemas if validation consistently fails

**User input handling:** Include explicit instructions for invalid or incompatible inputs. Models attempt schema adherence even with unrelated data, potentially causing hallucinations. Specify fallback responses (empty fields, error messages) for incompatible inputs.

**Output quality:** Structured outputs are designed to output schema compliance but not semantic accuracy. For persistent errors, refine instructions, add system message examples, or decompose complex tasks. See the [prompt engineering guide](/docs/prompting) for optimization techniques.

## Schema Requirements

Structured Outputs supports a [JSON Schema](https://json-schema.org/docs) subset with specific constraints for performance and reliability.

### Supported Data Types

- **Primitives:** String, Number, Boolean, Integer
- **Complex:** Object, Array, Enum
- **Composition:** anyOf (union types)

### Mandatory Constraints

**Required fields:** All schema properties must be marked as `required`. Optional fields are not supported.

**Closed objects:** All objects must set `additionalProperties: false` to prevent undefined properties. This ensures strict schema adherence.

**Union types:** Each schema within `anyOf` must comply with all subset restrictions:

**Reusable subschemas:** Define reusable components with `$defs` and reference them using `$ref`:

**Root recursion:** Use `#` to reference the root schema:

**Explicit recursion** through definition references:

JSON Object Mode provides basic JSON output validation without schema enforcement. Unlike Structured Outputs with `json_schema` mode, it is designed to output valid JSON syntax but not schema compliance. The endpoint will either return valid JSON or throw an error if the model cannot produce valid JSON syntax. Use [Structured Outputs](#introduction) when available for your use case.

Enable JSON Object Mode by setting `response_format` to `{ "type": "json_object" }`.

**Requirements and limitations:**
- Include explicit JSON instructions in your prompt (system message or user input)
- Outputs are syntactically valid JSON but may not match your intended schema
- Combine with validation libraries and retry logic for schema compliance

### Sentiment Analysis Example

This example shows prompt-guided JSON generation for sentiment analysis, adaptable to classification, extraction, or summarization tasks:

**Response structure:**
- **sentiment**: Classification (positive/negative/neutral) 
- **confidence_score**: Confidence level (0-1 scale)
- **key_phrases**: Extracted phrases with individual sentiment scores
- **summary**: Analysis overview and main findings

## Speech To Text: Translation (js)

URL: https://console.groq.com/docs/speech-to-text/scripts/translation

import fs from "fs";
import Groq from "groq-sdk";

// Initialize the Groq client
const groq = new Groq();
async function main() {
  // Create a translation job
  const translation = await groq.audio.translations.create({
    file: fs.createReadStream("sample_audio.m4a"), // Required path to audio file - replace with your audio file!
    model: "whisper-large-v3", // Required model to use for translation
    prompt: "Specify context or spelling", // Optional
    language: "en", // Optional ('en' only)
    response_format: "json", // Optional
    temperature: 0.0, // Optional
  });
  // Log the transcribed text
  console.log(translation.text);
}
main();

## Initialize the Groq client

URL: https://console.groq.com/docs/speech-to-text/scripts/transcription.py

```python
import os
import json
from groq import Groq

**Examples:**

Example 1 (json):
```json
{
    "query": "SELECT c.name, c.email, SUM(o.total_amount) as total_order_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY) AND o.total_amount > 500 GROUP BY c.customer_id, c.name, c.email ORDER BY total_order_amount DESC",
    "query_type": "SELECT",
    "tables_used": ["customers", "orders"],
    "estimated_complexity": "medium",
    "execution_notes": [
      "Query uses JOIN to connect customers and orders tables",
      "DATE_SUB function calculates 30 days ago from current date",
      "GROUP BY aggregates orders per customer",
      "Results ordered by total order amount descending"
    ],
    "validation_status": {
      "is_valid": true,
      "syntax_errors": []
    }
}
```

Example 2 (json):
```json
{
    "category": "urgent",
    "priority": "critical",
    "confidence_score": 0.95,
    "sentiment": "negative",
    "key_entities": [
        {
          "entity": "production server",
          "type": "system"
        },
        {
          "entity": "2:30 PM EST",
          "type": "datetime"
        },
        {
          "entity": "DevOps Team",
          "type": "organization"
        },
        {
          "entity": "customer-facing services",
          "type": "system"
        }
    ],
    "suggested_actions": [
        "Join emergency call immediately",
        "Escalate to senior DevOps team",
        "Activate incident response protocol",
        "Prepare customer communication",
        "Monitor service restoration progress"
    ],
    "requires_immediate_attention": true,
    "estimated_response_time": "immediate"
}
```

Example 3 (json):
```json
{
    "validation_result": {
        "is_valid": false,
        "status_code": 400,
        "error_count": 2
    },
    "field_validations": [
        {
            "field_name": "user_id",
            "field_type": "string",
            "is_valid": true,
            "error_message": "",
            "expected_format": "string"
        },
        {
            "field_name": "email",
            "field_type": "string",
            "is_valid": false,
            "error_message": "Invalid email format",
            "expected_format": "valid email address (e.g., user@example.com)"
        }
    ],
    "data_quality_score": 0.7,
    "suggested_fixes": [
        "Fix email format validation to ensure proper email structure",
        "Add proper error handling structure to response"
    ],
    "compliance_check": {
        "follows_rest_standards": false,
        "has_proper_error_handling": false,
        "includes_metadata": false
    }
}
```

Example 4 (json):
```json
{
    "category": "feature_request",
    "priority": "low",
    "urgency_score": 2.5,
    "customer_info": {
        "name": "Mike",
        "company": "StartupXYZ",
        "tier": "paid"
    },
    "technical_details": [
        {
            "component": "dashboard",
            "description": "Request for dark mode feature"
        },
        {
            "component": "user_interface",
            "description": "Request for keyboard shortcuts"
        }
    ],
    "keywords": ["dark mode", "dashboard", "keyboard shortcuts", "enhancement"],
    "requires_escalation": false,
    "estimated_resolution_hours": 40,
    "summary": "Feature request for dark mode and keyboard shortcuts from paying customer"
}
```

---
