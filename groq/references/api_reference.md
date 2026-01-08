# Groq - Api Reference

**Pages:** 7

---

## API Error Codes and Responses

**URL:** llms-txt#api-error-codes-and-responses

**Contents:**
- Success Codes
- Client Error Codes
- Server Error Codes
- Informational Codes
- Error Object Explanation
- Error Object Structure
- Components
- Toolhouse 🛠️🏠
- Toolhouse 🛠️🏠
  - Getting Started

Our API uses standard HTTP response status codes to indicate the success or failure of an API request. In cases of errors, the body of the response will contain a JSON object with details about the error. Below are the error codes you may encounter, along with their descriptions and example response bodies.

- **200 OK**: The request was successfully executed. No further action is needed.

## Client Error Codes

- **400 Bad Request**: The server could not understand the request due to invalid syntax. Review the request format and ensure it is correct.
- **401 Unauthorized**: The request was not successful because it lacks valid authentication credentials for the requested resource. Ensure the request includes the necessary authentication credentials and the api key is valid.
- **403 Forbidden**: The request is not allowed due to permission restrictions. Ensure the request includes the necessary permissions to access the resource or that your permissions are configured correctly to make a request to the resource.
- **404 Not Found**: The requested resource could not be found. Check the request URL and the existence of the resource.
- **413 Request Entity Too Large**: The request body is too large. Please reduce the size of the request body.
- **422 Unprocessable Entity**: The request was well-formed, but could not be followed due to semantic errors or model hallucinations. Verify the data provided for correctness and completeness or retry your request.
- **424 Failed Dependency**: The request failed because the dependent request failed. This may occur when using [Remote MCP](/docs/mcp) in the case of authentication issues.
- **429 Too Many Requests**: Too many requests were sent in a given timeframe. Implement request throttling and respect rate limits.
- **498 Custom: Flex Tier Capacity Exceeded**: This is a custom status code we use and will return in the event that the flex tier is at capacity and the request won't be processed. You can try again later.
- **499 Custom: Request Cancelled**: This is a custom status code we use in our logs page to signify when the request is cancelled by the caller.

## Server Error Codes

- **500 Internal Server Error**: A generic error occurred on the server. Try the request again later or contact support if the issue persists.
- **502 Bad Gateway**: The server received an invalid response from an upstream server. This may be a temporary issue; retrying the request might resolve it.
- **503 Service Unavailable**: The server is not ready to handle the request, often due to maintenance or overload. Wait before retrying the request.

## Informational Codes

- **206 Partial Content**: Only part of the resource is being delivered, usually in response to range headers sent by the client. Ensure this is expected for the request being made.

## Error Object Explanation

When an error occurs, our API returns a structured error object containing detailed information about the issue. This section explains the components of the error object to aid in troubleshooting and error handling.

## Error Object Structure

The error object follows a specific structure, providing a clear and actionable message alongside an error type classification:

- **`error` (object):** The primary container for error details.
  - **`message` (string):** A descriptive message explaining the nature of the error, intended to aid developers in diagnosing the problem.
  - **`type` (string):** A classification of the error type, such as `"invalid_request_error"`, indicating the general category of the problem encountered.

URL: https://console.groq.com/docs/toolhouse

## Toolhouse 🛠️🏠
[Toolhouse](https://toolhouse.ai) is the first Backend-as-a-Service for the agentic stack. Toolhouse allows you to define agents as configuration, and to deploy them as APIs. Toolhouse agents are automatically connected to 40+ tools including RAG, MCP servers, web search, webpage readers, memory, storage, statefulness and more. With Toolhouse, you can build both conversational and autonomous agents without the need to host and maintain your own infrastructure.

You can use Groq’s fast inference with Toolhouse. This page shows you how to use Llama 4 Maverick and Groq’s Compound Beta to build a Toolhouse agent.

#### Step 1: Download the Toolhouse CLI

Download the Toolhouse CLI by typing this command on your Terminal:

#### Step 2: Log into Toolhouse

Log into Toolhouse via the CLI:

Follow the instructions to create a free Sandbox account.

#### Step 3: Add your Groq API Key to Toolhouse

Generate a Groq API Key in your [Groq Console](https://console.groq.com/keys), then copy its value.

In the CLI, set your Groq API Key:

You’re all set! From now on, you’ll be able to use Groq models with your Toolhouse agents. For a list of supported models, refer to the [Toolhouse models page](https://docs.toolhouse.ai/toolhouse/bring-your-model#supported-models).

## Using Toolhouse with Llama 4 models

To use a specific model, simply reference the model identifier in your agent file, for example:

- For Llama 4 Maverick: `@groq/meta-llama/llama-4-maverick-17b-128e-instruct`
- For Llama 4 Scout: `@groq/meta-llama/llama-4-scout-17b-16e-instruct`

Here’s an example of a working agent file. You can copy this file and save it as `groq.yaml` . In this example, we use an image generation tool, along with Maverick.

You will see something like this:

If the results look good to you, you can deploy this agent using `th deploy groq.yaml`

## Using Toolhouse with Compound Beta

Compound Beta is an advanced AI system that is designed to agentically [search the web and execute code](/docs/agentic-tooling), while being optimized for latency.

To use Compound Beta, simply specify `@groq/compound-beta` or `@groq/compound-beta-mini` as the model identifier. In this example, Compound Beta will search the web under the hood. Save the following file as `groq.yaml`:

Run it with the following command:

You will see something like this:

Then to deploy the agent as an API:

## Flex Processing: Example1 (py)

URL: https://console.groq.com/docs/flex-processing/scripts/example1.py

## Flex Processing: Example1 (js)

URL: https://console.groq.com/docs/flex-processing/scripts/example1

## Flex Processing: Example1 (json)

URL: https://console.groq.com/docs/flex-processing/scripts/example1.json

URL: https://console.groq.com/docs/flex-processing

**Examples:**

Example 1 (json):
```json
{
  "error": {
    "message": "String - description of the specific error",
    "type": "invalid_request_error"
  }
}
```

Example 2 (bash):
```bash
npm i -g @toolhouseai/cli
```

Example 3 (bash):
```bash
th login
```

Example 4 (bash):
```bash
th secrets set GROQ_API_KEY=(replace this with your Groq API Key)
```

---

## Configure environment variables for Phoenix

**URL:** llms-txt#configure-environment-variables-for-phoenix

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

---

## Ensure your GROQ_API_KEY is set as an environment variable

**URL:** llms-txt#ensure-your-groq_api_key-is-set-as-an-environment-variable

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

user_query = "What were the main highlights from the latest Apple keynote event?"

---

## Groq API Reference

**URL:** llms-txt#groq-api-reference

**Contents:**
- Rate Limits

URL: https://console.groq.com/docs/rate-limits

---

## Groq Batch API

**URL:** llms-txt#groq-batch-api

**Contents:**
- What is Batch Processing?
- Overview
- Model Availability and Pricing
- Getting Started
  - 1. Prepare Your Batch File

Process large-scale workloads asynchronously with our Batch API.

## What is Batch Processing?
Batch processing lets you run thousands of API requests at scale by submitting your workload as an asynchronous batch of requests to Groq with 50% lower cost, no impact to your standard rate limits, and 24-hour to 7 day processing window.

## Overview
While some of your use cases may require synchronous API requests, asynchronous batch processing is perfect for use cases that don't need immediate reponses or for processing a large number of queries that standard rate limits cannot handle, such as processing large datasets, generating content in bulk, and running evaluations.

Compared to using our synchronous API endpoints, our Batch API has:
- **Higher rate limits:** Process thousands of requests per batch with no impact on your standard API rate limits
- **Cost efficiency:** 50% cost discount compared to synchronous APIs

## Model Availability and Pricing
The Batch API can currently be used to execute queries for chat completion (both text and vision), audio transcription, and audio translation inputs with the following models:

| Model ID                        | Model                          |
|---------------------------------|--------------------------------|
| openai/gpt-oss-20b                  | GPT-OSS 20B                  |
| openai/gpt-oss-120b                  | GPT-OSS 120B                  |
| meta-llama/llama-4-maverick-17b-128e-instruct | Llama 4 Maverick |
| meta-llama/llama-4-scout-17b-16e-instruct | Llama 4 Scout |
| llama-3.3-70b-versatile | Llama 3.3 70B |
| llama-3.1-8b-instant | Llama 3.1 8B Instant |
| meta-llama/llama-guard-4-12b | Llama Guard 4 12B |

Pricing is at a 50% cost discount compared to [synchronous API pricing.](https://groq.com/pricing)

## Getting Started
Our Batch API endpoints allow you to collect a group of requests into a single file, kick off a batch processing job to execute the requests within your file, query for the status of your batch, and eventually 
retrieve the results when your batch is complete.

Multiple batch jobs can be submitted at once.

Each batch has a processing window, during which we'll process as many requests as our capacity allows while maintaining service quality for all users. We allow for setting 
a batch window from 24 hours to 7 days and recommend setting a longer batch window allow us more time to complete your batch jobs instead of expiring them.

### 1. Prepare Your Batch File
A batch is composed of a list of API requests and every batch job starts with a JSON Lines (JSONL) file that contains the requests
you want processed. Each line in this file represents a single API call.

The Groq Batch API currently supports:
- Chat completion requests through `/v1/chat/completions`
- Audio transcription requests through `/v1/audio/transcriptions`
- Audio translation requests through `/v1/audio/translations`

The structure for each line must include:
- `custom_id`: Your unique identifier for tracking the batch request
- `method`: The HTTP method (currently `POST` only)
- `url`: The API endpoint to call (one of: `/v1/chat/completions`, `/v1/audio/transcriptions`, or `/v1/audio/translations`)
- `body`: The parameters of your request matching our synchronous API format.

The following is an example of a JSONL batch file with different types of requests:

#### Converting Sync Calls to Batch Format 
If you're familiar with making synchronous API calls, converting them to batch format is straightforward. Here's how a regular API call transforms
into a batch request:

**Examples:**

Example 1 (json):
```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+3?"}]}}
{"custom_id": "request-3", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "count up to 1000000. starting with 1, 2, 3. print all the numbers, do not stop until you get to 1000000."}]}}
```

---

## Responses API

**URL:** llms-txt#responses-api

**Contents:**
- Configuring OpenAI Client for Responses API
- Multi-turn Conversations
- Image Inputs
- Built-In Tools
  - Model Support
  - Code Execution Example
  - Browser Search Example
- Structured Outputs
  - Using a Schema Validation Library
- Reasoning

Groq's Responses API is fully compatible with OpenAI's Responses API, making it easy to integrate advanced conversational AI capabilities into your applications. The Responses API supports both text and image inputs while producing text outputs, stateful conversations, and function calling to connect with external systems.

## Configuring OpenAI Client for Responses API

To use the Responses API with OpenAI's client libraries, configure your client with your Groq API key and set the base URL to `https://api.groq.com/openai/v1`.

You can find your API key [here](/keys).

## Multi-turn Conversations

The Responses API on Groq doesn't support stateful conversations yet, so you'll need to keep track of the conversation history yourself and provide it in every request.

The Responses API supports image inputs with all [vision-capable models](/docs/vision). Here's an example of how to pass an image to the model.

In addition to a model's regular [tool use capabilities](/docs/tool-use), the Responses API supports various built-in tools to extend your model's capabilities.

While all models support the Responses API, these built-in tools are only supported for the following models:

| Model ID                        | Browser Search | Code Execution |
|---------------------------------|--------------------------------|--------------------------------|
| [openai/gpt-oss-20b](/docs/model/openai/gpt-oss-20b) | ✅ | ✅ |
| [openai/gpt-oss-120b](/docs/model/openai/gpt-oss-120b) | ✅ | ✅ |

### Code Execution Example

Enable your models to write and execute Python code for calculations, data analysis, and problem-solving - see our [code execution documentation](/docs/code-execution) for more details.

### Browser Search Example

Give your models access to real-time web content and up-to-date information - see our [browser search documentation](/docs/browser-search) for more details.

## Structured Outputs

Use structured outputs to ensure the model's response follows a specific JSON schema. This is useful for extracting structured data from text, ensuring consistent response formats, or integrating with downstream systems that expect specific data structures.

For a complete list of models that support structured outputs, see our [structured outputs documentation](/docs/structured-outputs).

### Using a Schema Validation Library

When working with Structured Outputs, you can use popular schema validation libraries like [Zod](https://zod.dev/) for TypeScript and [Pydantic](https://docs.pydantic.dev/latest/) for Python. These libraries provide type safety, runtime validation, and seamless integration with JSON Schema generation.

Use reasoning to let the model produce an internal chain of thought before generating a response. This is useful for complex problem solving, multi-step agentic workflow planning, and scientific analysis.

For a complete list of models that support reasoning, see our [reasoning documentation](/docs/reasoning).

## Model Context Protocol (MCP)

The Responses API also supports the [Model Context Protocol (MCP)](/docs/mcp), an open-source standard that enables AI applications to connect with external systems like databases, APIs, and tools. MCP provides a standardized way for AI models to access and interact with your data and workflows.

With MCP, you can build AI agents that access your codebase through GitHub, query databases with natural language, browse the web for real-time information, or connect to any API-based service like Slack, Notion, or Google Calendar.

Here's an example using [Hugging Face's MCP server](https://huggingface.co/settings/mcp) to search for trending AI models.

## Unsupported Features

Although Groq's Responses API is mostly compatible with OpenAI's Responses API, there are a few features we don't support just yet:

- `previous_response_id`
- `store`
- `truncation`
- `include`
- `safety_identifier`
- `prompt_cache_key`

Want to see one of these features supported? Let us know on our [Community forum](https://community.groq.com)!

## Detailed Usage Metrics

To include detailed usage metrics for each request (such as exact inference time), set the following header:

In the response body, the `metadata` field will include the following keys:
- `completion_time`: The time in seconds it took to generate the output
- `prompt_time`: The time in seconds it took to process the input prompt
- `queue_time`: The time in seconds the requests was queued before being processed
- `total_time`: The total time in seconds it took to process the request

To calculate output tokens per second, combine the information from the `usage` field with the `metadata` field:

Explore more advanced use cases in our built-in [browser search](/docs/browser-search) and [code execution](/docs/code-execution) documentation, or learn about connecting to external systems with [MCP](/docs/mcp).

## Vision: Vision (js)

URL: https://console.groq.com/docs/vision/scripts/vision

## Vision: Jsonmode (py)

URL: https://console.groq.com/docs/vision/scripts/jsonmode.py

from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List what you observe in this photo in JSON format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/SF_From_Marin_Highlands3.jpg"
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None,
)

print(completion.choices[0].message)

## Vision: Vision (json)

URL: https://console.groq.com/docs/vision/scripts/vision.json

## Function to encode the image

URL: https://console.groq.com/docs/vision/scripts/local.py

from groq import Groq
import base64
import os

**Examples:**

Example 1 (text):
```text
Groq-Beta: inference-metrics
```

Example 2 (json):
```json
{
  "metadata": {
    "completion_time": "2.567331286",
    "prompt_time": "0.003652567",
    "queue_time": "0.018393202",
    "total_time": "2.570983853"
  }
}
```

Example 3 (text):
```text
output_tokens_per_second = usage.output_tokens / metadata.completion_time
```

Example 4 (javascript):
```javascript
import { Groq } from 'groq-sdk';

const groq = new Groq();
async function main() {
  const chatCompletion = await groq.chat.completions.create({
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What's in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
            }
          }
        ]
      }
    ],
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "temperature": 1,
    "max_completion_tokens": 1024,
    "top_p": 1,
    "stream": false,
    "stop": null
  });

   console.log(chatCompletion.choices[0].message.content);
}

main();
```

---

## Set your API key

**URL:** llms-txt#set-your-api-key

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

---
