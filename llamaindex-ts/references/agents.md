# Llamaindex-Ts - Agents

**Pages:** 19

---

## 1. Setup

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/1_setup/

**Contents:**
- 1. Setup
- What is an Agent?
- Install LlamaIndex.TS
- Choose your model
- Get an OpenAI API key

In this guide we’ll walk you through the process of building an Agent in JavaScript using the LlamaIndex.TS library, starting from nothing and adding complexity in stages.

In LlamaIndex, an agent is a semi-autonomous piece of software powered by an LLM that is given a task and executes a series of steps towards solving that task. It is given a set of tools, which can be anything from arbitrary functions up to full LlamaIndex query engines, and it selects the best available tool to complete each step. When each step is completed, the agent judges whether the task is now complete, in which case it returns a result to the user, or whether it needs to take another step, in which case it loops back to the start.

You’ll need to have a recent version of Node.js installed. Then you can install LlamaIndex.TS by running

By default we’ll be using OpenAI with GPT-4, as it’s a powerful model and easy to get started with. If you’d prefer to run a local model, see using a local model.

If you don’t already have one, you can sign up for an OpenAI API key. You should then put the key in a .env file in the root of the project; the file should look like

We’ll use dotenv to pull the API key out of that .env file, so also run:

Now you’re ready to create your agent.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/readers @llamaindex/huggingface @llamaindex/workflow
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/readers @llamaindex/huggingface @llamaindex/workflow
```

Example 3 (unknown):
```unknown
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX
```

Example 4 (unknown):
```unknown
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX
```

---

## 2. Create a basic agent

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/2_create_agent/

**Contents:**
- 2. Create a basic agent
  - Load your dependencies
  - Initialize your LLM
  - Create a function
  - Turn the function into a tool for the agent
  - Create the agent
  - Ask the agent a question
  - Logging workflow events

We want to use await so we’re going to wrap all of our code in a main function, like this:

For the rest of this guide we’ll assume your code is wrapped like this so we can use await. You can run the code this way:

First we’ll need to pull in our dependencies. These are:

We need to tell our OpenAI class where its API key is, and which of OpenAI’s models to use. We’ll be using gpt-4o, which is capable while still being pretty cheap. This is a global setting, so anywhere an LLM is needed will use the same model.

We’re going to create a very simple function that adds two numbers together. This will be the tool we ask our agent to use.

Note that we’re passing in an object with two named parameters, a and b. This is a little unusual, but important for defining a tool that an LLM can use.

This is the most complicated part of creating an agent. We need to define a tool. We have to pass in:

We then wrap up the tools into an array. We could provide lots of tools this way, but for this example we’re just using the one.

With your LLM already set up and your tools defined, creating an agent is simple:

We can use the run method to ask our agent a question, and it will use the tools we’ve defined to find an answer.

You will see the following output:

To stream the response, you need to call runStream, which returns a stream of events. The agentStreamEvent provides chunks of the response as they become available. This allows you to display the response incrementally rather than waiting for the full response:

Note that we’re filtering for agentStreamEvent as an agent might return other events - more about that in the following section.

To log the workflow events, you can check the event type and log the event data.

Let’s see what running this looks like using npx tsx agent.ts

We’re seeing several workflow events being logged:

Great! We’ve built an agent that can understand requests and use tools to fulfill them. Next you can:

**Examples:**

Example 1 (javascript):
```javascript
// Your imports go here
async function main() {  // the rest of your code goes here}
main().catch(console.error);
```

Example 2 (javascript):
```javascript
// Your imports go here
async function main() {  // the rest of your code goes here}
main().catch(console.error);
```

Example 3 (unknown):
```unknown
npx tsx example.ts
```

Example 4 (unknown):
```unknown
npx tsx example.ts
```

---

## 3. Using a local model via Ollama

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/3_local_model/

**Contents:**
- 3. Using a local model via Ollama
  - Install Ollama
  - Pick and run a model
  - Switch the LLM in your code
  - Run local agent
  - Next steps

If you’re happy using OpenAI, you can skip this section, but many people are interested in using models they run themselves. The easiest way to do this is via the great work of our friends at Ollama, who provide a simple to use client that will download, install and run a growing range of models for you.

They provide a one-click installer for Mac, Linux and Windows on their home page.

Since we’re going to be doing agentic work, we’ll need a very capable model, but the largest models are hard to run on a laptop. We think mixtral 8x7b is a good balance between power and resources, but llama3 is another great option. You can run it simply by running

The first time you run it will also automatically download and install the model for you.

There are two changes you need to make to the code we already wrote in 1_agent to get Mixtral 8x7b to work. First, you need to switch to that model. Replace the call to Settings.llm with this:

You can also create local agent by importing agent from @llamaindex/workflow.

Now you’ve got a local agent, you can add Retrieval-Augmented Generation to your agent.

**Examples:**

Example 1 (json):
```json
ollama run mixtral:8x7b
```

Example 2 (json):
```json
ollama run mixtral:8x7b
```

Example 3 (css):
```css
Settings.llm = ollama({  model: "mixtral:8x7b",});
```

Example 4 (css):
```css
Settings.llm = ollama({  model: "mixtral:8x7b",});
```

---

## 4. Adding Retrieval-Augmented Generation (RAG)

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/4_agentic_rag/

**Contents:**
- 4. Adding Retrieval-Augmented Generation (RAG)
- Installation
  - New dependencies
  - Add an embedding model
  - Load data using SimpleDirectoryReader
  - Index our data
  - Use index.queryTool

While an agent that can perform math is nifty (LLMs are usually not very good at math), LLM-based applications are always more interesting when they work with large amounts of data. In this case, we’re going to use a 200-page PDF of the proposed budget of the city of San Francisco for fiscal years 2024-2024 and 2024-2025. It’s a great example because it’s extremely wordy and full of tables of figures, which present a challenge for humans and LLMs alike.

To learn more about RAG, we recommend this introduction from our Python docs. We’ll assume you know the basics:

We’re going to start with the same agent we built in step 1, but make a few changes. You can find the finished version in the repository.

We’ll be bringing in SimpleDirectoryReader, HuggingFaceEmbedding, VectorStoreIndex, and QueryEngineTool, OpenAIContextAwareAgent from LlamaIndex.TS, as well as the dependencies we previously used.

To encode our text into embeddings, we’ll need an embedding model. We could use OpenAI for this but to save on API calls we’re going to use a local embedding model from HuggingFace.

SimpleDirectoryReader is a flexible tool that can read various file formats. We will point it at our data directory, which contains a single PDF file, and retrieve a set of documents.

We will convert our text into embeddings using the VectorStoreIndex class through the fromDocuments method, which utilizes the embedding model defined earlier in Settings.

index.queryTool creates a QueryEngineTool that can be used be an agent to query data from the index:

The metadata that we’re setting helps the agent to decide when to use the tool. Note that by default LlamaIndex will retrieve just the 2 most relevant chunks of text. This document is complex though, so we’ll ask for more context by setting similarityTopK to 10.

Now, we can create an agent using the QueryEngineTool:

Once again we see a toolResult. You can see the query the LLM decided to send to the query engine (“total budget”), and the output the engine returned. In response.message you see that the LLM has returned the output from the tool almost verbatim, although it trimmed out the bit about 2024-2025 since we didn’t ask about that year.

So now we have an agent that can index complicated documents and answer questions about them. Let’s combine our math agent and our RAG agent!

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/huggingface
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/huggingface
```

Example 3 (sql):
```sql
import { QueryEngineTool, Settings, VectorStoreIndex } from "llamaindex";import { agent } from "@llamaindex/workflow";import { openai } from "@llamaindex/openai";import { HuggingFaceEmbedding } from "@llamaindex/huggingface";import { SimpleDirectoryReader } from "@llamaindex/readers/directory";
```

Example 4 (sql):
```sql
import { QueryEngineTool, Settings, VectorStoreIndex } from "llamaindex";import { agent } from "@llamaindex/workflow";import { openai } from "@llamaindex/openai";import { HuggingFaceEmbedding } from "@llamaindex/huggingface";import { SimpleDirectoryReader } from "@llamaindex/readers/directory";
```

---

## 5. A RAG agent that does math

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/5_rag_and_tools/

**Contents:**
- 5. A RAG agent that does math

In our third iteration of the agent we’ve combined the two previous agents, so we’ve defined both sumNumbers and a QueryEngineTool and created an array of two tools. The tools support both Zod and JSON Schema for parameter definition:

You can also use JSON Schema to define the tool parameters as an alternative to Zod.

These tool descriptions are identical to the ones we previously defined. Now let’s ask it 3 questions in a row:

We’ll abbreviate the output, but here are the important things to spot:

This is the first tool call, where it used the query engine to get the public health budget.

In the second tool call, it got the police budget also from the query engine.

In the final tool call, it used the sumNumbers function to add the two budgets together. Perfect! This leads to the final answer:

Great! Now let’s improve accuracy by improving our parsing with LlamaParse.

**Examples:**

Example 1 (typescript):
```typescript
// define the query engine as a toolconst tools = [  index.queryTool({    metadata: {      name: "san_francisco_budget_tool",      description: `This tool can answer detailed questions about the individual components of the budget of San Francisco in 2023-2024.`,    },    options: { similarityTopK: 10 },  }),  tool({    name: "sumNumbers",    description: "Use this function to sum two numbers",    parameters: z.object({      a: z.number({        description: "First number to sum",      }),      b: z.number({        description: "Second number to sum",      }),    }),    execute: ({ a, b }) => `${a + b}`,  }),];
```

Example 2 (typescript):
```typescript
// define the query engine as a toolconst tools = [  index.queryTool({    metadata: {      name: "san_francisco_budget_tool",      description: `This tool can answer detailed questions about the individual components of the budget of San Francisco in 2023-2024.`,    },    options: { similarityTopK: 10 },  }),  tool({    name: "sumNumbers",    description: "Use this function to sum two numbers",    parameters: z.object({      a: z.number({        description: "First number to sum",      }),      b: z.number({        description: "Second number to sum",      }),    }),    execute: ({ a, b }) => `${a + b}`,  }),];
```

Example 3 (css):
```css
tool(sumNumbers, {  name: "sumNumbers",  description: "Use this function to sum two numbers",  parameters: {    type: "object",    properties: {      a: {        type: "number",        description: "First number to sum",      },      b: {        type: "number",        description: "Second number to sum",      },    },    required: ["a", "b"],  },}),
```

Example 4 (css):
```css
tool(sumNumbers, {  name: "sumNumbers",  description: "Use this function to sum two numbers",  parameters: {    type: "object",    properties: {      a: {        type: "number",        description: "First number to sum",      },      b: {        type: "number",        description: "Second number to sum",      },    },    required: ["a", "b"],  },}),
```

---

## 7. Adding persistent vector storage

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/7_qdrant/

**Contents:**
- 7. Adding persistent vector storage
- Next steps

In the previous examples, we’ve been loading our data into memory each time we run the agent. This is fine for small datasets, but for larger datasets you’ll want to store your embeddings in a database. LlamaIndex.TS provides a VectorStore class that can store your embeddings in a variety of databases. We’re going to use Qdrant, a popular vector store, for this example.

We can get a local instance of Qdrant running very simply with Docker (make sure you install Docker first):

And in our code we initialize a VectorStore with the Qdrant URL:

Now once we have loaded our documents, we can instantiate an index with the vector store:

In the final iteration you can see that we have also implemented a very naive caching mechanism to avoid re-parsing the PDF each time we run the agent:

Since parsing a PDF can be slow, especially a large one, using the pre-parsed chunks in Qdrant can significantly speed up your agent.

In this guide you’ve learned how to

The next steps are up to you! Try creating more complex functions and query engines, and set your agent loose on the world.

**Examples:**

Example 1 (json):
```json
docker pull qdrant/qdrantdocker run -p 6333:6333 qdrant/qdrant
```

Example 2 (json):
```json
docker pull qdrant/qdrantdocker run -p 6333:6333 qdrant/qdrant
```

Example 3 (css):
```css
// initialize qdrant vector storeconst vectorStore = new QdrantVectorStore({  url: "http://localhost:6333",});
```

Example 4 (css):
```css
// initialize qdrant vector storeconst vectorStore = new QdrantVectorStore({  url: "http://localhost:6333",});
```

---

## Agents

**URL:** https://developers.llamaindex.ai/typescript/framework/migration/deprecated/agent/

**Contents:**
- Agents
- Getting Started
- Api References

Note: Agents are deprecated, use Agent Workflows instead.

An “agent” is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing that query in order to return the correct result. The key agent components can include, but are not limited to:

LlamaIndex.TS comes with a few built-in agents, but you can also create your own. The built-in agents include:

---

## Agent Workflows

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/agents/agent_workflow/

**Contents:**
- Agent Workflows
- Usage
  - Single Agent Workflow
  - Structured Output
  - Event Streaming
  - Multi-Agent Workflow

Agent Workflows are a powerful system that enables you to create and orchestrate one or multiple agents with tools to perform specific tasks. It’s built on top of the base Workflow system and provides a streamlined interface for agent interactions.

The simplest use case is creating a single agent with specific tools. Here’s an example of creating an assistant that tells jokes:

You can extract structured data from agent responses by providing a responseFormat with a Zod schema. This is useful when you need the agent’s response in a specific format for further processing:

Agent Workflows provide a unified interface for event streaming, making it easy to track and respond to different events during execution:

An Agent Workflow can orchestrate multiple agents, enabling complex interactions and task handoffs. Each agent in a multi-agent workflow requires:

Here’s an example of a multi-agent system that combines joke-telling and weather information:

The workflow will coordinate between agents, allowing them to handle different aspects of the request and hand off tasks when appropriate.

**Examples:**

Example 1 (swift):
```swift
import { tool } from "llamaindex";import { agent } from "@llamaindex/workflow";import { openai } from "@llamaindex/openai";
// Define a joke-telling toolconst jokeTool = tool(  () => "Baby Llama is called cria",  {    name: "joke",    description: "Use this tool to get a joke",  });
// Create an single agent workflow with the toolconst jokeAgent = agent({  tools: [jokeTool],  llm: openai({ model: "gpt-4o-mini" }),});
// Run the workflowconst result = await jokeAgent.run("Tell me something funny");console.log(result.data.result); // Baby Llama is called criaconsole.log(result.data.message); // { role: 'assistant', content: 'Baby Llama is called cria' }
```

Example 2 (swift):
```swift
import { tool } from "llamaindex";import { agent } from "@llamaindex/workflow";import { openai } from "@llamaindex/openai";
// Define a joke-telling toolconst jokeTool = tool(  () => "Baby Llama is called cria",  {    name: "joke",    description: "Use this tool to get a joke",  });
// Create an single agent workflow with the toolconst jokeAgent = agent({  tools: [jokeTool],  llm: openai({ model: "gpt-4o-mini" }),});
// Run the workflowconst result = await jokeAgent.run("Tell me something funny");console.log(result.data.result); // Baby Llama is called criaconsole.log(result.data.message); // { role: 'assistant', content: 'Baby Llama is called cria' }
```

Example 3 (swift):
```swift
import { z } from "zod";import { tool } from "llamaindex";import { agent } from "@llamaindex/workflow";import { openai } from "@llamaindex/openai";
// Define a weather toolconst weatherTool = tool({  name: "weatherTool",  description: "Get weather information",  parameters: z.object({    location: z.string(),  }),  execute: ({ location }) => {    return `The weather in ${location} is sunny. The temperature is 72 degrees. The humidity is 50%. The wind speed is 10 mph.`;  },});
// Define the structure you want for the responseconst responseSchema = z.object({  temperature: z.number(),  humidity: z.number(),  windSpeed: z.number(),});
// Create the agentconst weatherAgent = agent({  name: "weatherAgent",  tools: [weatherTool],  llm: openai({ model: "gpt-4.1-mini" }),});
// Run with structured outputconst result = await weatherAgent.run("What's the weather in Tokyo?", {  responseFormat: responseSchema,});
console.log("Natural language result:", result.data.result);console.log("Structured data:", result.data.object);// Output: { temperature: 72, humidity: 50, windSpeed: 10 }
```

Example 4 (swift):
```swift
import { z } from "zod";import { tool } from "llamaindex";import { agent } from "@llamaindex/workflow";import { openai } from "@llamaindex/openai";
// Define a weather toolconst weatherTool = tool({  name: "weatherTool",  description: "Get weather information",  parameters: z.object({    location: z.string(),  }),  execute: ({ location }) => {    return `The weather in ${location} is sunny. The temperature is 72 degrees. The humidity is 50%. The wind speed is 10 mph.`;  },});
// Define the structure you want for the responseconst responseSchema = z.object({  temperature: z.number(),  humidity: z.number(),  windSpeed: z.number(),});
// Create the agentconst weatherAgent = agent({  name: "weatherAgent",  tools: [weatherTool],  llm: openai({ model: "gpt-4.1-mini" }),});
// Run with structured outputconst result = await weatherAgent.run("What's the weather in Tokyo?", {  responseFormat: responseSchema,});
console.log("Natural language result:", result.data.result);console.log("Structured data:", result.data.object);// Output: { temperature: 72, humidity: 50, windSpeed: 10 }
```

---

## Basic Agent

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/basic_agent/

**Contents:**
- Basic Agent
- Set up
- Run agent

We have a comprehensive, step-by-step guide to building agents in LlamaIndex.TS that we recommend to learn what agents are and how to build them for production. But building a basic agent is simple:

Create the file example.ts. This code will:

You should expect output something like:

**Examples:**

Example 1 (python):
```python
npm initnpm i -D typescript @types/nodenpm i @llamaindex/openai @llamaindex/workflow llamaindex zod
```

Example 2 (python):
```python
npm initnpm i -D typescript @types/nodenpm i @llamaindex/openai @llamaindex/workflow llamaindex zod
```

Example 3 (unknown):
```unknown
npx tsx example.ts
```

Example 4 (unknown):
```unknown
npx tsx example.ts
```

---

## Define workflows using natural language

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/agents/natural_language_workflow/

**Contents:**
- Define workflows using natural language
- Usage
  - Define the events
  - Define the workflow

When working with Workflows, you have to write code to handle an event in the workflow. Often, the logic of the handler is not too complex so that it can be expressed using natural language and executed by an LLM. Besides the instructions, we just need the expected result event of the step, possible tool calls and optionally other events that can be emitted.

Let’s take an example of a workflow that generates a joke, gets a critique for it, and then improves it.

First, we define the events for our workflow. We need one for writing the joke, one for critiquing it, and one for the final result:

Note that your natural language workflows the events need to be created by the zodEvent function passing the zod schema as an argument. The agent needs the schema of the event data to correctly generate events. Also, we need a debugLabel so the LLM can identify the event to emit in the workflow.

As usual you first create the workflow:

Then you need to handle the events. For the handlers, instead of code, you’re now going to use natural language by calling the agentHandler function.

It only requires two parameters:

Then you will have a simple code to handle the step:

For advanced usage, you can add more functionality to agentHandler by using these parameters:

You can find more code examples in the examples folder.

**Examples:**

Example 1 (sql):
```sql
import { z } from "zod";import { zodEvent } from "@llamaindex/workflow";
const writeJokeSchema = z.object({  description: z    .string()    .describe("The topic to write a joke or describe the joke to improve."),  writtenJoke: z.optional(z.string()).describe("The written joke."),  retriedTimes: z    .number()    .default(0)    .describe(      "The retried times for writing the joke. Always increase this from the input retriedTimes.",    ),});
const critiqueSchema = z.object({  joke: z.string().describe("The joke to critique"),  retriedTimes: z.number().describe("The retried times for writing the joke."),});
const finalResultSchema = z.object({  joke: z.string().describe("The joke to critique"),  critique: z.string().describe("The critique of the joke"),});
const writeJokeEvent = zodEvent(writeJokeSchema, {  debugLabel: "writeJokeEvent",});const critiqueEvent = zodEvent(critiqueSchema, {  debugLabel: "critiqueEvent",});const finalResultEvent = zodEvent(finalResultSchema, {  debugLabel: "finalResultEvent",});
```

Example 2 (sql):
```sql
import { z } from "zod";import { zodEvent } from "@llamaindex/workflow";
const writeJokeSchema = z.object({  description: z    .string()    .describe("The topic to write a joke or describe the joke to improve."),  writtenJoke: z.optional(z.string()).describe("The written joke."),  retriedTimes: z    .number()    .default(0)    .describe(      "The retried times for writing the joke. Always increase this from the input retriedTimes.",    ),});
const critiqueSchema = z.object({  joke: z.string().describe("The joke to critique"),  retriedTimes: z.number().describe("The retried times for writing the joke."),});
const finalResultSchema = z.object({  joke: z.string().describe("The joke to critique"),  critique: z.string().describe("The critique of the joke"),});
const writeJokeEvent = zodEvent(writeJokeSchema, {  debugLabel: "writeJokeEvent",});const critiqueEvent = zodEvent(critiqueSchema, {  debugLabel: "critiqueEvent",});const finalResultEvent = zodEvent(finalResultSchema, {  debugLabel: "finalResultEvent",});
```

Example 3 (sql):
```sql
import { agentHandler, createWorkflow } from "@llamaindex/workflow";
const jokeFlow = createWorkflow();
```

Example 4 (sql):
```sql
import { agentHandler, createWorkflow } from "@llamaindex/workflow";
const jokeFlow = createWorkflow();
```

---

## Evaluating

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/evaluation/

**Contents:**
- Evaluating
- Concept
- Response Evaluation
- Usage

Evaluation and benchmarking are crucial concepts in LLM development. To improve the performance of an LLM app (RAG, agents) you must have a way to measure it.

LlamaIndex offers key modules to measure the quality of generated results. We also offer key modules to measure retrieval quality.

Evaluation of generated results can be difficult, since unlike traditional machine learning the predicted result is not a single number, and it can be hard to define quantitative metrics for this problem.

LlamaIndex offers LLM-based evaluation modules to measure the quality of results. This uses a “gold” LLM (e.g. GPT-4) to decide whether the predicted answer is correct in a variety of ways.

Note that many of these current evaluation modules do not require ground-truth labels. Evaluation can be done with some combination of the query, context, response, and combine these with LLM calls.

These evaluation modules are in the following forms:

Correctness: Whether the generated answer matches that of the reference answer given the query (requires labels).

Faithfulness: Evaluates if the answer is faithful to the retrieved contexts (in other words, whether if there’s hallucination).

Relevancy: Evaluates if the response from a query engine matches any source nodes.

---

## Gemini

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/gemini/

**Contents:**
- Gemini
- Installation
- Usage
  - Usage with Vertex AI
- Multimodal Usage
- Tool Calling
- Live API (Real-time Conversations)
- Load and index documents
- Query
- Full Example

To use Gemini via Vertex AI, you can specify the vertex configuration:

To authenticate for local development:

To authenticate for production you’ll have to use a service account. googleAuthOptions has credentials which might be useful for you.

Gemini supports multimodal inputs including text, images, audio, and video:

Gemini supports function calling with tools:

For real-time audio/video conversations using Gemini Live API.

The Live API is running directly in the frontend. That’s why you have to generate an ephemeral key first on the server side and pass it to the frontend.

To use the Live API, make sure to pass apiVersion: "v1alpha" to the httpOptions.

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/google
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/google
```

Example 3 (sql):
```sql
import { gemini, GEMINI_MODEL } from "@llamaindex/google";import { Settings } from "llamaindex";
Settings.llm = gemini({  model: GEMINI_MODEL.GEMINI_2_0_FLASH,});
```

Example 4 (sql):
```sql
import { gemini, GEMINI_MODEL } from "@llamaindex/google";import { Settings } from "llamaindex";
Settings.llm = gemini({  model: GEMINI_MODEL.GEMINI_2_0_FLASH,});
```

---

## Low-Level LLM Execution

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/agents/low-level/

**Contents:**
- Low-Level LLM Execution
- When to Use llm.exec
- Basic Usage
- Structured Output
- Agent Loop Pattern
- Streaming Support
- Return Values
- Best Practices

Sometimes your need more control over LLM interactions than what high-level agents provide. The llm.exec method makes it simple for you to make a single LLM call with tools but hides the complexity of executing the tools and generating the tool messages.

Use llm.exec when you need to:

The llm.exec method takes messages and tools as parameter and executes one LLM call. The LLM might either request to call one or more of the tools or generate an assistant message as result. For each tool call that is requested, llm.exec executes it and generates the two tool call messages (call and result). If no tool call is requested, just the assistant message is returned.

newMessages is an array as each tool call generates two messages: a tool call message and the tool call result message.

You can use responseFormat with a Zod schema to get structured data from the LLM response:

A common pattern is to use llm.exec in a loop until the LLM stops making tool calls:

For real-time responses, use the stream option to get the assistant’s response as streamed tokens:

newMessages is a function when streaming. The reason is that the result only is available after streaming. Calling it before, will throw an error.

llm.exec returns an object with:

For using llm.exec in an agent loop, take care to:

**Examples:**

Example 1 (javascript):
```javascript
import { openai } from "@llamaindex/openai";import { ChatMessage, tool } from "llamaindex";import z from "zod";
const llm = openai({ model: "gpt-4.1-mini" });const messages = [  {    content: "What's the weather like in San Francisco?",    role: "user",  } as ChatMessage,];
const { newMessages, toolCalls } = await llm.exec({  messages,  tools: [    tool({      name: "get_weather",      description: "Get the current weather for a location",      parameters: z.object({        address: z.string().describe("The address"),      }),      execute: ({ address }) => {        return `It's sunny in ${address}!`;      },    }),  ],});
// Add the new messages (including tool calls and responses) to your conversationmessages.push(...newMessages);
```

Example 2 (javascript):
```javascript
import { openai } from "@llamaindex/openai";import { ChatMessage, tool } from "llamaindex";import z from "zod";
const llm = openai({ model: "gpt-4.1-mini" });const messages = [  {    content: "What's the weather like in San Francisco?",    role: "user",  } as ChatMessage,];
const { newMessages, toolCalls } = await llm.exec({  messages,  tools: [    tool({      name: "get_weather",      description: "Get the current weather for a location",      parameters: z.object({        address: z.string().describe("The address"),      }),      execute: ({ address }) => {        return `It's sunny in ${address}!`;      },    }),  ],});
// Add the new messages (including tool calls and responses) to your conversationmessages.push(...newMessages);
```

Example 3 (javascript):
```javascript
import { openai } from "@llamaindex/openai";import { ChatMessage } from "llamaindex";import z from "zod";
const llm = openai({ model: "gpt-4.1-mini" });
const schema = z.object({  title: z.string(),  author: z.string(),  year: z.number(),});
const messages = [  {    role: "user",    content: "I have been reading La Divina Commedia by Dante Alighieri, published in 1321",  } as ChatMessage,];
const { newMessages, toolCalls, object } = await llm.exec({  messages,  responseFormat: schema,});
console.log(object); // { title: "La Divina Commedia", author: "Dante Alighieri", year: 1321 }
```

Example 4 (javascript):
```javascript
import { openai } from "@llamaindex/openai";import { ChatMessage } from "llamaindex";import z from "zod";
const llm = openai({ model: "gpt-4.1-mini" });
const schema = z.object({  title: z.string(),  author: z.string(),  year: z.number(),});
const messages = [  {    role: "user",    content: "I have been reading La Divina Commedia by Dante Alighieri, published in 1321",  } as ChatMessage,];
const { newMessages, toolCalls, object } = await llm.exec({  messages,  responseFormat: schema,});
console.log(object); // { title: "La Divina Commedia", author: "Dante Alighieri", year: 1321 }
```

---

## MCP Toolbox For Databases

**URL:** https://developers.llamaindex.ai/typescript/framework/integration/mcp-toolbox/

**Contents:**
- MCP Toolbox For Databases
- MCP Toolbox for Databases
  - Configure and deploy
  - Install client SDK
  - Loading Toolbox Tools
  - Advanced Toolbox Features

MCP Toolbox for Databases is an open source MCP server for databases. It was designed with enterprise-grade and production-quality in mind. It enables you to develop tools easier, faster, and more securely by handling the complexities such as connection pooling, authentication, and more.

Toolbox Tools can be seemlessly integrated with LlamaIndex applications. For more information on getting started or configuring Toolbox, see the documentation.

Toolbox is an open source server that you deploy and manage yourself. For more instructions on deploying and configuring, see the official Toolbox documentation:

LlamaIndex relies on the @toolbox-sdk/core node package to use Toolbox. Install the package before getting started:

Once your Toolbox server is configured and up and running, you can load tools from your server using the SDK:

Toolbox has a variety of features to make developing Gen AI tools for databases seamless. For more information, read more about the following:

**Examples:**

Example 1 (python):
```python
npm install @toolbox-sdk/core
```

Example 2 (python):
```python
npm install @toolbox-sdk/core
```

Example 3 (javascript):
```javascript
import { gemini, GEMINI_MODEL } from "@llamaindex/google";import { agent } from "@llamaindex/workflow";import { tool } from "llamaindex";import { ToolboxClient } from "@toolbox-sdk/core";
// Initialize LLMconst llm = gemini({  model: GEMINI_MODEL.GEMINI_2_0_FLASH,  apiKey: process.env.GOOGLE_API_KEY,});
// Replace with your Toolbox Server URLconst URL = 'https://127.0.0.1:5000';
const client = new ToolboxClient("http://127.0.0.1:5000");const toolboxTools = await client.loadToolset("my-toolset");
const getTool = (toolboxTool) => tool({  name: toolboxTool.getName(),  description: toolboxTool.getDescription(),  parameters: toolboxTool.getParamSchema(),  execute: toolboxTool});const tools = toolboxTools.map(getTool);
const myAgent = agent({  tools: tools,  llm,  memory,  systemPrompt: prompt,});const result = await myAgent.run(query);console.log(result);
```

Example 4 (javascript):
```javascript
import { gemini, GEMINI_MODEL } from "@llamaindex/google";import { agent } from "@llamaindex/workflow";import { tool } from "llamaindex";import { ToolboxClient } from "@toolbox-sdk/core";
// Initialize LLMconst llm = gemini({  model: GEMINI_MODEL.GEMINI_2_0_FLASH,  apiKey: process.env.GOOGLE_API_KEY,});
// Replace with your Toolbox Server URLconst URL = 'https://127.0.0.1:5000';
const client = new ToolboxClient("http://127.0.0.1:5000");const toolboxTools = await client.loadToolset("my-toolset");
const getTool = (toolboxTool) => tool({  name: toolboxTool.getName(),  description: toolboxTool.getDescription(),  parameters: toolboxTool.getParamSchema(),  execute: toolboxTool});const tools = toolboxTools.map(getTool);
const myAgent = agent({  tools: tools,  llm,  memory,  systemPrompt: prompt,});const result = await myAgent.run(query);console.log(result);
```

---

## OpenAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/openai/

**Contents:**
- OpenAI
- Installation
- Using OpenAI Responses API
  - Basic Setup
  - Message Content Types
  - Advanced Features
    - Built-in Tools
    - Response Tracking and Storage
    - Streaming Responses
  - Configuration Options

You can setup the apiKey on the environment variables, like:

You can optionally set a custom base URL, like:

The OpenAI Responses API provides enhanced functionality for handling complex interactions, including built-in tools, annotations, and streaming responses. Here’s how to use it:

The API supports different types of message content, including text and images:

The OpenAI Responses API supports various configuration options:

The API returns responses with rich metadata and optional annotations:

You can configure OpenAI to return responses in JSON format:

The OpenAI LLM supports different response formats to structure the output in specific ways. There are two main approaches to formatting responses:

The simplest way to get structured JSON responses is using the json_object response format:

For more robust type safety and validation, you can use Zod schemas to define the expected response structure:

The response format can be configured in two ways:

The response format options are:

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

The OpenAI Live LLM integration in LlamaIndex provides real-time chat capabilities with support for audio streaming and tool calling.

Tools are handled server-side, making it simple to pass them to the live session:

For audio capabilities:

Listen to events from the session:

The OpenAI Live LLM supports:

// Get an ephemeral key // Usually this code is run on the server and the ephemeral key is passed to the // client - the ephemeral key can be securely used on the client side

Creates a new live session.

Gets a temporary key for the session.

Sends a message to the assistant.

Closes the session and cleans up resources.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 3 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";import { Settings } from "llamaindex";
Settings.llm = new OpenAI({ model: "gpt-3.5-turbo", temperature: 0, apiKey: <YOUR_API_KEY> });
```

Example 4 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";import { Settings } from "llamaindex";
Settings.llm = new OpenAI({ model: "gpt-3.5-turbo", temperature: 0, apiKey: <YOUR_API_KEY> });
```

---

## Tools

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/agents/tool/

**Contents:**
- Tools
- Tool Function
  - Parameters with Zod
- Built-in tools
- MCP tools
- Function tool
  - Binding

A “tool” is a utility that can be called by an agent on behalf of an LLM. A tool can be called to perform custom actions, or retrieve extra information based on the LLM-generated input. A result from a tool call can be used by subsequent steps in a workflow, or to compute a final answer. For example, a “weather tool” could fetch some live weather information from a geographical location.

The tool function is a utility provided to define a tool that can be used by an agent. It takes a function and a configuration object as arguments. The configuration object includes the tool’s name, description, and parameters.

The parameters field in the tool configuration is defined using zod, a TypeScript-first schema declaration and validation library. zod allows you to specify the expected structure and types of the input parameters, ensuring that the data passed to the tool is valid.

In this example, z.object is used to define a schema for the parameters where question is expected to be a string. This ensures that any input to the tool adheres to the specified structure, providing a layer of type safety and validation.

You can import built-in tools from the @llamaindex/tools package.

If you have a MCP server running, you can fetch tools from the server and use them in your agents.

You can also use MCP Toolbox for Databases to interact with MCP tools.

You can still use the FunctionTool class to define a tool. A FunctionTool is constructed from a function with signature

An additional argument can be bound to a tool, each tool call will be passed

Note: calling the bind method will return a new FunctionTool instance, without modifying the tool which bind is called on.

Example to pass a userToken as additional argument:

**Examples:**

Example 1 (javascript):
```javascript
import { tool } from "llamaindex";import { agent } from "@llamaindex/workflow";import { z } from "zod";
// first arg is LLM input, second is bound argconst queryKnowledgeBase = async ({ question }, { userToken }) => {  const response = await fetch(`https://knowledge-base.com?token=${userToken}&query=${question}`);  // ...};
// define tool with zod validationconst kbTool = tool(queryKnowledgeBase, {  name: 'queryKnowledgeBase',  description: 'Query knowledge base',  parameters: z.object({    question: z.string({      description: 'The user question',    }),  }),});
```

Example 2 (javascript):
```javascript
import { tool } from "llamaindex";import { agent } from "@llamaindex/workflow";import { z } from "zod";
// first arg is LLM input, second is bound argconst queryKnowledgeBase = async ({ question }, { userToken }) => {  const response = await fetch(`https://knowledge-base.com?token=${userToken}&query=${question}`);  // ...};
// define tool with zod validationconst kbTool = tool(queryKnowledgeBase, {  name: 'queryKnowledgeBase',  description: 'Query knowledge base',  parameters: z.object({    question: z.string({      description: 'The user question',    }),  }),});
```

Example 3 (julia):
```julia
import { agent } from "@llamaindex/workflow";import { wiki } from "@llamaindex/tools";
const researchAgent = agent({  name: "WikiAgent",  description: "Gathering information from the internet",  systemPrompt: `You are a research agent. Your role is to gather information from the internet using the provided tools.`,  tools: [wiki()],});
```

Example 4 (julia):
```julia
import { agent } from "@llamaindex/workflow";import { wiki } from "@llamaindex/tools";
const researchAgent = agent({  name: "WikiAgent",  description: "Gathering information from the internet",  systemPrompt: `You are a research agent. Your role is to gather information from the internet using the provided tools.`,  tools: [wiki()],});
```

---

## Welcome to LlamaIndex.TS

**URL:** https://developers.llamaindex.ai/typescript/framework/

**Contents:**
- Welcome to LlamaIndex.TS
  - Introduction
  - Use cases
  - Getting started
  - LlamaCloud
  - Community
  - Related projects
- Introduction
  - What are agents?
  - What are workflows?

LlamaIndex.TS is a framework for utilizing context engineering to build generative AI applications with large language models. From rapid-prototyping RAG chatbots to deploying multi-agent workflows in production, LlamaIndex gives you everything you need — all in idiomatic TypeScript.

Built for modern JavaScript runtimes like Node.js Node.js, Deno Deno, Bun Bun, Cloudflare Workers Cloudflare Workers, and more.

Context engineering, agents & workflows — what do they mean?

See what you can build with LlamaIndex.TS.

Your first app in 5 lines of code.

Managed parsing, extraction & retrieval pipelines.

Join thousands of builders on Discord, Twitter, and more.

Connectors, demos & starter kits.

Agents are LLM-powered assistants that can reason, use external tools, and take actions to accomplish tasks such as research, data extraction, and automation. LlamaIndex.TS provides foundational building blocks for creating and orchestrating these agents.

Workflows are multi-step, event-driven processes that combine agents, data connectors, and other tools to solve complex problems. With LlamaIndex.TS you can chain together retrieval, generation, and tool-calling steps and then deploy the entire pipeline as a microservice.

LLMs come pre-trained on vast public corpora, but not on your private or domain-specific data. Context engineering bridges that gap by injecting the right pieces of your data into the LLM prompt at the right time. The most popular example is Retrieval-Augmented Generation (RAG), but the same idea powers agent memory, evaluation, extraction, summarisation, and more.

LlamaIndex.TS gives you:

You can learn more about these concepts in our concepts guide.

Popular scenarios include:

The fastest way to get started is in StackBlitz below — no local setup required:

Want to learn more? We have several tutorials to get you started:

Need an end-to-end managed pipeline? Check out LlamaCloud: best-in-class document parsing (LlamaParse), extraction (LlamaExtract), and indexing services with generous free tiers.

We 💜 contributors! View our contributing guide to get started.

---

## Workflows

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/workflows/

**Contents:**
- Workflows
- Getting Started
  - Defining Workflow Events
  - Setting up the Workflow with Stateful Middleware
  - Adding Handlers with Loops
  - Running the Workflow
  - Using Stream Utilities
- Next Steps

A Workflow in LlamaIndex is a lightweight, event-driven abstraction used to chain together several events. Workflows are made up of handlers, with each one responsible for processing specific event types and emitting new events.

Workflows are designed to be flexible and can be used to build agents, RAG flows, extraction flows, or anything else you want to implement.

Let’s explore a simple workflow example where a joke is generated and then critiqued and iterated on:

There are a few moving pieces here, so let’s go through this step by step.

Events are defined using the workflowEvent function and contain arbitrary data provided as a generic type. In this example, we have four events:

Our workflow is implemented using the createWorkflow() function, enhanced with the withState middleware. This middleware provides shared state across all handlers, which in this case tracks:

This state will be accessible within workflows by using the getContext().state function.

We have three key handlers in our workflow:

To run the workflow, we:

The stream returned by createContext contains utility functions to make working with event streams easier:

The stream utilities make it easier to work with the asynchronous event flow. In this example, we use:

You can combine these utilities with other stream operators like filter and map to create powerful processing pipelines.

To learn more about workflows, check out the Workflows documentation.

**Examples:**

Example 1 (python):
```python
npm i @llamaindex/workflow @llamaindex/openai
```

Example 2 (python):
```python
npm i @llamaindex/workflow @llamaindex/openai
```

Example 3 (typescript):
```typescript
const startEvent = workflowEvent<string>(); // Input topic for jokeconst jokeEvent = workflowEvent<{ joke: string }>(); // Intermediate jokeconst critiqueEvent = workflowEvent<{ joke: string; critique: string }>(); // Intermediate critiqueconst resultEvent = workflowEvent<{ joke: string; critique: string }>(); // Final joke + critique
```

Example 4 (typescript):
```typescript
const startEvent = workflowEvent<string>(); // Input topic for jokeconst jokeEvent = workflowEvent<{ joke: string }>(); // Intermediate jokeconst critiqueEvent = workflowEvent<{ joke: string; critique: string }>(); // Intermediate critiqueconst resultEvent = workflowEvent<{ joke: string; critique: string }>(); // Final joke + critique
```

---

## Workflows

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/agents/workflows/

**Contents:**
- Workflows

A Workflow in LlamaIndex is a lightweight, event-driven abstraction used to chain together several events. Workflows are made up of handlers, with each one responsible for processing specific event types and emitting new events.

Workflows are designed to be flexible and can be used to build agents, RAG flows, extraction flows, or anything else you want to implement.

To use workflows install this package:

This contains the core functionality for the workflow system. You can read more about the core concepts in the workflow-core section.

In contrast, the @llamaindex/workflow package contains more utiltities, such as prebuilt agents.

**Examples:**

Example 1 (python):
```python
npm i @llamaindex/workflow-core
```

Example 2 (python):
```python
npm i @llamaindex/workflow-core
```

Example 3 (python):
```python
npm i @llamaindex/workflow
```

Example 4 (python):
```python
npm i @llamaindex/workflow
```

---
