# Llamaindex-Ts - Getting Started

**Pages:** 10

---

## Code examples

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/examples/

**Contents:**
- Code examples
- Use examples locally
- Try examples online

Our GitHub repository has a wealth of examples to explore and try out. You can check out our examples folder to see them all at once, or browse the pages in this section for some selected highlights.

It may be useful to check out all the examples at once so you can try them out locally. To do this into a folder called my-new-project, run these commands:

Then you can run any example in the folder with tsx, e.g.:

You can also try the examples online using StackBlitz:

**Examples:**

Example 1 (unknown):
```unknown
npx degit run-llama/LlamaIndexTS/examples my-new-projectcd my-new-projectnpm i
```

Example 2 (unknown):
```unknown
npx degit run-llama/LlamaIndexTS/examples my-new-projectcd my-new-projectnpm i
```

Example 3 (unknown):
```unknown
export OPENAI_API_KEY=your-api-keynpx tsx ./agents/agent/openai.ts
```

Example 4 (unknown):
```unknown
export OPENAI_API_KEY=your-api-keynpx tsx ./agents/agent/openai.ts
```

---

## Concepts

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/rag/concepts/

**Contents:**
- Concepts
- Answering Questions Across Your Data
  - Indexing Stage
  - Querying Stage
    - Building Blocks
    - Pipelines

LlamaIndex.TS helps you build LLM-powered applications (e.g. Q&A, chatbot) over custom data.

In this high-level concepts guide, you will learn:

LlamaIndex uses a two stage method when using an LLM with your data:

This process is also known as Retrieval Augmented Generation (RAG).

LlamaIndex.TS provides the essential toolkit for making both steps super easy.

Let’s explore each stage in detail.

LlamaIndex.TS help you prepare the knowledge base with a suite of data connectors and indexes.

Data Loaders: A data connector (i.e. Reader) ingest data from different data sources and data formats into a simple Document representation (text and simple metadata).

Documents / Nodes: A Document is a generic container around any data source - for instance, a PDF, an API output, or retrieved data from a database. A Node is the atomic unit of data in LlamaIndex and represents a “chunk” of a source Document. It’s a rich representation that includes metadata and relationships (to other nodes) to enable accurate and expressive retrieval operations.

Data Indexes: Once you’ve ingested your data, LlamaIndex helps you index data into a format that’s easy to retrieve.

Under the hood, LlamaIndex parses the raw documents into intermediate representations, calculates vector embeddings, and stores your data in-memory or to disk.

In the querying stage, the query pipeline retrieves the most relevant context given a user query, and pass that to the LLM (along with the query) to synthesize a response.

This gives the LLM up-to-date knowledge that is not in its original training data, (also reducing hallucination).

The key challenge in the querying stage is retrieval, orchestration, and reasoning over (potentially many) knowledge bases.

LlamaIndex provides composable modules that help you build and integrate RAG pipelines for Q&A (query engine), chatbot (chat engine), or as part of an agent.

These building blocks can be customized to reflect ranking preferences, as well as composed to reason over multiple knowledge bases in a structured way.

Retrievers: A retriever defines how to efficiently retrieve relevant context from a knowledge base (i.e. index) when given a query. The specific retrieval logic differs for different indices, the most popular being dense retrieval against a vector index.

Response Synthesizers: A response synthesizer generates a response from an LLM, using a user query and a given set of retrieved text chunks.

Query Engines: A query engine is an end-to-end pipeline that allow you to ask question over your data. It takes in a natural language query, and returns a response, along with reference context retrieved and passed to the LLM.

Chat Engines: A chat engine is an end-to-end pipeline for having a conversation with your data (multiple back-and-forth instead of a single question & answer).

---

## Create-Llama

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/create_llama/

**Contents:**
- Create-Llama
- Learn more

create-llama is a powerful but easy to use command-line tool that generates a working, full-stack web application that allows you to chat with your data. You can learn more about it on the create-llama README page.

Run it once and it will ask you a series of questions about the kind of application you want to generate. Then you can customize your application to suit your use-case. To get started, run:

Once your app is generated, cd into your app directory and run

to start the development server. You can then visit http://localhost:3000 to see your app, which should look something like this:

**Examples:**

Example 1 (python):
```python
npx create-llama@latest
```

Example 2 (python):
```python
npx create-llama@latest
```

Example 3 (unknown):
```unknown
npm run dev
```

Example 4 (unknown):
```unknown
npm run dev
```

---

## High-Level Concepts

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/concepts/

**Contents:**
- High-Level Concepts
- Large Language Models (LLMs)
- Agentic Applications
- Agents
- Retrieval Augmented Generation (RAG)
- Use cases

This is a quick guide to the high-level concepts you’ll encounter frequently when building LLM applications.

LLMs are the fundamental innovation that launched LlamaIndex. They are an artificial intelligence (AI) computer system that can understand, generate, and manipulate natural language, including answering questions based on their training data or data provided to them at query time.

When an LLM is used within an application, it is often used to make decisions, take actions, and/or interact with the world. This is the core definition of an agentic application.

While the definition of an agentic application is broad, there are several key characteristics that define an agentic application:

In LlamaIndex, you can build agentic applications by using the workflows to orchestrate a sequence of steps and LLMs. You can learn more about workflows.

We define an agent as a specific instance of an “agentic application”. An agent is a piece of software that semi-autonomously performs tasks by combining LLMs with other tools and memory, orchestrated in a reasoning loop that decides which tool to use next (if any).

What this means in practice, is something like:

You can learn more about agents.

Retrieval-Augmented Generation (RAG) is a core technique for building data-backed LLM applications with LlamaIndex. It allows LLMs to answer questions about your private data by providing it to the LLM at query time, rather than training the LLM on your data. To avoid sending all of your data to the LLM every time, RAG indexes your data and selectively sends only the relevant parts along with your query. You can learn more about RAG.

There are endless use cases for data-backed LLM applications but they can be roughly grouped into four categories:

Agents: An agent is an automated decision-maker powered by an LLM that interacts with the world via a set of tools. Agents can take an arbitrary number of steps to complete a given task, dynamically deciding on the best course of action rather than following pre-determined steps. This gives it additional flexibility to tackle more complex tasks.

Workflows: A Workflow in LlamaIndex is a specific event-driven abstraction that allows you to orchestrate a sequence of steps and LLMs calls. Workflows can be used to implement any agentic application, and are a core component of LlamaIndex.

Structured Data Extraction: Pydantic extractors allow you to specify a precise data structure to extract from your data and use LLMs to fill in the missing pieces in a type-safe way. This is useful for extracting structured data from unstructured sources like PDFs, websites, and more, and is key to automating workflows.

Query Engines: A query engine is an end-to-end flow that allows you to ask questions over your data. It takes in a natural language query, and returns a response, along with reference context retrieved and passed to the LLM.

Chat Engines: A chat engine is an end-to-end flow for having a conversation with your data (multiple back-and-forth instead of a single question-and-answer).

---

## Installation

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/installation/

**Contents:**
- Installation
- Quick Start
- Environment Setup
  - API Keys
  - Loading Environment Variables
- TypeScript Configuration
- Running your first agent
  - Set up
  - Run the agent
- Performance Optimization

Install the core package:

In most cases, you’ll also need an LLM provider and the Workflow package:

Most LLM providers require API keys. Set your OpenAI key (or other provider):

For Node.js applications:

For other environments, see the deployment-specific guides below.

LlamaIndex.TS is built with TypeScript and provides excellent type safety. Add these settings to your tsconfig.json:

If you don’t already have a project, you can create a new one in a new folder:

Create the file example.ts. This code will:

You should expect output something like:

Install gpt-tokenizer for 60x faster tokenization (Node.js environments only):

LlamaIndex will automatically use this when available.

Choose your deployment target:

Server APIs & Backends

Go to LLM APIs and Embedding APIs to find out how to use different LLM and embedding providers beyond OpenAI.

Show me code examples

**Examples:**

Example 1 (unknown):
```unknown
npm i llamaindex
```

Example 2 (unknown):
```unknown
npm i llamaindex
```

Example 3 (python):
```python
npm i @llamaindex/openai @llamaindex/workflow
```

Example 4 (python):
```python
npm i @llamaindex/openai @llamaindex/workflow
```

---

## Next.js Applications

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/installation/nextjs/

**Contents:**
- Next.js Applications
- Essential Configuration
  - Next.js Config
- API Routes
  - App Router (Recommended)
  - Pages Router (Legacy)
- Server Components
- Edge Runtime
- Streaming Responses
- Client-side Integration

This guide covers integrating LlamaIndex.TS agents with Next.js applications.

Use withLlamaIndex to ensure compatibility:

Initialize agents in server components:

The Edge Runtime has limited Node.js API access:

Implement streaming for better user experience:

**Examples:**

Example 1 (javascript):
```javascript
import withLlamaIndex from "llamaindex/next";
/** @type {import('next').NextConfig} */const nextConfig = {  // Your existing config};
export default withLlamaIndex(nextConfig);
```

Example 2 (javascript):
```javascript
import withLlamaIndex from "llamaindex/next";
/** @type {import('next').NextConfig} */const nextConfig = {  // Your existing config};
export default withLlamaIndex(nextConfig);
```

Example 3 (javascript):
```javascript
import { agent } from "@llamaindex/workflow";import { tool } from "llamaindex";import { openai } from "@llamaindex/openai";import { z } from "zod";import { NextRequest, NextResponse } from "next/server";
// Initialize agent once (consider using a singleton pattern)let myAgent: any = null;
async function initializeAgent() {  if (myAgent) return myAgent;
  try {    const greetTool = tool({      name: "greet",      description: "Greets a user with their name",      parameters: z.object({        name: z.string(),      }),      execute: ({ name }) => `Hello, ${name}! How can I help you today?`,    });
    myAgent = agent({      tools: [greetTool],      llm: openai({ model: "gpt-4o-mini" }),    });
    return myAgent;  } catch (error) {    console.error("Failed to initialize agent:", error);    throw error;  }}
export async function POST(request: NextRequest) {  try {    const { message } = await request.json();
    if (!message || typeof message !== 'string') {      return NextResponse.json(        { error: "Message is required and must be a string" },        { status: 400 }      );    }
    const agent = await initializeAgent();    const result = await agent.run(message);
    return NextResponse.json({ response: result.data });  } catch (error) {    console.error("Chat error:", error);    return NextResponse.json(      { error: "Internal server error" },      { status: 500 }    );  }}
```

Example 4 (javascript):
```javascript
import { agent } from "@llamaindex/workflow";import { tool } from "llamaindex";import { openai } from "@llamaindex/openai";import { z } from "zod";import { NextRequest, NextResponse } from "next/server";
// Initialize agent once (consider using a singleton pattern)let myAgent: any = null;
async function initializeAgent() {  if (myAgent) return myAgent;
  try {    const greetTool = tool({      name: "greet",      description: "Greets a user with their name",      parameters: z.object({        name: z.string(),      }),      execute: ({ name }) => `Hello, ${name}! How can I help you today?`,    });
    myAgent = agent({      tools: [greetTool],      llm: openai({ model: "gpt-4o-mini" }),    });
    return myAgent;  } catch (error) {    console.error("Failed to initialize agent:", error);    throw error;  }}
export async function POST(request: NextRequest) {  try {    const { message } = await request.json();
    if (!message || typeof message !== 'string') {      return NextResponse.json(        { error: "Message is required and must be a string" },        { status: 400 }      );    }
    const agent = await initializeAgent();    const result = await agent.run(message);
    return NextResponse.json({ response: result.data });  } catch (error) {    console.error("Chat error:", error);    return NextResponse.json(      { error: "Internal server error" },      { status: 500 }    );  }}
```

---

## Retrieval Augmented Generation (RAG)

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/rag/

**Contents:**
- Retrieval Augmented Generation (RAG)
- Set up the project
- Run queries

One of the most common use-cases for LlamaIndex is Retrieval-Augmented Generation or RAG, in which your data is indexed and selectively retrieved to be given to an LLM as source material for responding to a query. You can learn more about the concepts behind RAG.

In a new folder, run:

Then, check out the installation steps to install LlamaIndex.TS and prepare an OpenAI key.

You can use other LLMs via their APIs; if you would prefer to use local models check out our local LLM example.

Create the file example.ts. This code will

Create a tsconfig.json file in the same folder:

Now you can run the code with

You should expect output something like:

Once you’ve mastered basic RAG, you may want to consider chatting with your data.

**Examples:**

Example 1 (python):
```python
npm initnpm i -D typescript @types/nodenpm i llamaindex
```

Example 2 (python):
```python
npm initnpm i -D typescript @types/nodenpm i llamaindex
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

## Serverless Functions

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/installation/serverless/

**Contents:**
- Serverless Functions
- Cloudflare Workers
- Vercel Functions
  - Node.js Runtime
  - Edge Runtime
- AWS Lambda
- Netlify Functions
- Next Steps

This guide covers adding LlamaIndex.TS agents to serverless environments where you have execution time and memory constraints.

**Examples:**

Example 1 (javascript):
```javascript
export default {  async fetch(request: Request, env: Env): Promise<Response> {    const { setEnvs } = await import("@llamaindex/env");    setEnvs(env);
    const { agent } = await import("@llamaindex/workflow");    const { openai } = await import("@llamaindex/openai");    const { tool } = await import("llamaindex");    const { z } = await import("zod");
    const timeTool = tool({      name: "getCurrentTime",      description: "Gets the current time",      parameters: z.object({}),      execute: () => new Date().toISOString(),    });
    const myAgent = agent({      tools: [timeTool],      llm: openai({ model: "gpt-4o-mini" }),    });
    try {      const { message } = await request.json();      const result = await myAgent.run(message);
      return new Response(JSON.stringify({ response: result.data }), {        headers: { "Content-Type": "application/json" },      });    } catch (error) {      return new Response(JSON.stringify({ error: error.message }), {        status: 500,        headers: { "Content-Type": "application/json" },      });    }  },};
```

Example 2 (javascript):
```javascript
export default {  async fetch(request: Request, env: Env): Promise<Response> {    const { setEnvs } = await import("@llamaindex/env");    setEnvs(env);
    const { agent } = await import("@llamaindex/workflow");    const { openai } = await import("@llamaindex/openai");    const { tool } = await import("llamaindex");    const { z } = await import("zod");
    const timeTool = tool({      name: "getCurrentTime",      description: "Gets the current time",      parameters: z.object({}),      execute: () => new Date().toISOString(),    });
    const myAgent = agent({      tools: [timeTool],      llm: openai({ model: "gpt-4o-mini" }),    });
    try {      const { message } = await request.json();      const result = await myAgent.run(message);
      return new Response(JSON.stringify({ response: result.data }), {        headers: { "Content-Type": "application/json" },      });    } catch (error) {      return new Response(JSON.stringify({ error: error.message }), {        status: 500,        headers: { "Content-Type": "application/json" },      });    }  },};
```

Example 3 (javascript):
```javascript
// pages/api/chat.ts or app/api/chat/route.tsimport { agent } from "@llamaindex/workflow";import { tool } from "llamaindex";import { openai } from "@llamaindex/openai";import { z } from "zod";
export default async function handler(req, res) {  if (req.method !== 'POST') {    return res.status(405).json({ error: 'Method not allowed' });  }
  const { message } = req.body;
  const weatherTool = tool({    name: "getWeather",    description: "Get weather information",    parameters: z.object({      city: z.string(),    }),    execute: ({ city }) => `Weather in ${city}: 72°F, sunny`,  });
  const myAgent = agent({    tools: [weatherTool],    llm: openai({ model: "gpt-4o-mini" }),  });
  try {    const result = await myAgent.run(message);    res.json({ response: result.data });  } catch (error) {    res.status(500).json({ error: error.message });  }}
```

Example 4 (javascript):
```javascript
// pages/api/chat.ts or app/api/chat/route.tsimport { agent } from "@llamaindex/workflow";import { tool } from "llamaindex";import { openai } from "@llamaindex/openai";import { z } from "zod";
export default async function handler(req, res) {  if (req.method !== 'POST') {    return res.status(405).json({ error: 'Method not allowed' });  }
  const { message } = req.body;
  const weatherTool = tool({    name: "getWeather",    description: "Get weather information",    parameters: z.object({      city: z.string(),    }),    execute: ({ city }) => `Weather in ${city}: 72°F, sunny`,  });
  const myAgent = agent({    tools: [weatherTool],    llm: openai({ model: "gpt-4o-mini" }),  });
  try {    const result = await myAgent.run(message);    res.json({ response: result.data });  } catch (error) {    res.status(500).json({ error: error.message });  }}
```

---

## Server APIs & Backends

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/installation/server-apis/

**Contents:**
- Server APIs & Backends
- Supported Runtimes
- Common Server Frameworks
  - Express.js
  - Fastify
  - Hono
- Streaming Responses
- Next Steps

This guide covers adding LlamaIndex.TS agents to traditional server environments where you have full Node.js runtime access.

LlamaIndex.TS works seamlessly with:

For real-time agent responses:

**Examples:**

Example 1 (javascript):
```javascript
import express from 'express';import { agent } from '@llamaindex/workflow';import { tool } from 'llamaindex';import { openai } from '@llamaindex/openai';import { z } from 'zod';
const app = express();app.use(express.json());
// Initialize agent once at startuplet myAgent: any;
async function initializeAgent() {  // Create tools for the agent  const sumTool = tool({    name: "sum",    description: "Adds two numbers",    parameters: z.object({      a: z.number(),      b: z.number(),    }),    execute: ({ a, b }) => a + b,  });
  const multiplyTool = tool({    name: "multiply",    description: "Multiplies two numbers",    parameters: z.object({      a: z.number(),      b: z.number(),    }),    execute: ({ a, b }) => a * b,  });
  // Create the agent  myAgent = agent({    tools: [sumTool, multiplyTool],    llm: openai({ model: "gpt-4o-mini" }),  });}
app.post('/api/chat', async (req, res) => {  try {    const { message } = req.body;    const result = await myAgent.run(message);    res.json({ response: result.data });  } catch (error) {    res.status(500).json({ error: 'Chat failed' });  }});
// Initialize and start serverinitializeAgent().then(() => {  app.listen(3000, () => {    console.log('Server running on port 3000');  });});
```

Example 2 (javascript):
```javascript
import express from 'express';import { agent } from '@llamaindex/workflow';import { tool } from 'llamaindex';import { openai } from '@llamaindex/openai';import { z } from 'zod';
const app = express();app.use(express.json());
// Initialize agent once at startuplet myAgent: any;
async function initializeAgent() {  // Create tools for the agent  const sumTool = tool({    name: "sum",    description: "Adds two numbers",    parameters: z.object({      a: z.number(),      b: z.number(),    }),    execute: ({ a, b }) => a + b,  });
  const multiplyTool = tool({    name: "multiply",    description: "Multiplies two numbers",    parameters: z.object({      a: z.number(),      b: z.number(),    }),    execute: ({ a, b }) => a * b,  });
  // Create the agent  myAgent = agent({    tools: [sumTool, multiplyTool],    llm: openai({ model: "gpt-4o-mini" }),  });}
app.post('/api/chat', async (req, res) => {  try {    const { message } = req.body;    const result = await myAgent.run(message);    res.json({ response: result.data });  } catch (error) {    res.status(500).json({ error: 'Chat failed' });  }});
// Initialize and start serverinitializeAgent().then(() => {  app.listen(3000, () => {    console.log('Server running on port 3000');  });});
```

Example 3 (javascript):
```javascript
import Fastify from 'fastify';import { agent } from '@llamaindex/workflow';import { tool } from 'llamaindex';import { openai } from '@llamaindex/openai';import { z } from 'zod';
const fastify = Fastify();let myAgent: any;
async function initializeAgent() {  const sumTool = tool({    name: "sum",    description: "Adds two numbers",    parameters: z.object({      a: z.number(),      b: z.number(),    }),    execute: ({ a, b }) => a + b,  });
  myAgent = agent({    tools: [sumTool],    llm: openai({ model: "gpt-4o-mini" }),  });}
fastify.post('/api/chat', async (request, reply) => {  try {    const { message } = request.body as { message: string };    const result = await myAgent.run(message);    return { response: result.data };  } catch (error) {    reply.status(500).send({ error: 'Chat failed' });  }});
const start = async () => {  await initializeAgent();  await fastify.listen({ port: 3000 });  console.log('Server running on port 3000');};
start();
```

Example 4 (javascript):
```javascript
import Fastify from 'fastify';import { agent } from '@llamaindex/workflow';import { tool } from 'llamaindex';import { openai } from '@llamaindex/openai';import { z } from 'zod';
const fastify = Fastify();let myAgent: any;
async function initializeAgent() {  const sumTool = tool({    name: "sum",    description: "Adds two numbers",    parameters: z.object({      a: z.number(),      b: z.number(),    }),    execute: ({ a, b }) => a + b,  });
  myAgent = agent({    tools: [sumTool],    llm: openai({ model: "gpt-4o-mini" }),  });}
fastify.post('/api/chat', async (request, reply) => {  try {    const { message } = request.body as { message: string };    const result = await myAgent.run(message);    return { response: result.data };  } catch (error) {    reply.status(500).send({ error: 'Chat failed' });  }});
const start = async () => {  await initializeAgent();  await fastify.listen({ port: 3000 });  console.log('Server running on port 3000');};
start();
```

---

## Troubleshooting

**URL:** https://developers.llamaindex.ai/typescript/framework/getting_started/installation/troubleshooting/

**Contents:**
- Troubleshooting
- Installation Issues
  - Module Resolution Errors
  - TypeScript Errors
  - Package Compatibility Issues
- Runtime Issues
  - Memory Errors
  - API Rate Limiting
  - Tokenization Performance
- Bundling Issues

This guide addresses common issues you might encounter when installing and deploying LlamaIndex.TS applications across different environments.

Problem: Import errors or module not found errors

Solution: Ensure your tsconfig.json is properly configured:

Alternative solution: Try different module resolution strategies:

Problem: TypeScript compilation errors with LlamaIndex imports

Solution: Ensure you have the correct TypeScript configuration:

Problem: Some packages don’t work in certain environments

Common incompatibilities:

Solution: Use environment-specific alternatives:

Problem: Out of memory errors during index creation or querying

Solution: Optimize memory usage:

For serverless environments:

Problem: Rate limiting errors from LLM providers

Solution: Implement retry logic with exponential backoff:

Problem: Slow tokenization affecting performance

Solution: Install faster tokenizer (Node.js only):

LlamaIndex will automatically use this for 60x faster tokenization.

Problem: Large bundle sizes affecting performance

Solution: Use dynamic imports and code splitting:

Problem: Bundler compatibility issues

Solution for Next.js:

Problem: Node.js version compatibility issues

Solution: Use supported Node.js versions:

Check your Node.js version:

Problem: Module not available in Cloudflare Workers

Solution: Use @llamaindex/env for environment compatibility:

Problem: Limited Node.js API access in Edge Runtime

Solution: Use standard runtime or adapt code:

Problem: Slow query performance

Solution: Implement caching and optimization:

Problem: Slow cold starts in serverless environments

Solution: Pre-warm your functions:

Problem: API key not found or invalid

Solution: Verify environment variable setup:

Problem: Environment variables not loading correctly

Solution: Use proper loading mechanisms:

Cause: Package not installed or module resolution issue

Cause: File system modules used in browser/edge environment

Cause: Global polyfill missing in browser environments

Cause: Query engine not properly initialized

If you’re still experiencing issues:

Check specific deployment guides:

Open an issue on GitHub with a minimal reproduction

Join our Discord for community support

**Examples:**

Example 1 (json):
```json
{  "compilerOptions": {    "moduleResolution": "bundler", // or "nodenext" | "node16" | "node"    "lib": ["DOM.AsyncIterable"],    "target": "es2020",    "module": "esnext"  }}
```

Example 2 (json):
```json
{  "compilerOptions": {    "moduleResolution": "bundler", // or "nodenext" | "node16" | "node"    "lib": ["DOM.AsyncIterable"],    "target": "es2020",    "module": "esnext"  }}
```

Example 3 (go):
```go
# Clear node_modules and reinstallrm -rf node_modules package-lock.jsonnpm install
# Or try with different package managerpnpm install# oryarn install
```

Example 4 (go):
```go
# Clear node_modules and reinstallrm -rf node_modules package-lock.jsonnpm install
# Or try with different package managerpnpm install# oryarn install
```

---
