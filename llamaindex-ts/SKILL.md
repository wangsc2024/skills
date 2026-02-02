---
name: llamaindex-ts
description: |
  LlamaIndex.TS - The leading TypeScript/JavaScript framework for building LLM-powered applications.
  Use this skill when building RAG pipelines, agents, chatbots, document parsing with LlamaParse,
  structured data extraction, or using LlamaCloud services.
triggers:
  - "llamaindex"
  - "llamaindex.ts"
  - "llamaindexts"
  - "rag typescript"
  - "rag javascript"
  - "llamaparse"
  - "llamacloud"
  - "llamaextract"
  - "vector store typescript"
  - "document qa typescript"
  - "llm agent typescript"
  - "create-llama"
---

# LlamaIndex.TS Skill

LlamaIndex.TS is the leading framework for building LLM-powered applications in JavaScript and TypeScript. It provides tools for RAG (Retrieval-Augmented Generation), agents, document parsing, and integration with LlamaCloud services.

## When to Use This Skill

- Building RAG (Retrieval-Augmented Generation) applications in TypeScript/JavaScript
- Creating LLM-powered agents with tools and function calling
- Parsing documents with LlamaParse (PDFs, Word docs, etc.)
- Extracting structured data from unstructured sources
- Building chatbots and Q&A systems over custom data
- Using LlamaCloud services (Index, Parse, Extract, Classify)
- Integrating with vector stores (Pinecone, Qdrant, Chroma, etc.)
- Working with embeddings and LLMs in Node.js

## Quick Start

### Installation

```bash
npm i llamaindex @llamaindex/openai @llamaindex/workflow
```

### Environment Setup

```bash
export OPENAI_API_KEY=sk-your-api-key
```

Or create a `.env` file and use `dotenv`:

```bash
npm i dotenv
```

### TypeScript Configuration

Required `tsconfig.json` settings:

```json
{
  "compilerOptions": {
    "moduleResolution": "bundler",
    "target": "es2020",
    "module": "esnext",
    "lib": ["DOM.AsyncIterable"]
  }
}
```

## Quick Reference

### Simple RAG Example

```typescript
import { Document, VectorStoreIndex } from "llamaindex";

// Create document from text
const document = new Document({ text: "Your content here", id_: "doc1" });

// Build index from documents
const index = await VectorStoreIndex.fromDocuments([document]);

// Create query engine and ask questions
const queryEngine = index.asQueryEngine();
const response = await queryEngine.query({ query: "What is the content about?" });
console.log(response.toString());
```

### Create an Agent with Tools

```typescript
import { agent, tool } from "@llamaindex/workflow";
import { openai } from "@llamaindex/openai";

// Define a tool
const sumNumbers = tool({
  name: "sumNumbers",
  description: "Adds two numbers together",
  parameters: {
    type: "object",
    properties: {
      a: { type: "number", description: "First number" },
      b: { type: "number", description: "Second number" }
    },
    required: ["a", "b"]
  },
  execute: ({ a, b }) => `${a + b}`
});

// Create agent with tools
const myAgent = agent({
  name: "math-agent",
  llm: openai({ model: "gpt-4o" }),
  tools: [sumNumbers]
});

// Run the agent
const result = await myAgent.run("What is 5 + 3?");
console.log(result.data.result);
```

### LLM Configuration

```typescript
import { Settings } from "llamaindex";
import { OpenAI } from "@llamaindex/openai";

// Set global LLM
Settings.llm = new OpenAI({ model: "gpt-4o", temperature: 0 });
```

### Using Local Models with Ollama

```typescript
import { ollama } from "@llamaindex/ollama";
import { Settings } from "llamaindex";

Settings.llm = ollama({ model: "mixtral:8x7b" });
```

### Embedding Models

```typescript
import { Settings } from "llamaindex";
import { OpenAIEmbedding } from "@llamaindex/openai";

Settings.embedModel = new OpenAIEmbedding({ model: "text-embedding-3-small" });
```

### Memory for Agents

```typescript
import { createMemory, staticBlock } from "llamaindex";
import { agent } from "@llamaindex/workflow";

const memory = createMemory({
  memoryBlocks: [
    staticBlock({
      content: "The user prefers TypeScript and is building a RAG application."
    })
  ]
});

const myAgent = agent({
  name: "assistant",
  llm: openai({ model: "gpt-4o" }),
  memory
});
```

### Loading Documents

```typescript
import { SimpleDirectoryReader } from "@llamaindex/readers";

// Load all documents from a directory
const reader = new SimpleDirectoryReader();
const documents = await reader.loadData("./data");
```

### Using LlamaParse for Complex Documents

```typescript
import { LlamaParseReader } from "@llamaindex/cloud";

const reader = new LlamaParseReader({ resultType: "markdown" });
const documents = await reader.loadData("./document.pdf");
```

### Vector Stores

```typescript
import { PineconeVectorStore } from "@llamaindex/pinecone";
import { VectorStoreIndex } from "llamaindex";

const vectorStore = new PineconeVectorStore({ indexName: "my-index" });
const index = await VectorStoreIndex.fromVectorStore(vectorStore);
```

## Core Concepts

### RAG Pipeline Stages

1. **Loading**: Ingest data from various sources (PDFs, APIs, databases)
2. **Indexing**: Create vector embeddings and store in index
3. **Querying**: Retrieve relevant context and generate responses

### Key Components

- **Document**: Container for data with text and metadata
- **Node**: Atomic unit representing a chunk of a Document
- **Index**: Stores processed data for efficient retrieval
- **Query Engine**: End-to-end pipeline for Q&A
- **Chat Engine**: Multi-turn conversation interface
- **Agent**: Autonomous task executor with tools

## Reference Files

Detailed documentation in `references/`:

- **getting_started.md** - Installation, concepts, and first steps
- **agents.md** - Building agents with tools, RAG agents, multi-agent systems
- **rag.md** - Vector stores, embeddings, query engines, retrievers
- **llamacloud.md** - LlamaParse, LlamaExtract, and cloud services

## Common Packages

| Package | Purpose |
|---------|---------|
| `llamaindex` | Core framework |
| `@llamaindex/openai` | OpenAI integration |
| `@llamaindex/workflow` | Agent workflows |
| `@llamaindex/readers` | Document loaders |
| `@llamaindex/cloud` | LlamaCloud services |
| `@llamaindex/ollama` | Local models |
| `@llamaindex/pinecone` | Pinecone vector store |
| `@llamaindex/qdrant` | Qdrant vector store |

## Scaffolding with create-llama

Generate a full-stack LLM application:

```bash
npx create-llama@latest
```

This creates a working app with:
- Next.js frontend
- RAG pipeline
- Chat interface
- API routes

## Resources

- [Official Documentation](https://developers.llamaindex.ai/typescript/)
- [GitHub Repository](https://github.com/run-llama/LlamaIndexTS)
- [LlamaCloud](https://cloud.llamaindex.ai/)
- [Examples](https://github.com/run-llama/LlamaIndexTS/tree/main/examples)

## Notes

- LlamaIndex.TS uses gpt-4o by default
- Install `gpt-tokenizer` for 60x faster tokenization
- Use `npx tsx` to run TypeScript files directly
- API keys should never be committed to repositories
