# Llamaindex-Ts - Llamacloud

**Pages:** 12

---

## 6. Adding LlamaParse

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/agents/6_llamaparse/

**Contents:**
- 6. Adding LlamaParse

Complicated PDFs can be very tricky for LLMs to understand. To help with this, LlamaIndex provides LlamaParse, a hosted service that parses complex documents including PDFs. To use it, get a LLAMA_CLOUD_API_KEY by signing up for LlamaCloud (it’s free for up to 1000 pages/day) and adding it to your .env file just as you did for your OpenAI key:

Then replace SimpleDirectoryReader with LlamaParseReader:

Now you will be able to ask more complicated questions of the same PDF and get better results. You can find this code in our repo.

Next up, let’s persist our embedded data so we don’t have to re-parse every time by using a vector store.

**Examples:**

Example 1 (unknown):
```unknown
LLAMA_CLOUD_API_KEY=llx-XXXXXXXXXXXXXXXX
```

Example 2 (unknown):
```unknown
LLAMA_CLOUD_API_KEY=llx-XXXXXXXXXXXXXXXX
```

Example 3 (javascript):
```javascript
const reader = new LlamaParseReader({ resultType: "markdown" });const documents = await reader.loadData("../data/sf_budget_2023_2024.pdf");
```

Example 4 (javascript):
```javascript
const reader = new LlamaParseReader({ resultType: "markdown" });const documents = await reader.loadData("../data/sf_budget_2023_2024.pdf");
```

---

## Image Retrieval

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/readers/llama_parse/images/

**Contents:**
- Image Retrieval
- Installation
- Usage
  - Multimodal Indexing
    - Text Documents
    - Image Documents

LlamaParse json mode supports extracting any images found in a page object by using the getImages function. They are downloaded to a local folder and can then be sent to a multimodal LLM for further processing.

We use the getImages method to input our array of JSON objects, download the images to a specified folder and get a list of ImageNodes.

You can create an index across both text and image nodes by requesting alternative text for the image from a multimodal LLM.

We use two helper functions to create documents from the text and image nodes provided.

To create documents from the text nodes of the json object, we just map the needed values to a new Document object. In this case we assign the text as text and the page number as metadata.

To create documents from the images, we need to use a multimodal LLM to generate alt text.

For this we create ImageNodes and add them as part of our message.

We can use the createMessageContent function to simplify this.

The returned imageDocs have the alt text assigned as text and the image path as metadata.

You can see the full example file here.

**Examples:**

Example 1 (python):
```python
npm i llamaindex llama-cloud-services @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex llama-cloud-services @llamaindex/openai
```

Example 3 (javascript):
```javascript
const reader = new LlamaParseReader();const jsonObjs = await reader.loadJson("../data/uber_10q_march_2022.pdf");const imageDicts = await reader.getImages(jsonObjs, "images");
```

Example 4 (javascript):
```javascript
const reader = new LlamaParseReader();const jsonObjs = await reader.loadJson("../data/uber_10q_march_2022.pdf");const imageDicts = await reader.getImages(jsonObjs, "images");
```

---

## Index

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/data_index/

**Contents:**
- Index
- API Reference

An index is the basic container for organizing your data. Besides managed indexes using LlamaCloud, LlamaIndex.TS supports three indexes:

**Examples:**

Example 1 (javascript):
```javascript
import { Document, VectorStoreIndex } from "llamaindex";
const document = new Document({ text: "test" });
const index = await VectorStoreIndex.fromDocuments([document]);
```

Example 2 (javascript):
```javascript
import { Document, VectorStoreIndex } from "llamaindex";
const document = new Document({ text: "test" });
const index = await VectorStoreIndex.fromDocuments([document]);
```

---

## JSON Mode

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/readers/llama_parse/json_mode/

**Contents:**
- JSON Mode
- Installation
- Usage
  - Output
    - Page objects
  - JSON Mode with SimpleDirectoryReader
- API Reference

In JSON mode, LlamaParse will return a data structure representing the parsed object.

For Json mode, you need to use loadJson. The resultType is automatically set with this method. More information about indexing the results on the next page.

The result format of the response, written to jsonObjs in the example, follows this structure:

Within page objects, the following keys may be present depending on your document.

All Readers share a loadData method with SimpleDirectoryReader that promises to return a uniform Document with Metadata. This makes JSON mode incompatible with SimpleDirectoryReader.

However, a simple work around is to create a new reader class that extends LlamaParseReader and adds a new method or overrides loadData, wrapping around JSON mode, extracting the required values, and returning a Document object.

Now we have documents with page number as metadata. This new reader can be used like any other and be integrated with SimpleDirectoryReader. Since it extends LlamaParseReader, you can use the same params.

You can assign any other values of the JSON response to the Document as needed.

**Examples:**

Example 1 (unknown):
```unknown
npm i llamaindex llama-cloud-services
```

Example 2 (unknown):
```unknown
npm i llamaindex llama-cloud-services
```

Example 3 (javascript):
```javascript
import { LlamaParseReader } from "llama-cloud-services";
const reader = new LlamaParseReader();async function main() {  // Load the file and return an array of json objects  const jsonObjs = await reader.loadJson("../data/uber_10q_march_2022.pdf");  // Access the first "pages" (=a single parsed file) object in the array  const jsonList = jsonObjs[0]["pages"];  // Further process the jsonList object as needed.}
```

Example 4 (javascript):
```javascript
import { LlamaParseReader } from "llama-cloud-services";
const reader = new LlamaParseReader();async function main() {  // Load the file and return an array of json objects  const jsonObjs = await reader.loadJson("../data/uber_10q_march_2022.pdf");  // Access the first "pages" (=a single parsed file) object in the array  const jsonList = jsonObjs[0]["pages"];  // Further process the jsonList object as needed.}
```

---

## LlamaParse

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/readers/llama_parse/

**Contents:**
- LlamaParse
- Usage
  - Params
    - General params:
    - Advanced params:
  - LlamaParse with SimpleDirectoryReader
- API Reference

LlamaParse is an API created by LlamaIndex to efficiently parse files, e.g. it’s great at converting PDF tables into markdown.

To use it, first login and get an API key from https://cloud.llamaindex.ai. Make sure to store the key as apiKey parameter or in the environment variable LLAMA_CLOUD_API_KEY.

Official documentation for LlamaParse can be found here.

You can then use the LlamaParseReader class to load local files and convert them into a parsed document that can be used by LlamaIndex. See reader.ts for a list of supported file types:

All options can be set with the LlamaParseReader constructor.

They can be divided into two groups.

Below a full example of LlamaParse integrated in SimpleDirectoryReader with additional options.

---

## Managed Index

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/data_index/managed/

**Contents:**
- Managed Index
- Access
- Create a Managed Index
- Use a Managed Index
- API Reference

LlamaCloud is a new generation of managed parsing, ingestion, and retrieval services, designed to bring production-grade context-augmentation to your LLM and RAG applications.

Visit LlamaCloud to sign in and get an API key.

Here’s an example of how to create a managed index by ingesting a couple of documents:

Here’s an example of how to use a managed index together with a chat engine:

---

## Metadata Extraction

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/ingestion_pipeline/transformations/metadata_extraction/

**Contents:**
- Metadata Extraction
- API Reference

You can use LLMs to automate metadata extraction with our Metadata Extractor modules.

Our metadata extractor modules include the following “feature extractors”:

Then you can chain the Metadata Extractors with the IngestionPipeline to extract metadata from a set of documents.

**Examples:**

Example 1 (javascript):
```javascript
import { Document, IngestionPipeline, TitleExtractor, QuestionsAnsweredExtractor } from "llamaindex";import { OpenAI } from "@llamaindex/openai";
async function main() {  const pipeline = new IngestionPipeline({    transformations: [      new TitleExtractor(),      new QuestionsAnsweredExtractor({        questions: 5,      }),    ],  });
  const nodes = await pipeline.run({    documents: [      new Document({ text: "I am 10 years old. John is 20 years old." }),    ],  });
  for (const node of nodes) {    console.log(node.metadata);  }}
main().then(() => console.log("done"));
```

Example 2 (javascript):
```javascript
import { Document, IngestionPipeline, TitleExtractor, QuestionsAnsweredExtractor } from "llamaindex";import { OpenAI } from "@llamaindex/openai";
async function main() {  const pipeline = new IngestionPipeline({    transformations: [      new TitleExtractor(),      new QuestionsAnsweredExtractor({        questions: 5,      }),    ],  });
  const nodes = await pipeline.run({    documents: [      new Document({ text: "I am 10 years old. John is 20 years old." }),    ],  });
  for (const node of nodes) {    console.log(node.metadata);  }}
main().then(() => console.log("done"));
```

---

## Node Parsers / Text Splitters

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/ingestion_pipeline/transformations/node-parser/

**Contents:**
- Node Parsers / Text Splitters
- SentenceSplitter
- MarkdownNodeParser
- CodeSplitter
- API Reference

Node parsers are a simple abstraction that take a list of Document objects, and chunk them into Node objects, such that each node is a specific chunk of the parent document. When a document is broken into nodes, all of it’s attributes are inherited to the children nodes (i.e. metadata, text and metadata templates, etc.). You can read more about Node and Document properties here.

By default, we will use Settings.nodeParser to split the document into nodes. You can also assign a custom NodeParser to the Settings object.

The SentenceSplitter is the default NodeParser in LlamaIndex. It will split the text from a Document into sentences.

The underlying text splitter will split text by sentences. It can also be used as a standalone module for splitting raw text.

The MarkdownNodeParser is a more advanced NodeParser that can handle markdown documents. It will split the markdown into nodes and then parse the nodes into a Document object.

The output metadata will be something like:

The CodeSplitter is a more advanced NodeParser that can handle code documents. It will split the code by AST nodes and then parse the nodes into a Document object.

You might setup WASM files for web-tree-sitter and use it in the browser.

In this example, you should put tree-sitter-typescript.wasm to the public folder for Next.js.

And also update the next.config.js to make @llamaindex/env work properly.

**Examples:**

Example 1 (sql):
```sql
import { TextFileReader } from '@llamaindex/readers/text'import { SentenceSplitter } from 'llamaindex';import { Settings } from 'llamaindex';
const nodeParser = new SentenceSplitter();Settings.nodeParser = nodeParser;//         ^?
```

Example 2 (sql):
```sql
import { TextFileReader } from '@llamaindex/readers/text'import { SentenceSplitter } from 'llamaindex';import { Settings } from 'llamaindex';
const nodeParser = new SentenceSplitter();Settings.nodeParser = nodeParser;//         ^?
```

Example 3 (sql):
```sql
import { SentenceSplitter } from "llamaindex";
const splitter = new SentenceSplitter({ chunkSize: 1 });
const texts = splitter.splitText("Hello World");//     ^?
```

Example 4 (sql):
```sql
import { SentenceSplitter } from "llamaindex";
const splitter = new SentenceSplitter({ chunkSize: 1 });
const texts = splitter.splitText("Hello World");//     ^?
```

---

## OVHcloud AI Endpoints

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/ovhcloud/

**Contents:**
- OVHcloud AI Endpoints
- Installation
- Authentication
- Basic Usage
- Standalone Usage
- Base URL
- Resources
- API Reference

OVHcloud AI Endpoints provide OpenAI-compatible embedding models. The service can be used for free with rate limits, or with an API key for higher limits.

OVHcloud is a global player and the leading European cloud provider operating over 450,000 servers within 40 data centers across 4 continents to reach 1.6 million customers in over 140 countries. Our product AI Endpoints offers access to various models with sovereignty, data privacy and GDPR compliance.

You can find the full list of models in the OVHcloud AI Endpoints catalog.

OVHcloud AI Endpoints can be used in two ways:

By default, OVHcloudEmbedding uses the BGE-M3 model. You can change the model by passing the model parameter to the constructor:

You can also set the maxRetries and timeout parameters when initializing OVHcloudEmbedding for better control over the request behavior:

The default base URL is https://oai.endpoints.kepler.ai.cloud.ovh.net/v1. You can override it if needed:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/ovhcloud
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/ovhcloud
```

Example 3 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { OVHcloudEmbedding } from "@llamaindex/ovhcloud";
// Update Embed Model (using free tier)Settings.embedModel = new OVHcloudEmbedding();
// Or with API key from environment variableimport { config } from "dotenv";config();Settings.embedModel = new OVHcloudEmbedding({  apiKey: process.env.OVHCLOUD_API_KEY || "",});
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { OVHcloudEmbedding } from "@llamaindex/ovhcloud";
// Update Embed Model (using free tier)Settings.embedModel = new OVHcloudEmbedding();
// Or with API key from environment variableimport { config } from "dotenv";config();Settings.embedModel = new OVHcloudEmbedding({  apiKey: process.env.OVHCLOUD_API_KEY || "",});
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## OVHcloud AI Endpoints

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/ovhcloud/

**Contents:**
- OVHcloud AI Endpoints
- Installation
- Authentication
- Basic Usage
- Load and index documents
- Query
- Full Example
- Streaming
- Base URL
- Resources

OVHcloud AI Endpoints provide serverless access to a variety of pre-trained AI models. The service is OpenAI-compatible and can be used for free with rate limits, or with an API key for higher limits.

OVHcloud is a global player and the leading European cloud provider operating over 450,000 servers within 40 data centers across 4 continents to reach 1.6 million customers in over 140 countries. Our product AI Endpoints offers access to various models with sovereignty, data privacy and GDPR compliance.

You can find the full list of models in the OVHcloud AI Endpoints catalog.

OVHcloud AI Endpoints can be used in two ways:

You can set the API key via environment variable:

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

OVHcloud AI Endpoints supports streaming responses:

The default base URL is https://oai.endpoints.kepler.ai.cloud.ovh.net/v1. You can override it if needed:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/ovhcloud
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/ovhcloud
```

Example 3 (julia):
```julia
import { OVHcloudLLM } from "@llamaindex/ovhcloud";import { Settings } from "llamaindex";
// Using without API key (free tier with rate limits)Settings.llm = new OVHcloudLLM({  model: "gpt-oss-120b",});
// Or with API key from environment variableimport { config } from "dotenv";config();Settings.llm = new OVHcloudLLM({  model: "gpt-oss-120b",  apiKey: process.env.OVHCLOUD_API_KEY || "",});
// Or with explicit API keySettings.llm = new OVHcloudLLM({  model: "gpt-oss-120b",  apiKey: "YOUR_API_KEY",});
```

Example 4 (julia):
```julia
import { OVHcloudLLM } from "@llamaindex/ovhcloud";import { Settings } from "llamaindex";
// Using without API key (free tier with rate limits)Settings.llm = new OVHcloudLLM({  model: "gpt-oss-120b",});
// Or with API key from environment variableimport { config } from "dotenv";config();Settings.llm = new OVHcloudLLM({  model: "gpt-oss-120b",  apiKey: process.env.OVHCLOUD_API_KEY || "",});
// Or with explicit API keySettings.llm = new OVHcloudLLM({  model: "gpt-oss-120b",  apiKey: "YOUR_API_KEY",});
```

---

## Router Query Engine

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/query_engines/router_query_engine/

**Contents:**
- Router Query Engine
- Setup
- Loading Data
- Service Context
- Creating Indices
- Creating Query Engines
- Creating a Router Query Engine
- Querying the Router Query Engine
- Full code
- API Reference

In this tutorial, we define a custom router query engine that selects one out of several candidate query engines to execute a query.

First, we need to install import the necessary modules from llamaindex:

Next, we need to load some data. We will use the SimpleDirectoryReader to load documents from a directory:

Next, we need to define some basic rules and parse the documents into nodes. We will use the SentenceSplitter to parse the documents into nodes and Settings to define the rules (eg. LLM API key, chunk size, etc.):

Next, we need to create some indices. We will create a VectorStoreIndex and a SummaryIndex:

Next, we need to create some query engines. We will create a VectorStoreQueryEngine and a SummaryQueryEngine:

Next, we need to create a router query engine. We will use the RouterQueryEngine to create a router query engine:

We’re defining two query engines, one for summarization and one for retrieving specific context. The router query engine will select the most appropriate query engine based on the query.

Finally, we can query the router query engine:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/readers
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/readers
```

Example 3 (sql):
```sql
import {  RouterQueryEngine,  SimpleDirectoryReader,  SentenceSplitter,  SummaryIndex,  VectorStoreIndex,  Settings,} from "llamaindex";import { OpenAI } from "@llamaindex/openai";import { SimpleDirectoryReader } from "@llamaindex/readers/directory";
```

Example 4 (sql):
```sql
import {  RouterQueryEngine,  SimpleDirectoryReader,  SentenceSplitter,  SummaryIndex,  VectorStoreIndex,  Settings,} from "llamaindex";import { OpenAI } from "@llamaindex/openai";import { SimpleDirectoryReader } from "@llamaindex/readers/directory";
```

---

## Structured data extraction

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/structured_data_extraction/

**Contents:**
- Structured data extraction
- Set up
- Extract data
- Using the exec method

Make sure you have installed LlamaIndex.TS and have an OpenAI key. If you haven’t, check out the installation guide.

You can use other LLMs via their APIs; if you would prefer to use local models check out our local LLM example.

Create the file example.ts. This code will:

You should expect output something like:

Many LLMs do not natively support structured output, and often rely exclusively on prompt or context engineering.

In this sense, we proved you with an alternative for structured data extraction, using the exec method with responseFormat.

For example, you can, in a new folder, install our Anthropic integration and zod v3:

And then try extracting data with this code:

The output should look like this:

**Examples:**

Example 1 (python):
```python
npm initnpm i -D typescript @types/nodenpm i @llamaindex/openai zod
```

Example 2 (python):
```python
npm initnpm i -D typescript @types/nodenpm i @llamaindex/openai zod
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
