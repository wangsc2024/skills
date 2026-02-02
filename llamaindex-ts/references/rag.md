# Llamaindex-Ts - Rag

**Pages:** 62

---

## Anthropic

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/anthropic/

**Contents:**
- Anthropic
- Installation
- Usage
- Load and index documents
- Query
- Full Example
- API Reference

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/anthropic
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/anthropic
```

Example 3 (sql):
```sql
import { Settings } from "llamaindex";import { Anthropic } from "@llamaindex/anthropic";
Settings.llm = new Anthropic({  apiKey: "<YOUR_API_KEY>",});
```

Example 4 (sql):
```sql
import { Settings } from "llamaindex";import { Anthropic } from "@llamaindex/anthropic";
Settings.llm = new Anthropic({  apiKey: "<YOUR_API_KEY>",});
```

---

## Azure OpenAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/azure/

**Contents:**
- Azure OpenAI
- Installation
- Usage
- Examples
- API Reference

To use Azure OpenAI, you only need to install the @llamaindex/azure package:

The class AzureOpenAI is used for setting the LLM and AzureOpenAIEmbedding is used for setting the embedding model, e.g.:

Instead of explicitly setting the API key, deployment, version, and endpoint in the constructor, you can use the following environment variables: AZURE_OPENAI_DEPLOYMENT for the model deployment name, AZURE_OPENAI_KEY for your API key, AZURE_OPENAI_ENDPOINT for your Azure endpoint URL, and AZURE_OPENAI_API_VERSION for the API version.

See the Azure examples for more examples of how to use Azure OpenAI.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/azure
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/azure
```

Example 3 (sql):
```sql
import { Settings } from "llamaindex";import { AzureOpenAI, AzureOpenAIEmbedding } from "@llamaindex/azure";
Settings.llm = new AzureOpenAI({  apiKey: '[key]',  deployment: '[model]',  apiVersion: '[version]',  endpoint: `https://[deployment].openai.azure.com/`,});Settings.embedModel = new AzureOpenAIEmbedding({  apiKey: '[key]',  deployment: '[embedding-model]',  apiVersion: '[version]',  endpoint: `https://[deployment].openai.azure.com/`,});
```

Example 4 (sql):
```sql
import { Settings } from "llamaindex";import { AzureOpenAI, AzureOpenAIEmbedding } from "@llamaindex/azure";
Settings.llm = new AzureOpenAI({  apiKey: '[key]',  deployment: '[model]',  apiVersion: '[version]',  endpoint: `https://[deployment].openai.azure.com/`,});Settings.embedModel = new AzureOpenAIEmbedding({  apiKey: '[key]',  deployment: '[embedding-model]',  apiVersion: '[version]',  endpoint: `https://[deployment].openai.azure.com/`,});
```

---

## Bedrock

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/bedrock/

**Contents:**
- Bedrock
- Installation
- Usage
- Full Example
- Agent Example

Supported models are listed below (accessible by BEDROCK_MODELS).

You can also use Bedrock’s Inference endpoints by using the model names (accessible by INFERENCE_BEDROCK_MODELS). Note that the region must be set correctly.

Sonnet, Haiku and Opus are multimodal, image_url only supports base64 data url format, e.g. data:image/jpeg;base64,SGVsbG8sIFdvcmxkIQ==

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/aws
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/aws
```

Example 3 (sql):
```sql
import { BEDROCK_MODELS, Bedrock } from "@llamaindex/aws";
Settings.llm = new Bedrock({  model: BEDROCK_MODELS.ANTHROPIC_CLAUDE_3_HAIKU,  region: "us-east-1", // can be provided via env AWS_REGION  credentials: {    accessKeyId: "...", // optional and can be provided via env AWS_ACCESS_KEY_ID    secretAccessKey: "...", // optional and can be provided via env AWS_SECRET_ACCESS_KEY  },});
```

Example 4 (sql):
```sql
import { BEDROCK_MODELS, Bedrock } from "@llamaindex/aws";
Settings.llm = new Bedrock({  model: BEDROCK_MODELS.ANTHROPIC_CLAUDE_3_HAIKU,  region: "us-east-1", // can be provided via env AWS_REGION  credentials: {    accessKeyId: "...", // optional and can be provided via env AWS_ACCESS_KEY_ID    secretAccessKey: "...", // optional and can be provided via env AWS_SECRET_ACCESS_KEY  },});
```

---

## ChatEngine

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/chat_engine/

**Contents:**
- ChatEngine
- Api References

The chat engine is a quick and simple way to chat with the data in your index.

In short, you can use the chat engine by calling index.asChatEngine(). It will return a ContextChatEngine to start chatting.

You can also pass in options to the chat engine.

The chat function also supports streaming, just add stream: true as an option:

**Examples:**

Example 1 (javascript):
```javascript
const retriever = index.asRetriever();const chatEngine = new ContextChatEngine({ retriever });
// start chattingconst response = await chatEngine.chat({ message: query });
```

Example 2 (javascript):
```javascript
const retriever = index.asRetriever();const chatEngine = new ContextChatEngine({ retriever });
// start chattingconst response = await chatEngine.chat({ message: query });
```

Example 3 (javascript):
```javascript
const chatEngine = index.asChatEngine();
```

Example 4 (javascript):
```javascript
const chatEngine = index.asChatEngine();
```

---

## Chat Stores

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/chat_stores/

**Contents:**
- Chat Stores
- Available Chat Stores
- API Reference

Chat stores manage chat history by storing sequences of messages in a structured way, ensuring the order of messages is maintained for accurate conversation flow.

Check the LlamaIndexTS Github for the most up to date overview of integrations.

---

## Cohere Reranker

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/node_postprocessors/cohere_reranker/

**Contents:**
- Cohere Reranker
- Setup
- Load and index documents
- Increase similarity topK to retrieve more results
- Create a new instance of the CohereRerank class
- Create a query engine with the retriever and node postprocessor
- API Reference

The Cohere Reranker is a postprocessor that uses the Cohere API to rerank the results of a search query.

Firstly, you will need to install the llamaindex package.

Now, you will need to sign up for an API key at Cohere. Once you have your API key you can import the necessary modules and create a new instance of the CohereRerank class.

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

The default value for similarityTopK is 2. This means that only the most similar document will be returned. To retrieve more results, you can increase the value of similarityTopK.

Then you can create a new instance of the CohereRerank class and pass in your API key and the number of results you want to return.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/cohere @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/cohere @llamaindex/openai
```

Example 3 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";import { CohereRerank } from "@llamaindex/cohere";import { Document, Settings, VectorStoreIndex } from "llamaindex";
```

Example 4 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";import { CohereRerank } from "@llamaindex/cohere";import { Document, Settings, VectorStoreIndex } from "llamaindex";
```

---

## Correctness Evaluator

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/evaluation/correctness/

**Contents:**
- Correctness Evaluator
- Usage
- API Reference

Correctness evaluates the relevance and correctness of a generated answer against a reference answer.

This is useful for measuring if the response was correct. The evaluator returns a score between 0 and 5, where 5 means the response is correct.

Firstly, you need to install the package:

Set the OpenAI API key:

Import the required modules:

Let’s setup gpt-4 for better results:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 3 (unknown):
```unknown
export OPENAI_API_KEY=your-api-key
```

Example 4 (unknown):
```unknown
export OPENAI_API_KEY=your-api-key
```

---

## Custom Model Per Request

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/custom_model_per_request/

**Contents:**
- Custom Model Per Request

There are scenarios, such as the case of a multi-tenant backend API, where it may be required to handle each request with a custom model.

In such a scenario, modifying the Settings object directly as follows is not recommended:

Setting llm and embedModel directly will lead to unpredictable responses, since Settings is global and mutable. This can lead to race conditions, as each request modifies Settings.embedModel or Settings.llm.

The recommended approach is to use Settings.withEmbedModel or Settings.withLLM as follows:

The full example can be found here.

**Examples:**

Example 1 (sql):
```sql
import { Settings } from 'llamaindex';import { OpenAIEmbedding } from '@llamaindex/embeddings-openai';
Settings.embedModel = new OpenAIEmbedding({ apiKey: 'CLIENT_API_KEY' });Settings.llm = openai({ apiKey: key,  model: 'gpt-4o' })
```

Example 2 (sql):
```sql
import { Settings } from 'llamaindex';import { OpenAIEmbedding } from '@llamaindex/embeddings-openai';
Settings.embedModel = new OpenAIEmbedding({ apiKey: 'CLIENT_API_KEY' });Settings.llm = openai({ apiKey: key,  model: 'gpt-4o' })
```

Example 3 (javascript):
```javascript
const embedModel = new OpenAIEmbedding({  apiKey: process.env.OPENAI_API_KEY,});const llm = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const llmResponse = await Settings.withEmbedModel(embedModel, async () => {  return Settings.withLLM(llm, async () => {    const path = "node_modules/llamaindex/examples/abramov.txt";    const essay = await fs.readFile(path, "utf-8");    // Create Document object with essay    const document = new Document({ text: essay, id_: path });    // Split text and create embeddings. Store them in a VectorStoreIndex    const index = await VectorStoreIndex.fromDocuments([document]);    // Query the index    const queryEngine = index.asQueryEngine();    const { message, sourceNodes } = await queryEngine.query({      query: "What did the author do in college?",    });    // Return response with sources    return message.content;  });});
```

Example 4 (javascript):
```javascript
const embedModel = new OpenAIEmbedding({  apiKey: process.env.OPENAI_API_KEY,});const llm = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const llmResponse = await Settings.withEmbedModel(embedModel, async () => {  return Settings.withLLM(llm, async () => {    const path = "node_modules/llamaindex/examples/abramov.txt";    const essay = await fs.readFile(path, "utf-8");    // Create Document object with essay    const document = new Document({ text: essay, id_: path });    // Split text and create embeddings. Store them in a VectorStoreIndex    const index = await VectorStoreIndex.fromDocuments([document]);    // Query the index    const queryEngine = index.asQueryEngine();    const { message, sourceNodes } = await queryEngine.query({      query: "What did the author do in college?",    });    // Return response with sources    return message.content;  });});
```

---

## DeepInfra

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/deepinfra/

**Contents:**
- DeepInfra
- Installation
- API Reference

To use DeepInfra embeddings, you need to import DeepInfraEmbedding from llamaindex. Check out available embedding models here.

By default, DeepInfraEmbedding is using the sentence-transformers/clip-ViT-B-32 model. You can change the model by passing the model parameter to the constructor. For example:

You can also set the maxRetries and timeout parameters when initializing DeepInfraEmbedding for better control over the request behavior.

For questions or feedback, please contact us at feedback@deepinfra.com

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/deepinfra
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/deepinfra
```

Example 3 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { DeepInfraEmbedding } from "@llamaindex/deepinfra";
// Update Embed ModelSettings.embedModel = new DeepInfraEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { DeepInfraEmbedding } from "@llamaindex/deepinfra";
// Update Embed ModelSettings.embedModel = new DeepInfraEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## DeepInfra

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/deepinfra/

**Contents:**
- DeepInfra
- Installation
- Load and index documents
- Query
- Full Example
- Feedback
- API Reference

Check out available LLMs here.

You can setup the apiKey on the environment variables, like:

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

If you have any feedback, please reach out to us at feedback@deepinfra.com

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/deepinfra
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/deepinfra
```

Example 3 (sql):
```sql
import { DeepInfra } from "@llamaindex/deepinfra";import { Settings } from "llamaindex";
// Get the API key from `DEEPINFRA_API_TOKEN` environment variableimport { config } from "dotenv";config();Settings.llm = new DeepInfra();
// Set the API keyapiKey = "YOUR_API_KEY";Settings.llm = new DeepInfra({ apiKey });
```

Example 4 (sql):
```sql
import { DeepInfra } from "@llamaindex/deepinfra";import { Settings } from "llamaindex";
// Get the API key from `DEEPINFRA_API_TOKEN` environment variableimport { config } from "dotenv";config();Settings.llm = new DeepInfra();
// Set the API keyapiKey = "YOUR_API_KEY";Settings.llm = new DeepInfra({ apiKey });
```

---

## DeepSeek LLM

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/deepseek/

**Contents:**
- DeepSeek LLM
- Usage
- Example
- Limitations
- API Reference

Currently does not support function calling.

Currently does not support json-output param while still is very good at json generating.

**Examples:**

Example 1 (sql):
```sql
import { Settings } from "llamaindex";import { DeepSeekLLM } from "@llamaindex/deepseek";
Settings.llm = new DeepSeekLLM({  apiKey: "<YOUR_API_KEY>",  model: "deepseek-coder", // or "deepseek-chat"});
```

Example 2 (sql):
```sql
import { Settings } from "llamaindex";import { DeepSeekLLM } from "@llamaindex/deepseek";
Settings.llm = new DeepSeekLLM({  apiKey: "<YOUR_API_KEY>",  model: "deepseek-coder", // or "deepseek-chat"});
```

Example 3 (javascript):
```javascript
import { Document, VectorStoreIndex, Settings } from "llamaindex";import { DeepSeekLLM } from "@llamaindex/deepseek";
const deepseekLlm = new DeepSeekLLM({  apiKey: "<YOUR_API_KEY>",  model: "deepseek-coder", // or "deepseek-chat"});
async function main() {  const response = await llm.deepseekLlm.chat({    messages: [      {        role: "system",        content: "You are an AI assistant",      },      {        role: "user",        content: "Tell me about San Francisco",      },    ],    stream: false,  });  console.log(response);}
```

Example 4 (javascript):
```javascript
import { Document, VectorStoreIndex, Settings } from "llamaindex";import { DeepSeekLLM } from "@llamaindex/deepseek";
const deepseekLlm = new DeepSeekLLM({  apiKey: "<YOUR_API_KEY>",  model: "deepseek-coder", // or "deepseek-chat"});
async function main() {  const response = await llm.deepseekLlm.chat({    messages: [      {        role: "system",        content: "You are an AI assistant",      },      {        role: "user",        content: "Tell me about San Francisco",      },    ],    stream: false,  });  console.log(response);}
```

---

## DiscordReader

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/readers/discord/

**Contents:**
- DiscordReader
- Installation
- Usage
  - Params
    - DiscordReader()
    - DiscordReader.loadData
- API Reference

DiscordReader is a simple data loader that reads all messages in a given Discord channel and returns them as Document objects. It uses the @discordjs/rest library to fetch the messages.

First step is to create a Discord Application and generating a bot token here. In your Discord Application, go to the OAuth2 tab and generate an invite URL by selecting bot and click Read Messages/View Channels as wells as Read Message History. This will invite the bot with the necessary permissions to read messages. Copy the URL in your browser and select the server you want your bot to join.

**Examples:**

Example 1 (python):
```python
npm install @llamaindex/discord
```

Example 2 (python):
```python
npm install @llamaindex/discord
```

---

## Documents and Nodes

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/

**Contents:**
- Documents and Nodes
- API Reference

Documents and Nodes are the basic building blocks of data in LlamaIndexTS. While the API for these objects is similar, Document objects represent entire files, while Nodes are smaller pieces of that original document, that are suitable for an LLM and Q&A.

**Examples:**

Example 1 (sql):
```sql
import { Document } from "llamaindex";
document = new Document({ text: "text", metadata: { key: "val" } });
```

Example 2 (sql):
```sql
import { Document } from "llamaindex";
document = new Document({ text: "text", metadata: { key: "val" } });
```

---

## Document Stores

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/doc_stores/

**Contents:**
- Document Stores
- Available Document Stores
- Using PostgreSQL as Document Store
- API Reference

Document stores contain ingested document chunks, i.e. Nodes.

Check the LlamaIndexTS Github for the most up to date overview of integrations.

You can configure the schemaName, tableName, namespace, and connectionString. If a connectionString is not provided, it will use the environment variables PGHOST, PGUSER, PGPASSWORD, PGDATABASE and PGPORT.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/postgres
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/postgres
```

Example 3 (javascript):
```javascript
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";import { PostgresDocumentStore } from "@llamaindex/postgres";
const storageContext = await storageContextFromDefaults({  docStore: new PostgresDocumentStore(),});
```

Example 4 (javascript):
```javascript
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";import { PostgresDocumentStore } from "@llamaindex/postgres";
const storageContext = await storageContextFromDefaults({  docStore: new PostgresDocumentStore(),});
```

---

## Embedding

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/

**Contents:**
- Embedding
- Installation
- Local Embedding
- Local Ollama Embeddings With Remote Host
- Available Embeddings
- API Reference

The embedding model in LlamaIndex is responsible for creating numerical representations of text. By default, LlamaIndex will use the text-embedding-ada-002 model from OpenAI.

This can be explicitly updated through Settings.embedModel.

For local embeddings, you can use the HuggingFace embedding model.

Ollama provides a way to run embedding models locally or connect to a remote Ollama instance. This is particularly useful when you need to:

The ENV variable method you will find elsewhere sometimes may not work with the OllamaEmbedding class. Also note, you’ll need to change the host in the Ollama server to 0.0.0.0 to allow connections from other machines.

To use Ollama embeddings with a remote host, you need to specify the host URL in the configuration like this:

Most available embeddings are listed in the sidebar on the left. Additionally the following integrations exist without separate documentation:

Check the LlamaIndexTS Github for the most up to date overview of integrations.

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
import { OpenAIEmbedding } from "@llamaindex/openai";import { Settings } from "llamaindex";
Settings.embedModel = new OpenAIEmbedding({  model: "text-embedding-ada-002",});
```

Example 4 (sql):
```sql
import { OpenAIEmbedding } from "@llamaindex/openai";import { Settings } from "llamaindex";
Settings.embedModel = new OpenAIEmbedding({  model: "text-embedding-ada-002",});
```

---

## Faithfulness Evaluator

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/evaluation/faithfulness/

**Contents:**
- Faithfulness Evaluator
- Usage
- API Reference

Faithfulness is a measure of whether the generated answer is faithful to the retrieved contexts. In other words, it measures whether there is any hallucination in the generated answer.

This uses the FaithfulnessEvaluator module to measure if the response from a query engine matches any source nodes.

This is useful for measuring if the response was hallucinated. The evaluator returns a score between 0 and 1, where 1 means the response is faithful to the retrieved contexts.

Firstly, you need to install the package:

Set the OpenAI API key:

Import the required modules:

Let’s setup gpt-4 for better results:

Now, let’s create a vector index and query engine with documents and query engine respectively. Then, we can evaluate the response with the query and response from the query engine.:

Now, let’s evaluate the response:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 3 (unknown):
```unknown
export OPENAI_API_KEY=your-api-key
```

Example 4 (unknown):
```unknown
export OPENAI_API_KEY=your-api-key
```

---

## Fireworks LLM

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/fireworks/

**Contents:**
- Fireworks LLM
- Usage
- Load and index documents
- Query
- Full Example
- API Reference

Fireworks.ai focus on production use cases for open source LLMs, offering speed and quality.

For this example, we will load the Berkshire Hathaway 2022 annual report pdf

**Examples:**

Example 1 (sql):
```sql
import { Settings } from "llamaindex";import { FireworksLLM } from "@llamaindex/fireworks";
Settings.llm = new FireworksLLM({  apiKey: "<YOUR_API_KEY>",});
```

Example 2 (sql):
```sql
import { Settings } from "llamaindex";import { FireworksLLM } from "@llamaindex/fireworks";
Settings.llm = new FireworksLLM({  apiKey: "<YOUR_API_KEY>",});
```

Example 3 (javascript):
```javascript
const reader = new PDFReader();const documents = await reader.loadData("../data/brk-2022.pdf");
// Split text and create embeddings. Store them in a VectorStoreIndexconst index = await VectorStoreIndex.fromDocuments(documents);
```

Example 4 (javascript):
```javascript
const reader = new PDFReader();const documents = await reader.loadData("../data/brk-2022.pdf");
// Split text and create embeddings. Store them in a VectorStoreIndexconst index = await VectorStoreIndex.fromDocuments(documents);
```

---

## Gemini

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/gemini/

**Contents:**
- Gemini
- Installation
- API Reference

To use Gemini embeddings, you need to import GeminiEmbedding from @llamaindex/google.

Per default, GeminiEmbedding is using the gemini-pro model. You can change the model by passing the model parameter to the constructor. For example:

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
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { GeminiEmbedding, GEMINI_MODEL } from "@llamaindex/google";
// Update Embed ModelSettings.embedModel = new GeminiEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { GeminiEmbedding, GEMINI_MODEL } from "@llamaindex/google";
// Update Embed ModelSettings.embedModel = new GeminiEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## Groq

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/groq/

**Contents:**
- Groq
- Installation
- Usage
- Load and index documents
- Query
- Full Example
- API Reference

First, create an API key at the Groq Console. Then save it in your environment:

The initialize the Groq module.

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/groq
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/groq
```

Example 3 (unknown):
```unknown
export GROQ_API_KEY=<your-api-key>
```

Example 4 (unknown):
```unknown
export GROQ_API_KEY=<your-api-key>
```

---

## HuggingFace

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/huggingface/

**Contents:**
- HuggingFace
- Installation
- API Reference

To use HuggingFace embeddings, you need to import HuggingFaceEmbedding from @llamaindex/huggingface.

Per default, HuggingFaceEmbedding is using the Xenova/all-MiniLM-L6-v2 model. You can change the model by passing the modelType parameter to the constructor. If you’re not using a quantized model, set the quantized parameter to false.

For example, to use the not quantized BAAI/bge-small-en-v1.5 model, you can use the following code:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/huggingface
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/huggingface
```

Example 3 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { HuggingFaceEmbedding } from "@llamaindex/huggingface";
// Update Embed ModelSettings.embedModel = new HuggingFaceEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { HuggingFaceEmbedding } from "@llamaindex/huggingface";
// Update Embed ModelSettings.embedModel = new HuggingFaceEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## Index Stores

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/index_stores/

**Contents:**
- Index Stores
- Available Index Stores
- Using PostgreSQL as Index Store
- API Reference

Index stores are underlying storage components that contain metadata(i.e. information created when indexing) about the index itself.

Check the LlamaIndexTS Github for the most up to date overview of integrations.

You can configure the schemaName, tableName, namespace, and connectionString. If a connectionString is not provided, it will use the environment variables PGHOST, PGUSER, PGPASSWORD, PGDATABASE and PGPORT.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/postgres
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/postgres
```

Example 3 (javascript):
```javascript
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";import { PostgresIndexStore } from "@llamaindex/postgres";
const storageContext = await storageContextFromDefaults({  indexStore: new PostgresIndexStore(),});
```

Example 4 (javascript):
```javascript
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";import { PostgresIndexStore } from "@llamaindex/postgres";
const storageContext = await storageContextFromDefaults({  indexStore: new PostgresIndexStore(),});
```

---

## Ingestion Pipeline

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/ingestion_pipeline/

**Contents:**
- Ingestion Pipeline
- Installation
- Usage Pattern
- Connecting to Vector Databases
- API Reference

An IngestionPipeline uses a concept of Transformations that are applied to input data. These Transformations are applied to your input data, and the resulting nodes are either returned or inserted into a vector database (if given).

The simplest usage is to instantiate an IngestionPipeline like so:

When running an ingestion pipeline, you can also chose to automatically insert the resulting nodes into a remote vector store.

Then, you can construct an index from that vector store later on.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/qdrant
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/qdrant
```

Example 3 (javascript):
```javascript
import fs from "node:fs/promises";import { OpenAI, OpenAIEmbedding } from "@llamaindex/openai";import {  Document,  IngestionPipeline,  MetadataMode,  TitleExtractor,  SentenceSplitter,} from "llamaindex";
async function main() {  // Load essay from abramov.txt in Node  const path = "node_modules/llamaindex/examples/abramov.txt";
  const essay = await fs.readFile(path, "utf-8");
  // Create Document object with essay  const document = new Document({ text: essay, id_: path });  const pipeline = new IngestionPipeline({    transformations: [      new SentenceSplitter({ chunkSize: 1024, chunkOverlap: 20 }),      new TitleExtractor(),      new OpenAIEmbedding(),    ],  });
  // run the pipeline  const nodes = await pipeline.run({ documents: [document] });
  // print out the result of the pipeline run  for (const node of nodes) {    console.log(node.getContent(MetadataMode.NONE));  }}
main().catch(console.error);
```

Example 4 (javascript):
```javascript
import fs from "node:fs/promises";import { OpenAI, OpenAIEmbedding } from "@llamaindex/openai";import {  Document,  IngestionPipeline,  MetadataMode,  TitleExtractor,  SentenceSplitter,} from "llamaindex";
async function main() {  // Load essay from abramov.txt in Node  const path = "node_modules/llamaindex/examples/abramov.txt";
  const essay = await fs.readFile(path, "utf-8");
  // Create Document object with essay  const document = new Document({ text: essay, id_: path });  const pipeline = new IngestionPipeline({    transformations: [      new SentenceSplitter({ chunkSize: 1024, chunkOverlap: 20 }),      new TitleExtractor(),      new OpenAIEmbedding(),    ],  });
  // run the pipeline  const nodes = await pipeline.run({ documents: [document] });
  // print out the result of the pipeline run  for (const node of nodes) {    console.log(node.getContent(MetadataMode.NONE));  }}
main().catch(console.error);
```

---

## Jina AI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/jinaai/

**Contents:**
- Jina AI
- API Reference

To use Jina AI embeddings, you need to import JinaAIEmbedding from @llamaindex/jinaai.

**Examples:**

Example 1 (javascript):
```javascript
import { Settings } from "llamaindex";import { JinaAIEmbedding } from "@llamaindex/jinaai";
Settings.embedModel = new JinaAIEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 2 (javascript):
```javascript
import { Settings } from "llamaindex";import { JinaAIEmbedding } from "@llamaindex/jinaai";
Settings.embedModel = new JinaAIEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## Jina AI Reranker

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/node_postprocessors/jinaai_reranker/

**Contents:**
- Jina AI Reranker
- Setup
- Load and index documents
- Increase similarity topK to retrieve more results
- Create a new instance of the JinaAIReranker class
- Create a query engine with the retriever and node postprocessor
- API Reference

The Jina AI Reranker is a postprocessor that uses the Jina AI Reranker API to rerank the results of a search query.

Firstly, you will need to install the llamaindex package.

Now, you will need to sign up for an API key at Jina AI. Once you have your API key you can import the necessary modules and create a new instance of the JinaAIReranker class.

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

The default value for similarityTopK is 2. This means that only the most similar document will be returned. To retrieve more results, you can increase the value of similarityTopK.

Then you can create a new instance of the JinaAIReranker class and pass in the number of results you want to return. The Jina AI Reranker API key is set in the JINAAI_API_KEY environment variable.

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
import { OpenAI } from "@llamaindex/openai";import { Document, Settings, VectorStoreIndex, JinaAIReranker } from "llamaindex";
```

Example 4 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";import { Document, Settings, VectorStoreIndex, JinaAIReranker } from "llamaindex";
```

---

## JSONReader

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/readers/json/

**Contents:**
- JSONReader
- Installation
- Usage
  - Options
    - Examples
- API Reference

A simple JSON data loader with various options. Either parses the entire string, cleaning it and treat each line as an embedding or performs a recursive depth-first traversal yielding JSON paths. Supports streaming of large JSON data using @discoveryjs/json-ext

streamingThreshold?: The threshold for using streaming mode in MB of the JSON Data. CEstimates characters by calculating bytes: (streamingThreshold * 1024 * 1024) / 2 and comparing against .length of the JSON string. Set undefined to disable streaming or 0 to always use streaming. Default is 50 MB.

ensureAscii?: Wether to ensure only ASCII characters be present in the output by converting non-ASCII characters to their unicode escape sequence. Default is false.

isJsonLines?: Wether the JSON is in JSON Lines format. If true, will split into lines, remove empty one and parse each line as JSON. Note: Uses a custom streaming parser, most likely less robust than json-ext. Default is false

cleanJson?: Whether to clean the JSON by filtering out structural characters ({}, [], and ,). If set to false, it will just parse the JSON, not removing structural characters. Default is true.

logger?: A placeholder for a custom logger function.

Depth-First-Traversal:

levelsBack?: Specifies how many levels up the JSON structure to include in the output. cleanJson will be ignored. If set to 0, all levels are included. If undefined, parses the entire JSON, treat each line as an embedding and create a document per top-level array. Default is undefined

collapseLength?: The maximum length of JSON string representation to be collapsed into a single line. Only applicable when levelsBack is set. Default is undefined

LevelsBack = undefined & cleanJson = true

Depth-First Traversal all levels:

Depth-First Traversal and Collapse:

levelsBack = 0 & collapseLength = 35

Depth-First Traversal limited levels:

levelsBack = undefined & cleanJson = false

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/readers
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/readers
```

Example 3 (sql):
```sql
import { JSONReader } from "@llamaindex/readers/json";
const file = "../../PATH/TO/FILE";const content = new TextEncoder().encode("JSON_CONTENT");
const reader = new JSONReader({ levelsBack: 0, collapseLength: 100 });const docsFromFile = reader.loadData(file);const docsFromContent = reader.loadDataAsContent(content);
```

Example 4 (sql):
```sql
import { JSONReader } from "@llamaindex/readers/json";
const file = "../../PATH/TO/FILE";const content = new TextEncoder().encode("JSON_CONTENT");
const reader = new JSONReader({ levelsBack: 0, collapseLength: 100 });const docsFromFile = reader.loadData(file);const docsFromContent = reader.loadDataAsContent(content);
```

---

## Key-Value Stores

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/kv_stores/

**Contents:**
- Key-Value Stores
- Available Key-Value Stores
- API Reference

Key-Value Stores represent underlying storage components used in Document Stores and Index Stores

Check the LlamaIndexTS Github for the most up to date overview of integrations.

---

## Langtrace

**URL:** https://developers.llamaindex.ai/typescript/framework/integration/lang-trace/

**Contents:**
- Langtrace
- Install
- Initialize

Enhance your observability with Langtrace, a robust open-source tool supports OpenTelemetry and is designed to trace, evaluate, and manage LLM applications seamlessly. Langtrace integrates directly with LlamaIndex, offering detailed, real-time insights into performance metrics such as accuracy, evaluations, and latency.

**Examples:**

Example 1 (python):
```python
npm i @langtrase/typescript-sdk
```

Example 2 (python):
```python
npm i @langtrase/typescript-sdk
```

Example 3 (typescript):
```typescript
import * as Langtrace from "@langtrase/typescript-sdk";Langtrace.init({ api_key: "<YOUR_API_KEY>" });
```

Example 4 (typescript):
```typescript
import * as Langtrace from "@langtrase/typescript-sdk";Langtrace.init({ api_key: "<YOUR_API_KEY>" });
```

---

## Large Language Models (LLMs)

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/

**Contents:**
- Large Language Models (LLMs)
- Installation
- Azure OpenAI
- Local LLM
- Available LLMs
- API Reference

The LLM is responsible for reading text and generating natural language responses to queries. By default, LlamaIndex.TS uses gpt-4o.

The LLM can be explicitly updated through Settings.

To use Azure OpenAI, you only need to set a few environment variables.

For local LLMs, currently we recommend the use of Ollama LLM.

Most available LLMs are listed in the sidebar on the left. Additionally the following integrations exist without separate documentation:

Check the LlamaIndexTS Github for the most up to date overview of integrations.

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
Settings.llm = new OpenAI({ model: "gpt-3.5-turbo", temperature: 0 });
```

Example 4 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";import { Settings } from "llamaindex";
Settings.llm = new OpenAI({ model: "gpt-3.5-turbo", temperature: 0 });
```

---

## LLama2

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/llama2/

**Contents:**
- LLama2
- Installation
- Usage
- Usage with Replication
- Load and index documents
- Query
- Full Example
- API Reference

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/replicate
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/replicate
```

Example 3 (sql):
```sql
import { LlamaDeuce, DeuceChatStrategy } from "@llamaindex/replicate";import { Document, VectorStoreIndex, Settings } from "llamaindex";
Settings.llm = new LlamaDeuce({ chatStrategy: DeuceChatStrategy.META });
```

Example 4 (sql):
```sql
import { LlamaDeuce, DeuceChatStrategy } from "@llamaindex/replicate";import { Document, VectorStoreIndex, Settings } from "llamaindex";
Settings.llm = new LlamaDeuce({ chatStrategy: DeuceChatStrategy.META });
```

---

## Loading Data

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/readers/

**Contents:**
- Loading Data
- SimpleDirectoryReader
  - Example
- Tips when using in non-Node.js environments
- Load file natively using Node.js Customization Hooks
- API Reference

Before you can start indexing your documents, you need to load them into memory. A reader is a module that loads data from a file into a Document object.

To install readers call:

If you want to use the reader module, you need to install @llamaindex/readers

We offer readers for different file formats.

LlamaIndex.TS supports easy loading of files from folders using the SimpleDirectoryReader class.

It is a simple reader that reads all files from a directory and its subdirectories and delegates the actual reading to the reader specified in the fileExtToReader map.

Currently, the following readers are mapped to specific file types:

You can modify the reader three different ways:

SimpleDirectoryReader supports up to 9 concurrent requests. Use the numWorkers option to set the number of concurrent requests. By default it runs in sequential mode, i.e. set to 1.

When using @llamaindex/readers in a non-Node.js environment (such as Vercel Edge, Cloudflare Workers, etc.) Some classes are not exported from top-level entry file.

The reason is that some classes are only compatible with Node.js runtime, (e.g. PDFReader) which uses Node.js specific APIs (like fs, child_process, crypto).

If you need any of those classes, you have to import them instead directly through their file path in the package.

As the PDFReader is not working with the Edge runtime, here’s how to use the SimpleDirectoryReader with the LlamaParseReader to load PDFs:

Note: Reader classes have to be added explicitly to the fileExtToReader map in the Edge version of the SimpleDirectoryReader.

You’ll find a complete example with LlamaIndexTS here: https://github.com/run-llama/create_llama_projects/tree/main/nextjs-edge-llamaparse

We have a helper utility to allow you to import a file in Node.js script.

**Examples:**

Example 1 (python):
```python
npm i @llamaindex/readers
```

Example 2 (python):
```python
npm i @llamaindex/readers
```

Example 3 (sql):
```sql
import { CSVReader } from '@llamaindex/readers/csv';import { DocxReader } from '@llamaindex/readers/docx';import { HTMLReader } from '@llamaindex/readers/html';import { ImageReader } from '@llamaindex/readers/image';import { JSONReader } from '@llamaindex/readers/json';import { MarkdownReader } from '@llamaindex/readers/markdown';import { ObsidianReader } from '@llamaindex/readers/obsidian';import { PDFReader } from '@llamaindex/readers/pdf';import { TextFileReader } from '@llamaindex/readers/text';
```

Example 4 (sql):
```sql
import { CSVReader } from '@llamaindex/readers/csv';import { DocxReader } from '@llamaindex/readers/docx';import { HTMLReader } from '@llamaindex/readers/html';import { ImageReader } from '@llamaindex/readers/image';import { JSONReader } from '@llamaindex/readers/json';import { MarkdownReader } from '@llamaindex/readers/markdown';import { ObsidianReader } from '@llamaindex/readers/obsidian';import { PDFReader } from '@llamaindex/readers/pdf';import { TextFileReader } from '@llamaindex/readers/text';
```

---

## Local LLMs

**URL:** https://developers.llamaindex.ai/typescript/framework/tutorials/local_llm/

**Contents:**
- Local LLMs
- Using a local model via Ollama
  - Install Ollama
  - Pick and run a model
  - Switch the LLM in your code
  - Use local embeddings
  - Try it out

LlamaIndex.TS supports OpenAI and other remote LLM APIs. You can also run a local LLM on your machine!

The easiest way to run a local LLM is via the great work of our friends at Ollama, who provide a simple to use client that will download, install and run a growing range of models for you.

They provide a one-click installer for Mac, Linux and Windows on their home page.

Since we’re going to be doing agentic work, we’ll need a very capable model, but the largest models are hard to run on a laptop. We think mixtral 8x7b is a good balance between power and resources, but llama3 is another great option. You can run Mixtral by running

The first time you run it will also automatically download and install the model for you.

To switch the LLM in your code, you first need to make sure to install the package for the Ollama model provider:

Then, to tell LlamaIndex to use a local LLM, use the Settings object:

If you’re doing retrieval-augmented generation, LlamaIndex.TS will also call out to OpenAI to index and embed your data. To be entirely local, you can use a local embedding model from Huggingface like this:

First install the Huggingface model provider package:

And then set the embedding model in your code:

The first time this runs it will download the embedding model to run it.

With a local LLM and local embeddings in place, you can perform RAG as usual and everything will happen on your machine without calling an API:

You can see the full example file.

**Examples:**

Example 1 (json):
```json
ollama run mixtral:8x7b
```

Example 2 (json):
```json
ollama run mixtral:8x7b
```

Example 3 (python):
```python
npm i @llamaindex/ollama
```

Example 4 (python):
```python
npm i @llamaindex/ollama
```

---

## Memory

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/memory/

**Contents:**
- Memory
- Concept
- Usage
  - Configuring Memory for an Agent
  - Using Vercel format
- Customizing Memory
  - Short-Term Memory
  - Long-Term Memory
    - VectorBlock Configuration Options
- Persistence with Snapshots

Memory is a core component of agentic systems. It allows you to store and retrieve information from the past.

In LlamaIndexTS, you can create memory by using the createMemory function. This function will return a Memory object, which you can then use to store and retrieve information.

As the agent runs, it will make calls to add() to store information, and get() to retrieve information.

A Memory object has both short-term memory (i.e. a FIFO queue of messages) and optionally long-term memory (i.e. extracting information over time).

get() always returns all messages stored in the memory. The longer the agent runs, this will exceed the context window of the agent. To avoid this, the agent is using the getLLM method to get the last X messages that fit into the context window.

Here we’re creating a memory with a static block (read more about memory blocks) that contains some information about the user.

You can also put messages in Vercel format directly to the memory:

If you call get, messages are usually retrieved in the LlamaIndexTS format (type ChatMessage). If you specify the type parameter using get, you can return the messages in different formats. E.g.: using type: "vercel", you can return the messages in Vercel format:

The Memory object will store all the messages that are added to the Memory object. Unless you call clear(), no messages are removed from the memory. This is the short-term memory (usually you will store the memory of one user session there) which is augmented by the long-term memory.

Calling getLLM will retrieve messages from long-term memory and ensure that the given tokenLimit is not reached. These are the messages that you will sent to the LLM.

For initialization, you call createMemory with the following options:

Long-term memory is represented as Memory Block objects. These objects contain information that are from previous user sessions or from the beginning of the current conversation. When memory is retrieved (by calling getLLM), the short-term and long-term memories are merged together within the given tokenLimit.

Currently, there are three predefined memory blocks:

This sounds a bit complicated, but it’s actually quite simple. Let’s look at an example:

Here, we’ve setup three memory blocks:

You’ll also notice that we’ve set the priority for the factExtractionBlock block. This is used to determine the handling when the memory blocks content (i.e. long-term memory) + short-term memory exceeds the token limit on the Memory object.

Now, let’s pass these blocks into the createMemory function:

When memory is retrieved (using getLLM), the short-term and long-term memories are merged together. The Memory object will ensure that the short-term memory + long-term memory content is less than or equal to the tokenLimit. If it is longer, messages are retrieved in the following order:

The amount of short-term memory included is specified by the shortTermTokenLimitRatio. If it’s set to 0.7, 70% of the tokenLimit is used for short-term memory (not including the static memory block).

The vectorBlock offers several configuration options to customize its behavior:

Key Configuration Options:

The vectorBlock automatically adds a session filter using the block’s ID to ensure that memories from different sessions don’t interfere with each other. This filter uses the sessionFilterKey (default: “session_id”) and can be customized if needed.

Save and restore memory state:

Want to learn more about the Memory class? Check out our example codes in Github.

**Examples:**

Example 1 (javascript):
```javascript
import { openai } from "@llamaindex/openai";import { agent } from "@llamaindex/workflow";import { createMemory, staticBlock } from "llamaindex";
const llm = openai({ model: "gpt-4.1-mini" });
// Create memory with predefined contextconst memory = createMemory({  memoryBlocks: [    staticBlock({      content:        "The user is a software engineer who loves TypeScript and LlamaIndex.",    }),  ],});
// Create an agent with the memoryconst workflow = agent({  name: "assistant",  llm,  memory,});
const result = await workflow.run("What is my name?");console.log("Response:", result.data.result);
```

Example 2 (javascript):
```javascript
import { openai } from "@llamaindex/openai";import { agent } from "@llamaindex/workflow";import { createMemory, staticBlock } from "llamaindex";
const llm = openai({ model: "gpt-4.1-mini" });
// Create memory with predefined contextconst memory = createMemory({  memoryBlocks: [    staticBlock({      content:        "The user is a software engineer who loves TypeScript and LlamaIndex.",    }),  ],});
// Create an agent with the memoryconst workflow = agent({  name: "assistant",  llm,  memory,});
const result = await workflow.run("What is my name?");console.log("Response:", result.data.result);
```

Example 3 (css):
```css
await memory.add({  id: "1",  createdAt: new Date(),  role: "user",  content: "Hello!",  options: {    parts: [      {        type: "file",        data: "base64...",        mimeType: "image/png",      },    ],  },});
```

Example 4 (css):
```css
await memory.add({  id: "1",  createdAt: new Date(),  role: "user",  content: "Hello!",  options: {    parts: [      {        type: "file",        data: "base64...",        mimeType: "image/png",      },    ],  },});
```

---

## Metadata Filtering

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/query_engines/metadata_filtering/

**Contents:**
- Metadata Filtering
- Setup
- Creating documents with metadata
- Creating a ChromaDB vector store
- Querying the index with metadata filtering
- Full Code
- API Reference

Metadata filtering is a way to filter the documents that are returned by a query based on the metadata associated with the documents. This is useful when you want to filter the documents based on some metadata that is not part of the document text.

You can also check our multi-tenancy blog post to see how metadata filtering can be used in a multi-tenant environment. [https://blog.llamaindex.ai/building-multi-tenancy-rag-system-with-llamaindex-0d6ab4e0c44b] (the article uses the Python version of LlamaIndex, but the concepts are the same).

Firstly if you haven’t already, you need to install the llamaindex package:

Then you can import the necessary modules from llamaindex:

You can create documents with metadata using the Document class:

You can create a ChromaVectorStore to store the documents:

Now you can query the index with metadata filtering using the preFilters option:

Besides using the equal operator (==), you can also use a whole set of different operators to filter your documents.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/chroma
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/chroma
```

Example 3 (sql):
```sql
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";import { ChromaVectorStore } from "@llamaindex/chroma";
const collectionName = "dog_colors";
```

Example 4 (sql):
```sql
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";import { ChromaVectorStore } from "@llamaindex/chroma";
const collectionName = "dog_colors";
```

---

## Migrating from v0.8 to v0.9

**URL:** https://developers.llamaindex.ai/typescript/framework/migration/08-to-09/

**Contents:**
- Migrating from v0.8 to v0.9
- Major Changes
  - Installing Provider Packages
  - Updating Imports
  - 1. AI Model Providers
  - 2. Storage Providers
  - 3. Data Loaders
  - 4. Prefer using llamaindex instead of @llamaindex/core
- Benefits of the Changes
- Need Help?

Version 0.9 of LlamaIndex.TS introduces significant architectural changes to improve package size and runtime compatibility. The main goals of this release are:

In v0.9, you need to explicitly install the provider packages you want to use. The main llamaindex package no longer includes these dependencies by default.

You’ll need to update your imports to get classes directly from their respective provider packages. Here’s how to migrate different components:

Note: This examples requires installing the @llamaindex/openai package:

For more details on available AI model providers and their configuration, see the LLMs documentation and the Embedding Models documentation.

For more information about available storage options, refer to the Data Stores documentation.

For more details about available data loaders and their usage, check the Loading Data.

llamaindex is now re-exporting most of @llamaindex/core. To simplify imports, just use import { ... } from "llamaindex" instead of import { ... } from "@llamaindex/core". This is possible because llamaindex is now a smaller package.

We might change imports internally in @llamaindex/core in the future. Let us know if you’re missing something.

If you encounter any issues during migration, please:

**Examples:**

Example 1 (sql):
```sql
import { OpenAI } from "llamaindex";
```

Example 2 (sql):
```sql
import { OpenAI } from "llamaindex";
```

Example 3 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";
```

Example 4 (sql):
```sql
import { OpenAI } from "@llamaindex/openai";
```

---

## MistralAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/mistral/

**Contents:**
- MistralAI
- Installation
- API Reference

To use MistralAI embeddings, you need to import MistralAIEmbedding from @llamaindex/mistral.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/mistral
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/mistral
```

Example 3 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { MistralAIEmbedding } from "@llamaindex/mistral";
// Update Embed ModelSettings.embedModel = new MistralAIEmbedding({  apiKey: "<YOUR_API_KEY>",});
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (sql):
```sql
import { Document, Settings, VectorStoreIndex } from "llamaindex";import { MistralAIEmbedding } from "@llamaindex/mistral";
// Update Embed ModelSettings.embedModel = new MistralAIEmbedding({  apiKey: "<YOUR_API_KEY>",});
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## Mistral

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/mistral/

**Contents:**
- Mistral
- Installation
- Usage
- Load and index documents
- Query
- Full Example
- API Reference

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/mistral
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/mistral
```

Example 3 (sql):
```sql
import { MistralAI } from "@llamaindex/mistral";import { Settings } from "llamaindex";
Settings.llm = new MistralAI({  model: "mistral-tiny",  apiKey: "<YOUR_API_KEY>",});
```

Example 4 (sql):
```sql
import { MistralAI } from "@llamaindex/mistral";import { Settings } from "llamaindex";
Settings.llm = new MistralAI({  model: "mistral-tiny",  apiKey: "<YOUR_API_KEY>",});
```

---

## MixedbreadAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/node_postprocessors/mixedbreadiai_reranker/

**Contents:**
- MixedbreadAI
- Table of Contents
- Setup
- Usage with LlamaIndex
  - Step 1: Load and Index Documents
  - Step 2: Increase Similarity TopK
  - Step 3: Create a MixedbreadAIReranker Instance
  - Step 4: Create a Query Engine
- Simple Reranking Guide
  - Step 1: Create an Instance of MixedbreadAIReranker

Welcome to the mixedbread ai reranker guide! This guide will help you use mixedbread ai’s API to rerank search query results, ensuring you get the most relevant information, just like picking the freshest bread from the bakery.

To find out more about the latest features and updates, visit the mixedbread.ai.

First, you will need to install the llamaindex package.

Next, sign up for an API key at mixedbread.ai. Once you have your API key, you can import the necessary modules and create a new instance of the MixedbreadAIReranker class.

This section will guide you through integrating mixedbread’s reranker with LlamaIndex.

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index, like a variety of breads in a bakery.

The default value for similarityTopK is 2, which means only the most similar document will be returned. To get more results, like picking a variety of fresh breads, you can increase the value of similarityTopK.

Create a new instance of the MixedbreadAIReranker class.

Combine the retriever and node postprocessor to create a query engine. This setup ensures that your queries are processed and reranked to provide the best results, like arranging the bread in the order of freshness and quality.

With mixedbread’s Reranker, you’re all set to serve up the most relevant and well-ordered results, just like a skilled baker arranging their best breads for eager customers. Enjoy the perfect blend of technology and culinary delight!

This section will guide you through a simple reranking process using mixedbread ai.

Create a new instance of the MixedbreadAIReranker class, passing in your API key and the number of results you want to return. It’s like setting up your bakery to offer a specific number of freshly baked items.

Define the nodes (documents) you want to rerank and the query.

Use the postprocessNodes method to rerank the nodes based on the query.

This section will guide you through reranking when working with objects.

Create a new instance of the MixedbreadAIReranker class, just like before.

Define the documents (objects) you want to rerank and the query.

Use the rerank method to reorder the documents based on the query.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/mixedbread
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai @llamaindex/mixedbread
```

Example 3 (sql):
```sql
import {  Document,  VectorStoreIndex,  Settings,} from "llamaindex";import { OpenAI } from "@llamaindex/openai";import { MixedbreadAIReranker } from "@llamaindex/mixedbread";
```

Example 4 (sql):
```sql
import {  Document,  VectorStoreIndex,  Settings,} from "llamaindex";import { OpenAI } from "@llamaindex/openai";import { MixedbreadAIReranker } from "@llamaindex/mixedbread";
```

---

## MixedbreadAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/mixedbreadai/

**Contents:**
- MixedbreadAI
- Table of Contents
- Setup
- Usage with LlamaIndex
  - Step 1: Load and Index Documents
  - Step 2: Create a Query Engine
- Embeddings with Custom Parameters
  - Step 1: Create an Instance of MixedbreadAIEmbeddings
  - Step 2: Define Texts
  - Step 3: Generate Embeddings

Welcome to the mixedbread embeddings guide! This guide will help you use the mixedbread ai’s API to generate embeddings for your text documents, ensuring you get the most relevant information, just like picking the freshest bread from the bakery.

To find out more about the latest features, updates, and available models, visit mixedbread.ai.

Next, sign up for an API key at mixedbread.ai. Once you have your API key, you can import the necessary modules and create a new instance of the MixedbreadAIEmbeddings class.

This section will guide you through integrating mixedbread embeddings with LlamaIndex for more advanced usage.

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index, like a variety of breads in a bakery.

Combine the retriever and the embed model to create a query engine. This setup ensures that your queries are processed to provide the best results, like arranging the bread in the order of freshness and quality.

Models can require prompts to generate embeddings for queries, in the ‘mixedbread-ai/mxbai-embed-large-v1’ model’s case, the prompt is Represent this sentence for searching relevant passages:.

This section will guide you through generating embeddings with custom parameters and usage with f.e. matryoshka and binary embeddings.

Create a new instance of the MixedbreadAIEmbeddings class with custom parameters. For example, to use the mixedbread-ai/mxbai-embed-large-v1 model with a batch size of 64, normalized embeddings, and binary encoding format:

Define the texts you want to generate embeddings for.

Use the embedDocuments method to generate embeddings for the texts.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/mixedbread
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/mixedbread
```

Example 3 (sql):
```sql
import { MixedbreadAIEmbeddings } from "@llamaindex/mixedbread";import { Document, Settings } from "llamaindex";
```

Example 4 (sql):
```sql
import { MixedbreadAIEmbeddings } from "@llamaindex/mixedbread";import { Document, Settings } from "llamaindex";
```

---

## More

**URL:** https://developers.llamaindex.ai/typescript/framework/more/

**Contents:**
- More
- 🗺️ Ecosystem
- Community

To download or contribute, find LlamaIndex on:

Need help? Have a feature suggestion? Join the LlamaIndex community:

---

## Node Postprocessors

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/node_postprocessors/

**Contents:**
- Node Postprocessors
- Installation
- Concept
- Usage Pattern
- Using Node Postprocessors in LlamaIndex
  - Using Node Postprocessors in a Query Engine
  - Using with retrieved nodes
- API Reference

Node postprocessors are a set of modules that take a set of nodes, and apply some kind of transformation or filtering before returning them.

In LlamaIndex, node postprocessors are most commonly applied within a query engine, after the node retrieval step and before the response synthesis step.

LlamaIndex offers several node postprocessors for immediate use, while also providing a simple API for adding your own custom postprocessors.

An example of using a node postprocessors is below:

Now you can use the filteredNodes and rerankedNodes in your application.

Most commonly, node-postprocessors will be used in a query engine, where they are applied to the nodes returned from a retriever, and before the response synthesis step.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/cohere @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/cohere @llamaindex/openai
```

Example 3 (typescript):
```typescript
import { CohereRerank } from "@llamaindex/cohere";import { Node, NodeWithScore, SimilarityPostprocessor, TextNode } from "llamaindex";
const nodes: NodeWithScore[] = [  {    node: new TextNode({ text: "hello world" }),    score: 0.8,  },  {    node: new TextNode({ text: "LlamaIndex is the best" }),    score: 0.6,  },];
// similarity postprocessor: filter nodes below 0.75 similarity scoreconst processor = new SimilarityPostprocessor({  similarityCutoff: 0.7,});
const filteredNodes = await processor.postprocessNodes(nodes);
// cohere rerank: rerank nodes given query using trained modelconst reranker = new CohereRerank({  apiKey: "<COHERE_API_KEY>",  topN: 2,});
const rerankedNodes = await reranker.postprocessNodes(nodes, "<user_query>");
console.log(filteredNodes, rerankedNodes);
```

Example 4 (typescript):
```typescript
import { CohereRerank } from "@llamaindex/cohere";import { Node, NodeWithScore, SimilarityPostprocessor, TextNode } from "llamaindex";
const nodes: NodeWithScore[] = [  {    node: new TextNode({ text: "hello world" }),    score: 0.8,  },  {    node: new TextNode({ text: "LlamaIndex is the best" }),    score: 0.6,  },];
// similarity postprocessor: filter nodes below 0.75 similarity scoreconst processor = new SimilarityPostprocessor({  similarityCutoff: 0.7,});
const filteredNodes = await processor.postprocessNodes(nodes);
// cohere rerank: rerank nodes given query using trained modelconst reranker = new CohereRerank({  apiKey: "<COHERE_API_KEY>",  topN: 2,});
const rerankedNodes = await reranker.postprocessNodes(nodes, "<user_query>");
console.log(filteredNodes, rerankedNodes);
```

---

## Ollama

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/ollama/

**Contents:**
- Ollama
- Installation
- API Reference

To use Ollama embeddings, you need to import OllamaEmbedding from @llamaindex/ollama.

Note that you need to pull the embedding model first before using it.

In the example below, we’re using the nomic-embed-text model, so you have to call:

**Examples:**

Example 1 (unknown):
```unknown
ollama pull nomic-embed-text
```

Example 2 (unknown):
```unknown
ollama pull nomic-embed-text
```

Example 3 (python):
```python
npm i llamaindex @llamaindex/ollama
```

Example 4 (python):
```python
npm i llamaindex @llamaindex/ollama
```

---

## Ollama

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/ollama/

**Contents:**
- Ollama
- Installation
- Usage
- Load and index documents
- Query
- Using JSON Response Format
- Full Example
- API Reference

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

You can configure Ollama to return responses in JSON format:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/ollama
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/ollama
```

Example 3 (sql):
```sql
import { Ollama } from "@llamaindex/ollama";import { Settings } from "llamaindex";
Settings.llm = ollamaLLM;Settings.embedModel = ollamaLLM;
```

Example 4 (sql):
```sql
import { Ollama } from "@llamaindex/ollama";import { Settings } from "llamaindex";
Settings.llm = ollamaLLM;Settings.embedModel = ollamaLLM;
```

---

## OpenAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/openai/

**Contents:**
- OpenAI
- Installation
- API Reference

To use OpenAI embeddings, you need to import OpenAIEmbedding from @llamaindex/openai.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 3 (javascript):
```javascript
import { OpenAIEmbedding } from "@llamaindex/openai";import { Document, Settings, VectorStoreIndex } from "llamaindex";
Settings.embedModel = new OpenAIEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (javascript):
```javascript
import { OpenAIEmbedding } from "@llamaindex/openai";import { Document, Settings, VectorStoreIndex } from "llamaindex";
Settings.embedModel = new OpenAIEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## OpenLLMetry

**URL:** https://developers.llamaindex.ai/typescript/framework/integration/open-llm-metry/

**Contents:**
- OpenLLMetry
  - Usage Pattern

OpenLLMetry is an open-source project based on OpenTelemetry for tracing and monitoring LLM applications. It connects to all major observability platforms and installs in minutes.

**Examples:**

Example 1 (python):
```python
npm i @traceloop/node-server-sdk
```

Example 2 (python):
```python
npm i @traceloop/node-server-sdk
```

Example 3 (sql):
```sql
import * as traceloop from "@traceloop/node-server-sdk";
traceloop.initialize({  apiKey: process.env.TRACELOOP_API_KEY,  disableBatch: true});
```

Example 4 (sql):
```sql
import * as traceloop from "@traceloop/node-server-sdk";
traceloop.initialize({  apiKey: process.env.TRACELOOP_API_KEY,  disableBatch: true});
```

---

## Perplexity LLM

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/perplexity/

**Contents:**
- Perplexity LLM
- Installation
- Usage
- Example
- Full Example
- Available Models
- Limitations
- API Reference

The following models are available:

Currently does not support function calling.

**Examples:**

Example 1 (python):
```python
npm i @llamaindex/perplexity
```

Example 2 (python):
```python
npm i @llamaindex/perplexity
```

Example 3 (sql):
```sql
import { Settings } from "llamaindex";import { perplexity } from "@llamaindex/perplexity";Settings.llm = perplexity({apiKey: "<YOUR_API_KEY>",model: "sonar", // or available models});
```

Example 4 (sql):
```sql
import { Settings } from "llamaindex";import { perplexity } from "@llamaindex/perplexity";Settings.llm = perplexity({apiKey: "<YOUR_API_KEY>",model: "sonar", // or available models});
```

---

## Portkey LLM

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/portkey/

**Contents:**
- Portkey LLM
- Installation
- Usage
- Load and index documents
- Query
- Full Example
- API Reference

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/portkey-ai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/portkey-ai
```

Example 3 (sql):
```sql
import { Portkey } from "@llamaindex/portkey-ai";import { Settings } from "llamaindex";
Settings.llm = new Portkey({  apiKey: "<YOUR_API_KEY>",});
```

Example 4 (sql):
```sql
import { Portkey } from "@llamaindex/portkey-ai";import { Settings } from "llamaindex";
Settings.llm = new Portkey({  apiKey: "<YOUR_API_KEY>",});
```

---

## Prompts

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/prompt/

**Contents:**
- Prompts
- Usage Pattern
  - 1. Customizing the default prompt on initialization
  - 2. Customizing submodules prompt
- API Reference

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion, perform traversal during querying, and to synthesize the final answer.

Users may also provide their own prompt templates to further customize the behavior of the framework. The best method for customizing is copying the default prompt from the link above, and using that as the base for any modifications.

Currently, there are two ways to customize prompts in LlamaIndex:

For both methods, you will need to create an function that overrides the default prompt.

The first method is to create a new instance of a Response Synthesizer (or the module you would like to update the prompt) by using the getResponseSynthesizer function. Instead of passing the custom prompt to the deprecated responseBuilder parameter, call getResponseSynthesizer with the mode as the first argument and supply the new prompt via the options parameter.

The second method is that most of the modules in LlamaIndex have a getPrompts and a updatePrompt method that allows you to override the default prompt. This method is useful when you want to change the prompt on the fly or in submodules on a more granular level.

**Examples:**

Example 1 (typescript):
```typescript
// Define a custom promptconst newTextQaPrompt: TextQaPrompt = ({ context, query }) => {  return `Context information is below.---------------------${context}---------------------Given the context information and not prior knowledge, answer the query.Answer the query in the style of a Sherlock Holmes detective novel.Query: ${query}Answer:`;};
```

Example 2 (typescript):
```typescript
// Define a custom promptconst newTextQaPrompt: TextQaPrompt = ({ context, query }) => {  return `Context information is below.---------------------${context}---------------------Given the context information and not prior knowledge, answer the query.Answer the query in the style of a Sherlock Holmes detective novel.Query: ${query}Answer:`;};
```

Example 3 (javascript):
```javascript
// Create an instance of Response Synthesizer
// Deprecated usage:const responseSynthesizer = new ResponseSynthesizer({  responseBuilder: new CompactAndRefine(undefined, newTextQaPrompt),});
// Current usage:const responseSynthesizer = getResponseSynthesizer('compact', {  textQATemplate: newTextQaPrompt})
// Create indexconst index = await VectorStoreIndex.fromDocuments([document]);
// Query the indexconst queryEngine = index.asQueryEngine({ responseSynthesizer });
const response = await queryEngine.query({  query: "What did the author do in college?",});
```

Example 4 (javascript):
```javascript
// Create an instance of Response Synthesizer
// Deprecated usage:const responseSynthesizer = new ResponseSynthesizer({  responseBuilder: new CompactAndRefine(undefined, newTextQaPrompt),});
// Current usage:const responseSynthesizer = getResponseSynthesizer('compact', {  textQATemplate: newTextQaPrompt})
// Create indexconst index = await VectorStoreIndex.fromDocuments([document]);
// Query the indexconst queryEngine = index.asQueryEngine({ responseSynthesizer });
const response = await queryEngine.query({  query: "What did the author do in college?",});
```

---

## Qdrant Vector Store

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/vector_stores/qdrant/

**Contents:**
- Qdrant Vector Store
- Installation
- Importing the modules
- Load the documents
- Setup Qdrant
- Setup the index
- Query the index
- Full code
- API Reference

To run this example, you need to have a Qdrant instance running. You can run it with Docker:

**Examples:**

Example 1 (json):
```json
docker pull qdrant/qdrantdocker run -p 6333:6333 qdrant/qdrant
```

Example 2 (json):
```json
docker pull qdrant/qdrantdocker run -p 6333:6333 qdrant/qdrant
```

Example 3 (python):
```python
npm i llamaindex @llamaindex/qdrant
```

Example 4 (python):
```python
npm i llamaindex @llamaindex/qdrant
```

---

## QueryEngine

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/query_engines/

**Contents:**
- QueryEngine
- Sub Question Query Engine
  - Getting Started
  - Tools
- API Reference

A query engine wraps a Retriever and a ResponseSynthesizer into a pipeline, that will use the query string to fetch nodes and then send them to the LLM to generate a response.

The query function also supports streaming, just add stream: true as an option:

The basic concept of the Sub Question Query Engine is that it splits a single query into multiple queries, gets an answer for each of those queries, and then combines those different answers into a single coherent response for the user. You can think of it as the “think this through step by step” prompt technique but iterating over your data sources!

The easiest way to start trying the Sub Question Query Engine is running the subquestion.ts file in examples.

SubQuestionQueryEngine is implemented with Tools. The basic idea of Tools is that they are executable options for the large language model. In this case, our SubQuestionQueryEngine relies on QueryEngineTool, which as you guessed it is a tool to run queries on a QueryEngine. This allows us to give the model an option to query different documents for different questions for example. You could also imagine that the SubQuestionQueryEngine could use a Tool that searches for something on the web or gets an answer using Wolfram Alpha.

You can learn more about Tools by taking a look at the LlamaIndex Python documentation https://gpt-index.readthedocs.io/en/latest/core_modules/agent_modules/tools/root.html

**Examples:**

Example 1 (javascript):
```javascript
const queryEngine = index.asQueryEngine();const response = await queryEngine.query({ query: "query string" });
```

Example 2 (javascript):
```javascript
const queryEngine = index.asQueryEngine();const response = await queryEngine.query({ query: "query string" });
```

Example 3 (javascript):
```javascript
const stream = await queryEngine.query({ query: "query string", stream: true });for await (const chunk of stream) {  process.stdout.write(chunk.response);}
```

Example 4 (javascript):
```javascript
const stream = await queryEngine.query({ query: "query string", stream: true });for await (const chunk of stream) {  process.stdout.write(chunk.response);}
```

---

## Relevancy Evaluator

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/evaluation/relevancy/

**Contents:**
- Relevancy Evaluator
- Usage
- API Reference

Relevancy measure if the response from a query engine matches any source nodes.

It is useful for measuring if the response was relevant to the query. The evaluator returns a score between 0 and 1, where 1 means the response is relevant to the query.

Firstly, you need to install the package:

Set the OpenAI API key:

Import the required modules:

Let’s setup gpt-4 for better results:

Now, let’s create a vector index and query engine with documents and query engine respectively. Then, we can evaluate the response with the query and response from the query engine.:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/openai
```

Example 3 (unknown):
```unknown
export OPENAI_API_KEY=your-api-key
```

Example 4 (unknown):
```unknown
export OPENAI_API_KEY=your-api-key
```

---

## Response Synthesizer

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/response_synthesizer/

**Contents:**
- Response Synthesizer
- API Reference

The ResponseSynthesizer is responsible for sending the query, nodes, and prompt templates to the LLM to generate a response. There are a few key modes for generating a response:

The synthesize function also supports streaming, just add stream: true as an option:

**Examples:**

Example 1 (javascript):
```javascript
import { NodeWithScore, TextNode, getResponseSynthesizer, responseModeSchema } from "llamaindex";
// you can also use responseModeSchema.Enum.refine, responseModeSchema.Enum.tree_summarize, responseModeSchema.Enum.multi_modal// or you can use the CompactAndRefine, Refine, TreeSummarize, or MultiModal classes directlyconst responseSynthesizer = getResponseSynthesizer(responseModeSchema.Enum.compact);
const nodesWithScore: NodeWithScore[] = [  {    node: new TextNode({ text: "I am 10 years old." }),    score: 1,  },  {    node: new TextNode({ text: "John is 20 years old." }),    score: 0.5,  },];
const response = await responseSynthesizer.synthesize({  query: "What age am I?",  nodesWithScore,});console.log(response.response);
```

Example 2 (javascript):
```javascript
import { NodeWithScore, TextNode, getResponseSynthesizer, responseModeSchema } from "llamaindex";
// you can also use responseModeSchema.Enum.refine, responseModeSchema.Enum.tree_summarize, responseModeSchema.Enum.multi_modal// or you can use the CompactAndRefine, Refine, TreeSummarize, or MultiModal classes directlyconst responseSynthesizer = getResponseSynthesizer(responseModeSchema.Enum.compact);
const nodesWithScore: NodeWithScore[] = [  {    node: new TextNode({ text: "I am 10 years old." }),    score: 1,  },  {    node: new TextNode({ text: "John is 20 years old." }),    score: 0.5,  },];
const response = await responseSynthesizer.synthesize({  query: "What age am I?",  nodesWithScore,});console.log(response.response);
```

Example 3 (javascript):
```javascript
const stream = await responseSynthesizer.synthesize({  query: "What age am I?",  nodesWithScore,  stream: true,});for await (const chunk of stream) {  process.stdout.write(chunk.response);}
```

Example 4 (javascript):
```javascript
const stream = await responseSynthesizer.synthesize({  query: "What age am I?",  nodesWithScore,  stream: true,});for await (const chunk of stream) {  process.stdout.write(chunk.response);}
```

---

## Retriever

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/rag/retriever/

**Contents:**
- Retriever

A retriever in LlamaIndex is what is used to fetch Nodes from an index using a query string.

**Examples:**

Example 1 (javascript):
```javascript
const retriever = vectorIndex.asRetriever({  similarityTopK: 3,});
// Fetch nodes!const nodesWithScore = await retriever.retrieve({ query: "query string" });
```

Example 2 (javascript):
```javascript
const retriever = vectorIndex.asRetriever({  similarityTopK: 3,});
// Fetch nodes!const nodesWithScore = await retriever.retrieve({ query: "query string" });
```

---

## Storage

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/

**Contents:**
- Storage
- Local Storage
- API Reference

Storage in LlamaIndex.TS works automatically once you’ve configured a StorageContext object.

Per default a local directory is used for storage. Depening on the storage type (i.e. doc stores, index stores or vector stores), you can configure a different persistence layer. Most commonly a vector database is used as vector store.

You can configure the persistDir to define where to store the data locally.

**Examples:**

Example 1 (javascript):
```javascript
import {  Document,  VectorStoreIndex,  storageContextFromDefaults,} from "llamaindex";
const storageContext = await storageContextFromDefaults({  persistDir: "./storage",});
const document = new Document({ text: "Test Text" });const index = await VectorStoreIndex.fromDocuments([document], {  storageContext,});
```

Example 2 (javascript):
```javascript
import {  Document,  VectorStoreIndex,  storageContextFromDefaults,} from "llamaindex";
const storageContext = await storageContextFromDefaults({  persistDir: "./storage",});
const document = new Document({ text: "Test Text" });const index = await VectorStoreIndex.fromDocuments([document], {  storageContext,});
```

---

## Supabase Vector Store

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/vector_stores/supabase/

**Contents:**
- Supabase Vector Store
- Installation
- Database Setup
- Importing the modules
- Setup Supabase
- Setup the index
- Query the index
- Query with filters
- Full code
- API Reference

To use this vector store, you need a Supabase project. You can create one at supabase.com.

Before using the vector store, you need to:

— Create a function for similarity search with filtering support

You can filter documents based on metadata when querying:

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/supabase
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/supabase
```

Example 3 (sql):
```sql
create table documents (id uuid primary key,content text,metadata jsonb,embedding vector(1536));
```

Example 4 (sql):
```sql
create table documents (id uuid primary key,content text,metadata jsonb,embedding vector(1536));
```

---

## Together

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/together/

**Contents:**
- Together
- API Reference

To use together embeddings, you need to import TogetherEmbedding from @llamaindex/together.

**Examples:**

Example 1 (javascript):
```javascript
import { Settings } from "llamaindex";import { TogetherEmbedding } from "@llamaindex/together";
Settings.embedModel = new TogetherEmbedding({  apiKey: "<YOUR_API_KEY>",});
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 2 (javascript):
```javascript
import { Settings } from "llamaindex";import { TogetherEmbedding } from "@llamaindex/together";
Settings.embedModel = new TogetherEmbedding({  apiKey: "<YOUR_API_KEY>",});
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---

## Together LLM

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/llms/together/

**Contents:**
- Together LLM
- Installation
- Usage
- Load and index documents
- Query
- Full Example
- API Reference

For this example, we will use a single document. In a real-world scenario, you would have multiple documents to index.

**Examples:**

Example 1 (python):
```python
npm i @llamaindex/together
```

Example 2 (python):
```python
npm i @llamaindex/together
```

Example 3 (sql):
```sql
import { Settings } from "llamaindex";import { TogetherLLM } from "@llamaindex/together";
Settings.llm = new TogetherLLM({  apiKey: "<YOUR_API_KEY>",});
```

Example 4 (sql):
```sql
import { Settings } from "llamaindex";import { TogetherLLM } from "@llamaindex/together";
Settings.llm = new TogetherLLM({  apiKey: "<YOUR_API_KEY>",});
```

---

## Transformations

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/ingestion_pipeline/transformations/

**Contents:**
- Transformations
- Usage Pattern
- Custom Transformations
- API Reference

A transformation is something that takes a list of nodes as an input, and returns a list of nodes. Each component that implements the Transformation class has both a transform definition responsible for transforming the nodes.

Currently, the following components are Transformation objects:

While transformations are best used with with an IngestionPipeline, they can also be used directly.

You can implement any transformation yourself by implementing the TransformComponent.

The following custom transformation will remove any special characters or punctuation in text.

These can then be used directly or in any IngestionPipeline.

**Examples:**

Example 1 (javascript):
```javascript
import { SentenceSplitter, TitleExtractor, Document } from "llamaindex";
async function main() {  let nodes = new SentenceSplitter().getNodesFromDocuments([    new Document({ text: "I am 10 years old. John is 20 years old." }),  ]);
  const titleExtractor = new TitleExtractor();
  nodes = await titleExtractor.transform(nodes);
  for (const node of nodes) {    console.log(node.getContent(MetadataMode.NONE));  }}
main().catch(console.error);
```

Example 2 (javascript):
```javascript
import { SentenceSplitter, TitleExtractor, Document } from "llamaindex";
async function main() {  let nodes = new SentenceSplitter().getNodesFromDocuments([    new Document({ text: "I am 10 years old. John is 20 years old." }),  ]);
  const titleExtractor = new TitleExtractor();
  nodes = await titleExtractor.transform(nodes);
  for (const node of nodes) {    console.log(node.getContent(MetadataMode.NONE));  }}
main().catch(console.error);
```

Example 3 (csharp):
```csharp
import { TransformComponent, TextNode } from "llamaindex";
export class RemoveSpecialCharacters extends TransformComponent {  async transform(nodes: TextNode[]): Promise<TextNode[]> {    for (const node of nodes) {      node.text = node.text.replace(/[^\w\s]/gi, "");    }
    return nodes;  }}
```

Example 4 (csharp):
```csharp
import { TransformComponent, TextNode } from "llamaindex";
export class RemoveSpecialCharacters extends TransformComponent {  async transform(nodes: TextNode[]): Promise<TextNode[]> {    for (const node of nodes) {      node.text = node.text.replace(/[^\w\s]/gi, "");    }
    return nodes;  }}
```

---

## Using @llamaindex/chat-ui

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/ui/

**Contents:**
- Using @llamaindex/chat-ui

@llamaindex/chat-ui is a library that provides a set of components for building chat user interfaces. It is built on top of Shadcn UI.

Check out our chat-ui documentation or try running examples on the ui.llamaindex.ai website.

---

## Using LlamaIndex Server

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/ui/llamaindex-server/

**Contents:**
- Using LlamaIndex Server
- LlamaIndex Server
- Features
- Quick Start

LlamaIndexServer is a Next.js-based application that allows you to quickly launch your LlamaIndex Workflows as an API server with an optional chat UI. It provides a complete environment for running LlamaIndex workflows with both API endpoints and a user interface for interaction.

Check the latest information on the NPM package page: https://www.npmjs.com/package/@llamaindex/server

---

## Vector Stores

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/data/stores/vector_stores/

**Contents:**
- Vector Stores
- Available Vector Stores

Vector stores save embedding vectors of your ingested document chunks.

Available Vector Stores are shown on the sidebar to the left. Additionally the following integrations exist without separate documentation:

Check the LlamaIndexTS Github for the most up to date overview of integrations.

---

## Vercel

**URL:** https://developers.llamaindex.ai/typescript/framework/integration/vercel/

**Contents:**
- Vercel
- Setup
- Using Vercel AI’s Model Providers
- Use Indexes
  - Using VectorStoreIndex
  - Using LlamaCloud
- Next Steps

LlamaIndex provides integration with Vercel’s AI SDK, allowing you to create powerful search and retrieval applications. You can:

First, install the required dependencies:

Using the VercelLLM adapter, it’s easy to use any of Vercel AI’s model providers as LLMs in LlamaIndex. Here’s an example of how to use OpenAI’s GPT-4o model:

Here’s how to create a simple vector store index and query it using Vercel’s AI SDK:

Note: the Vercel AI model referenced in the llamaindex function is used by the response synthesizer to generate a response for the tool call.

For production deployments, you can use LlamaCloud to store and manage your documents:

**Examples:**

Example 1 (python):
```python
npm i @llamaindex/vercel ai
```

Example 2 (python):
```python
npm i @llamaindex/vercel ai
```

Example 3 (javascript):
```javascript
const llm = new VercelLLM({ model: openai("gpt-4o") });const result = await llm.complete({  prompt: "What is the capital of France?",  stream: false, // Set to true if you want streaming responses});console.log(result.text);
```

Example 4 (javascript):
```javascript
const llm = new VercelLLM({ model: openai("gpt-4o") });const result = await llm.complete({  prompt: "What is the capital of France?",  stream: false, // Set to true if you want streaming responses});console.log(result.text);
```

---

## VoyageAI

**URL:** https://developers.llamaindex.ai/typescript/framework/modules/models/embeddings/voyageai/

**Contents:**
- VoyageAI
- Installation
- API Reference

To use VoyageAI embeddings, you need to import VoyageAIEmbedding from @llamaindex/voyage-ai.

**Examples:**

Example 1 (python):
```python
npm i llamaindex @llamaindex/voyage-ai
```

Example 2 (python):
```python
npm i llamaindex @llamaindex/voyage-ai
```

Example 3 (javascript):
```javascript
import { VoyageAIEmbedding } from "@llamaindex/voyage-ai";import { Document, Settings, VectorStoreIndex } from "llamaindex";
Settings.embedModel = new VoyageAIEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

Example 4 (javascript):
```javascript
import { VoyageAIEmbedding } from "@llamaindex/voyage-ai";import { Document, Settings, VectorStoreIndex } from "llamaindex";
Settings.embedModel = new VoyageAIEmbedding();
const document = new Document({ text: essay, id_: "essay" });
const index = await VectorStoreIndex.fromDocuments([document]);
const queryEngine = index.asQueryEngine();
const query = "What is the meaning of life?";
const results = await queryEngine.query({  query,});
```

---
