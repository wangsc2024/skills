# RAG Fundamentals

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval. Instead of relying solely on training data, RAG systems:

1. **Retrieve** relevant documents from an external knowledge base
2. **Augment** the user's query with retrieved context
3. **Generate** a response grounded in actual data

## Why RAG?

RAG addresses two critical LLM limitations:

| Limitation | How RAG Solves It |
|------------|-------------------|
| **Static Knowledge** | Access real-time, updated information |
| **Finite Context** | Retrieve only relevant portions from large corpora |
| **Hallucinations** | Ground responses in factual retrieved content |
| **Domain Specificity** | Incorporate proprietary/specialized data |

## RAG Architecture

### Basic Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                           │
├─────────────────────────────────────────────────────────────────┤
│  Documents → Chunking → Embedding → Vector Database            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│  User Query → Embed Query → Vector Search → Retrieve Top-K     │
│      ↓                                                          │
│  Augmented Prompt (Query + Context) → LLM → Response           │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Document Loaders**
   - Ingest data from various sources (PDFs, web, databases)
   - Extract text and metadata

2. **Text Splitters**
   - Break large documents into manageable chunks
   - Maintain semantic coherence

3. **Embedding Models**
   - Convert text to numerical vectors
   - Capture semantic meaning

4. **Vector Stores**
   - Store and index embeddings efficiently
   - Enable fast similarity search

5. **Retrievers**
   - Find relevant documents for a query
   - Return top-k matches

6. **LLM (Generator)**
   - Generate response using query + retrieved context
   - Synthesize information

## RAG Patterns

### 1. Basic RAG (Naive RAG)

```python
# Pseudocode
docs = retrieve(query, k=4)
context = format(docs)
response = llm(f"Context: {context}\n\nQuestion: {query}")
```

**Pros**: Simple, fast, predictable
**Cons**: May miss relevant info, no validation

### 2. Agentic RAG

```python
# Pseudocode
agent = create_agent(tools=[retriever])
# Agent decides when to retrieve, can do multiple searches
response = agent.run(query)
```

**Pros**: Flexible, multi-step reasoning
**Cons**: Higher latency, more complex

### 3. Corrective RAG (CRAG)

```python
# Pseudocode
docs = retrieve(query)
if not relevant(docs, query):
    docs = web_search(query)  # Fallback
response = llm(context=docs, query=query)
```

**Pros**: Self-correcting, handles edge cases
**Cons**: More LLM calls, higher cost

### 4. Graph RAG

```python
# Pseudocode
entities = extract_entities(query)
related = graph.traverse(entities)
docs = retrieve(query, filter=related)
response = llm(context=docs, query=query)
```

**Pros**: Captures relationships, structured knowledge
**Cons**: Requires knowledge graph, complex setup

## The RAG Triad

Three key metrics for RAG quality:

1. **Context Relevance**: Are retrieved documents relevant to the query?
2. **Groundedness**: Is the answer supported by the context?
3. **Answer Relevance**: Does the answer address the question?

```
         Query
           ↓
    ┌──────────────┐
    │  Retrieval   │ ← Context Relevance
    └──────────────┘
           ↓
       Context
           ↓
    ┌──────────────┐
    │  Generation  │ ← Groundedness
    └──────────────┘
           ↓
       Answer       ← Answer Relevance
```

## When to Use RAG vs Fine-Tuning

| Use RAG When | Use Fine-Tuning When |
|--------------|---------------------|
| Data changes frequently | Need specific writing style |
| Need source attribution | Want faster inference |
| Have large knowledge bases | Have stable, small dataset |
| Need factual grounding | Need domain-specific behavior |
| Want to avoid retraining | Have compute budget |

## Implementation Checklist

- [ ] Define knowledge sources
- [ ] Choose chunking strategy
- [ ] Select embedding model
- [ ] Set up vector store
- [ ] Implement retrieval pipeline
- [ ] Design prompt template
- [ ] Add evaluation metrics
- [ ] Test with representative queries
- [ ] Monitor and iterate

## Common Pitfalls

1. **Wrong chunk size**: Too big = noisy, too small = fragmented
2. **Ignoring metadata**: Lose valuable filtering capabilities
3. **Fixed k value**: Should adapt to query complexity
4. **No reranking**: Wastes context on irrelevant docs
5. **Poor prompting**: LLM doesn't know how to use context
6. **No evaluation**: Can't improve what you don't measure

## Resources

- [LangChain RAG Docs](https://docs.langchain.com/oss/python/langchain/rag)
- [LlamaIndex RAG Guide](https://developers.llamaindex.ai/python/framework/)
- [AWS RAG Overview](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
