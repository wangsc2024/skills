---
name: rag
description: |
  Retrieval-Augmented Generation (RAG) comprehensive guide. Use for building RAG systems, vector databases, embeddings, chunking strategies, hybrid search, reranking, and evaluation with RAGAS.
  Use when: building RAG systems, implementing vector search, managing embeddings, document chunking, knowledge retrieval, or when user mentions RAG, vector database, embeddings, retrieval, 向量資料庫, 檢索, RAGAS, semantic search.
  Triggers: "RAG", "vector database", "embeddings", "retrieval", "向量資料庫", "檢索", "RAGAS", "semantic search", "chunking", "reranking", "knowledge base", "document retrieval"
---

# RAG (Retrieval-Augmented Generation) Skill

Comprehensive guide to building production-ready RAG systems that combine LLMs with external knowledge retrieval for accurate, grounded AI applications.

## When to Use This Skill

This skill should be triggered when:
- Building RAG pipelines for question-answering systems
- Implementing document retrieval with vector databases
- Choosing chunking and embedding strategies
- Setting up hybrid search (semantic + keyword)
- Implementing reranking for improved relevance
- Evaluating RAG systems with RAGAS metrics
- Working with LangChain, LlamaIndex, or similar frameworks
- Optimizing retrieval quality and performance

## Quick Reference

### RAG Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Indexing  │    │  Retrieval  │    │ Augmentation│    │ Generation  │
├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤
│ Load docs   │───▶│ Query embed │───▶│ Inject into │───▶│ LLM creates │
│ Chunk       │    │ Vector search│    │ prompt      │    │ response    │
│ Embed       │    │ Rerank      │    │             │    │             │
│ Store       │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Installation

```bash
# LangChain
pip install langchain langchain-openai langchain-chroma

# LlamaIndex
pip install llama-index llama-index-vector-stores-chroma

# Vector Databases
pip install chromadb          # Chroma (local/embedded)
pip install pinecone-client   # Pinecone (cloud)
pip install qdrant-client     # Qdrant
pip install weaviate-client   # Weaviate

# Evaluation
pip install ragas
```

## Core Components

### 1. Document Loading

```python
# LangChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader
)

# Load PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Load directory
loader = DirectoryLoader("./docs", glob="**/*.txt")
docs = loader.load()

# Load web page
loader = WebBaseLoader("https://example.com/page")
docs = loader.load()
```

```python
# LlamaIndex
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

### 2. Chunking Strategies

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Recommended: Recursive Character Splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # 10-20% overlap recommended
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
```

**Chunking Best Practices:**
| Strategy | Use Case | Chunk Size |
|----------|----------|------------|
| Fixed size | General purpose | 500-1000 chars |
| Recursive | Structured text | 1000 chars, 200 overlap |
| Semantic | Natural boundaries | Variable |
| Token-based | LLM alignment | 256-512 tokens |

**Key Insight:** Chunking is the most important factor for RAG performance. Too large = loses specificity, too small = loses context.

### 3. Embeddings

```python
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Open Source Alternatives
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Popular Embedding Models:**
| Model | Dimensions | Use Case |
|-------|------------|----------|
| text-embedding-3-small | 1536 | General (OpenAI) |
| text-embedding-3-large | 3072 | High accuracy (OpenAI) |
| all-MiniLM-L6-v2 | 384 | Fast, open source |
| bge-large-en-v1.5 | 1024 | High quality, open source |
| E5-large-v2 | 1024 | Instruction-tuned |

### 4. Vector Store

```python
# Chroma (Local)
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Pinecone (Cloud)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("your-index")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
```

```python
# LlamaIndex
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
```

### 5. Retrieval

```python
# Basic Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Retrieve documents
docs = retriever.invoke("What is RAG?")
```

### 6. RAG Chain (Complete Example)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query
response = rag_chain.invoke("What is RAG?")
```

## Advanced Techniques

### Hybrid Search (Semantic + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (keyword-based)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# Vector (semantic)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Ensemble with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Tune based on your data
)
```

### Reranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Cohere Reranker
reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

# Cross-Encoder Reranker (open source)
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_docs(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
```

### Query Expansion / HyDE

```python
# Hypothetical Document Embeddings (HyDE)
from langchain.chains import HypotheticalDocumentEmbedder

hyde = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"
)

# Query Rewriting
rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite this query to be more specific and searchable:
Original: {query}
Rewritten:
""")

def expand_query(query):
    return llm.invoke(rewrite_prompt.format(query=query)).content
```

### Multi-Query Retrieval

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
# Generates multiple query variations for better recall
```

### Parent Document Retrieval

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store full documents, retrieve by child chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
```

## Evaluation with RAGAS

### Installation & Setup

```python
pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
```

### Core Metrics

| Metric | What it Measures |
|--------|------------------|
| **Faithfulness** | Is the answer factually grounded in the context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are retrieved docs actually relevant? |
| **Context Recall** | Did we retrieve all relevant info? |

### Evaluation Example

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Prepare evaluation data
eval_data = {
    "question": ["What is RAG?", "How does chunking work?"],
    "answer": ["RAG combines retrieval with generation...", "Chunking splits documents..."],
    "contexts": [
        ["RAG is a technique that...", "Retrieved context 2..."],
        ["Chunking divides text...", "More context..."]
    ],
    "ground_truth": ["RAG is...", "Chunking is..."]  # Optional
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92, ...}
```

## RAG Architecture Patterns

### 1. Basic RAG (2-Step)
```
Query → Retrieve → Generate
```
- Simple, fast, predictable
- Best for: FAQs, documentation search

### 2. Agentic RAG
```
Query → Agent decides when/what to retrieve → Generate
```
- Flexible, multi-step reasoning
- Best for: Complex research, multi-source queries

### 3. Corrective RAG (CRAG)
```
Query → Retrieve → Validate relevance → Re-retrieve if needed → Generate
```
- Self-correcting, higher quality
- Best for: High-stakes applications

### 4. Graph RAG
```
Query → Graph traversal + Vector search → Generate
```
- Captures relationships between entities
- Best for: Knowledge graphs, connected data

## Best Practices

### Chunking
1. Use 10-20% overlap between chunks
2. Respect natural boundaries (paragraphs, sections)
3. Keep chunk size aligned with embedding model limits
4. Include metadata (source, page number, section)

### Retrieval
1. Start with k=4-6 documents
2. Use hybrid search for production systems
3. Add reranking for quality-critical applications
4. Filter by metadata when possible

### Generation
1. Always include source citations
2. Set clear instructions for handling missing info
3. Use structured output for consistency
4. Implement guardrails for hallucination

### Evaluation
1. Track: Precision@k, Recall@k, MRR, NDCG
2. Use RAGAS for automated evaluation
3. Include human evaluation for critical systems
4. A/B test retrieval strategies

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Missing relevant docs | Increase k, use hybrid search |
| Too much irrelevant context | Add reranking, reduce k |
| Hallucinations | Improve faithfulness prompt, add verification |
| Slow retrieval | Use approximate NN (HNSW), cache embeddings |
| Poor chunk quality | Adjust chunk size/overlap, use semantic chunking |

## Technology Stack Recommendations

### For Prototyping
- **Framework**: LangChain or LlamaIndex
- **Vector DB**: Chroma (local)
- **Embeddings**: OpenAI or HuggingFace
- **LLM**: GPT-4o-mini or Claude Haiku

### For Production
- **Framework**: LangChain + LangGraph
- **Vector DB**: Pinecone, Qdrant, or Weaviate
- **Embeddings**: text-embedding-3-large or BGE
- **LLM**: GPT-4o or Claude Sonnet
- **Reranker**: Cohere or Cross-Encoder
- **Evaluation**: RAGAS + custom metrics

## Resources

### Official Documentation
- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)
- [LlamaIndex Framework](https://developers.llamaindex.ai/python/framework/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [RAGAS Documentation](https://docs.ragas.io/en/stable/)

### Learning Resources
- [AWS RAG Overview](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [Google Cloud RAG](https://cloud.google.com/use-cases/retrieval-augmented-generation)
- [IBM RAG Guide](https://www.ibm.com/think/topics/retrieval-augmented-generation)

### Advanced Topics
- [Chunking Strategies 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Advanced RAG Techniques](https://neo4j.com/blog/genai/advanced-rag-techniques/)

## Reference Files

This skill includes detailed documentation in `references/`:

| File | Description |
|------|-------------|
| **fundamentals.md** | RAG concepts, architecture, components |
| **chunking.md** | Chunking strategies and best practices |
| **embeddings.md** | Embedding models and usage |
| **vector_stores.md** | Vector database options |
| **advanced.md** | Hybrid search, reranking, query expansion |
| **evaluation.md** | RAGAS metrics and evaluation |
| **frameworks.md** | LangChain, LlamaIndex examples |

## Notes

- This skill was compiled from official documentation and industry best practices (2025)
- RAG is evolving rapidly; verify against latest documentation
- Chunking and retrieval quality are the biggest factors in RAG performance
- Always evaluate with domain-specific test data

## Key Takeaways

1. **Chunking is critical** - 10-20% overlap, respect natural boundaries
2. **Hybrid search improves recall** - Combine semantic + keyword
3. **Reranking improves precision** - Use cross-encoders for quality
4. **Evaluate systematically** - RAGAS for faithfulness, relevancy, precision, recall
5. **Start simple, iterate** - Basic RAG → Hybrid → Reranking → Agentic
