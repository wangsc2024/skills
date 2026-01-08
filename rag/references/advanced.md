# Advanced RAG Techniques

## Overview

Advanced RAG techniques address limitations of basic RAG:
- Missing relevant documents
- Retrieving irrelevant content
- Poor ranking of results
- Query-document vocabulary mismatch

## Hybrid Search

Combines semantic (vector) and lexical (keyword) search for better coverage.

### Why Hybrid Search?

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Semantic** | Understands meaning, synonyms | Misses exact terms, IDs, codes |
| **Keyword (BM25)** | Exact matching, proper nouns | Misses synonyms, paraphrases |
| **Hybrid** | Best of both | More complex setup |

### Implementation

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Keyword retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# Vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Tune based on your data
)

results = ensemble_retriever.invoke("What is product SKU-12345?")
```

### Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(results_list, k=60):
    """
    Merge multiple ranked lists using RRF.
    k is a constant (typically 60) that controls ranking smoothness.
    """
    fused_scores = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0}
            fused_scores[doc_id]["score"] += 1 / (rank + k)

    # Sort by fused score
    sorted_docs = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    return [item["doc"] for item in sorted_docs]
```

## Reranking

Re-scores initial retrieval results for better precision.

### Why Rerank?

Initial retrieval optimizes for recall (find everything relevant).
Reranking optimizes for precision (keep only the best).

### Cross-Encoder Reranking

Most accurate, but slower.

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, docs, top_k=3):
    # Create query-document pairs
    pairs = [(query, doc.page_content) for doc in docs]

    # Score each pair
    scores = cross_encoder.predict(pairs)

    # Sort by score
    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in doc_scores[:top_k]]

# Usage
initial_docs = retriever.invoke(query)  # Get 20 docs
reranked_docs = rerank_with_cross_encoder(query, initial_docs, top_k=5)
```

### Cohere Rerank

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

results = compression_retriever.invoke(query)
```

### ColBERT (Late Interaction)

Balances accuracy and speed.

```python
# ColBERT encodes query and documents separately
# Then computes token-level similarity efficiently
# More accurate than bi-encoders, faster than cross-encoders
```

## Query Expansion

Improves recall by generating query variations.

### Multi-Query Retrieval

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# LLM generates multiple query variations
# Each variation retrieves documents
# Results are combined
results = multi_retriever.invoke("How does RAG work?")
```

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, use it for retrieval.

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"
)

# 1. LLM generates hypothetical answer to query
# 2. Hypothetical answer is embedded
# 3. Embedding used for similarity search
# 4. Retrieved docs used for final answer
```

### Query Rewriting

```python
rewrite_prompt = """
Rewrite this query to be more specific and searchable.
Add relevant technical terms if appropriate.

Original: {query}
Rewritten:
"""

def rewrite_query(query):
    response = llm.invoke(rewrite_prompt.format(query=query))
    return response.content

# Usage
original = "how to make my app faster"
rewritten = rewrite_query(original)
# "optimize application performance latency caching"
```

## Self-Corrective RAG (CRAG)

Validates and corrects retrieval quality.

```python
def corrective_rag(query, retriever, web_search):
    # Step 1: Initial retrieval
    docs = retriever.invoke(query)

    # Step 2: Grade relevance
    relevant_docs = []
    for doc in docs:
        if is_relevant(doc, query):  # LLM-based grading
            relevant_docs.append(doc)

    # Step 3: Fallback if poor results
    if len(relevant_docs) < 2:
        web_docs = web_search(query)
        relevant_docs.extend(web_docs)

    # Step 4: Generate with validated context
    return generate(query, relevant_docs)

def is_relevant(doc, query):
    prompt = f"""
    Is this document relevant to the query?
    Query: {query}
    Document: {doc.page_content[:500]}
    Answer (yes/no):
    """
    response = llm.invoke(prompt)
    return "yes" in response.content.lower()
```

## Contextual Retrieval

Adds context to chunks before embedding.

```python
def add_context_to_chunks(document, chunks):
    """
    Prepend document-level context to each chunk.
    Improves retrieval by capturing global context.
    """
    doc_summary = summarize(document)

    contextualized_chunks = []
    for chunk in chunks:
        contextualized = f"""
        Document: {doc_summary}
        Section: {chunk.metadata.get('section', 'Unknown')}

        {chunk.page_content}
        """
        contextualized_chunks.append(contextualized)

    return contextualized_chunks
```

## Agentic RAG

LLM decides when and how to retrieve.

```python
from langgraph.prebuilt import create_react_agent

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return web_search_tool(query)

agent = create_react_agent(
    llm,
    tools=[search_documents, search_web],
    system_message="You are a helpful research assistant..."
)

# Agent decides which tool to use, can do multiple searches
response = agent.invoke({"messages": [("user", query)]})
```

## Graph RAG

Combines knowledge graphs with vector search.

```python
# 1. Extract entities and relationships from documents
# 2. Build knowledge graph
# 3. Query expansion using graph traversal
# 4. Hybrid retrieval: graph + vector

from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

def graph_enhanced_retrieval(query):
    # Extract entities from query
    entities = extract_entities(query)

    # Find related entities in graph
    related = graph.query(f"""
        MATCH (e)-[r]-(related)
        WHERE e.name IN {entities}
        RETURN related.name, type(r)
    """)

    # Expand query with related entities
    expanded_query = f"{query} {' '.join(related)}"

    # Vector search with expanded query
    return retriever.invoke(expanded_query)
```

## Summary: When to Use What

| Technique | Use When | Complexity |
|-----------|----------|------------|
| **Hybrid Search** | Need exact + semantic matching | Medium |
| **Reranking** | Initial retrieval has noise | Medium |
| **Query Expansion** | Vocabulary mismatch issues | Low |
| **CRAG** | High-stakes, need validation | High |
| **Agentic RAG** | Complex, multi-step queries | High |
| **Graph RAG** | Connected knowledge, entities | High |

## Implementation Order

1. **Start simple**: Basic RAG with good chunking
2. **Add hybrid search**: Vector + BM25
3. **Add reranking**: Cross-encoder or Cohere
4. **Query enhancement**: Multi-query or HyDE
5. **Self-correction**: CRAG for validation
6. **Go agentic**: When you need multi-step reasoning

## Resources

- [Optimizing RAG with Hybrid Search](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Advanced RAG Techniques (Neo4j)](https://neo4j.com/blog/genai/advanced-rag-techniques/)
- [9 Advanced RAG Techniques](https://www.meilisearch.com/blog/rag-techniques)
- [DataCamp Advanced RAG](https://www.datacamp.com/blog/rag-advanced)
