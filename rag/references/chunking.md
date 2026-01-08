# Chunking Strategies for RAG

## Why Chunking Matters

Chunking is arguably the **most important factor** for RAG performance. How you split documents directly affects:

- **Retrieval quality**: Can the system find relevant information?
- **Context coherence**: Does the chunk make sense standalone?
- **Embedding accuracy**: Does the vector capture the right meaning?

> "Even a perfect retrieval system fails if it searches over poorly prepared data."

## The Chunking Dilemma

| Chunk Size | Pros | Cons |
|------------|------|------|
| **Too Large** | More context | Loses specificity, noisy embeddings |
| **Too Small** | Specific | Loses context, fragmented meaning |

**Goal**: Find the sweet spot where chunks are specific enough to be useful but contain enough context to be meaningful.

## Chunking Strategies

### 1. Fixed-Size Chunking

Split by character or token count with overlap.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)
chunks = splitter.split_documents(docs)
```

**Best for**: Simple, unstructured text
**Recommended settings**: 500-1000 chars, 10-20% overlap

### 2. Recursive Character Splitting

Tries multiple separators in order of preference.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
```

**Best for**: Most use cases, respects document structure
**Recommended**: Default choice for production

### 3. Token-Based Splitting

Aligns with LLM token limits.

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

**Best for**: When you need precise token control
**Note**: Slower due to tokenization

### 4. Semantic Chunking

Splits based on meaning changes.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)
chunks = splitter.split_documents(docs)
```

**Best for**: Narratives, documents with natural topic shifts
**Trade-off**: Requires embedding calls, slower

### 5. Document-Specific Splitting

#### Markdown

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
chunks = splitter.split_text(markdown_text)
```

#### Code

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)
chunks = splitter.split_documents(python_docs)
```

#### HTML

```python
from langchain.text_splitter import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_string)
```

## Overlap Best Practices

Overlap ensures complete thoughts aren't split across chunks.

| Overlap % | Use Case |
|-----------|----------|
| 0% | When chunks are naturally independent |
| 10-20% | Standard recommendation |
| 30%+ | Highly interconnected content |

```python
# Calculate overlap
chunk_size = 1000
overlap_percentage = 0.15  # 15%
chunk_overlap = int(chunk_size * overlap_percentage)  # 150
```

## Advanced Techniques

### Parent Document Retrieval

Store large chunks but retrieve by smaller child chunks.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Large parent chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Small child chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
retriever.add_documents(docs)

# Retrieves by small chunk, returns parent
results = retriever.get_relevant_documents("query")
```

### Late Chunking

Embed full documents first, then chunk—preserves cross-chunk references.

```python
# Conceptual approach
# 1. Embed entire document with long-context model
# 2. Split into chunks post-embedding
# 3. Each chunk's embedding contains full document context
```

### Hierarchical Chunking

Create multiple granularity levels.

```python
# Level 1: Document summaries
# Level 2: Section summaries
# Level 3: Paragraphs
# Level 4: Sentences

# Search across levels, return appropriate detail
```

## Metadata Best Practices

Always attach metadata to chunks:

```python
from langchain.schema import Document

chunk = Document(
    page_content="chunk text here",
    metadata={
        "source": "document.pdf",
        "page": 5,
        "section": "Introduction",
        "chunk_index": 3,
        "total_chunks": 25,
        "created_at": "2025-01-03"
    }
)
```

Useful metadata fields:
- `source`: Origin file/URL
- `page`: Page number (for PDFs)
- `section`: Section header
- `chunk_index`: Position in sequence
- `date`: Document date
- `author`: Content author

## Evaluation

### Metrics to Track

1. **Chunk coherence**: Does each chunk make sense alone?
2. **Retrieval precision**: Are retrieved chunks relevant?
3. **Boundary quality**: Are important ideas split?

### Testing Approach

```python
# 1. Create test queries
test_queries = [
    "What is X?",
    "How does Y work?",
    "Compare X and Y"
]

# 2. Retrieve for each chunking strategy
strategies = ["fixed_500", "fixed_1000", "recursive", "semantic"]

# 3. Measure precision@k, recall@k
# 4. Human evaluation of top results
```

## Recommendations by Content Type

| Content Type | Strategy | Chunk Size | Overlap |
|-------------|----------|------------|---------|
| Technical docs | Recursive | 1000 | 200 |
| Legal documents | Semantic | Variable | 15% |
| Code | Language-aware | 2000 | 200 |
| Chat logs | Fixed | 500 | 50 |
| Research papers | Section-based | 1500 | 200 |
| Product descriptions | Small fixed | 300 | 50 |

## Common Mistakes

1. **Ignoring document structure**: Use headers, sections
2. **No overlap**: Ideas get split
3. **One size fits all**: Different content needs different strategies
4. **Forgetting metadata**: Lose valuable filtering capability
5. **Not testing**: Can't optimize what you don't measure

## Resources

- [Chunking Strategies 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Weaviate Chunking Guide](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Stack Overflow: Breaking Up RAG](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
