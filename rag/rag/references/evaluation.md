# RAG Evaluation with RAGAS

## Why Evaluate RAG?

RAG systems have multiple components that can fail:
- Retriever may miss relevant documents
- Retrieved docs may be irrelevant
- LLM may not use context properly
- Answer may not address the question

Systematic evaluation helps identify and fix issues.

## RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) provides automated, reference-free evaluation of RAG pipelines.

### Installation

```bash
pip install ragas
```

### Core Metrics

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Faithfulness** | Is the answer grounded in context? | 0-1 |
| **Answer Relevancy** | Does the answer address the query? | 0-1 |
| **Context Precision** | Are top-ranked docs relevant? | 0-1 |
| **Context Recall** | Did we retrieve all relevant info? | 0-1 |

### The RAG Triad

```
         Query ──────────────────┐
           │                     │
           ▼                     ▼
    ┌─────────────┐      Answer Relevancy
    │  Retrieval  │
    └─────────────┘
           │
           ▼
       Context ─────── Context Precision
           │          Context Recall
           ▼
    ┌─────────────┐
    │  Generation │
    └─────────────┘
           │
           ▼
       Answer ─────── Faithfulness
```

## Basic Evaluation

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Prepare evaluation data
eval_data = {
    "question": [
        "What is RAG?",
        "How does chunking work?"
    ],
    "answer": [
        "RAG combines retrieval with generation to ground LLM responses.",
        "Chunking splits documents into smaller pieces for embedding."
    ],
    "contexts": [
        [
            "RAG (Retrieval-Augmented Generation) is a technique...",
            "RAG systems retrieve relevant documents..."
        ],
        [
            "Chunking divides text into segments...",
            "Common chunking strategies include..."
        ]
    ],
    "ground_truth": [  # Optional, needed for context_recall
        "RAG is a technique that enhances LLMs with external knowledge.",
        "Chunking breaks documents into smaller chunks for processing."
    ]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92,
#  'context_precision': 0.78, 'context_recall': 0.88}
```

## Metric Details

### Faithfulness

Measures if the answer is factually supported by the context.

```python
from ragas.metrics import faithfulness

# High faithfulness: Answer claims are in context
# Low faithfulness: Answer contains hallucinations
```

**Interpretation:**
- 1.0: All claims in answer are supported by context
- 0.5: Half the claims are unsupported
- 0.0: No claims are supported (complete hallucination)

### Answer Relevancy

Measures if the answer addresses the question.

```python
from ragas.metrics import answer_relevancy

# Works by generating questions from the answer
# and comparing to original question
```

**Interpretation:**
- 1.0: Answer perfectly addresses the question
- 0.5: Answer is partially relevant
- 0.0: Answer is completely off-topic

### Context Precision

Measures if retrieved documents are relevant and well-ranked.

```python
from ragas.metrics import context_precision

# Evaluates: Are the top-ranked docs actually relevant?
```

**Interpretation:**
- 1.0: All retrieved docs are relevant, best ones ranked first
- 0.5: Mix of relevant and irrelevant docs
- 0.0: Retrieved docs are not relevant

### Context Recall

Measures if all relevant information was retrieved.

```python
from ragas.metrics import context_recall

# Requires ground_truth to compare against
```

**Interpretation:**
- 1.0: All ground truth info is in retrieved context
- 0.5: Half the relevant info is missing
- 0.0: None of the relevant info was retrieved

## Advanced Metrics

### Factual Correctness

```python
from ragas.metrics import factual_correctness

# Compares answer facts to ground truth facts
```

### Semantic Similarity

```python
from ragas.metrics import answer_similarity

# Embedding-based similarity between answer and ground truth
```

### Noise Sensitivity

```python
from ragas.metrics import noise_sensitivity

# How robust is the answer to irrelevant context?
```

## Evaluation Pipeline

### 1. Collect Test Data

```python
test_cases = []

# Option A: Manual curation
test_cases.append({
    "question": "What is X?",
    "ground_truth": "X is...",
    "contexts": [],  # Will be filled by retriever
    "answer": ""     # Will be filled by RAG pipeline
})

# Option B: Synthetic generation
from ragas.testset import TestsetGenerator

generator = TestsetGenerator.from_langchain(llm, embeddings)
testset = generator.generate_with_langchain_docs(docs, test_size=50)
```

### 2. Run RAG Pipeline

```python
for case in test_cases:
    # Retrieve
    docs = retriever.invoke(case["question"])
    case["contexts"] = [doc.page_content for doc in docs]

    # Generate
    case["answer"] = rag_chain.invoke(case["question"])
```

### 3. Evaluate

```python
from datasets import Dataset
from ragas import evaluate

dataset = Dataset.from_list(test_cases)
results = evaluate(dataset, metrics=[...])
```

### 4. Analyze Results

```python
# Per-sample analysis
df = results.to_pandas()
print(df[df["faithfulness"] < 0.7])  # Find low-faithfulness samples

# Aggregate metrics
print(f"Mean Faithfulness: {df['faithfulness'].mean():.2f}")
print(f"Mean Context Precision: {df['context_precision'].mean():.2f}")
```

## Continuous Evaluation

### Integration with CI/CD

```python
# test_rag_quality.py
import pytest
from ragas import evaluate

def test_rag_quality():
    results = evaluate(test_dataset, metrics=[faithfulness, answer_relevancy])

    assert results["faithfulness"] > 0.8, "Faithfulness below threshold"
    assert results["answer_relevancy"] > 0.8, "Relevancy below threshold"
```

### Monitoring Dashboard

Track metrics over time:
- Faithfulness trends
- Precision/Recall by query type
- Failure case analysis
- A/B test results

## Interpreting Results

### Good RAG System

| Metric | Target | Meaning |
|--------|--------|---------|
| Faithfulness | > 0.85 | Minimal hallucination |
| Answer Relevancy | > 0.85 | On-topic responses |
| Context Precision | > 0.75 | Good retrieval ranking |
| Context Recall | > 0.80 | Not missing relevant info |

### Debugging Low Scores

| Low Score | Likely Issue | Fix |
|-----------|-------------|-----|
| Faithfulness | LLM hallucinating | Better prompting, guardrails |
| Answer Relevancy | Off-topic answers | Improve context quality |
| Context Precision | Irrelevant docs retrieved | Better chunking, reranking |
| Context Recall | Missing relevant docs | Increase k, hybrid search |

## Comparison with Other Tools

| Tool | Focus | Best For |
|------|-------|----------|
| **RAGAS** | RAG-specific metrics | RAG pipeline evaluation |
| **DeepEval** | General LLM testing | Broader LLM evaluation |
| **Giskard** | AI testing platform | Vulnerability scanning |
| **TruLens** | Feedback functions | Observability + evaluation |

## Best Practices

1. **Create domain-specific test sets**: Generic tests miss domain nuances
2. **Include edge cases**: Ambiguous queries, out-of-scope questions
3. **Balance metrics**: Don't optimize one at the expense of others
4. **Regular re-evaluation**: Knowledge bases change
5. **Human validation**: Automated metrics aren't perfect

## Resources

- [RAGAS Documentation](https://docs.ragas.io/en/stable/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGAS Paper (arXiv)](https://arxiv.org/abs/2309.15217)
- [Evaluating RAG in 2025](https://www.cohorte.co/blog/evaluating-rag-systems-in-2025-ragas-deep-dive-giskard-showdown-and-the-future-of-context)
