---
name: dspy
description: |
  DSPy is Stanford's framework for programming and optimizing language model prompts. Use this skill when building LM-powered pipelines, optimizing prompts with teleprompters, creating modular AI programs with signatures and modules, evaluating and fine-tuning language model systems, or implementing retrieval-augmented generation (RAG) with DSPy.
  Use when: optimizing LLM prompts programmatically, building modular AI systems, implementing teleprompters, evaluating prompt performance, or when user mentions DSPy, teleprompter, signature, module, MIPRO, prompt optimization, 提示優化.
  Triggers: "DSPy", "teleprompter", "signature", "module", "MIPRO", "prompt optimization", "提示優化", "declarative LM", "LM programming", "prompt tuning"
version: 1.0.0
---

# DSPy Skill

DSPy 是 Stanford 開發的框架，用於程式化語言模型提示詞優化。透過模組化設計和自動優化，讓開發者能建構可靠、可組合的 AI 應用程式。

## When to Use This Skill

This skill should be triggered when:
- Building LM-powered pipelines with modular components
- Optimizing prompts using teleprompters (MIPROv2, COPRO, BootstrapFewShot)
- Creating modular AI programs with signatures and modules
- Implementing retrieval-augmented generation (RAG) with DSPy
- Fine-tuning and evaluating language model systems
- Building tool-using agents with ReAct pattern

## Quick Reference

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Signature** | 定義輸入輸出的結構化規格 |
| **Module** | 可組合的 LM 元件（Predict, ChainOfThought, ReAct） |
| **Optimizer** | 自動優化提示詞和權重（MIPROv2, COPRO, BootstrapFewShot） |
| **Metric** | 評估模型輸出品質的函數 |

### Basic Setup

```python
import dspy

# Configure the language model
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

### Signatures - Define Input/Output Structure

```python
# Inline signature (simple)
classify = dspy.Predict('sentence -> sentiment: bool')
response = classify(sentence="it's a charming journey.")
print(response.sentiment)

# Class-based signature (complex)
class QuestionAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context: str = dspy.InputField(desc="may contain relevant facts")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

### Modules - Building Blocks

#### dspy.Predict - Basic Module
```python
qa = dspy.Predict('question: str -> response: str')
response = qa(question="What is the capital of France?")
print(response.response)
```

#### dspy.ChainOfThought - Step-by-Step Reasoning
```python
# ChainOfThought adds reasoning before the answer
cot = dspy.ChainOfThought('question -> response')
prediction = cot(question="Should curly braces appear on their own line?")
print(prediction.reasoning)  # Step-by-step reasoning
print(prediction.response)   # Final answer
```

#### dspy.ReAct - Tool-Using Agents
```python
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny."

react = dspy.ReAct(signature="question -> answer", tools=[get_weather])
pred = react(question="What is the weather in Tokyo?")
print(pred.answer)
```

#### dspy.ProgramOfThought - Code Generation
```python
pot = dspy.ProgramOfThought("question -> answer", max_iters=3)
result = pot(question="What is 15 * 23 + 42?")
print(result.answer)
```

### Building Complex Modules

```python
class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question, k=self.num_docs)  # Your retrieval function
        return self.respond(context=context, question=question)

rag = RAG()
response = rag(question="What is task decomposition?")
```

### Multi-Stage Pipelines

```python
class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section = self.draft_section(
                topic=outline.title,
                section_heading=heading,
                section_subheadings=subheadings
            )
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)
```

## Optimizers

### MIPROv2 - Production-Ready Optimizer

```python
# Best for production - optimizes prompts automatically
tp = dspy.MIPROv2(
    metric=dspy.SemanticF1(),
    auto="medium",
    num_threads=24
)

optimized_rag = tp.compile(
    RAG(),
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2
)
```

### BootstrapFewShot - Quick Few-Shot Learning

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4
)

teleprompter = BootstrapFewShotWithRandomSearch(metric=your_metric, **config)
optimized_program = teleprompter.compile(your_program, trainset=trainset)
```

### KNNFewShot - Semantic Example Selection

```python
from sentence_transformers import SentenceTransformer
from dspy.teleprompt import KNNFewShot

knn_optimizer = KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

qa_compiled = knn_optimizer.compile(student=dspy.ChainOfThought("question -> answer"))
```

### GEPA - Feedback-Driven Optimization

```python
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_lm=dspy.LM(model="gpt-4o", temperature=1.0)
)

optimized = optimizer.compile(student=your_module, trainset=trainset)
```

## RAG Implementation

### Complete RAG Example

```python
import dspy

def search(question, k=5):
    """Your retrieval function - ColBERT, Pinecone, etc."""
    results = dspy.ColBERTv2(url='http://your-server')(question, k=k)
    return [x['text'] for x in results]

class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question, k=self.num_docs)
        return self.respond(context=context, question=question)

# Optimize with MIPROv2
tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
optimized_rag = tp.compile(RAG(), trainset=trainset,
                           max_bootstrapped_demos=2, max_labeled_demos=2)

# Use the optimized RAG
pred = optimized_rag(question="How does task decomposition work?")
print(pred.response)
```

## Evaluation

```python
from dspy import Evaluate

# Define your metric
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Evaluate
evaluate = Evaluate(devset=devset, metric=exact_match, num_threads=4)
score = evaluate(your_module)
print(f"Accuracy: {score}")
```

## Best Practices

### 1. Start Simple
```python
# Start with Predict, upgrade to ChainOfThought if needed
simple = dspy.Predict('question -> answer')
complex = dspy.ChainOfThought('question -> answer')  # Adds reasoning
```

### 2. Use Typed Outputs
```python
from typing import Literal

class Classification(dspy.Signature):
    text: str = dspy.InputField()
    label: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
```

### 3. Custom Instructions
```python
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults or harassment.",
    )
)
```

### 4. Inspect Optimization History
```python
# Debug and understand what the optimizer did
dspy.inspect_history()
```

## Reference Files

This skill includes comprehensive documentation in `references/`:
- **modules.md** - Detailed module documentation

## Resources

- **Official Docs**: https://dspy.ai
- **GitHub**: https://github.com/stanfordnlp/dspy
- **Tutorials**: https://dspy.ai/tutorials

## Notes

- DSPy automatically handles prompt engineering through optimization
- Use `dspy.configure()` to set default LM for all modules
- Optimizers like MIPROv2 can significantly improve performance (10%+ gains typical)
- Start with smaller trainsets (50-100 examples) for faster iteration
