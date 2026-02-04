---
name: vllm
description: vLLM is a fast and easy-to-use library for LLM inference and serving. Use this skill when working with vLLM deployment, model serving, inference optimization, PagedAttention, continuous batching, tensor parallelism, or high-throughput LLM serving.
---

# vLLM Skill

vLLM 是高效能的 LLM 推論和服務函式庫，透過 PagedAttention 實現業界領先的吞吐量。

## When to Use This Skill

- 部署 LLM 推論服務
- 需要高吞吐量的 LLM 服務
- 使用 OpenAI 相容 API
- 多 GPU 張量並行部署
- LoRA 動態載入
- 量化模型推論

## 核心特性

```
vLLM 核心技術
├── PagedAttention       # 高效記憶體管理，提升吞吐量 24x
├── Continuous Batching  # 連續批次處理，最大化 GPU 利用率
├── Tensor Parallelism   # 多 GPU 張量並行
├── Speculative Decoding # 推測解碼加速
├── Chunked Prefill      # 分塊預填充
└── Quantization         # AWQ/GPTQ/FP8 量化支援
```

## 安裝

```bash
# 基本安裝（CUDA 12.1）
pip install vllm

# 指定 CUDA 版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 離線推論

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### OpenAI 相容伺服器

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 進階配置

### 多 GPU 張量並行

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9
)
```

### 量化模型

```python
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ", quantization="gptq")
```

### LoRA 動態載入

```python
from vllm.lora.request import LoRARequest
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_lora=True)
lora_request = LoRARequest("my-lora", 1, "/path/to/lora")
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```

## 常用參數

### SamplingParams

| 參數 | 說明 | 預設值 |
|------|------|--------|
| temperature | 取樣溫度 | 1.0 |
| top_p | Nucleus 取樣 | 1.0 |
| max_tokens | 最大生成 token | 16 |
| stop | 停止詞列表 | None |

### LLM 引擎參數

| 參數 | 說明 |
|------|------|
| tensor_parallel_size | GPU 數量 |
| gpu_memory_utilization | GPU 記憶體利用率 |
| max_model_len | 最大上下文長度 |
| quantization | 量化方式 (awq/gptq/fp8) |

## Docker 部署

```bash
docker run --runtime nvidia --gpus all \
    -p 8000:8000 vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

## 常見問題

### OOM 解決

```python
llm = LLM(
    model="model-name",
    gpu_memory_utilization=0.8,
    max_model_len=2048,
    enforce_eager=True
)
```

## Reference Files

| 檔案 | 內容 |
|------|------|
| getting_started.md | 快速入門 |
| api.md | API 參考 |
| deployment.md | 部署指南 |
| performance.md | 效能優化 |

## Resources

- [官方文件](https://docs.vllm.ai/)
- [GitHub](https://github.com/vllm-project/vllm)
