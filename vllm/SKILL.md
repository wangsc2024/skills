---
name: vllm
description: |
  vLLM 高效能 LLM 推理引擎部署與優化指南。支援 PagedAttention、Tensor Parallel、Continuous Batching 等技術，實現高吞吐量模型部署。
  Use when: 部署 LLM 推理服務、優化模型效能、設定 OpenAI 相容 API、配置分散式推理，or when user mentions vLLM, inference, serving, 推理, 部署.
  Triggers: "vLLM", "vllm", "inference", "serving", "PagedAttention", "推理", "部署", "模型部署", "LLM serving", "tensor parallel", "continuous batching", "quantization", "高吞吐量"
---

# vLLM Skill

vLLM 是高效能 LLM 推理引擎，專為生產環境設計，提供極高的吞吐量和低延遲。

## When to Use This Skill

- 部署 LLM 推理服務（OpenAI 相容 API）
- 優化模型推理效能
- 配置 Tensor Parallel 分散式推理
- 使用 PagedAttention 提升記憶體效率
- 量化模型（AWQ, GPTQ, FP8）
- 設定 Continuous Batching

## Quick Reference

### 安裝

```bash
pip install vllm
```

### 啟動 OpenAI 相容 API Server

```bash
# 基本啟動
vllm serve meta-llama/Llama-3.1-8B-Instruct

# 進階配置
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### Python API 使用

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# 設定取樣參數
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 批次推理
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI Client 呼叫

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM 不驗證 key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 常用配置參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--tensor-parallel-size` | GPU 數量（Tensor Parallel） | 1 |
| `--max-model-len` | 最大上下文長度 | 模型預設 |
| `--gpu-memory-utilization` | GPU 記憶體使用率 | 0.9 |
| `--quantization` | 量化方式（awq, gptq, fp8） | None |
| `--dtype` | 資料類型（auto, float16, bfloat16） | auto |
| `--max-num-seqs` | 最大並行序列數 | 256 |

### 量化部署

```bash
# AWQ 量化模型
vllm serve TheBloke/Llama-2-70B-AWQ --quantization awq

# GPTQ 量化模型
vllm serve TheBloke/Llama-2-70B-GPTQ --quantization gptq
```

## Reference Files

詳細文檔請參考 `references/` 目錄：

| 檔案 | 內容 |
|------|------|
| `getting_started.md` | 快速入門指南 |
| `api.md` | 完整 API 參考 |
| `deployment.md` | 部署指南與最佳實踐 |
| `models.md` | 支援的模型列表 |
| `features.md` | 功能特性說明 |
| `performance.md` | 效能優化指南 |
| `configuration.md` | 配置參數詳解 |

## Resources

- 官方文檔: https://docs.vllm.ai/
- GitHub: https://github.com/vllm-project/vllm
