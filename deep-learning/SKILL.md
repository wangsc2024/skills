---
name: deep-learning
description: |
  深度學習核心技術。涵蓋 CNN、RNN、LSTM、Transformer 架構，以及 Dropout、Batch Normalization 等優化技術。
  Use when: understanding deep learning architectures, implementing neural networks, optimizing models, or when user mentions CNN, RNN, LSTM, Transformer, 神經網路, 深度學習, neural network, Dropout, Batch Normalization.
  Triggers: "CNN", "RNN", "LSTM", "Transformer", "神經網路", "深度學習", "neural network", "Dropout", "Batch Normalization", "卷積", "循環神經網路"
version: 1.0.0
---

# 深度學習技術 Skill

全面涵蓋深度學習的核心架構與技術，包括 CNN、RNN、Transformer，以及正則化、優化等關鍵技術。

## 適用情境

當你需要：
- 理解和實作深度學習模型
- 選擇適合任務的神經網路架構
- 應用 CNN 進行影像處理
- 使用 RNN/LSTM 處理序列資料
- 實作 Transformer 和注意力機制
- 應用正則化技術防止過擬合
- 優化模型訓練過程

## 神經網路架構總覽

### 架構演進

```
ANN (感知器)
  ↓
CNN (視覺)
  ↓
RNN (序列)
  ↓
LSTM/GRU (長序列)
  ↓
Transformer (並行 + 長距離依賴)
  ↓
現代 LLM (GPT, BERT, etc.)
```

### 架構比較

| 架構 | 最適任務 | 優勢 | 限制 |
|------|---------|------|------|
| **CNN** | 影像 | 空間特徵提取 | 難處理序列 |
| **RNN** | 序列 | 時間依賴 | 梯度消失 |
| **LSTM** | 長序列 | 長期記憶 | 計算較慢 |
| **Transformer** | NLP/多模態 | 並行化、長距離 | 計算量大 |

## 1. 卷積神經網路 (CNN)

### 核心概念

CNN 專為處理網格狀資料（如影像）設計，透過卷積運算提取特徵。

### 架構組成

```
輸入影像 → [卷積層 → 激活 → 池化層] × N → 展平 → 全連接層 → 輸出

典型流程：
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  輸入    │   │  卷積層  │   │  池化層  │   │  全連接  │
│  影像    │──▶│  特徵    │──▶│  降維    │──▶│  分類    │
│ 224×224  │   │  提取    │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### 核心層次

#### 1. 卷積層 (Convolution Layer)

```python
import torch.nn as nn

# 卷積層定義
conv = nn.Conv2d(
    in_channels=3,      # 輸入通道 (RGB=3)
    out_channels=64,    # 輸出特徵圖數量
    kernel_size=3,      # 卷積核大小 3×3
    stride=1,           # 步幅
    padding=1           # 填充
)
```

**卷積運算視覺化：**
```
輸入特徵圖         卷積核 (3×3)       輸出特徵圖
┌─────────────┐   ┌───────────┐   ┌─────────────┐
│ 1  2  3  4  │   │ 1  0  1   │   │             │
│ 5  6  7  8  │ ⊛ │ 0  1  0   │ = │  特徵值     │
│ 9  10 11 12 │   │ 1  0  1   │   │             │
│ 13 14 15 16 │   └───────────┘   └─────────────┘
└─────────────┘

運算：對應位置相乘後求和
```

**特徵提取層次：**
```
淺層 → 邊緣、顏色、紋理
中層 → 形狀、圖案
深層 → 物體部位、語義特徵
```

#### 2. 池化層 (Pooling Layer)

```python
# 最大池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

**池化作用：**
- 降低特徵圖尺寸
- 提供平移不變性
- 減少計算量

```
最大池化 (2×2, stride=2)：

┌───┬───┬───┬───┐        ┌───┬───┐
│ 1 │ 2 │ 5 │ 6 │        │   │   │
├───┼───┼───┼───┤   →    │ 4 │ 8 │
│ 3 │ 4 │ 7 │ 8 │        ├───┼───┤
├───┼───┼───┼───┤        │   │   │
│ 9 │ 10│ 13│ 14│   →    │12 │16 │
├───┼───┼───┼───┤        └───┴───┘
│ 11│ 12│ 15│ 16│
└───┴───┴───┴───┘
```

### 完整 CNN 範例

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 特徵提取
        self.features = nn.Sequential(
            # 第一卷積區塊
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 第二卷積區塊
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 第三卷積區塊
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 分類器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## 2. 循環神經網路 (RNN)

### 核心概念

RNN 專為處理序列資料設計，具有「記憶」能力。

### 基本結構

```
RNN 展開圖：

  h₀    h₁    h₂    h₃    h₄
   │     │     │     │     │
   ▼     ▼     ▼     ▼     ▼
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ A  │→│ A  │→│ A  │→│ A  │→│ A  │
└────┘ └────┘ └────┘ └────┘ └────┘
   ▲     ▲     ▲     ▲     ▲
   │     │     │     │     │
  x₀    x₁    x₂    x₃    x₄

A = 相同的神經網路單元
h = 隱藏狀態 (記憶)
x = 輸入
```

### 數學表達

```
hₜ = tanh(Wₕₕ · hₜ₋₁ + Wₓₕ · xₜ + b)
yₜ = Wₕᵧ · hₜ

其中：
- hₜ = 時間步 t 的隱藏狀態
- xₜ = 時間步 t 的輸入
- W = 權重矩陣
```

### RNN 的問題：梯度消失/爆炸

```
長序列的問題：

x₀ ─────────────────────────────────▶ yₙ
    ↑
    需要記住很多步之前的資訊

反向傳播時梯度連乘：
∂L/∂W = ∂L/∂hₙ × ∂hₙ/∂hₙ₋₁ × ... × ∂h₁/∂W
        ─────────────────────────────
                連乘很多項

→ 梯度消失 (值 < 1 連乘) 或 梯度爆炸 (值 > 1 連乘)
```

## 3. LSTM (長短期記憶)

### 核心概念

LSTM 透過「門控機制」解決 RNN 的梯度問題，能學習長期依賴。

### LSTM 單元結構

```
                    ┌─────────────────────────────────┐
                    │           LSTM Cell             │
                    │                                 │
    cₜ₋₁ ─────────────▶ ×  ────────▶ +  ──────────▶ cₜ
                       ▲         ▲    ▲
                       │         │    │
                    ┌──┴──┐   ┌──┴──┐ │
                    │遺忘門│   │輸入門│ │
                    │  fₜ │   │ iₜ  │ │
                    └─────┘   └─────┘ │
                       ▲         ▲    │
                       │         │    │
    hₜ₋₁ ──┬───────────┴─────────┴────┤
           │                          │
           │    ┌─────┐               │
           └───▶│輸出門│──▶ tanh ──▶ hₜ
                │  oₜ │
                └─────┘
                   ▲
                   │
    xₜ ────────────┴
```

### 三個門的功能

| 門 | 功能 | 公式 |
|----|------|------|
| **遺忘門 (f)** | 決定丟棄什麼 | σ(Wf · [hₜ₋₁, xₜ] + bf) |
| **輸入門 (i)** | 決定儲存什麼 | σ(Wi · [hₜ₋₁, xₜ] + bi) |
| **輸出門 (o)** | 決定輸出什麼 | σ(Wo · [hₜ₋₁, xₜ] + bo) |

### PyTorch 實現

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True  # 雙向 LSTM
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 因為雙向

    def forward(self, x):
        # LSTM 輸出
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最後時間步
        out = self.fc(lstm_out[:, -1, :])
        return out
```

## 4. Transformer

### 核心概念

Transformer 完全基於注意力機制，捨棄循環結構，實現高度並行化。

### 架構總覽

```
┌─────────────────────────────────────────────────────────┐
│                      Transformer                         │
├──────────────────────────┬──────────────────────────────┤
│        Encoder           │          Decoder              │
│                          │                               │
│  ┌──────────────────┐    │    ┌──────────────────┐      │
│  │  Self-Attention  │    │    │ Masked Attention │      │
│  └────────┬─────────┘    │    └────────┬─────────┘      │
│           ↓              │             ↓                 │
│  ┌──────────────────┐    │    ┌──────────────────┐      │
│  │   Feed Forward   │    │    │ Cross-Attention  │      │
│  └────────┬─────────┘    │    └────────┬─────────┘      │
│           ↓              │             ↓                 │
│        (× N)             │    ┌──────────────────┐      │
│                          │    │   Feed Forward   │      │
│                          │    └────────┬─────────┘      │
│                          │             ↓                 │
│                          │          (× N)               │
└──────────────────────────┴──────────────────────────────┘
```

### 自注意力機制 (Self-Attention)

**核心公式：**
```
Attention(Q, K, V) = softmax(QK^T / √dₖ) × V

Q = Query (查詢)
K = Key (鍵)
V = Value (值)
dₖ = Key 的維度
```

**計算流程：**
```
1. 計算相似度分數
   ┌───────────────────────────────────────┐
   │   scores = Q × K^T                    │
   │   ┌─────────────────┐                 │
   │   │ q₁·k₁  q₁·k₂ ... │               │
   │   │ q₂·k₁  q₂·k₂ ... │               │
   │   │  ...    ...  ... │               │
   │   └─────────────────┘                 │
   └───────────────────────────────────────┘

2. 縮放並 Softmax
   ┌───────────────────────────────────────┐
   │   attention_weights = softmax(scores/√dₖ) │
   └───────────────────────────────────────┘

3. 加權求和
   ┌───────────────────────────────────────┐
   │   output = attention_weights × V     │
   └───────────────────────────────────────┘
```

### 多頭注意力 (Multi-Head Attention)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 線性投影並分成多頭
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 計算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # 合併多頭
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(context)
```

### 位置編碼 (Positional Encoding)

由於 Transformer 沒有循環結構，需要額外注入位置資訊：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶數位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇數位置

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## 5. 正則化技術

### Dropout

**原理：** 訓練時隨機「丟棄」神經元，防止過度依賴。

```python
# PyTorch 實現
dropout = nn.Dropout(p=0.5)  # 50% 的神經元被丟棄

# 使用
x = dropout(x)  # 訓練時隨機丟棄
                # 推理時自動關閉
```

**視覺化：**
```
訓練時：                    推理時：
○─○─○─○                    ●─●─●─●
 ╲ ╱ ╲ ╱                    │ │ │ │
  ×   ○─×                   ●─●─●─●
 ╱ ╲   ╲                    │ │ │ │
○   ○───○                   ●─●─●─●

○ = 活躍    × = 被丟棄      ● = 全部活躍
```

**Dropout 變體：**

| 變體 | 特點 |
|------|------|
| **DropConnect** | 丟棄權重而非神經元 |
| **Spatial Dropout** | 丟棄整個特徵圖（用於 CNN） |
| **Variational Dropout** | RNN 中使用相同遮罩 |

### Batch Normalization

**原理：** 對每個 mini-batch 進行標準化，加速訓練、穩定梯度。

```python
# PyTorch 實現
bn = nn.BatchNorm2d(num_features=64)  # 用於 CNN
bn = nn.BatchNorm1d(num_features=256) # 用於全連接層

# 使用
x = bn(x)
```

**計算過程：**
```
1. 計算 mini-batch 的均值和方差
   μ = (1/m) Σ xᵢ
   σ² = (1/m) Σ (xᵢ - μ)²

2. 標準化
   x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

3. 縮放和偏移 (可學習參數)
   yᵢ = γ × x̂ᵢ + β
```

### Layer Normalization

**與 Batch Norm 的區別：**
```
Batch Norm: 沿 batch 維度標準化
Layer Norm: 沿 feature 維度標準化

         Batch 維度
            ↓
┌───┬───┬───┬───┐
│ x │ x │ x │ x │ ← BatchNorm (這一行)
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘
    ↑
    LayerNorm (這一列)
```

```python
# PyTorch 實現
layer_norm = nn.LayerNorm(normalized_shape=256)
```

### L1/L2 正則化

```python
# PyTorch 中使用 weight_decay (L2)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             weight_decay=1e-4)  # L2 正則化

# L1 需要手動添加
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm
```

## 6. 優化技術

### 學習率調度

```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
```

### 常用優化器

| 優化器 | 特點 | 適用場景 |
|--------|------|----------|
| **SGD** | 簡單穩定 | 大規模訓練 |
| **Adam** | 自適應學習率 | 大多數情況 |
| **AdamW** | Adam + 權重衰減 | Transformer |
| **LAMB** | 大 batch 訓練 | BERT 預訓練 |

### 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 快速參考

### 架構選擇

| 任務類型 | 推薦架構 |
|----------|----------|
| 影像分類 | CNN (ResNet, EfficientNet) |
| 物體檢測 | CNN (YOLO, Faster R-CNN) |
| 時序預測 | LSTM, GRU, Transformer |
| 文本分類 | Transformer (BERT) |
| 文本生成 | Transformer (GPT) |
| 跨模態 | Vision Transformer |

### 超參數建議

| 參數 | 建議值 |
|------|--------|
| 學習率 | 1e-3 ~ 1e-5 |
| Batch Size | 16, 32, 64, 128 |
| Dropout | 0.1 ~ 0.5 |
| Weight Decay | 1e-4 ~ 1e-2 |
| Warmup Steps | 總步數的 5-10% |

## 資源

### 官方文檔
- [PyTorch 官方文檔](https://pytorch.org/docs/)
- [TensorFlow 官方文檔](https://www.tensorflow.org/api_docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### 學習資源
- [CS231n - CNN for Visual Recognition](https://cs231n.github.io/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Dive into Deep Learning](https://d2l.ai/)

### 論文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [BERT](https://arxiv.org/abs/1810.04805)

## 注意事項

- 從預訓練模型開始，使用遷移學習
- 監控訓練曲線，及早發現過擬合
- 使用混合精度訓練加速
- 合理使用數據增強
- 確保可重現性（設定 random seed）
