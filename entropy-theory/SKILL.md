---
name: entropy-theory
version: "1.0"
description: |
  熵（Entropy）理論與應用的深度智能技能。涵蓋熱力學熵、信息熵、
  交叉熵等核心理論，以及在壓縮、分詞、機器學習、密碼學、決策樹、
  猜數字遊戲等領域的實際應用。當用戶提到熵、entropy、信息論、
  壓縮原理、交叉熵損失、KL散度等概念時觸發此技能。
triggers:
  - "entropy"
  - "熵"
  - "信息熵"
  - "交叉熵"
  - "cross entropy"
  - "KL divergence"
  - "KL散度"
  - "information theory"
  - "信息論"
  - "熱寂"
  - "heat death"
  - "壓縮原理"
  - "information gain"
  - "信息增益"
---

# 熵理論與應用 (Entropy Theory & Applications)

熵是橫跨物理學、信息論、機器學習的核心概念。本技能提供從理論到實踐的完整知識體系。

## 概覽：熵的三大支柱

```
┌─────────────────────────────────────────────────────────────────┐
│                        熵 (Entropy)                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   熱力學熵       │    信息熵        │      應用熵                  │
│ Thermodynamic   │  Information    │    Applied                  │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • 第二定律       │ • Shannon 熵    │ • 壓縮演算法                 │
│ • 增熵原理       │ • 條件熵        │ • 機器學習損失函數           │
│ • 熱寂假說       │ • 互信息        │ • 自然語言處理               │
│ • Boltzmann 公式 │ • KL 散度       │ • 密碼學安全性               │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

---

# 第一部分：熵的理論基礎

## 1. 熱力學熵 (Thermodynamic Entropy)

### 1.1 基本定義

熱力學熵描述系統的**無序程度**，由 Rudolf Clausius 於 1865 年提出。

**Boltzmann 熵公式：**
```
S = k_B × ln(W)

其中：
- S = 熵（單位：J/K）
- k_B = Boltzmann 常數 ≈ 1.38 × 10⁻²³ J/K
- W = 微觀態數量（系統可能的排列組合數）
```

### 1.2 熱力學第二定律

> 孤立系統的熵永不減少。

```
ΔS_universe ≥ 0

• 可逆過程：ΔS = 0
• 不可逆過程：ΔS > 0（自然界絕大多數過程）
```

**生活例子：**
| 過程 | 熵變化 | 說明 |
|------|--------|------|
| 冰融化 | 增加 | 有序晶體 → 無序液體 |
| 氣體擴散 | 增加 | 集中 → 分散 |
| 打碎杯子 | 增加 | 完整 → 碎片 |
| 生命維持 | 局部減少 | 但環境熵增加更多 |

### 1.3 熱寂假說 (Heat Death)

William Thomson（Lord Kelvin）於 1851 年提出：

> 宇宙最終將達到熱力學平衡，熵達到最大值，無法再進行任何有用功。

**時間線：**
```
現在 → 10^14 年後恆星燃盡 → 10^40 年後質子衰變 → 10^100 年後黑洞蒸發 → 熱寂
```

**特徵：**
- 溫度均勻（無溫度梯度）
- 無自由能
- 無生命、無結構
- 宇宙成為均勻的「冷湯」

**爭議：** Max Planck 等科學家質疑「宇宙熵」是否有意義，因為宇宙沒有明確邊界。

---

## 2. 信息熵 (Information Entropy)

### 2.1 Shannon 熵

Claude Shannon 於 1948 年在《通信的數學理論》中提出：

**離散隨機變數的熵：**
```
H(X) = -Σ p(xᵢ) × log₂ p(xᵢ)

其中：
- H(X) = 信息熵（單位：bit）
- p(xᵢ) = 事件 xᵢ 發生的機率
- Σ 對所有可能事件求和
```

**直觀理解：**
- **熵 = 平均不確定性 = 平均信息量**
- 熵越高 → 越隨機 → 需要更多 bit 來描述
- 熵越低 → 越確定 → 可以更簡潔描述

### 2.2 熵的計算範例

**範例 1：公平硬幣**
```python
import math

# 公平硬幣：P(正) = P(反) = 0.5
p_heads = 0.5
p_tails = 0.5

H = -p_heads * math.log2(p_heads) - p_tails * math.log2(p_tails)
# H = -0.5 * (-1) - 0.5 * (-1) = 1 bit

print(f"公平硬幣熵: {H} bit")  # 輸出: 1 bit
```

**範例 2：不公平硬幣**
```python
# 不公平硬幣：P(正) = 0.9, P(反) = 0.1
p_heads = 0.9
p_tails = 0.1

H = -p_heads * math.log2(p_heads) - p_tails * math.log2(p_tails)
# H ≈ 0.469 bit（更可預測，熵更低）

print(f"不公平硬幣熵: {H:.3f} bit")  # 輸出: 0.469 bit
```

**範例 3：骰子**
```python
# 公平骰子：6 面各 1/6 機率
H = -6 * (1/6) * math.log2(1/6)
# H ≈ 2.585 bit

print(f"骰子熵: {H:.3f} bit")  # 輸出: 2.585 bit
```

### 2.3 熵的性質

| 性質 | 數學表達 | 說明 |
|------|----------|------|
| 非負性 | H(X) ≥ 0 | 熵永不為負 |
| 最大熵 | H(X) ≤ log₂(n) | 均勻分布時達最大 |
| 確定性 | H(X) = 0 當且僅當 | 某事件機率為 1 |
| 鏈式法則 | H(X,Y) = H(X) + H(Y\|X) | 聯合熵分解 |

### 2.4 條件熵 (Conditional Entropy)

給定 Y 後 X 的不確定性：

```
H(X|Y) = -Σᵢ Σⱼ p(xᵢ, yⱼ) × log₂ p(xᵢ|yⱼ)
```

**意義：** 知道 Y 後，X 還有多少「驚喜」？

### 2.5 互信息 (Mutual Information)

X 和 Y 共享的信息量：

```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

**直觀：** 知道 Y 後，X 的不確定性減少了多少？

```
        ┌───────────┐
        │   H(X)    │
        │   ┌───┐   │
        │   │I(X;Y) │◄── 共享信息
        │   └───┘   │
        │           │
        └───────────┘
```

---

## 3. 交叉熵與 KL 散度

### 3.1 交叉熵 (Cross-Entropy)

衡量用分布 Q 來編碼來自分布 P 的數據所需的平均 bit 數：

```
H(P, Q) = -Σ p(x) × log₂ q(x)
```

**與熵的關係：**
```
H(P, Q) = H(P) + D_KL(P || Q)

交叉熵 = 真實熵 + 額外代價（KL 散度）
```

### 3.2 KL 散度 (Kullback-Leibler Divergence)

又稱**相對熵**，衡量兩個分布的差異：

```
D_KL(P || Q) = Σ p(x) × log₂(p(x) / q(x))
```

**重要性質：**
- D_KL ≥ 0（非負）
- D_KL = 0 當且僅當 P = Q
- **不對稱**：D_KL(P||Q) ≠ D_KL(Q||P)

**Python 計算：**
```python
import numpy as np
from scipy.special import rel_entr

def kl_divergence(p, q):
    """計算 KL 散度"""
    return np.sum(rel_entr(p, q))

# 範例
p = np.array([0.4, 0.3, 0.3])
q = np.array([0.33, 0.33, 0.34])

kl = kl_divergence(p, q)
print(f"KL(P||Q) = {kl:.4f}")
```

---

# 第二部分：熵的應用

## 4. 壓縮演算法

### 4.1 理論極限

**Shannon 信源編碼定理：**
> 無損壓縮的平均碼長不能低於信源的熵。

```
平均碼長 ≥ H(X)
```

**實際意義：**
- 英文字母熵約 4.03 bit/字符
- 考慮上下文後約 1-1.5 bit/字符
- 因此英文可壓縮到原大小的 12-19%

### 4.2 Huffman 編碼

根據字符頻率構建最優前綴碼：

```python
import heapq
from collections import Counter

def huffman_encoding(text):
    """Huffman 編碼實現"""
    # 統計頻率
    freq = Counter(text)

    # 構建優先隊列
    heap = [[count, [char, ""]] for char, count in freq.items()]
    heapq.heapify(heap)

    # 構建 Huffman 樹
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return dict(heap[0][1:])

# 範例
text = "AAAAABBBCCCD"
codes = huffman_encoding(text)
print(codes)
# {'A': '0', 'B': '10', 'C': '110', 'D': '111'}

# 壓縮率計算
original_bits = len(text) * 8  # 假設 ASCII
compressed_bits = sum(len(codes[c]) for c in text)
ratio = compressed_bits / original_bits
print(f"壓縮率: {ratio:.2%}")
```

### 4.3 算術編碼 (Arithmetic Coding)

比 Huffman 更接近熵極限：

```
壓縮後大小 ≈ H(X) × 訊息長度
```

**優勢：**
- 可以用小數 bit（如 1.3 bit/符號）
- 更適合機率分布不均的情況

---

## 5. 機器學習中的熵

### 5.1 交叉熵損失函數

**二元分類（Binary Cross-Entropy）：**
```python
import torch
import torch.nn as nn

# 預測值（經過 sigmoid）
predictions = torch.tensor([0.9, 0.2, 0.8])
# 真實標籤
targets = torch.tensor([1.0, 0.0, 1.0])

# 交叉熵損失
bce_loss = nn.BCELoss()
loss = bce_loss(predictions, targets)
print(f"BCE Loss: {loss.item():.4f}")
```

**多類分類（Categorical Cross-Entropy）：**
```python
# 預測 logits
logits = torch.tensor([[2.0, 0.5, 0.1],
                       [0.1, 3.0, 0.2]])
# 真實類別
targets = torch.tensor([0, 1])

# 交叉熵損失（含 softmax）
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, targets)
print(f"CE Loss: {loss.item():.4f}")
```

**為什麼交叉熵有效？**
```
最小化交叉熵 ⟺ 最小化 KL 散度 ⟺ 讓模型分布接近真實分布
```

### 5.2 決策樹與信息增益

**信息增益 (Information Gain)：**
```
IG(D, A) = H(D) - H(D|A)
        = 父節點熵 - 子節點加權平均熵
```

```python
import numpy as np

def entropy(y):
    """計算熵"""
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def information_gain(X_column, y, threshold):
    """計算信息增益"""
    # 父節點熵
    parent_entropy = entropy(y)

    # 分割數據
    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0

    # 子節點加權熵
    n = len(y)
    n_left, n_right = np.sum(left_mask), np.sum(right_mask)
    e_left, e_right = entropy(y[left_mask]), entropy(y[right_mask])
    child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

    return parent_entropy - child_entropy

# 範例：根據「是否有房」預測「是否貸款」
y = np.array([1, 1, 1, 0, 0, 0, 1, 0])  # 貸款結果
has_house = np.array([1, 1, 0, 0, 0, 1, 1, 0])  # 是否有房

ig = information_gain(has_house, y, 0.5)
print(f"信息增益: {ig:.4f}")
```

**ID3/C4.5 算法選擇：**
- 選擇信息增益最大的特徵作為分裂點
- 遞歸構建決策樹

### 5.3 VAE 中的 KL 散度

**VAE 損失函數：**
```
L = 重建損失 + β × KL(q(z|x) || p(z))

其中：
- q(z|x) = 編碼器輸出的潛在分布
- p(z) = 先驗分布（通常是 N(0, I)）
- β = 平衡係數
```

```python
def vae_loss(x_recon, x, mu, log_var):
    """VAE 損失函數"""
    # 重建損失
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 散度（解析解）
    # KL(N(μ, σ²) || N(0, 1)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss
```

---

## 6. 自然語言處理中的熵

### 6.1 語言模型困惑度 (Perplexity)

```
PPL = 2^H(P, Q) = 2^(-1/N × Σ log₂ P(wᵢ|w₁...wᵢ₋₁))
```

**直觀理解：**
- PPL = 100 → 平均每個詞有 100 個等可能選項
- PPL 越低 → 模型越好

```python
import math

def perplexity(log_probs):
    """計算困惑度"""
    avg_log_prob = sum(log_probs) / len(log_probs)
    return 2 ** (-avg_log_prob)

# 範例
log_probs = [-2.5, -3.0, -2.0, -4.0]  # 對數機率
ppl = perplexity(log_probs)
print(f"Perplexity: {ppl:.2f}")
```

### 6.2 基於熵的中文分詞

**新詞發現原理：**
- **左右熵**：詞語兩側出現字符的多樣性
- **互信息**：詞內部字符的黏合度

```python
from collections import defaultdict
import math

def calculate_entropy(counter):
    """計算熵"""
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((c/total) * math.log2(c/total)
                for c in counter.values() if c > 0)

def find_new_words(text, min_count=5, min_entropy=1.0):
    """基於熵的新詞發現"""
    # 統計 n-gram
    ngrams = defaultdict(int)
    left_chars = defaultdict(lambda: defaultdict(int))
    right_chars = defaultdict(lambda: defaultdict(int))

    for n in range(2, 6):  # 2-5 字詞
        for i in range(len(text) - n + 1):
            word = text[i:i+n]
            ngrams[word] += 1

            # 記錄左右字符
            if i > 0:
                left_chars[word][text[i-1]] += 1
            if i + n < len(text):
                right_chars[word][text[i+n]] += 1

    # 篩選候選詞
    candidates = []
    for word, count in ngrams.items():
        if count < min_count:
            continue

        left_entropy = calculate_entropy(left_chars[word])
        right_entropy = calculate_entropy(right_chars[word])

        # 左右熵都要高
        if left_entropy >= min_entropy and right_entropy >= min_entropy:
            candidates.append({
                'word': word,
                'count': count,
                'left_entropy': left_entropy,
                'right_entropy': right_entropy
            })

    return sorted(candidates, key=lambda x: x['count'], reverse=True)

# 範例
text = "自然語言處理是人工智能的重要分支自然語言處理技術日益成熟"
words = find_new_words(text, min_count=2, min_entropy=0.5)
for w in words[:5]:
    print(f"{w['word']}: 頻次={w['count']}, 左熵={w['left_entropy']:.2f}, 右熵={w['right_entropy']:.2f}")
```

### 6.3 文本分類的熵特徵

```python
def text_entropy(text):
    """計算文本字符熵"""
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    total = len(text)
    entropy = 0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

# 高熵文本（隨機）vs 低熵文本（重複）
random_text = "asdfjkl;qwertyuiopzxcvbnm"
repetitive_text = "aaaaaabbbbbbcccccc"

print(f"隨機文本熵: {text_entropy(random_text):.2f}")
print(f"重複文本熵: {text_entropy(repetitive_text):.2f}")
```

---

## 7. 密碼學中的熵

### 7.1 密碼強度計算

**密碼熵公式：**
```
E = log₂(R^L) = L × log₂(R)

其中：
- E = 熵（bits）
- R = 字符集大小
- L = 密碼長度
```

| 字符集 | R | 8字符熵 | 12字符熵 |
|--------|---|---------|----------|
| 純數字 | 10 | 26.6 bits | 39.9 bits |
| 小寫字母 | 26 | 37.6 bits | 56.4 bits |
| 大小寫字母 | 52 | 45.6 bits | 68.4 bits |
| 字母+數字+符號 | 95 | 52.6 bits | 78.8 bits |

**安全建議：**
```
< 28 bits  → 極弱（秒破）
28-35 bits → 弱
36-59 bits → 中等
60-127 bits → 強
≥ 128 bits → 極強
```

```python
import math
import string

def password_entropy(password):
    """計算密碼熵"""
    charset_size = 0

    if any(c.islower() for c in password):
        charset_size += 26
    if any(c.isupper() for c in password):
        charset_size += 26
    if any(c.isdigit() for c in password):
        charset_size += 10
    if any(c in string.punctuation for c in password):
        charset_size += 32

    if charset_size == 0:
        return 0

    entropy = len(password) * math.log2(charset_size)
    return entropy

# 範例
passwords = [
    "password",
    "P@ssw0rd",
    "correct-horse-battery-staple",
    "Tr0ub4dor&3"
]

for pwd in passwords:
    e = password_entropy(pwd)
    print(f"{pwd}: {e:.1f} bits")
```

### 7.2 隨機數生成

**真隨機 vs 偽隨機：**
- **真隨機**：熱噪聲、放射性衰變（硬體 RNG）
- **偽隨機**：CSPRNG（如 `/dev/urandom`）

```python
import secrets
import os

# 生成密碼學安全的隨機數
random_bytes = secrets.token_bytes(32)  # 256 bits 熵
random_hex = secrets.token_hex(16)      # 128 bits 熵
random_url = secrets.token_urlsafe(16)  # URL 安全的隨機字串

print(f"Hex token: {random_hex}")
print(f"URL-safe token: {random_url}")
```

---

## 8. 博弈論與猜測遊戲

### 8.1 猜數字遊戲

**最優策略：二分搜尋**

```
目標：在 1-100 中猜數字
最優策略：每次猜中點

平均猜測次數 = log₂(100) ≈ 6.64 次
最差情況 = ⌈log₂(100)⌉ = 7 次
```

**資訊增益分析：**
```python
import math

def expected_guesses(n):
    """猜 1-n 的期望次數"""
    return math.log2(n)

def binary_search_guesses(n):
    """二分搜尋的最差次數"""
    return math.ceil(math.log2(n))

for n in [10, 100, 1000, 1000000]:
    exp = expected_guesses(n)
    worst = binary_search_guesses(n)
    print(f"1-{n}: 期望 {exp:.2f} 次, 最差 {worst} 次")
```

### 8.2 二十問遊戲 (20 Questions)

**策略：最大化信息增益**

```python
def best_question(candidates, questions):
    """選擇最佳問題（最大信息增益）"""
    best_ig = 0
    best_q = None

    for question, answers in questions.items():
        # 計算分割後的熵
        yes_count = sum(1 for c in candidates if answers.get(c, False))
        no_count = len(candidates) - yes_count

        if yes_count == 0 or no_count == 0:
            continue

        # 信息增益 ≈ 越接近 50/50 越好
        p_yes = yes_count / len(candidates)
        p_no = no_count / len(candidates)

        # 條件熵
        H_after = -p_yes * math.log2(p_yes) - p_no * math.log2(p_no) if p_yes > 0 and p_no > 0 else 0

        if H_after > best_ig:
            best_ig = H_after
            best_q = question

    return best_q, best_ig
```

### 8.3 Wordle 策略

**熵最優開局詞：**
```
根據信息熵分析，最佳開局詞能最大化期望信息增益：
- "SALET" - 信息熵 ≈ 5.87 bits
- "REAST" - 信息熵 ≈ 5.84 bits
- "CRATE" - 信息熵 ≈ 5.83 bits
```

---

## 9. 熵在其他領域的應用

### 9.1 圖像處理

**圖像熵 = 像素分布的不確定性**

```python
import numpy as np
from PIL import Image

def image_entropy(image_path):
    """計算圖像熵"""
    img = Image.open(image_path).convert('L')  # 轉灰度
    histogram = img.histogram()

    total = sum(histogram)
    entropy = 0
    for count in histogram:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy

# 高熵圖像：細節豐富、噪聲
# 低熵圖像：大面積純色、簡單
```

### 9.2 異常檢測

**原理：異常數據會增加熵**

```python
def detect_anomaly_by_entropy(data, window_size=100, threshold=2.0):
    """基於熵變化的異常檢測"""
    anomalies = []

    for i in range(window_size, len(data)):
        window = data[i-window_size:i]

        # 計算窗口熵
        hist, _ = np.histogram(window, bins=50)
        hist = hist / hist.sum()
        entropy = -np.sum(h * np.log2(h + 1e-10) for h in hist if h > 0)

        # 如果熵突變，標記為異常
        if i > window_size:
            if abs(entropy - prev_entropy) > threshold:
                anomalies.append(i)

        prev_entropy = entropy

    return anomalies
```

### 9.3 網路流量分析

```python
def packet_entropy(packet_sizes):
    """計算封包大小分布的熵"""
    hist = {}
    for size in packet_sizes:
        hist[size] = hist.get(size, 0) + 1

    total = len(packet_sizes)
    entropy = 0
    for count in hist.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

# 正常流量：中等熵（有模式但有變化）
# DDoS 攻擊：低熵（大量相同大小封包）
# 加密流量：高熵（近似隨機）
```

---

# 第三部分：實用工具與範例

## 10. Python 熵計算工具包

```python
"""
entropy_toolkit.py - 熵計算實用工具
"""

import math
import numpy as np
from collections import Counter
from typing import List, Union, Dict

class EntropyToolkit:
    """熵計算工具類"""

    @staticmethod
    def shannon_entropy(data: Union[List, str, np.ndarray]) -> float:
        """
        計算 Shannon 熵

        Args:
            data: 離散數據（列表、字串或數組）

        Returns:
            float: 熵值（bits）
        """
        counter = Counter(data)
        total = len(data)

        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """
        計算交叉熵 H(P, Q)

        Args:
            p: 真實分布
            q: 預測分布

        Returns:
            float: 交叉熵值
        """
        # 避免 log(0)
        q = np.clip(q, 1e-15, 1)
        return -np.sum(p * np.log2(q))

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        計算 KL 散度 D_KL(P || Q)

        Args:
            p: 真實分布
            q: 近似分布

        Returns:
            float: KL 散度
        """
        p = np.clip(p, 1e-15, 1)
        q = np.clip(q, 1e-15, 1)
        return np.sum(p * np.log2(p / q))

    @staticmethod
    def conditional_entropy(joint_probs: np.ndarray) -> float:
        """
        計算條件熵 H(X|Y)

        Args:
            joint_probs: 聯合機率表 P(X, Y)

        Returns:
            float: 條件熵
        """
        # P(Y)
        p_y = joint_probs.sum(axis=0)

        H = 0
        for j in range(joint_probs.shape[1]):
            if p_y[j] > 0:
                # P(X|Y=j)
                p_x_given_y = joint_probs[:, j] / p_y[j]
                H += p_y[j] * EntropyToolkit.shannon_entropy(p_x_given_y)

        return H

    @staticmethod
    def mutual_information(joint_probs: np.ndarray) -> float:
        """
        計算互信息 I(X; Y)

        Args:
            joint_probs: 聯合機率表 P(X, Y)

        Returns:
            float: 互信息
        """
        # 邊際分布
        p_x = joint_probs.sum(axis=1)
        p_y = joint_probs.sum(axis=0)

        # H(X)
        H_X = -np.sum(p_x * np.log2(p_x + 1e-15))

        # H(X|Y)
        H_X_given_Y = EntropyToolkit.conditional_entropy(joint_probs)

        return H_X - H_X_given_Y

    @staticmethod
    def password_entropy(password: str) -> Dict[str, float]:
        """
        計算密碼熵

        Args:
            password: 密碼字串

        Returns:
            dict: 包含熵值和強度評估
        """
        import string

        charset_size = 0
        if any(c.islower() for c in password):
            charset_size += 26
        if any(c.isupper() for c in password):
            charset_size += 26
        if any(c.isdigit() for c in password):
            charset_size += 10
        if any(c in string.punctuation for c in password):
            charset_size += 32
        if any(c == ' ' for c in password):
            charset_size += 1

        if charset_size == 0:
            return {'entropy': 0, 'strength': 'empty'}

        entropy = len(password) * math.log2(charset_size)

        # 強度評估
        if entropy < 28:
            strength = 'very_weak'
        elif entropy < 36:
            strength = 'weak'
        elif entropy < 60:
            strength = 'moderate'
        elif entropy < 128:
            strength = 'strong'
        else:
            strength = 'very_strong'

        return {
            'entropy': entropy,
            'charset_size': charset_size,
            'length': len(password),
            'strength': strength
        }


# 使用範例
if __name__ == "__main__":
    toolkit = EntropyToolkit()

    # Shannon 熵
    text = "hello world"
    print(f"Text entropy: {toolkit.shannon_entropy(text):.4f} bits")

    # 交叉熵
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    print(f"Cross-entropy: {toolkit.cross_entropy(p, q):.4f}")

    # KL 散度
    print(f"KL divergence: {toolkit.kl_divergence(p, q):.4f}")

    # 密碼熵
    passwords = ["password", "P@ssw0rd!", "correct horse battery staple"]
    for pwd in passwords:
        result = toolkit.password_entropy(pwd)
        print(f"'{pwd}': {result['entropy']:.1f} bits ({result['strength']})")
```

---

## 11. 快速參考表

### 熵公式速查

| 概念 | 公式 | 用途 |
|------|------|------|
| Shannon 熵 | H(X) = -Σ p(x) log₂ p(x) | 衡量不確定性 |
| 條件熵 | H(X\|Y) = -Σ p(x,y) log₂ p(x\|y) | Y 已知時 X 的不確定性 |
| 聯合熵 | H(X,Y) = -Σ p(x,y) log₂ p(x,y) | 兩變數總不確定性 |
| 互信息 | I(X;Y) = H(X) - H(X\|Y) | 共享信息量 |
| 交叉熵 | H(P,Q) = -Σ p(x) log₂ q(x) | 用 Q 編碼 P 的代價 |
| KL 散度 | D_KL(P\|\|Q) = Σ p(x) log₂(p/q) | 分布差異 |

### 應用場景速查

| 領域 | 熵的應用 | 關鍵技術 |
|------|----------|----------|
| 壓縮 | 無損壓縮極限 | Huffman, 算術編碼 |
| ML 分類 | 損失函數 | 交叉熵損失 |
| 決策樹 | 特徵選擇 | 信息增益, Gini |
| NLP | 語言模型評估 | Perplexity |
| 分詞 | 新詞發現 | 左右熵, 互信息 |
| 密碼學 | 安全性度量 | 密碼熵, 隨機性 |
| 生成模型 | VAE, GAN | KL 散度正則化 |
| 遊戲策略 | 最優猜測 | 信息增益最大化 |

---

## 參考資源

### 經典論文
- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Kullback, S. & Leibler, R. A. (1951). "On Information and Sufficiency"

### 推薦書籍
- "Elements of Information Theory" - Cover & Thomas
- "Information Theory, Inference and Learning Algorithms" - MacKay

### 線上資源
- [Wikipedia: Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Stanford Entropy Course](https://ee.stanford.edu/~gray/it.pdf)
- [3Blue1Brown: Information Theory](https://www.youtube.com/watch?v=v68zYyaEmEA)

---

**Generated by Skill Seekers** | Entropy Theory & Applications | v1.0
