# 自然語言處理中的熵應用

## 語言模型與困惑度

### 語言模型的熵

給定語言 L 和模型 M，語言的交叉熵：

```
H(L, M) = lim(n→∞) -1/n × Σ P(w₁w₂...wₙ) × log₂ M(w₁w₂...wₙ)
```

**實際計算（使用測試集近似）：**
```python
def model_cross_entropy(model, test_sentences):
    """計算語言模型在測試集上的交叉熵"""
    total_log_prob = 0
    total_tokens = 0

    for sentence in test_sentences:
        tokens = tokenize(sentence)
        for i, token in enumerate(tokens):
            # 條件機率
            context = tokens[:i]
            prob = model.probability(token, context)

            total_log_prob += np.log2(prob + 1e-10)
            total_tokens += 1

    return -total_log_prob / total_tokens
```

### 困惑度 (Perplexity)

```
PPL = 2^H(L, M)
```

**直觀理解：**
- PPL = 100 意味著模型平均對每個詞有 100 個等可能的選擇
- PPL 越低，模型越好

```python
def perplexity(model, test_corpus):
    """計算困惑度"""
    cross_entropy = model_cross_entropy(model, test_corpus)
    return 2 ** cross_entropy

# 不同模型的典型 PPL
# - 隨機猜測（50k 詞彙）: PPL ≈ 50,000
# - Unigram: PPL ≈ 1000
# - Bigram: PPL ≈ 200-400
# - Trigram + smoothing: PPL ≈ 100-200
# - RNN/LSTM: PPL ≈ 50-100
# - GPT-2: PPL ≈ 20-30
# - GPT-4: PPL ≈ 10-15
```

---

## 基於熵的分詞

### 理論基礎

**假設：**
- 詞的內部凝聚度高（互信息高）
- 詞的邊界自由度高（左右熵高）

### 互信息（凝聚度）

```python
def pointwise_mutual_information(word, corpus):
    """
    計算詞的點互信息（衡量內部凝聚度）

    PMI(xy) = log₂(P(xy) / (P(x) × P(y)))
    """
    # 統計頻率
    word_freq = corpus.count(word)
    total = len(corpus)

    if len(word) < 2:
        return 0

    # 計算所有分割點的 PMI，取最小值
    min_pmi = float('inf')

    for i in range(1, len(word)):
        left = word[:i]
        right = word[i:]

        p_word = word_freq / total
        p_left = corpus.count(left) / total
        p_right = corpus.count(right) / total

        if p_left > 0 and p_right > 0:
            pmi = np.log2(p_word / (p_left * p_right + 1e-10))
            min_pmi = min(min_pmi, pmi)

    return min_pmi if min_pmi != float('inf') else 0
```

### 左右熵（邊界自由度）

```python
from collections import defaultdict
import math

class EntropySegmenter:
    def __init__(self, corpus, min_freq=5, min_entropy=1.0, min_pmi=2.0):
        self.corpus = corpus
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_pmi = min_pmi

        self.word_freq = defaultdict(int)
        self.left_chars = defaultdict(lambda: defaultdict(int))
        self.right_chars = defaultdict(lambda: defaultdict(int))

    def calculate_entropy(self, char_freq):
        """計算熵"""
        total = sum(char_freq.values())
        if total == 0:
            return 0

        entropy = 0
        for count in char_freq.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0

        return entropy

    def extract_ngrams(self, max_n=5):
        """提取 n-gram 及其左右字符"""
        text = self.corpus

        for n in range(2, max_n + 1):
            for i in range(len(text) - n + 1):
                word = text[i:i+n]
                self.word_freq[word] += 1

                # 左字符
                if i > 0:
                    self.left_chars[word][text[i-1]] += 1

                # 右字符
                if i + n < len(text):
                    self.right_chars[word][text[i+n]] += 1

    def find_words(self):
        """發現新詞"""
        self.extract_ngrams()

        candidates = []

        for word, freq in self.word_freq.items():
            if freq < self.min_freq:
                continue

            # 計算左右熵
            left_entropy = self.calculate_entropy(self.left_chars[word])
            right_entropy = self.calculate_entropy(self.right_chars[word])

            # 計算凝聚度（簡化版 PMI）
            pmi = self.calculate_cohesion(word)

            # 篩選條件
            if (left_entropy >= self.min_entropy and
                right_entropy >= self.min_entropy and
                pmi >= self.min_pmi):

                candidates.append({
                    'word': word,
                    'freq': freq,
                    'left_entropy': left_entropy,
                    'right_entropy': right_entropy,
                    'cohesion': pmi
                })

        # 按頻率排序
        return sorted(candidates, key=lambda x: x['freq'], reverse=True)

    def calculate_cohesion(self, word):
        """計算凝聚度"""
        if len(word) < 2:
            return 0

        freq = self.word_freq[word]
        total = len(self.corpus)

        min_cohesion = float('inf')

        for i in range(1, len(word)):
            left = word[:i]
            right = word[i:]

            p_word = freq / total
            p_left = self.word_freq.get(left, 0) / total
            p_right = self.word_freq.get(right, 0) / total

            if p_left > 0 and p_right > 0:
                cohesion = math.log2(p_word / (p_left * p_right + 1e-10))
                min_cohesion = min(min_cohesion, cohesion)

        return min_cohesion if min_cohesion != float('inf') else 0


# 使用範例
corpus = """
自然語言處理是人工智能的重要分支
自然語言處理技術日益成熟
人工智能改變了我們的生活
深度學習推動了自然語言處理的發展
""" * 100  # 重複以增加頻率

segmenter = EntropySegmenter(corpus, min_freq=50, min_entropy=0.5, min_pmi=1.0)
new_words = segmenter.find_words()

print("發現的新詞：")
for w in new_words[:10]:
    print(f"  {w['word']}: 頻率={w['freq']}, "
          f"左熵={w['left_entropy']:.2f}, "
          f"右熵={w['right_entropy']:.2f}, "
          f"凝聚度={w['cohesion']:.2f}")
```

---

## 最大熵模型

### 理論基礎

**最大熵原則：**
> 在滿足已知約束的情況下，選擇熵最大的分布。

**數學形式：**
```
P*(x) = argmax_P H(P)
        subject to: E_P[f_i] = E_empirical[f_i]
```

### 最大熵分類器實現

```python
import numpy as np
from scipy.optimize import minimize

class MaxEntClassifier:
    """最大熵分類器（Logistic Regression 的等價形式）"""

    def __init__(self, max_iter=100, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None

    def feature_function(self, x, y):
        """特徵函數"""
        # 簡化：將 x 和 y 拼接
        features = np.zeros(len(x) * self.n_classes)
        start = y * len(x)
        features[start:start + len(x)] = x
        return features

    def compute_probabilities(self, x):
        """計算條件機率 P(y|x)"""
        scores = []
        for y in range(self.n_classes):
            features = self.feature_function(x, y)
            score = np.dot(self.weights, features)
            scores.append(score)

        # Softmax
        scores = np.array(scores)
        scores -= np.max(scores)  # 數值穩定性
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        return probs

    def fit(self, X, y):
        """訓練模型"""
        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        self.weights = np.zeros(n_features * self.n_classes)

        for iteration in range(self.max_iter):
            gradient = np.zeros_like(self.weights)

            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]

                # 經驗分布的期望
                empirical = self.feature_function(x_i, y_i)

                # 模型分布的期望
                probs = self.compute_probabilities(x_i)
                model_expect = np.zeros_like(self.weights)
                for c in range(self.n_classes):
                    model_expect += probs[c] * self.feature_function(x_i, c)

                # 梯度 = 經驗期望 - 模型期望
                gradient += empirical - model_expect

            # 更新權重
            self.weights += self.learning_rate * gradient / len(X)

        return self

    def predict(self, X):
        """預測"""
        predictions = []
        for x in X:
            probs = self.compute_probabilities(x)
            predictions.append(np.argmax(probs))
        return np.array(predictions)
```

### 最大熵馬爾可夫模型 (MEMM)

用於序列標註（POS tagging, NER）：

```python
class MEMM:
    """最大熵馬爾可夫模型"""

    def __init__(self):
        self.feature_weights = {}
        self.labels = set()

    def extract_features(self, sentence, position, prev_label):
        """提取特徵"""
        features = []
        word = sentence[position]

        # 當前詞特徵
        features.append(f'word={word}')
        features.append(f'prev_label={prev_label}')

        # 詞形特徵
        if word[0].isupper():
            features.append('is_capitalized')
        if word.isdigit():
            features.append('is_digit')

        # 上下文特徵
        if position > 0:
            features.append(f'prev_word={sentence[position-1]}')
        if position < len(sentence) - 1:
            features.append(f'next_word={sentence[position+1]}')

        # 前綴後綴
        features.append(f'prefix2={word[:2]}')
        features.append(f'suffix2={word[-2:]}')

        return features

    def compute_transition_prob(self, features, label):
        """計算轉移機率"""
        score = sum(self.feature_weights.get((f, label), 0) for f in features)
        return score

    def viterbi(self, sentence):
        """Viterbi 解碼"""
        n = len(sentence)
        labels = list(self.labels)

        # DP 表
        dp = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # 初始化
        features = self.extract_features(sentence, 0, '<START>')
        for label in labels:
            dp[0][label] = self.compute_transition_prob(features, label)
            backpointer[0][label] = '<START>'

        # 遞推
        for i in range(1, n):
            for label in labels:
                best_score = float('-inf')
                best_prev = None

                for prev_label in labels:
                    features = self.extract_features(sentence, i, prev_label)
                    score = dp[i-1][prev_label] + self.compute_transition_prob(features, label)

                    if score > best_score:
                        best_score = score
                        best_prev = prev_label

                dp[i][label] = best_score
                backpointer[i][label] = best_prev

        # 回溯
        best_final = max(dp[n-1], key=dp[n-1].get)
        path = [best_final]

        for i in range(n-1, 0, -1):
            path.append(backpointer[i][path[-1]])

        return list(reversed(path))
```

---

## 文本熵分析

### 字符級熵

```python
def character_entropy(text):
    """計算文本的字符級熵"""
    from collections import Counter

    freq = Counter(text)
    total = len(text)

    entropy = 0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

# 不同語言的字符熵比較
# 英文：約 4.0-4.5 bits/字符
# 中文：約 9.5-10 bits/字符（但每字符信息量更大）
# 隨機文本：log₂(字符集大小) bits/字符
```

### 詞級熵

```python
def word_entropy(text, tokenizer=None):
    """計算文本的詞級熵"""
    if tokenizer is None:
        words = text.split()
    else:
        words = tokenizer(text)

    freq = Counter(words)
    total = len(words)

    entropy = 0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

# 英文詞熵：約 8-10 bits/詞
```

### 條件熵（n-gram）

```python
def conditional_entropy(text, n=2):
    """計算 n-gram 條件熵 H(Xₙ|X₁...Xₙ₋₁)"""
    # 統計 n-gram
    ngram_freq = Counter()
    context_freq = Counter()

    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        context = text[i:i+n-1]

        ngram_freq[ngram] += 1
        context_freq[context] += 1

    # 計算條件熵
    total = sum(ngram_freq.values())
    entropy = 0

    for ngram, count in ngram_freq.items():
        context = ngram[:-1]
        p_joint = count / total
        p_context = context_freq[context] / total

        if p_context > 0:
            p_conditional = count / context_freq[context]
            entropy -= p_joint * math.log2(p_conditional)

    return entropy

# Shannon 的英文實驗結果：
# n=1: H ≈ 4.03 bits/字符
# n=2: H ≈ 2.8 bits/字符
# n→∞: H ≈ 1.0-1.3 bits/字符
```

---

## 文本複雜度與熵

### 閱讀難度估計

```python
def text_complexity_by_entropy(text):
    """使用熵估計文本複雜度"""
    # 字符熵
    char_ent = character_entropy(text)

    # 詞熵
    word_ent = word_entropy(text)

    # 詞彙多樣性
    words = text.split()
    type_token_ratio = len(set(words)) / len(words)

    # 綜合得分
    complexity = (char_ent / 5 + word_ent / 12 + type_token_ratio) / 3

    return {
        'char_entropy': char_ent,
        'word_entropy': word_ent,
        'type_token_ratio': type_token_ratio,
        'complexity_score': complexity
    }
```

### 文本壓縮性分析

```python
import zlib

def compression_entropy_ratio(text):
    """使用壓縮估計熵"""
    text_bytes = text.encode('utf-8')
    original_size = len(text_bytes)

    compressed = zlib.compress(text_bytes, level=9)
    compressed_size = len(compressed)

    # 壓縮比近似於熵比
    compression_ratio = compressed_size / original_size

    # 估計的每字節熵（假設最大 8 bits/byte）
    estimated_entropy = compression_ratio * 8

    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'estimated_entropy_per_byte': estimated_entropy
    }
```

---

## 主題模型與熵

### LDA 與信息論

LDA 使用 KL 散度來衡量文檔-主題和主題-詞分布：

```python
def topic_coherence_entropy(topic_word_dist, word_cooccurrence):
    """基於熵的主題連貫性"""
    coherence_scores = []

    for topic in topic_word_dist:
        # 取 top-k 詞
        top_words = np.argsort(topic)[-10:]

        # 計算詞對的 PMI
        pmi_sum = 0
        count = 0

        for i, w1 in enumerate(top_words):
            for w2 in top_words[i+1:]:
                p_w1_w2 = word_cooccurrence[w1, w2]
                p_w1 = topic[w1]
                p_w2 = topic[w2]

                if p_w1 > 0 and p_w2 > 0 and p_w1_w2 > 0:
                    pmi = np.log2(p_w1_w2 / (p_w1 * p_w2))
                    pmi_sum += pmi
                    count += 1

        if count > 0:
            coherence_scores.append(pmi_sum / count)

    return np.mean(coherence_scores)
```

### 主題多樣性熵

```python
def topic_diversity(doc_topic_distributions):
    """計算文檔集的主題多樣性"""
    # 平均主題分布
    mean_dist = np.mean(doc_topic_distributions, axis=0)

    # 熵（多樣性度量）
    entropy = -np.sum(mean_dist * np.log2(mean_dist + 1e-10))

    # 最大可能熵
    n_topics = len(mean_dist)
    max_entropy = np.log2(n_topics)

    # 正規化多樣性得分
    diversity = entropy / max_entropy

    return diversity
```
