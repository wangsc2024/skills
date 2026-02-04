# 機器學習中的熵應用

## 交叉熵損失函數詳解

### 為什麼使用交叉熵而非 MSE？

**問題：Sigmoid + MSE 的梯度消失**

```python
# Sigmoid 函數
σ(z) = 1 / (1 + e^(-z))

# 導數
σ'(z) = σ(z) × (1 - σ(z))

# 當 z 很大或很小時，σ' → 0
# 導致學習停滯
```

**解決方案：交叉熵**

```
L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

∂L/∂z = ŷ - y  # 梯度簡潔，不會消失！
```

### PyTorch 實現細節

```python
import torch
import torch.nn as nn

# 方法 1: BCELoss（需要先 sigmoid）
criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
loss = criterion(sigmoid(logits), targets)

# 方法 2: BCEWithLogitsLoss（數值更穩定）
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)  # 內部做 sigmoid

# 方法 3: 帶權重的 BCE
pos_weight = torch.tensor([5.0])  # 正樣本權重
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 多類別交叉熵

```python
# 方法 1: CrossEntropyLoss（softmax + NLL）
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, class_indices)  # targets 是類別索引

# 方法 2: NLLLoss + LogSoftmax
log_softmax = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()
loss = criterion(log_softmax(logits), class_indices)

# Label Smoothing（正則化技巧）
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## KL 散度在深度學習中的應用

### 1. 變分自編碼器 (VAE)

**VAE 損失函數：**
```
L = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
  = 重建損失 - KL 正則項
```

**KL 散度的解析解（高斯分布）：**
```python
def kl_divergence_gaussian(mu, log_var):
    """
    計算 KL(N(μ, σ²) || N(0, 1))

    公式推導：
    D_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

**完整 VAE 實現：**
```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    """β-VAE 損失函數"""
    # 重建損失
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss
```

### 2. 知識蒸餾 (Knowledge Distillation)

```python
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=3.0, alpha=0.7):
    """
    知識蒸餾損失

    Args:
        student_logits: 學生模型輸出
        teacher_logits: 教師模型輸出
        labels: 真實標籤
        temperature: 溫度參數（軟化分布）
        alpha: 軟標籤損失權重
    """
    # 軟標籤損失（KL 散度）
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
    soft_loss = nn.functional.kl_div(soft_student, soft_targets, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)

    # 硬標籤損失（交叉熵）
    hard_loss = nn.functional.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 3. 強化學習中的 KL 約束 (PPO/TRPO)

```python
def ppo_loss(old_probs, new_probs, advantages, clip_ratio=0.2):
    """
    PPO 裁剪損失

    通過限制策略更新幅度來穩定訓練
    """
    ratio = new_probs / old_probs

    # 裁剪
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # 取較小值（悲觀估計）
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    return loss.mean()

def approximate_kl(old_probs, new_probs):
    """近似 KL 散度（用於早停）"""
    return (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(-1).mean()
```

---

## 決策樹與信息增益

### ID3 算法實現

```python
import numpy as np
from collections import Counter

class DecisionTreeID3:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        """計算熵"""
        counter = Counter(y)
        total = len(y)
        ent = 0
        for count in counter.values():
            p = count / total
            ent -= p * np.log2(p) if p > 0 else 0
        return ent

    def information_gain(self, X, y, feature_idx, threshold):
        """計算信息增益"""
        parent_entropy = self.entropy(y)

        # 分割
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        # 加權子節點熵
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        e_left = self.entropy(y[left_mask])
        e_right = self.entropy(y[right_mask])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def best_split(self, X, y):
        """找最佳分割點"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        """遞歸構建樹"""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # 停止條件
        if (depth >= self.max_depth or
            n_classes == 1 or
            n_samples < self.min_samples_split):
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}

        # 找最佳分割
        feature, threshold, gain = self.best_split(X, y)

        if gain == 0:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}

        # 分割數據
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 遞歸
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self

    def predict_sample(self, x, node):
        if node['leaf']:
            return node['class']

        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        else:
            return self.predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])
```

### 其他分裂指標

**Gini 不純度：**
```python
def gini(y):
    """Gini 不純度"""
    counter = Counter(y)
    total = len(y)
    return 1 - sum((count/total)**2 for count in counter.values())
```

**增益比（C4.5）：**
```python
def gain_ratio(X, y, feature_idx, threshold):
    """增益比 = 信息增益 / 分裂信息"""
    ig = information_gain(X, y, feature_idx, threshold)

    left_mask = X[:, feature_idx] <= threshold
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = n - n_left

    # 分裂信息（避免偏向多值特徵）
    split_info = 0
    if n_left > 0:
        p = n_left / n
        split_info -= p * np.log2(p)
    if n_right > 0:
        p = n_right / n
        split_info -= p * np.log2(p)

    return ig / split_info if split_info > 0 else 0
```

---

## 信息熵在特徵選擇中的應用

### 互信息特徵選擇

```python
from sklearn.feature_selection import mutual_info_classif

def select_features_by_mi(X, y, n_features=10):
    """使用互信息選擇特徵"""
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # 排序
    feature_ranking = np.argsort(mi_scores)[::-1]
    selected_features = feature_ranking[:n_features]

    return selected_features, mi_scores[selected_features]
```

### 最大相關最小冗餘 (mRMR)

```python
def mrmr_selection(X, y, n_features=10):
    """
    mRMR 特徵選擇

    目標：最大化與目標的相關性，最小化特徵間冗餘
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import mutual_info_score

    n_total_features = X.shape[1]
    selected = []
    remaining = list(range(n_total_features))

    # 與目標的互信息
    mi_with_target = mutual_info_classif(X, y, random_state=42)

    for _ in range(n_features):
        best_score = -np.inf
        best_feature = None

        for f in remaining:
            # 相關性項
            relevance = mi_with_target[f]

            # 冗餘項
            redundancy = 0
            if selected:
                for s in selected:
                    redundancy += mutual_info_score(
                        np.digitize(X[:, f], np.percentile(X[:, f], [25, 50, 75])),
                        np.digitize(X[:, s], np.percentile(X[:, s], [25, 50, 75]))
                    )
                redundancy /= len(selected)

            # mRMR 得分
            score = relevance - redundancy

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected
```

---

## GAN 與 KL/JS 散度

### 原始 GAN 損失與 JS 散度

```
原始 GAN 損失實際上最小化 P_data 和 P_G 的 JS 散度：

D_JS(P || Q) = 0.5 × D_KL(P || M) + 0.5 × D_KL(Q || M)

其中 M = 0.5 × (P + Q)
```

### f-GAN（使用各種 f-散度）

```python
def f_divergence_loss(D_real, D_fake, divergence='kl'):
    """
    f-GAN 損失

    Args:
        D_real: 判別器對真實樣本的輸出
        D_fake: 判別器對生成樣本的輸出
        divergence: 'kl', 'reverse_kl', 'js', 'hellinger'
    """
    if divergence == 'kl':
        # KL 散度
        d_loss = -torch.mean(D_real) + torch.mean(torch.exp(D_fake - 1))
        g_loss = -torch.mean(D_fake)

    elif divergence == 'reverse_kl':
        # 反向 KL
        d_loss = torch.mean(torch.exp(D_real)) - 1 - torch.mean(D_fake)
        g_loss = torch.mean(torch.exp(-D_fake))

    elif divergence == 'js':
        # JS 散度
        d_loss = -torch.mean(torch.log(torch.sigmoid(D_real) + 1e-10)) \
                 - torch.mean(torch.log(1 - torch.sigmoid(D_fake) + 1e-10))
        g_loss = -torch.mean(torch.log(torch.sigmoid(D_fake) + 1e-10))

    return d_loss, g_loss
```

### Wasserstein GAN（更穩定）

```python
def wasserstein_loss(D_real, D_fake):
    """
    WGAN 損失

    使用 Earth Mover 距離代替 JS 散度
    """
    d_loss = -torch.mean(D_real) + torch.mean(D_fake)
    g_loss = -torch.mean(D_fake)
    return d_loss, g_loss
```
