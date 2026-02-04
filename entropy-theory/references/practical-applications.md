# 熵的實用應用

## 數據壓縮

### Shannon 信源編碼定理

**定理：**
對於無記憶信源，存在編碼使得平均碼長 L 滿足：
```
H(X) ≤ L < H(X) + 1
```

**意義：**
- 熵是壓縮的理論下限
- 不可能壓縮到比熵更小
- 但可以逼近熵

### Huffman 編碼完整實現

```python
import heapq
from collections import Counter
from typing import Dict, Tuple

class HuffmanNode:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoding:
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}
        self.root = None

    def build_tree(self, text: str) -> None:
        """構建 Huffman 樹"""
        # 統計頻率
        freq = Counter(text)

        # 建立優先隊列
        heap = [HuffmanNode(char, f) for char, f in freq.items()]
        heapq.heapify(heap)

        # 合併節點直到只剩一個
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = HuffmanNode(freq=left.freq + right.freq)
            merged.left = left
            merged.right = right

            heapq.heappush(heap, merged)

        self.root = heap[0] if heap else None

    def _generate_codes(self, node: HuffmanNode, code: str) -> None:
        """遞歸生成編碼"""
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = code
            self.reverse_codes[code] = node.char
            return

        self._generate_codes(node.left, code + '0')
        self._generate_codes(node.right, code + '1')

    def encode(self, text: str) -> Tuple[str, Dict[str, str]]:
        """編碼文本"""
        self.build_tree(text)
        self._generate_codes(self.root, '')

        encoded = ''.join(self.codes[char] for char in text)
        return encoded, self.codes

    def decode(self, encoded: str) -> str:
        """解碼"""
        decoded = []
        current = self.root

        for bit in encoded:
            current = current.left if bit == '0' else current.right

            if current.char is not None:
                decoded.append(current.char)
                current = self.root

        return ''.join(decoded)

    def compression_stats(self, text: str) -> Dict:
        """壓縮統計"""
        encoded, codes = self.encode(text)

        original_bits = len(text) * 8  # ASCII
        compressed_bits = len(encoded)

        # 計算熵
        freq = Counter(text)
        total = len(text)
        entropy = -sum((f/total) * math.log2(f/total) for f in freq.values())

        # 平均碼長
        avg_code_length = sum(len(codes[c]) * freq[c] for c in codes) / total

        return {
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': compressed_bits / original_bits,
            'entropy': entropy,
            'avg_code_length': avg_code_length,
            'efficiency': entropy / avg_code_length,  # 越接近 1 越好
            'codes': codes
        }


# 使用範例
import math

text = "AAAAABBBCCCCCCCCDDDEEEEEEEEE"
huffman = HuffmanCoding()
stats = huffman.compression_stats(text)

print(f"原始大小: {stats['original_bits']} bits")
print(f"壓縮後: {stats['compressed_bits']} bits")
print(f"壓縮比: {stats['compression_ratio']:.2%}")
print(f"熵: {stats['entropy']:.3f} bits/symbol")
print(f"平均碼長: {stats['avg_code_length']:.3f} bits/symbol")
print(f"效率: {stats['efficiency']:.2%}")
print(f"編碼表: {stats['codes']}")
```

### 算術編碼

```python
from decimal import Decimal, getcontext

getcontext().prec = 50  # 高精度

class ArithmeticCoding:
    def __init__(self):
        self.prob_ranges = {}

    def build_model(self, text: str) -> None:
        """建立機率模型"""
        freq = Counter(text)
        total = len(text)

        # 累積機率
        cumulative = Decimal(0)
        for char in sorted(freq.keys()):
            prob = Decimal(freq[char]) / Decimal(total)
            self.prob_ranges[char] = (cumulative, cumulative + prob)
            cumulative += prob

    def encode(self, text: str) -> Decimal:
        """編碼"""
        self.build_model(text)

        low = Decimal(0)
        high = Decimal(1)

        for char in text:
            range_width = high - low
            char_low, char_high = self.prob_ranges[char]

            high = low + range_width * char_high
            low = low + range_width * char_low

        # 返回區間中的一個值
        return (low + high) / 2

    def decode(self, encoded: Decimal, length: int) -> str:
        """解碼"""
        decoded = []
        value = encoded

        for _ in range(length):
            for char, (low, high) in self.prob_ranges.items():
                if low <= value < high:
                    decoded.append(char)
                    range_width = high - low
                    value = (value - low) / range_width
                    break

        return ''.join(decoded)


# 範例
text = "ABRACADABRA"
ac = ArithmeticCoding()
encoded = ac.encode(text)
decoded = ac.decode(encoded, len(text))

print(f"原文: {text}")
print(f"編碼: {encoded}")
print(f"解碼: {decoded}")
```

---

## 密碼學應用

### 密碼強度計算器

```python
import string
import math
import re

class PasswordStrengthAnalyzer:
    def __init__(self):
        self.common_passwords = {
            'password', '123456', 'qwerty', 'abc123', 'monkey',
            'master', 'dragon', 'letmein', 'login', 'admin'
        }

        self.common_patterns = [
            r'(.)\1{2,}',           # 重複字符 aaa
            r'(012|123|234|345)',   # 連續數字
            r'(abc|bcd|cde)',       # 連續字母
            r'(qwerty|asdf|zxcv)',  # 鍵盤序列
        ]

    def calculate_entropy(self, password: str) -> float:
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
        if any(c == ' ' for c in password):
            charset_size += 1

        if charset_size == 0:
            return 0

        return len(password) * math.log2(charset_size)

    def check_patterns(self, password: str) -> list:
        """檢查常見模式"""
        issues = []

        # 常見密碼
        if password.lower() in self.common_passwords:
            issues.append("常見密碼")

        # 危險模式
        for pattern in self.common_patterns:
            if re.search(pattern, password.lower()):
                issues.append(f"包含危險模式: {pattern}")

        return issues

    def estimate_crack_time(self, entropy: float) -> str:
        """估計破解時間"""
        # 假設每秒 10 億次猜測（現代硬體）
        guesses_per_second = 1e9
        total_guesses = 2 ** entropy

        seconds = total_guesses / guesses_per_second / 2  # 平均情況

        if seconds < 1:
            return "瞬間"
        elif seconds < 60:
            return f"{seconds:.0f} 秒"
        elif seconds < 3600:
            return f"{seconds/60:.0f} 分鐘"
        elif seconds < 86400:
            return f"{seconds/3600:.0f} 小時"
        elif seconds < 31536000:
            return f"{seconds/86400:.0f} 天"
        elif seconds < 31536000 * 100:
            return f"{seconds/31536000:.0f} 年"
        elif seconds < 31536000 * 1e6:
            return f"{seconds/31536000/1000:.0f} 千年"
        else:
            return "超過宇宙年齡"

    def analyze(self, password: str) -> dict:
        """完整分析"""
        entropy = self.calculate_entropy(password)
        issues = self.check_patterns(password)
        crack_time = self.estimate_crack_time(entropy)

        # 評級
        if entropy < 28:
            rating = "極弱"
            score = 1
        elif entropy < 36:
            rating = "弱"
            score = 2
        elif entropy < 60:
            rating = "中等"
            score = 3
        elif entropy < 80:
            rating = "強"
            score = 4
        elif entropy < 100:
            rating = "很強"
            score = 5
        else:
            rating = "極強"
            score = 5

        # 有問題則降級
        if issues:
            score = max(1, score - len(issues))

        return {
            'password': password,
            'length': len(password),
            'entropy': entropy,
            'rating': rating,
            'score': score,
            'crack_time': crack_time,
            'issues': issues,
            'suggestions': self.get_suggestions(password, entropy)
        }

    def get_suggestions(self, password: str, entropy: float) -> list:
        """改進建議"""
        suggestions = []

        if len(password) < 12:
            suggestions.append("增加長度至少到 12 字符")
        if not any(c.isupper() for c in password):
            suggestions.append("添加大寫字母")
        if not any(c.islower() for c in password):
            suggestions.append("添加小寫字母")
        if not any(c.isdigit() for c in password):
            suggestions.append("添加數字")
        if not any(c in string.punctuation for c in password):
            suggestions.append("添加特殊字符")
        if entropy < 60:
            suggestions.append("考慮使用密碼短語（如 correct-horse-battery-staple）")

        return suggestions


# 使用範例
analyzer = PasswordStrengthAnalyzer()

passwords = [
    "password",
    "P@ssw0rd!",
    "MyD0g$Nam3IsM@x",
    "correct horse battery staple",
    "Tr0ub4dor&3"
]

for pwd in passwords:
    result = analyzer.analyze(pwd)
    print(f"\n密碼: {result['password']}")
    print(f"  熵: {result['entropy']:.1f} bits")
    print(f"  評級: {result['rating']} ({result['score']}/5)")
    print(f"  破解時間: {result['crack_time']}")
    if result['issues']:
        print(f"  問題: {', '.join(result['issues'])}")
    if result['suggestions']:
        print(f"  建議: {', '.join(result['suggestions'][:2])}")
```

### 安全隨機數生成

```python
import secrets
import os
import hashlib

class SecureRandomGenerator:
    """密碼學安全的隨機數生成器"""

    @staticmethod
    def random_bytes(n: int) -> bytes:
        """生成 n 字節的隨機數據"""
        return secrets.token_bytes(n)

    @staticmethod
    def random_int(min_val: int, max_val: int) -> int:
        """生成範圍內的隨機整數"""
        return secrets.randbelow(max_val - min_val + 1) + min_val

    @staticmethod
    def random_password(length: int = 16,
                       include_special: bool = True) -> str:
        """生成安全密碼"""
        alphabet = string.ascii_letters + string.digits
        if include_special:
            alphabet += string.punctuation

        while True:
            password = ''.join(secrets.choice(alphabet) for _ in range(length))
            # 確保包含各種字符類型
            if (any(c.islower() for c in password)
                and any(c.isupper() for c in password)
                and any(c.isdigit() for c in password)):
                if not include_special or any(c in string.punctuation for c in password):
                    return password

    @staticmethod
    def random_passphrase(num_words: int = 4) -> str:
        """生成密碼短語（Diceware 風格）"""
        # 簡化的詞表
        wordlist = [
            "correct", "horse", "battery", "staple", "apple", "orange",
            "quantum", "neural", "cipher", "matrix", "vector", "tensor",
            "rocket", "planet", "galaxy", "nebula", "quasar", "pulsar",
            "mountain", "river", "forest", "ocean", "desert", "island"
        ]

        return '-'.join(secrets.choice(wordlist) for _ in range(num_words))

    @staticmethod
    def secure_compare(a: bytes, b: bytes) -> bool:
        """常數時間比較（防止時序攻擊）"""
        return secrets.compare_digest(a, b)


# 使用範例
rng = SecureRandomGenerator()

print(f"隨機字節 (hex): {rng.random_bytes(16).hex()}")
print(f"隨機整數 [1-100]: {rng.random_int(1, 100)}")
print(f"安全密碼: {rng.random_password(16)}")
print(f"密碼短語: {rng.random_passphrase(4)}")
```

---

## 猜測遊戲策略

### 猜數字遊戲

```python
import math
from typing import Tuple, List

class NumberGuessingGame:
    """最優猜數字策略（基於信息論）"""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.target = None
        self.guesses = []

    def set_target(self, target: int):
        """設置目標數字"""
        self.target = target
        self.guesses = []

    def optimal_guess(self, low: int, high: int) -> int:
        """最優猜測（二分法）"""
        return (low + high) // 2

    def information_gain(self, guess: int, low: int, high: int) -> float:
        """計算猜測的期望信息增益"""
        total_range = high - low + 1

        # 猜測後的兩個子範圍
        left_range = guess - low
        right_range = high - guess

        # 信息增益
        p_left = left_range / total_range
        p_right = right_range / total_range

        if p_left > 0 and p_right > 0:
            current_entropy = math.log2(total_range)
            expected_entropy = (p_left * math.log2(left_range) if left_range > 0 else 0) + \
                              (p_right * math.log2(right_range) if right_range > 0 else 0)
            return current_entropy - expected_entropy

        return 0

    def play_optimal(self) -> List[Tuple[int, str]]:
        """使用最優策略玩遊戲"""
        if self.target is None:
            raise ValueError("請先設置目標數字")

        low, high = self.low, self.high
        history = []

        while low <= high:
            guess = self.optimal_guess(low, high)
            self.guesses.append(guess)

            if guess == self.target:
                history.append((guess, "correct"))
                break
            elif guess < self.target:
                history.append((guess, "too low"))
                low = guess + 1
            else:
                history.append((guess, "too high"))
                high = guess - 1

        return history

    def theoretical_guesses(self) -> float:
        """理論最優猜測次數"""
        return math.log2(self.high - self.low + 1)


# 使用範例
game = NumberGuessingGame(1, 100)
game.set_target(73)

print(f"範圍: 1-100")
print(f"目標: 73")
print(f"理論最優: {game.theoretical_guesses():.2f} 次")
print("\n猜測歷史:")

history = game.play_optimal()
for i, (guess, result) in enumerate(history, 1):
    print(f"  第 {i} 次: 猜 {guess} → {result}")

print(f"\n實際使用: {len(history)} 次")
```

### 20 問遊戲

```python
class TwentyQuestionsGame:
    """二十問遊戲的信息論最優策略"""

    def __init__(self, items: List[dict]):
        """
        items: 候選項目列表，每個是 {'name': str, 'features': dict}
        """
        self.items = items
        self.remaining = list(items)

    def calculate_entropy(self, items: List[dict]) -> float:
        """計算當前候選集的熵"""
        if len(items) <= 1:
            return 0
        return math.log2(len(items))

    def question_value(self, feature: str, value: any) -> float:
        """評估問題的價值（信息增益）"""
        yes_count = sum(1 for item in self.remaining
                       if item['features'].get(feature) == value)
        no_count = len(self.remaining) - yes_count

        if yes_count == 0 or no_count == 0:
            return 0

        # 越接近 50/50 越好
        p_yes = yes_count / len(self.remaining)
        p_no = no_count / len(self.remaining)

        current_entropy = self.calculate_entropy(self.remaining)

        # 條件熵
        entropy_if_yes = math.log2(yes_count) if yes_count > 1 else 0
        entropy_if_no = math.log2(no_count) if no_count > 1 else 0
        expected_entropy = p_yes * entropy_if_yes + p_no * entropy_if_no

        return current_entropy - expected_entropy

    def best_question(self) -> Tuple[str, any, float]:
        """找出最佳問題"""
        best_gain = -1
        best_feature = None
        best_value = None

        # 收集所有可能的特徵-值對
        all_features = set()
        for item in self.remaining:
            for feature, value in item['features'].items():
                all_features.add((feature, value))

        # 評估每個問題
        for feature, value in all_features:
            gain = self.question_value(feature, value)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value

        return best_feature, best_value, best_gain

    def ask(self, feature: str, value: any, answer: bool) -> None:
        """根據答案過濾候選"""
        if answer:
            self.remaining = [item for item in self.remaining
                            if item['features'].get(feature) == value]
        else:
            self.remaining = [item for item in self.remaining
                            if item['features'].get(feature) != value]


# 範例：猜動物
animals = [
    {'name': '狗', 'features': {'哺乳類': True, '會飛': False, '四條腿': True, '會游泳': True}},
    {'name': '貓', 'features': {'哺乳類': True, '會飛': False, '四條腿': True, '會游泳': False}},
    {'name': '鳥', 'features': {'哺乳類': False, '會飛': True, '四條腿': False, '會游泳': False}},
    {'name': '魚', 'features': {'哺乳類': False, '會飛': False, '四條腿': False, '會游泳': True}},
    {'name': '蝙蝠', 'features': {'哺乳類': True, '會飛': True, '四條腿': True, '會游泳': False}},
]

game = TwentyQuestionsGame(animals)

print(f"初始候選: {len(game.remaining)} 項")
print(f"初始熵: {game.calculate_entropy(game.remaining):.2f} bits")

# 找最佳問題
feature, value, gain = game.best_question()
print(f"\n最佳問題: '{feature}' 是否為 {value}?")
print(f"期望信息增益: {gain:.2f} bits")
```

---

## 異常檢測

### 基於熵的異常檢測

```python
import numpy as np
from collections import deque

class EntropyAnomalyDetector:
    """基於熵變化的異常檢測"""

    def __init__(self, window_size: int = 100, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.baseline_entropy = None

    def calculate_entropy(self, data: np.ndarray) -> float:
        """計算數據的熵"""
        # 離散化
        hist, _ = np.histogram(data, bins=50)
        hist = hist / hist.sum()

        # 移除零值
        hist = hist[hist > 0]

        return -np.sum(hist * np.log2(hist))

    def fit(self, normal_data: np.ndarray) -> None:
        """用正常數據建立基線"""
        entropies = []

        for i in range(self.window_size, len(normal_data)):
            window = normal_data[i-self.window_size:i]
            entropy = self.calculate_entropy(window)
            entropies.append(entropy)

        self.baseline_entropy = np.mean(entropies)
        self.baseline_std = np.std(entropies)

    def detect(self, value: float) -> Tuple[bool, float]:
        """檢測單個值是否異常"""
        self.window.append(value)

        if len(self.window) < self.window_size:
            return False, 0.0

        current_entropy = self.calculate_entropy(np.array(self.window))

        # 計算 z-score
        z_score = abs(current_entropy - self.baseline_entropy) / self.baseline_std

        is_anomaly = z_score > self.threshold

        return is_anomaly, z_score


# 使用範例
np.random.seed(42)

# 正常數據：高斯分布
normal_data = np.random.normal(0, 1, 1000)

# 異常數據：混入一些異常值
test_data = np.concatenate([
    np.random.normal(0, 1, 200),      # 正常
    np.random.normal(5, 0.1, 50),     # 異常（均值偏移，方差減小）
    np.random.normal(0, 1, 200),      # 正常
])

detector = EntropyAnomalyDetector(window_size=50, threshold=2.5)
detector.fit(normal_data)

print("異常檢測結果:")
anomalies = []
for i, value in enumerate(test_data):
    is_anomaly, z_score = detector.detect(value)
    if is_anomaly:
        anomalies.append(i)
        print(f"  位置 {i}: z-score = {z_score:.2f}")

print(f"\n檢測到 {len(anomalies)} 個異常區域")
```

---

## 圖像熵分析

```python
from PIL import Image
import numpy as np

def image_entropy(image_path: str) -> dict:
    """分析圖像的熵"""
    img = Image.open(image_path)

    # 轉灰度
    gray = img.convert('L')
    gray_arr = np.array(gray)

    # 計算灰度直方圖熵
    hist, _ = np.histogram(gray_arr.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    gray_entropy = -np.sum(hist * np.log2(hist))

    # RGB 通道分析
    if img.mode == 'RGB':
        rgb_arr = np.array(img)
        channel_entropies = []

        for i, channel in enumerate(['R', 'G', 'B']):
            hist, _ = np.histogram(rgb_arr[:,:,i].flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            channel_entropies.append(-np.sum(hist * np.log2(hist)))

        avg_rgb_entropy = np.mean(channel_entropies)
    else:
        channel_entropies = [gray_entropy]
        avg_rgb_entropy = gray_entropy

    # 局部熵（紋理複雜度）
    block_size = 16
    local_entropies = []

    for i in range(0, gray_arr.shape[0] - block_size, block_size):
        for j in range(0, gray_arr.shape[1] - block_size, block_size):
            block = gray_arr[i:i+block_size, j:j+block_size]
            hist, _ = np.histogram(block.flatten(), bins=32, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            if len(hist) > 0:
                local_entropies.append(-np.sum(hist * np.log2(hist)))

    return {
        'gray_entropy': gray_entropy,
        'channel_entropies': channel_entropies,
        'avg_rgb_entropy': avg_rgb_entropy,
        'local_entropy_mean': np.mean(local_entropies),
        'local_entropy_std': np.std(local_entropies),
        'max_possible_entropy': 8.0,  # log2(256)
        'complexity_ratio': gray_entropy / 8.0
    }


# 圖像熵的典型值：
# - 純色圖像: ~0 bits（熵很低）
# - 簡單圖形: 1-3 bits
# - 自然照片: 6-7 bits
# - 噪聲/隨機: ~8 bits（最大熵）
```
