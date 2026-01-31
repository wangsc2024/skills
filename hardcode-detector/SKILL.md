---
name: hardcode-detector
description: |
  Detect hardcoded values, fake implementations, mock data, magic numbers, embedded secrets, stubbed functions, and test-mode bypasses in production code.
  Use when: auditing code quality, reviewing for production readiness, checking for secrets, finding magic numbers, or when user mentions 硬編碼, hardcode, magic number, 寫死, 假資料, fake data, mock殘留, secrets洩漏.
  Triggers: "hardcode", "magic number", "寫死", "硬編碼", "fake", "mock", "stub", "secrets", "檢查寫死的值"
allowed-tools: Read, Grep, Glob
version: 1.0.0
---

# Hardcode & Fake Implementation Detector

審查程式碼中的硬編碼值、虛假實作、與測試繞過機制，確保生產環境程式碼品質。

## 檢測類別

| 類別 | 風險等級 | 說明 |
|------|----------|------|
| **Secrets 洩漏** | 🚨 Critical | API keys, 密碼, tokens |
| **虛假實作** | 🔴 High | Stub/Mock 殘留於生產碼 |
| **測試模式繞過** | 🔴 High | if TEST_MODE 分支 |
| **Magic Numbers** | 🟡 Medium | 未命名的常數值 |
| **硬編碼配置** | 🟡 Medium | URLs, paths, IDs |
| **假資料** | 🟡 Medium | Lorem ipsum, test@test.com |

## 🚨 Critical: Secrets 洩漏

### 檢測模式

```python
# ❌ 硬編碼 API Key
API_KEY = "sk-1234567890abcdef"
api_key = "AIzaSyB-abc123xyz"

# ❌ 硬編碼密碼
password = "admin123"
db_password = "P@ssw0rd!"

# ❌ 硬編碼 Token
jwt_secret = "my-super-secret-key"
auth_token = "Bearer eyJhbGciOiJIUzI1NiIs..."

# ❌ 連線字串含密碼
DATABASE_URL = "postgresql://user:password123@localhost/db"
REDIS_URL = "redis://:secretpass@localhost:6379"
```

### 正確做法

```python
# ✅ 使用環境變數
import os

API_KEY = os.environ["API_KEY"]
DATABASE_URL = os.environ.get("DATABASE_URL")

# ✅ 使用 secrets manager
from aws_secrets import get_secret
api_key = get_secret("prod/api-key")

# ✅ 使用 .env 檔案（不進版控）
from dotenv import load_dotenv
load_dotenv()
```

### Grep 搜尋指令

```bash
# 搜尋可能的 secrets
grep -rE "(api[_-]?key|secret|password|token|credential)\s*[=:]\s*['\"][^'\"]{8,}" --include="*.py" --include="*.js" --include="*.ts"

# 搜尋 AWS keys
grep -rE "AKIA[0-9A-Z]{16}" .

# 搜尋 JWT tokens
grep -rE "eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+" .
```

## 🔴 High: 虛假實作

### Stub/Mock 殘留

```python
# ❌ 假的 API 回應
def get_user(user_id):
    return {"id": user_id, "name": "Test User", "email": "test@test.com"}

# ❌ 空實作回傳成功
def send_email(to, subject, body):
    return True  # TODO: implement

# ❌ Sleep 模擬處理時間
def process_payment(amount):
    time.sleep(2)  # Simulate processing
    return {"status": "success"}

# ❌ 隨機但固定的回傳
def generate_id():
    return "usr_123456"  # Should be random
```

### 正確做法

```python
# ✅ 真實實作
def get_user(user_id):
    response = db.users.find_one({"_id": user_id})
    if not response:
        raise UserNotFoundError(user_id)
    return response

# ✅ 明確拋出未實作
def send_email(to, subject, body):
    raise NotImplementedError("Email service not configured")

# ✅ 真實 ID 生成
import uuid
def generate_id():
    return f"usr_{uuid.uuid4().hex[:12]}"
```

### 檢測模式

```bash
# 搜尋 TODO/FIXME 標記
grep -rE "(TODO|FIXME|XXX|HACK|STUB):" --include="*.py" --include="*.js"

# 搜尋假回傳
grep -rE "return\s+(True|False|\{\}|\[\]|None)\s*#" --include="*.py"

# 搜尋 sleep 模擬
grep -rE "(time\.sleep|setTimeout|Thread\.sleep)" --include="*.py" --include="*.js" --include="*.java"
```

## 🔴 High: 測試模式繞過

### 危險模式

```python
# ❌ 測試模式分支
if os.environ.get("TEST_MODE"):
    return mock_data
else:
    return real_api_call()

# ❌ 開發模式跳過驗證
if settings.DEBUG:
    return True  # Skip auth in dev

# ❌ 環境判斷繞過
if ENV != "production":
    user = {"role": "admin"}  # Bypass auth
```

### 正確做法

```python
# ✅ 使用依賴注入
class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

# 測試時注入 mock
# 生產時注入真實 repository

# ✅ 使用介面/抽象
from abc import ABC, abstractmethod

class PaymentGateway(ABC):
    @abstractmethod
    def charge(self, amount: int) -> PaymentResult:
        pass

# 測試用 FakePaymentGateway
# 生產用 StripePaymentGateway
```

### 檢測指令

```bash
# 搜尋測試模式判斷
grep -rE "if.*(TEST|DEBUG|DEV|MOCK).*:" --include="*.py"
grep -rE "if.*process\.env\.(NODE_ENV|TEST)" --include="*.js" --include="*.ts"

# 搜尋環境判斷
grep -rE "if.*['\"]production['\"]" --include="*.py" --include="*.js"
```

## 🟡 Medium: Magic Numbers

### 問題範例

```python
# ❌ 未命名常數
if retry_count > 3:
    raise TooManyRetriesError()

if len(password) < 8:
    raise ValidationError()

discount = price * 0.15

timeout = 30000  # What unit? What for?
```

### 正確做法

```python
# ✅ 命名常數
MAX_RETRY_ATTEMPTS = 3
MIN_PASSWORD_LENGTH = 8
STANDARD_DISCOUNT_RATE = 0.15
API_TIMEOUT_MS = 30_000

if retry_count > MAX_RETRY_ATTEMPTS:
    raise TooManyRetriesError()

if len(password) < MIN_PASSWORD_LENGTH:
    raise ValidationError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
```

### 檢測指令

```bash
# 搜尋 magic numbers（常見閾值）
grep -rE ">\s*[0-9]{2,}|<\s*[0-9]{2,}" --include="*.py" --include="*.js"

# 搜尋浮點數常數
grep -rE "\*\s*0\.[0-9]+" --include="*.py" --include="*.js"
```

## 🟡 Medium: 硬編碼配置

### 問題範例

```python
# ❌ 硬編碼 URL
response = requests.get("https://api.example.com/v1/users")

# ❌ 硬編碼路徑
config_path = "/etc/myapp/config.json"
log_path = "C:\\Users\\admin\\logs\\"

# ❌ 硬編碼 ID
ADMIN_USER_ID = "usr_abc123"
DEFAULT_TENANT_ID = 42
```

### 正確做法

```python
# ✅ 使用配置
from config import settings

response = requests.get(f"{settings.API_BASE_URL}/users")
config_path = settings.CONFIG_PATH
log_path = settings.LOG_DIRECTORY

# ✅ 使用常數檔
# constants.py
class UserRoles:
    ADMIN = "admin"
    USER = "user"
```

## 🟡 Medium: 假資料殘留

### 問題範例

```python
# ❌ 測試用假資料
email = "test@test.com"
name = "John Doe"
phone = "123-456-7890"
address = "123 Main St"

# ❌ Lorem ipsum
description = "Lorem ipsum dolor sit amet..."

# ❌ 範例資料
users = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
]
```

### 檢測指令

```bash
# 搜尋測試 email
grep -rE "test@|example\.com|fake@|dummy@" --include="*.py" --include="*.js"

# 搜尋 Lorem ipsum
grep -ri "lorem ipsum" --include="*.py" --include="*.js" --include="*.tsx"

# 搜尋範例姓名
grep -rE "(John Doe|Jane Doe|Alice|Bob|Test User)" --include="*.py" --include="*.js"
```

## 審查報告模板

```markdown
# 硬編碼與虛假實作審查報告

## 摘要
- 📁 掃描檔案數: XX
- 🚨 Critical 問題: X
- 🔴 High 問題: X
- 🟡 Medium 問題: X

## 🚨 Critical Issues

### [C-001] API Key 硬編碼
- **檔案**: `src/services/api.py:42`
- **問題**:
  ```python
  API_KEY = "sk-live-abc123..."
  ```
- **風險**: Secret 洩漏至版本控制
- **修復**: 移至環境變數 `os.environ["API_KEY"]`

## 🔴 High Issues

### [H-001] 虛假實作殘留
- **檔案**: `src/services/email.py:15`
- **問題**:
  ```python
  def send_email(to, subject, body):
      return True  # TODO: implement
  ```
- **風險**: 功能未真正實作
- **修復**: 實作真實 email 發送或拋出 NotImplementedError

## 🟡 Medium Issues

### [M-001] Magic Number
- **檔案**: `src/utils/validation.py:28`
- **問題**:
  ```python
  if len(password) < 8:
  ```
- **修復**: 定義常數 `MIN_PASSWORD_LENGTH = 8`

## 修復優先順序

1. 🚨 立即修復所有 Critical 問題
2. 🔴 發布前修復所有 High 問題
3. 🟡 排入技術債清理週期
```

## 自動化掃描腳本

```python
#!/usr/bin/env python3
"""hardcode_scanner.py - 掃描硬編碼與虛假實作"""

import re
import sys
from pathlib import Path

PATTERNS = {
    "critical": [
        (r"['\"]sk-[a-zA-Z0-9]{20,}['\"]", "Possible API key"),
        (r"['\"]AKIA[0-9A-Z]{16}['\"]", "AWS Access Key"),
        (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
        (r"['\"]eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+", "JWT token"),
    ],
    "high": [
        (r"if.*(TEST_MODE|DEBUG|MOCK).*:", "Test mode bypass"),
        (r"return\s+(True|False)\s*#.*TODO", "Stub implementation"),
        (r"time\.sleep\([0-9]+\)", "Sleep simulation"),
    ],
    "medium": [
        (r"test@test\.com|example\.com", "Test email"),
        (r"lorem ipsum", "Placeholder text"),
        (r"John Doe|Jane Doe", "Placeholder name"),
    ],
}

def scan_file(path: Path) -> list:
    issues = []
    try:
        content = path.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), 1):
            for severity, patterns in PATTERNS.items():
                for pattern, desc in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            "file": str(path),
                            "line": line_num,
                            "severity": severity,
                            "description": desc,
                            "content": line.strip()[:80],
                        })
    except Exception as e:
        pass
    return issues

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go"}

    all_issues = []
    for ext in extensions:
        for path in root.rglob(f"*{ext}"):
            if "node_modules" in str(path) or ".venv" in str(path):
                continue
            all_issues.extend(scan_file(path))

    # 輸出結果
    for issue in sorted(all_issues, key=lambda x: x["severity"]):
        print(f"[{issue['severity'].upper()}] {issue['file']}:{issue['line']}")
        print(f"  {issue['description']}: {issue['content']}")
        print()

if __name__ == "__main__":
    main()
```

## Checklist

### Secrets
- [ ] 無 API keys 硬編碼
- [ ] 無密碼硬編碼
- [ ] 無 tokens 硬編碼
- [ ] 連線字串使用環境變數
- [ ] .env 檔案已加入 .gitignore

### 實作完整性
- [ ] 無 TODO/FIXME 在關鍵路徑
- [ ] 無 stub 函式回傳假值
- [ ] 無 sleep 模擬處理時間
- [ ] 無空的 try/catch 區塊

### 測試隔離
- [ ] 無 if TEST_MODE 分支
- [ ] 無 if DEBUG 繞過
- [ ] 無環境判斷跳過驗證
- [ ] 使用依賴注入而非條件分支

### 配置管理
- [ ] URL 使用配置檔
- [ ] 路徑使用配置檔
- [ ] Magic numbers 已命名為常數
- [ ] 無假資料殘留
