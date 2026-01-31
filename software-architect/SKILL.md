---
name: software-architect
description: |
  Apply software architecture principles including Clean Architecture, SOLID, design patterns, and system design best practices. Guides architectural decisions, code organization, and scalability planning.
  Use when: designing systems, making architectural decisions, reviewing architecture, planning refactoring, discussing design patterns, or when user mentions 架構, architecture, SOLID, 設計模式, design pattern, clean architecture, 重構, 系統設計.
  Triggers: "architecture", "design pattern", "SOLID", "clean architecture", "架構設計", "系統設計", "怎麼設計", "結構", "分層"
version: 1.0.0
---

# Software Architecture

軟體架構設計原則與最佳實踐。

## 核心架構原則

### Clean Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    External Systems                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │                  Frameworks                      │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │            Interface Adapters            │    │    │
│  │  │  ┌─────────────────────────────────┐    │    │    │
│  │  │  │         Application Layer        │    │    │    │
│  │  │  │  ┌─────────────────────────┐    │    │    │    │
│  │  │  │  │    Domain/Entities      │    │    │    │    │
│  │  │  │  │   (Business Logic)      │    │    │    │    │
│  │  │  │  └─────────────────────────┘    │    │    │    │
│  │  │  └─────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

依賴方向：外層 → 內層
內層不知道外層的存在
```

### 分層職責

| 層級 | 職責 | 範例 |
|------|------|------|
| **Domain** | 核心業務邏輯 | Entity, Value Object, Domain Service |
| **Application** | 用例協調 | Use Case, Command, Query |
| **Interface** | 轉換格式 | Controller, Presenter, Gateway |
| **Infrastructure** | 外部系統 | Database, API Client, File System |

## SOLID 原則

### S - Single Responsibility (單一職責)

```python
# ❌ 違反 SRP
class UserService:
    def create_user(self, data): pass
    def send_email(self, user): pass      # 不應該在這裡
    def generate_report(self, users): pass # 不應該在這裡

# ✅ 遵守 SRP
class UserService:
    def create_user(self, data): pass

class EmailService:
    def send_email(self, recipient, content): pass

class ReportService:
    def generate_user_report(self, users): pass
```

### O - Open/Closed (開放封閉)

```python
# ❌ 違反 OCP - 每次新增付款方式都要修改
class PaymentProcessor:
    def process(self, payment_type, amount):
        if payment_type == "credit_card":
            # 信用卡邏輯
        elif payment_type == "paypal":
            # PayPal 邏輯
        elif payment_type == "crypto":  # 新增需修改
            # 加密貨幣邏輯

# ✅ 遵守 OCP - 對擴展開放，對修改封閉
class PaymentProcessor(ABC):
    @abstractmethod
    def process(self, amount): pass

class CreditCardProcessor(PaymentProcessor):
    def process(self, amount): pass

class PayPalProcessor(PaymentProcessor):
    def process(self, amount): pass

class CryptoProcessor(PaymentProcessor):  # 新增不需修改現有程式碼
    def process(self, amount): pass
```

### L - Liskov Substitution (里氏替換)

```python
# ❌ 違反 LSP
class Rectangle:
    def set_width(self, width): self.width = width
    def set_height(self, height): self.height = height

class Square(Rectangle):  # 正方形不應繼承矩形
    def set_width(self, width):
        self.width = self.height = width  # 破壞了父類別的契約

# ✅ 遵守 LSP
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self): return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side
    def area(self): return self.side ** 2
```

### I - Interface Segregation (介面隔離)

```python
# ❌ 違反 ISP - 肥大的介面
class Worker(ABC):
    @abstractmethod
    def work(self): pass
    @abstractmethod
    def eat(self): pass     # 機器人不需要
    @abstractmethod
    def sleep(self): pass   # 機器人不需要

# ✅ 遵守 ISP - 細粒度的介面
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Human(Workable, Eatable):
    def work(self): pass
    def eat(self): pass

class Robot(Workable):
    def work(self): pass
```

### D - Dependency Inversion (依賴反轉)

```python
# ❌ 違反 DIP - 高層依賴低層
class OrderService:
    def __init__(self):
        self.db = MySQLDatabase()  # 直接依賴具體實作

# ✅ 遵守 DIP - 依賴抽象
class DatabaseInterface(ABC):
    @abstractmethod
    def save(self, data): pass

class OrderService:
    def __init__(self, db: DatabaseInterface):
        self.db = db  # 依賴注入抽象

# 使用
mysql_db = MySQLDatabase()  # 實作 DatabaseInterface
service = OrderService(mysql_db)
```

## 常用設計模式

### 創建型模式

```python
# Factory Method
class NotificationFactory:
    @staticmethod
    def create(type: str) -> Notification:
        if type == "email":
            return EmailNotification()
        elif type == "sms":
            return SMSNotification()
        raise ValueError(f"Unknown type: {type}")

# Builder
class QueryBuilder:
    def __init__(self):
        self.query = ""

    def select(self, fields):
        self.query += f"SELECT {fields} "
        return self

    def from_table(self, table):
        self.query += f"FROM {table} "
        return self

    def where(self, condition):
        self.query += f"WHERE {condition} "
        return self

    def build(self):
        return self.query

# 使用
query = QueryBuilder().select("*").from_table("users").where("active = 1").build()
```

### 結構型模式

```python
# Repository Pattern
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, id: str) -> User: pass

    @abstractmethod
    def save(self, user: User) -> None: pass

class SQLUserRepository(UserRepository):
    def __init__(self, session):
        self.session = session

    def find_by_id(self, id: str) -> User:
        return self.session.query(User).filter(User.id == id).first()

    def save(self, user: User) -> None:
        self.session.add(user)
        self.session.commit()

# Adapter Pattern
class LegacyPaymentSystem:
    def make_payment(self, amount_cents: int): pass

class PaymentAdapter:
    def __init__(self, legacy: LegacyPaymentSystem):
        self.legacy = legacy

    def pay(self, amount_dollars: float):
        amount_cents = int(amount_dollars * 100)
        return self.legacy.make_payment(amount_cents)
```

### 行為型模式

```python
# Strategy Pattern
class PricingStrategy(ABC):
    @abstractmethod
    def calculate(self, base_price: float) -> float: pass

class RegularPricing(PricingStrategy):
    def calculate(self, base_price): return base_price

class PremiumPricing(PricingStrategy):
    def calculate(self, base_price): return base_price * 0.9

class VIPPricing(PricingStrategy):
    def calculate(self, base_price): return base_price * 0.8

class Order:
    def __init__(self, pricing: PricingStrategy):
        self.pricing = pricing

    def total(self, base_price):
        return self.pricing.calculate(base_price)

# Observer Pattern
class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event: str, callback):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def emit(self, event: str, data=None):
        for callback in self._listeners.get(event, []):
            callback(data)
```

## 專案結構範例

### Python (Clean Architecture)

```
src/
├── domain/                 # 核心業務邏輯
│   ├── entities/
│   │   ├── user.py
│   │   └── order.py
│   ├── value_objects/
│   │   ├── email.py
│   │   └── money.py
│   └── services/
│       └── pricing_service.py
├── application/            # 用例層
│   ├── use_cases/
│   │   ├── create_order.py
│   │   └── process_payment.py
│   └── interfaces/
│       ├── repositories.py
│       └── services.py
├── infrastructure/         # 外部系統
│   ├── persistence/
│   │   ├── sqlalchemy/
│   │   └── repositories/
│   ├── external/
│   │   ├── payment_gateway.py
│   │   └── email_service.py
│   └── config/
│       └── settings.py
└── presentation/           # API/UI
    ├── api/
    │   ├── routes/
    │   └── schemas/
    └── cli/
```

### TypeScript (Clean Architecture)

```
src/
├── domain/
│   ├── entities/
│   ├── value-objects/
│   └── services/
├── application/
│   ├── use-cases/
│   └── ports/
├── infrastructure/
│   ├── repositories/
│   ├── services/
│   └── config/
└── presentation/
    ├── http/
    │   ├── controllers/
    │   └── middleware/
    └── graphql/
```

## 架構決策記錄 (ADR)

```markdown
# ADR-001: 選擇事件驅動架構

## 狀態
已接受

## 背景
系統需要處理高併發訂單，且多個服務需要對訂單事件做出反應。

## 決策
採用事件驅動架構，使用 Apache Kafka 作為訊息佇列。

## 後果
### 優點
- 服務解耦
- 可擴展性
- 事件溯源能力

### 缺點
- 增加系統複雜度
- 需要處理最終一致性
- 運維成本增加
```

## 系統設計考量

### 可擴展性

```markdown
水平擴展策略：
1. 無狀態服務設計
2. 資料庫讀寫分離
3. 快取層（Redis）
4. CDN 靜態資源
5. 負載均衡
```

### 可靠性

```markdown
容錯機制：
1. 熔斷器模式（Circuit Breaker）
2. 重試機制（Exponential Backoff）
3. 服務降級（Graceful Degradation）
4. 健康檢查（Health Check）
```

### 可維護性

```markdown
程式碼品質：
1. 高內聚、低耦合
2. 命名清晰
3. 適當的抽象層級
4. 完整的測試覆蓋
5. 文件與註解
```

## Checklist

架構設計時確認：

- [ ] 遵守 SOLID 原則
- [ ] 依賴方向正確（外層→內層）
- [ ] 業務邏輯與技術細節分離
- [ ] 適當使用設計模式
- [ ] 考慮可擴展性
- [ ] 考慮可測試性
- [ ] 有 ADR 記錄重要決策
- [ ] 專案結構清晰
