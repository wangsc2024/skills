---
name: test-driven-development
description: |
  Implement features and fix bugs using strict Test-Driven Development (TDD). Enforces RED-GREEN-REFACTOR cycle where tests are written before any production code.
  Use when: implementing new features, fixing bugs, refactoring code, writing tests, adding test coverage, or when user mentions TDD, 測試驅動, 寫測試, 單元測試, unit test, pytest, jest, 紅綠重構.
  Triggers: "write test first", "test driven", "TDD", "red green refactor", "failing test", "測試先行", "先寫測試"
version: 1.0.0
---

# Test-Driven Development (TDD)

嚴格執行測試驅動開發，確保程式碼品質。

## 核心原則

> **NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST**
> 沒有失敗的測試，就不能寫生產程式碼

如果先寫了程式碼再補測試，必須**刪除程式碼重新開始**。

## TDD 循環

```
┌─────────────────────────────────────────┐
│                                         │
│   🔴 RED → 🟢 GREEN → 🔄 REFACTOR      │
│     ↑                         │         │
│     └─────────────────────────┘         │
│                                         │
└─────────────────────────────────────────┘
```

### Phase 1: 🔴 RED（紅燈）

**寫一個失敗的測試**

```python
# test_calculator.py
def test_add_two_numbers():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5
```

```bash
# 執行測試 - 必須看到失敗
pytest test_calculator.py -v
# FAILED - NameError: Calculator is not defined
```

**重要**：必須親眼看到測試失敗，確認測試有效。

### Phase 2: 🟢 GREEN（綠燈）

**寫最少的程式碼讓測試通過**

```python
# calculator.py
class Calculator:
    def add(self, a, b):
        return a + b  # 最小實作
```

```bash
# 執行測試 - 必須看到通過
pytest test_calculator.py -v
# PASSED
```

**注意**：只寫剛好讓測試通過的程式碼，不多不少。

### Phase 3: 🔄 REFACTOR（重構）

**在綠燈狀態下改善程式碼**

```python
# 重構後仍保持測試通過
class Calculator:
    def add(self, a: int, b: int) -> int:
        """Add two numbers and return the result."""
        return a + b
```

```bash
# 重構後再次確認
pytest test_calculator.py -v
# PASSED
```

## 完整工作流程

```markdown
1. 📝 寫測試（描述預期行為）
2. 🔴 執行測試（確認失敗）
3. 💻 寫最小實作
4. 🟢 執行測試（確認通過）
5. 🔄 重構（改善程式碼品質）
6. 🟢 再次執行測試（確認仍通過）
7. 📦 Commit（描述性訊息）
8. 🔁 重複下一個測試案例
```

## 測試案例設計

### 從簡單到複雜

```python
# 1. Happy path（正常路徑）
def test_add_positive_numbers():
    assert Calculator().add(2, 3) == 5

# 2. Edge cases（邊界條件）
def test_add_zero():
    assert Calculator().add(5, 0) == 5

def test_add_negative_numbers():
    assert Calculator().add(-2, -3) == -5

# 3. Error cases（錯誤情況）
def test_add_with_none_raises_error():
    with pytest.raises(TypeError):
        Calculator().add(None, 5)
```

### 測試命名規範

```python
# 格式: test_<what>_<condition>_<expected>
def test_login_with_valid_credentials_returns_token():
    pass

def test_login_with_invalid_password_raises_auth_error():
    pass

def test_withdraw_exceeding_balance_raises_insufficient_funds():
    pass
```

## Bug 修復流程

```markdown
1. 📝 寫一個會觸發 bug 的測試
2. 🔴 確認測試失敗（重現 bug）
3. 💻 修復 bug
4. 🟢 確認測試通過
5. 📦 Commit: "fix: <描述> - 新增回歸測試"
```

```python
# 範例：修復除以零的 bug
def test_divide_by_zero_raises_error():
    """Bug #123: 除以零應該拋出錯誤而非崩潰"""
    with pytest.raises(ZeroDivisionError):
        Calculator().divide(10, 0)
```

## 常見藉口（不接受）

| 藉口 | 為什麼不接受 |
|------|-------------|
| "太簡單不需要測試" | 簡單的程式碼更容易測試 |
| "趕時間之後補測試" | 之後的測試無法驗證需求 |
| "我已經手動測過了" | 手動測試無法重複、無法自動化 |
| "這只是原型" | 原型也會變成生產程式碼 |
| "測試太麻煩" | 沒測試的 debug 更麻煩 |

## 合理例外（需明確核准）

- 拋棄式原型（確定不會進生產）
- 自動生成的程式碼
- 純 UI 樣式調整（無邏輯）

## 測試框架快速參考

### Python (pytest)

```bash
# 執行所有測試
pytest

# 執行特定檔案
pytest test_calculator.py

# 執行特定測試
pytest test_calculator.py::test_add_two_numbers

# 顯示詳細輸出
pytest -v

# 顯示 print 輸出
pytest -s

# 失敗時停止
pytest -x
```

### JavaScript (Jest/Vitest)

```bash
# Jest
npm test
npm test -- --watch

# Vitest
npx vitest
npx vitest run
```

### Go

```bash
go test ./...
go test -v ./...
go test -run TestAdd
```

## Checklist

每次實作前確認：

- [ ] 測試已寫好
- [ ] 測試已執行且失敗
- [ ] 失敗原因符合預期
- [ ] 只寫最小實作
- [ ] 測試通過
- [ ] 程式碼已重構（如需要）
- [ ] 所有測試仍通過
- [ ] 已 commit

## 反模式警告

```markdown
🚫 先寫程式碼再補測試 → 測試通過不代表正確
🚫 一次寫太多測試 → 失去回饋循環
🚫 測試太大太複雜 → 難以定位問題
🚫 跳過紅燈階段 → 無法確認測試有效
🚫 重構時加新功能 → 混淆問題來源
```
