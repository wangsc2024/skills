---
name: systematic-debugging
description: |
  Debug issues using a disciplined four-phase approach that prioritizes root cause investigation before attempting fixes. Prevents wasted time from guessing at solutions.
  Use when: encountering bugs, errors, exceptions, unexpected behavior, system failures, crashes, or when user mentions 除錯, debug, 找bug, 錯誤, error, exception, traceback, stack trace, 根因分析.
  Triggers: "debug", "bug", "error", "exception", "not working", "fails", "broken", "除錯", "錯誤", "為什麼不動", "出問題"
version: 1.0.0
---

# Systematic Debugging

系統化除錯方法論，先找根因再修復。

## 核心原則

> **NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST**
> 沒有根因分析，就不能嘗試修復

亂猜解法只會浪費時間並引入新問題。

## 四階段流程

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Phase 1        Phase 2        Phase 3    Phase 4  │
│  ┌───────┐     ┌───────┐     ┌───────┐   ┌──────┐ │
│  │ 調查  │ ──▶ │ 分析  │ ──▶ │ 假設  │ ─▶│ 修復 │ │
│  └───────┘     └───────┘     └───────┘   └──────┘ │
│                                                     │
│  收集證據      比對模式      驗證理論   實作修復   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Phase 1: 根因調查 🔍

**必須完成此階段才能進入修復**

### 1.1 完整閱讀錯誤訊息

```python
# ❌ 只看最後一行
TypeError: cannot unpack non-iterable NoneType object

# ✅ 看完整 stack trace
Traceback (most recent call last):
  File "app.py", line 45, in process_order
    user_id, order_id = get_order_info(data)
  File "utils.py", line 23, in get_order_info
    return None  # <-- 問題在這裡！
TypeError: cannot unpack non-iterable NoneType object
```

### 1.2 穩定重現問題

```markdown
重現步驟：
1. 輸入：[具體輸入值]
2. 操作：[具體操作步驟]
3. 預期結果：[應該發生什麼]
4. 實際結果：[實際發生什麼]
5. 重現率：[100% / 偶發]
```

### 1.3 檢查最近變更

```bash
# 查看最近的 commits
git log --oneline -10

# 查看特定檔案的變更歷史
git log -p --follow -- path/to/file.py

# 比對兩個版本
git diff HEAD~5 HEAD -- src/

# 找出問題是何時引入的
git bisect start
git bisect bad HEAD
git bisect good v1.2.0
```

### 1.4 追蹤資料流

```python
# 在關鍵點加入診斷日誌
def process_order(data):
    print(f"[DEBUG] Input data: {data}")

    order_info = get_order_info(data)
    print(f"[DEBUG] order_info: {order_info}, type: {type(order_info)}")

    user_id, order_id = order_info  # 錯誤發生點
    print(f"[DEBUG] user_id: {user_id}, order_id: {order_id}")
```

### 1.5 多元件系統診斷

```markdown
在每個邊界加入檢查點：

[Client] → 請求發出？資料正確？
    ↓
[API Gateway] → 收到請求？轉發正確？
    ↓
[Backend Service] → 處理邏輯正確？
    ↓
[Database] → 查詢正確？資料存在？
    ↓
[Response] → 回應格式正確？
```

## Phase 2: 模式分析 📊

### 2.1 比對正常 vs 異常

```python
# 正常運作的程式碼
def get_user(user_id):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise UserNotFoundError(user_id)
    return user

# 有問題的程式碼
def get_order_info(data):
    order = db.query(Order).filter(Order.id == data.get('id')).first()
    return order  # ❌ 沒有處理 None 的情況！
```

### 2.2 識別差異

| 面向 | 正常 | 異常 |
|------|------|------|
| 輸入資料 | `{"id": 123}` | `{"id": null}` |
| 環境 | 開發環境 | 生產環境 |
| 時間 | 白天 | 凌晨（cron job） |
| 使用者 | 一般用戶 | 管理員 |

### 2.3 依賴分析

```markdown
問題函式的依賴：
├── get_order_info()
│   ├── db.query() - 資料庫連線正常？
│   ├── Order model - schema 正確？
│   └── data.get('id') - 輸入驗證？
```

## Phase 3: 假設與測試 🧪

### 3.1 建立明確假設

```markdown
假設格式：
"我認為 [X] 導致 [Y]，因為 [Z]"

範例：
"我認為 get_order_info 回傳 None 是因為 data['id'] 為 null，
 導致資料庫查詢找不到對應的訂單。"
```

### 3.2 設計驗證實驗

```python
# 驗證假設的最小測試
def test_hypothesis():
    # 模擬問題情境
    data = {"id": None}
    result = get_order_info(data)

    # 驗證假設
    assert result is None, "假設正確：None id 導致 None 結果"
```

### 3.3 逐步縮小範圍

```markdown
1. 問題在前端還是後端？ → 後端
2. 問題在 API 層還是 Service 層？ → Service 層
3. 問題在 get_order_info 還是 process_order？ → get_order_info
4. 問題是輸入驗證還是資料庫查詢？ → 輸入驗證缺失
```

## Phase 4: 實作修復 🔧

### 4.1 先寫測試

```python
# 寫一個會觸發 bug 的測試
def test_get_order_info_with_none_id_raises_error():
    """Bug #456: None id 應該拋出錯誤而非回傳 None"""
    data = {"id": None}
    with pytest.raises(InvalidOrderIdError):
        get_order_info(data)
```

### 4.2 實作根因修復

```python
# 修復根本原因，而非症狀
def get_order_info(data):
    order_id = data.get('id')

    # 根因修復：驗證輸入
    if order_id is None:
        raise InvalidOrderIdError("Order ID cannot be None")

    order = db.query(Order).filter(Order.id == order_id).first()

    if order is None:
        raise OrderNotFoundError(order_id)

    return order.user_id, order.id
```

### 4.3 驗證修復

```bash
# 1. 新測試通過
pytest test_order.py::test_get_order_info_with_none_id_raises_error -v

# 2. 所有現有測試仍通過
pytest test_order.py -v

# 3. 手動驗證原始問題已解決
```

## 三次失敗規則

```markdown
如果同一個問題嘗試修復 3 次以上仍失敗：

🛑 停止繼續嘗試修補
🤔 質疑底層架構是否有問題
💬 尋求團隊討論
📐 可能需要重新設計而非修補
```

## 常見藉口（不接受）

| 藉口 | 為什麼不接受 |
|------|-------------|
| "很簡單，試一下就知道" | 簡單問題更應該系統化處理 |
| "我們很趕時間" | 亂猜更浪費時間 |
| "先試這個快速修復" | 快速修復常引入新問題 |
| "我之前遇過類似的" | 每個 bug 都需要獨立分析 |

## 除錯工具箱

### 日誌分析

```bash
# 搜尋錯誤
grep -r "ERROR\|Exception" logs/

# 時間範圍過濾
awk '/2024-01-15 10:00/,/2024-01-15 11:00/' app.log

# 統計錯誤類型
grep "ERROR" app.log | cut -d':' -f4 | sort | uniq -c | sort -rn
```

### 網路診斷

```bash
# 檢查連線
curl -v http://api.example.com/health

# DNS 解析
nslookup api.example.com

# 埠口監聽
netstat -tlnp | grep 8080
```

### 資料庫診斷

```sql
-- 檢查最近的錯誤
SELECT * FROM logs WHERE level = 'ERROR' ORDER BY created_at DESC LIMIT 10;

-- 檢查資料一致性
SELECT COUNT(*) FROM orders WHERE user_id IS NULL;
```

## Checklist

除錯前確認：

- [ ] 完整閱讀錯誤訊息和 stack trace
- [ ] 能穩定重現問題
- [ ] 檢查過最近的變更
- [ ] 追蹤過資料流
- [ ] 比對過正常與異常的差異
- [ ] 建立了明確的假設
- [ ] 驗證了假設
- [ ] 寫了會觸發 bug 的測試
- [ ] 修復了根本原因（非症狀）
- [ ] 所有測試通過
