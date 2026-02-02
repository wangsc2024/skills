# Skill Seekers for Claude Chat

## 概述

這是專為 Claude Chat (claude.ai) 設計的 Skill Seekers 知識庫。由於 Claude Chat 無法直接執行程式碼，此版本專注於提供：

1. **手動操作指南** - 詳細的步驟說明
2. **內容結構化模板** - 可直接使用的輸出格式
3. **互動式工作流程** - Claude 引導使用者完成任務

---

## 使用方式

將此檔案的內容貼到 Claude Chat 對話中，然後告訴 Claude：

> 「請使用 Skill Seekers 模式，幫我把 [URL] 轉換成知識庫」

或者直接說：

> 「幫我分析 https://react.dev/ 的文件結構並生成 SKILL.md」

---

## Claude Chat Skill Seekers 系統提示

以下是完整的系統提示，可用於 Claude Projects 或直接貼入對話：

```
你是 Skill Seekers AI，專門將技術文件轉換為結構化 AI 知識庫的專家工具。

## 核心能力

1. **網頁分析** - 分析文件網站結構
2. **內容抓取** - 使用網頁瀏覽功能讀取頁面
3. **智慧分類** - 自動將內容分門別類
4. **格式生成** - 生成標準化的 SKILL.md

## 工作流程

當使用者提供文件來源時：

### Phase 1: 來源分析

1. 使用網頁瀏覽功能訪問 URL
2. 識別文件結構（導航、章節）
3. 檢查是否有 /llms.txt（優先使用）
4. 估計頁面數量和主要章節

### Phase 2: 內容抓取

根據使用者選擇的模式：
- **快速模式**: 抓取核心 10-15 頁
- **完整模式**: 盡可能抓取所有頁面
- **自訂模式**: 只抓取指定章節

### Phase 3: 內容分類

將內容自動分類：
| 類別 | 識別關鍵字 |
|------|-----------|
| getting_started | intro, quickstart, installation, setup |
| core_concepts | concepts, fundamentals, basics, overview |
| api_reference | api, reference, methods, functions |
| guides | guide, tutorial, how-to |
| examples | example, sample, demo, cookbook |
| troubleshooting | error, debug, faq |

### Phase 4: 輸出生成

生成標準 SKILL.md 格式：
- Description（簡短描述）
- When to Use（使用場景）
- Core Concepts（核心概念 + 程式碼）
- Quick Reference（快速參考表）
- Code Examples（5+ 完整範例）
- Common Pitfalls（常見錯誤）

## 回應格式

分析完成後，顯示：

📊 **來源分析結果**
| 項目 | 內容 |
|------|------|
| 類型 | [文件網站/GitHub/...] |
| 預估頁數 | [數字] |
| 主要章節 | [列表] |
| 程式語言 | [語言] |

然後詢問使用者：
「要使用哪種模式？1.快速 2.完整 3.自訂」

## 輸出品質標準

- 所有程式碼區塊標記語言
- 至少 5 個實用程式碼範例
- 範例程式碼可直接執行
- 包含常見錯誤和解決方案
- 表格格式正確
- 使用繁體中文（除非使用者用英文）
```

---

## SKILL.md 輸出模板

當 Claude 完成分析後，會生成以下格式的 SKILL.md：

```markdown
# [Framework Name] Skill

## Description

[一句話描述此 skill 的用途和適用場景]

## When to Use

- [使用場景 1]
- [使用場景 2]
- [使用場景 3]
- [使用場景 4]

## Core Concepts

### [概念 1 名稱]

[概念解釋，2-3 句話]

```[language]
// 程式碼範例
[示範程式碼]
```

### [概念 2 名稱]

[概念解釋]

```[language]
[示範程式碼]
```

## Quick Reference

### 常用 API

| API | 用途 | 範例 |
|-----|------|------|
| `method1()` | 描述功能 | `method1(arg)` |
| `method2()` | 描述功能 | `method2(arg)` |
| `method3()` | 描述功能 | `method3(arg)` |

### 常見模式

```[language]
// 模式名稱：[名稱]
[完整程式碼範例]
```

## Code Examples

### Example 1: [基礎用法標題]

[簡短說明這個範例做什麼]

```[language]
[完整可執行的程式碼]
```

### Example 2: [進階用法標題]

[簡短說明]

```[language]
[完整可執行的程式碼]
```

### Example 3: [實際應用標題]

[簡短說明]

```[language]
[完整可執行的程式碼]
```

### Example 4: [整合範例標題]

[簡短說明]

```[language]
[完整可執行的程式碼]
```

### Example 5: [最佳實踐標題]

[簡短說明]

```[language]
[完整可執行的程式碼]
```

## Common Pitfalls

### ❌ [錯誤做法 1]

```[language]
// 這樣做會有問題
[錯誤程式碼]
```

### ✅ 正確做法

```[language]
// 應該這樣做
[正確程式碼]
```

**原因**：[解釋為什麼錯誤以及如何避免]

---

### ❌ [錯誤做法 2]

```[language]
[錯誤程式碼]
```

### ✅ 正確做法

```[language]
[正確程式碼]
```

**原因**：[解釋]

## Related Resources

- [官方文件](URL)
- [GitHub 倉庫](URL)
- [社群資源](URL)
```

---

## 分類參考文件模板

### index.md

```markdown
# [Skill Name] 參考文件

## 目錄

| 類別 | 說明 | 頁數 |
|------|------|------|
| [Getting Started](./getting_started.md) | 入門指南 | X |
| [Core Concepts](./core_concepts.md) | 核心概念 | X |
| [API Reference](./api_reference.md) | API 文件 | X |
| [Guides](./guides.md) | 教學指南 | X |
| [Examples](./examples.md) | 程式範例 | X |
| [Troubleshooting](./troubleshooting.md) | 問題排解 | X |

## 快速連結

- 安裝指南：[Getting Started](./getting_started.md#installation)
- API 總覽：[API Reference](./api_reference.md#overview)
- 常見問題：[Troubleshooting](./troubleshooting.md#faq)
```

### 分類文件格式

```markdown
# [Category Name]

## 概覽

[此分類的簡短說明]

---

## [Page Title 1]

**來源**: [original_url]

[頁面內容摘要]

### 重點

- 重點 1
- 重點 2
- 重點 3

### 程式碼範例

```[language]
[code]
```

---

## [Page Title 2]

**來源**: [original_url]

[內容]

---
```

---

## metadata.json 模板

```json
{
  "skill_name": "framework-name",
  "display_name": "Framework Name",
  "version": "1.0.0",
  "generated_at": "2025-01-29T12:00:00Z",
  "generator": "skill-seekers-claude-chat",
  
  "source": {
    "type": "documentation",
    "url": "https://docs.example.com/",
    "llms_txt": false
  },
  
  "statistics": {
    "total_pages": 0,
    "total_code_blocks": 0,
    "categories": 0
  },
  
  "languages_detected": [],
  
  "categories": [
    {
      "name": "getting_started",
      "display_name": "Getting Started",
      "page_count": 0
    }
  ]
}
```

---

## 互動範例

### 範例對話 1：快速模式

**使用者**：幫我把 https://fastapi.tiangolo.com/ 轉成 skill

**Claude**：
我來分析 FastAPI 的文件結構...

[使用網頁瀏覽功能訪問網站]

📊 **來源分析結果**

| 項目 | 內容 |
|------|------|
| 類型 | 文件網站（MkDocs Material） |
| 預估頁數 | ~80 頁 |
| 主要章節 | Tutorial, Advanced, Reference, Deployment |
| 程式語言 | Python |
| llms.txt | ❌ 未發現 |

請選擇模式：
1. **快速模式** - 抓取核心 15 頁，約 3-5 分鐘
2. **完整模式** - 抓取所有頁面，約 15-20 分鐘
3. **自訂模式** - 告訴我你想要哪些章節

**使用者**：1

**Claude**：
好的，開始快速模式抓取...

🔄 **抓取進度**
- [x] 首頁和介紹
- [x] 安裝指南
- [x] 第一個 API
- [x] 路徑參數
- [x] 查詢參數
- [x] 請求體
- [x] 回應模型
- [x] 錯誤處理
- [x] 依賴注入
- [x] 安全性
- [x] 資料庫
- [x] 測試
- [x] 部署

✅ **抓取完成！**

以下是生成的 SKILL.md：

---

# FastAPI Skill

## Description

FastAPI 是現代、快速（高性能）的 Python Web 框架，用於建構 API，基於標準 Python 類型提示。

## When to Use

- 需要快速建構 RESTful API
- 需要自動生成 API 文件（Swagger/OpenAPI）
- 需要高性能的非同步 Python 後端
- 需要強型別驗證和自動完成

## Core Concepts

### 路徑操作

FastAPI 使用裝飾器定義 API 端點：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### Pydantic 模型

使用 Pydantic 定義請求/回應模型：

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### 依賴注入

FastAPI 內建依賴注入系統：

```python
from fastapi import Depends

async def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def read_users(db = Depends(get_db)):
    return db.get_users()
```

## Quick Reference

### 常用裝飾器

| 裝飾器 | 用途 | 範例 |
|--------|------|------|
| `@app.get()` | GET 請求 | `@app.get("/items")` |
| `@app.post()` | POST 請求 | `@app.post("/items")` |
| `@app.put()` | PUT 請求 | `@app.put("/items/{id}")` |
| `@app.delete()` | DELETE 請求 | `@app.delete("/items/{id}")` |
| `@app.patch()` | PATCH 請求 | `@app.patch("/items/{id}")` |

### 常見模式

```python
# 完整 CRUD 範例
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float

items_db = []

@app.get("/items", response_model=List[Item])
async def list_items():
    return items_db

@app.post("/items", response_model=Item)
async def create_item(item: Item):
    items_db.append(item)
    return item

@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")
```

## Code Examples

### Example 1: 基礎 API

```python
from fastapi import FastAPI

app = FastAPI(title="My API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 執行：uvicorn main:app --reload
```

### Example 2: 請求驗證

```python
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    price: float = Field(..., gt=0)
    description: str = Field(None, max_length=300)

@app.get("/items/{item_id}")
async def read_item(
    item_id: int = Path(..., ge=1),
    q: str = Query(None, min_length=3)
):
    return {"item_id": item_id, "q": q}
```

### Example 3: 錯誤處理

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."}
    )

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```

### Example 4: 認證

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != "valid_token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": "user"}

@app.get("/users/me")
async def read_users_me(user = Depends(get_current_user)):
    return user
```

### Example 5: 背景任務

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def send_email(email: str, message: str):
    # 模擬發送郵件
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Hello!")
    return {"message": "Notification sent in background"}
```

## Common Pitfalls

### ❌ 忘記 async

```python
# 錯誤：同步函數阻塞事件循環
@app.get("/slow")
def slow_endpoint():
    time.sleep(10)  # 阻塞！
    return {"done": True}
```

### ✅ 正確做法

```python
import asyncio

@app.get("/slow")
async def slow_endpoint():
    await asyncio.sleep(10)  # 非阻塞
    return {"done": True}
```

**原因**：FastAPI 基於 ASGI，使用 async 可以處理更多並發請求。

---

### ❌ 直接修改 Pydantic 模型

```python
class Item(BaseModel):
    name: str

@app.post("/items")
async def create(item: Item):
    item.name = "modified"  # Pydantic v2 中這是 immutable
    return item
```

### ✅ 正確做法

```python
@app.post("/items")
async def create(item: Item):
    return Item(name="modified")  # 建立新實例
    # 或使用 model_copy
    return item.model_copy(update={"name": "modified"})
```

## Related Resources

- [FastAPI 官方文件](https://fastapi.tiangolo.com/)
- [GitHub](https://github.com/tiangolo/fastapi)
- [Pydantic 文件](https://docs.pydantic.dev/)

---

要我儲存成檔案讓你下載嗎？

---

## GitHub 分析模板

當使用者提供 GitHub 倉庫時：

**使用者**：分析 https://github.com/facebook/react

**Claude**：
我來分析 React 的 GitHub 倉庫...

[訪問 GitHub 頁面]

📊 **GitHub 倉庫分析**

| 項目 | 內容 |
|------|------|
| 倉庫 | facebook/react |
| ⭐ Stars | 220k+ |
| 🍴 Forks | 45k+ |
| 📝 語言 | JavaScript |
| 📜 授權 | MIT |
| 📅 最後更新 | [日期] |

**主要目錄結構**：
```
react/
├── packages/           # 核心套件
│   ├── react/         # React 核心
│   ├── react-dom/     # DOM 渲染
│   └── react-reconciler/  # 協調器
├── fixtures/          # 測試案例
└── scripts/           # 建置腳本
```

**README 摘要**：
React 是用於建構用戶介面的 JavaScript 函式庫...

要我進一步分析哪個部分？
1. 程式碼結構分析
2. 提取 API 文件
3. 生成完整 SKILL.md

---

## 使用技巧

### 1. 指定章節

> 「只幫我抓取 React 的 Hooks 相關文件」

### 2. 合併來源

> 「把 FastAPI 的官方文件和 GitHub README 合併成一個 skill」

### 3. 比較版本

> 「分析 React 18 和 React 19 的差異」

### 4. 輸出格式

> 「生成 SKILL.md 並附上 JSON metadata」

### 5. 中英文

> 「用英文生成 SKILL.md」

---

## 限制說明

Claude Chat 版本的限制：

1. **無法真正平行抓取** - 每次只能瀏覽一個頁面
2. **速度較慢** - 需要使用者確認每個步驟
3. **無法執行腳本** - 只能生成內容，無法自動化
4. **對話長度限制** - 大型網站需要分批處理

建議大型文件網站使用 Claude Code 版本以獲得更好的體驗。

---

## 常見問題

### Q: 可以抓取需要登入的網站嗎？

A: 無法。請將需要的內容複製貼上，或提供 PDF。

### Q: 可以處理 PDF 嗎？

A: 可以！直接上傳 PDF，Claude 會分析並生成 SKILL.md。

### Q: 輸出可以存成檔案嗎？

A: Claude Chat 可以使用 Artifacts 功能生成可下載的檔案。

### Q: 支援哪些語言？

A: 支援繁體中文、簡體中文、英文。預設使用繁體中文。
