# 輸出範例

## 範例 1：React 知識庫

以下是一個完整的 React Skill 輸出範例：

---

# React Skill

## Description

React 18+ 現代前端開發完整知識庫，涵蓋 Hooks、Server Components、狀態管理與效能優化最佳實踐。

## When to Use

- 建立互動式 Web 應用程式
- 需要元件化、可重用的 UI 架構
- 使用 Next.js、Remix 等 React 生態系框架
- 需要高效能的單頁應用程式 (SPA)

## Core Concepts

### JSX

JSX 是 JavaScript 的語法擴展，讓你在 JS 中撰寫類似 HTML 的標記。

```jsx
const element = <h1>Hello, {userName}</h1>;
```

### Components

元件是 React 的基本建構單元，現代 React 推薦使用函數元件：

```jsx
function Welcome({ name }) {
  return <h1>Hello, {name}</h1>;
}
```

### Hooks

Hooks 讓函數元件擁有狀態和生命週期功能：

| Hook | 用途 |
|------|------|
| `useState` | 管理狀態 |
| `useEffect` | 處理副作用 |
| `useContext` | 跨元件共享資料 |
| `useMemo` | 快取計算結果 |
| `useCallback` | 快取函數 |

## Quick Reference

### 狀態更新模式

```jsx
// 基本更新
setState(newValue);

// 基於前值更新
setState(prev => prev + 1);

// 物件更新
setState(prev => ({ ...prev, key: value }));

// 陣列新增
setItems(prev => [...prev, newItem]);

// 陣列移除
setItems(prev => prev.filter(item => item.id !== targetId));
```

### useEffect 依賴

```jsx
// 每次 render 執行
useEffect(() => { });

// 只在 mount 執行
useEffect(() => { }, []);

// 依賴變化時執行
useEffect(() => { }, [dep]);

// 清理函數
useEffect(() => {
  const sub = subscribe();
  return () => sub.unsubscribe();
}, []);
```

## Code Examples

### Example 1: Counter

```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>計數: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}
```

### Example 2: 表單處理

```jsx
import { useState } from 'react';

function ContactForm() {
  const [form, setForm] = useState({ name: '', email: '' });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('提交:', form);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="name" value={form.name} onChange={handleChange} />
      <input name="email" value={form.email} onChange={handleChange} />
      <button type="submit">送出</button>
    </form>
  );
}
```

### Example 3: API 資料抓取

```jsx
import { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => {
        setUser(data);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>載入中...</div>;
  return <div>{user.name}</div>;
}
```

### Example 4: 自訂 Hook

```jsx
import { useState, useEffect } from 'react';

function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : initialValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}

// 使用
function App() {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  return <button onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}>
    {theme}
  </button>;
}
```

### Example 5: Context 狀態管理

```jsx
import { createContext, useContext, useState } from 'react';

const AuthContext = createContext(null);

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const login = (data) => setUser(data);
  const logout = () => setUser(null);
  
  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

function useAuth() {
  return useContext(AuthContext);
}

// 使用
function Navbar() {
  const { user, logout } = useAuth();
  return user ? <button onClick={logout}>登出</button> : <a href="/login">登入</a>;
}
```

## Common Pitfalls

- ❌ 直接修改 state
  ```jsx
  items.push(newItem); // 錯誤
  ```
  ✅ 建立新陣列
  ```jsx
  setItems([...items, newItem]); // 正確
  ```

- ❌ useEffect 缺少依賴
  ```jsx
  useEffect(() => {
    fetchData(userId);
  }, []); // 缺少 userId
  ```
  ✅ 加入依賴
  ```jsx
  useEffect(() => {
    fetchData(userId);
  }, [userId]);
  ```

- ❌ 在條件式中使用 Hooks
  ```jsx
  if (condition) {
    const [state, setState] = useState(); // 錯誤
  }
  ```
  ✅ 始終在頂層呼叫
  ```jsx
  const [state, setState] = useState();
  if (condition) { /* 使用 state */ }
  ```

## Related Resources

- [React 官方文件](https://react.dev)
- [React GitHub](https://github.com/facebook/react)

---

## 範例 2：FastAPI 知識庫（精簡版）

# FastAPI Skill

## Description

FastAPI 是現代、高效能的 Python Web 框架，用於建立 API，支援自動文件生成和型別檢查。

## When to Use

- 建立 RESTful API
- 需要自動生成 OpenAPI 文件
- 需要高效能的 Python 後端
- 使用 Python 型別提示

## Core Concepts

### 基本路由

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### Pydantic 模型

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
def create_item(item: Item):
    return item
```

## Quick Reference

| 裝飾器 | HTTP 方法 |
|--------|-----------|
| `@app.get()` | GET |
| `@app.post()` | POST |
| `@app.put()` | PUT |
| `@app.delete()` | DELETE |

## Code Examples

### Example 1: CRUD API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
items = {}

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    items[item_id] = item
    return item

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

## Related Resources

- [FastAPI 官方文件](https://fastapi.tiangolo.com)
- [GitHub](https://github.com/tiangolo/fastapi)
