---
name: react
description: React framework for building user interfaces. Use for React components, hooks, state management, JSX, and modern frontend development.
---

# React Skill

React 是用於建構使用者介面的 JavaScript 函式庫。

## When to Use This Skill

- 建構 React 元件和應用
- 使用 React Hooks
- 管理元件狀態
- 處理副作用和生命週期
- 實作 Context 和狀態管理

## 核心概念

```
React 核心概念
├── Components       # 可重用的 UI 單元
├── JSX              # JavaScript + XML 語法
├── Props            # 元件間資料傳遞
├── State            # 元件內部狀態
├── Hooks            # 函式元件的狀態和副作用
└── Context          # 跨層級狀態共享
```

## Quick Start

### 函式元件

```jsx
function Welcome({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// 使用
<Welcome name="React" />
```

### useState - 狀態管理

```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
      <button onClick={() => setCount(prev => prev - 1)}>-1</button>
    </div>
  );
}
```

### useEffect - 副作用

```jsx
import { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => {
        setUser(data);
        setLoading(false);
      });
    
    // Cleanup function
    return () => console.log('Cleanup');
  }, [userId]); // 依賴陣列

  if (loading) return <p>Loading...</p>;
  return <div>{user?.name}</div>;
}
```

## 常用 Hooks

### useRef - 保存參考

```jsx
import { useRef, useEffect } from 'react';

function TextInput() {
  const inputRef = useRef(null);
  
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return <input ref={inputRef} />;
}
```

### useMemo - 記憶計算值

```jsx
import { useMemo } from 'react';

function ExpensiveComponent({ items, filter }) {
  const filteredItems = useMemo(() => {
    return items.filter(item => item.includes(filter));
  }, [items, filter]);

  return <ul>{filteredItems.map(item => <li key={item}>{item}</li>)}</ul>;
}
```

### useCallback - 記憶函式

```jsx
import { useCallback, useState } from 'react';

function Parent() {
  const [count, setCount] = useState(0);
  
  const handleClick = useCallback(() => {
    setCount(prev => prev + 1);
  }, []);

  return <Child onClick={handleClick} />;
}
```

### useContext - 跨層級狀態

```jsx
import { createContext, useContext, useState } from 'react';

const ThemeContext = createContext('light');

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function ThemedButton() {
  const { theme, setTheme } = useContext(ThemeContext);
  return (
    <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      Current: {theme}
    </button>
  );
}
```

### useReducer - 複雜狀態邏輯

```jsx
import { useReducer } from 'react';

const reducer = (state, action) => {
  switch (action.type) {
    case 'increment': return { count: state.count + 1 };
    case 'decrement': return { count: state.count - 1 };
    default: return state;
  }
};

function Counter() {
  const [state, dispatch] = useReducer(reducer, { count: 0 });
  
  return (
    <div>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>+</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
    </div>
  );
}
```

## 表單處理

### 受控元件

```jsx
function Form() {
  const [formData, setFormData] = useState({ name: '', email: '' });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="name" value={formData.name} onChange={handleChange} />
      <input name="email" value={formData.email} onChange={handleChange} />
      <button type="submit">Submit</button>
    </form>
  );
}
```

## 列表和 Key

```jsx
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
}
```

## 條件渲染

```jsx
function Greeting({ isLoggedIn }) {
  return (
    <div>
      {isLoggedIn ? <UserDashboard /> : <LoginForm />}
      {isLoggedIn && <LogoutButton />}
    </div>
  );
}
```

## 自訂 Hook

```jsx
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : initialValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}

// 使用
function App() {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  // ...
}
```

## 效能優化

### React.memo

```jsx
const ExpensiveComponent = React.memo(function({ data }) {
  // 只在 data 改變時重新渲染
  return <div>{data}</div>;
});
```

### lazy + Suspense

```jsx
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <HeavyComponent />
    </Suspense>
  );
}
```

## React 19 新特性

### use Hook

```jsx
import { use } from 'react';

function Comments({ commentsPromise }) {
  const comments = use(commentsPromise);
  return comments.map(c => <p key={c.id}>{c.text}</p>);
}
```

### Server Components

```jsx
// app/page.tsx (Server Component)
async function Page() {
  const data = await fetch('https://api.example.com/data');
  return <ClientComponent data={data} />;
}
```

## Reference Files

| 檔案 | 內容 |
|------|------|
| hooks.md | Hooks API 完整參考 |
| components.md | 內建元件參考 |
| api.md | React API 參考 |
| state.md | 狀態管理指南 |

## Resources

- [React 官方文件](https://react.dev/)
- [React GitHub](https://github.com/facebook/react)
- [React TypeScript 指南](https://react-typescript-cheatsheet.netlify.app/)
