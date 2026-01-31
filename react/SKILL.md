---
name: react
description: |
  React 前端框架開發指南。涵蓋元件開發、Hooks、狀態管理、JSX 語法等核心概念。
  Use when: 開發 React 應用、建立元件、管理狀態、使用 Hooks，or when user mentions React, component, hook, useState, JSX.
  Triggers: "React", "react", "component", "元件", "hook", "useState", "useEffect", "useContext", "useRef", "JSX", "props", "state", "前端開發", "單頁應用", "SPA"
version: 1.0.0
---

# React Skill

React 是用於建立使用者介面的 JavaScript 函式庫，採用元件化開發模式。

## When to Use This Skill

- 開發 React 應用程式
- 建立可重用元件
- 管理應用狀態
- 使用 React Hooks
- 處理表單與事件
- 整合外部 API

## Quick Reference

### 建立專案

```bash
# 使用 Vite（推薦）
npm create vite@latest my-app -- --template react-ts

# 使用 Create React App
npx create-react-app my-app --template typescript
```

### 函數元件基礎

```jsx
function Welcome({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// 使用
<Welcome name="React" />
```

### 常用 Hooks

```jsx
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

function Example() {
  // 狀態管理
  const [count, setCount] = useState(0);

  // 副作用
  useEffect(() => {
    document.title = `Count: ${count}`;
    return () => { /* cleanup */ };
  }, [count]);

  // DOM 參照
  const inputRef = useRef(null);

  // 記憶化回調
  const handleClick = useCallback(() => {
    setCount(c => c + 1);
  }, []);

  // 記憶化計算
  const doubled = useMemo(() => count * 2, [count]);

  return (
    <div>
      <p>Count: {count}, Doubled: {doubled}</p>
      <button onClick={handleClick}>Increment</button>
      <input ref={inputRef} />
    </div>
  );
}
```

### 條件渲染與列表

```jsx
function List({ items, showAll }) {
  return (
    <ul>
      {showAll && <li>All Items:</li>}
      {items.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
}
```

### 表單處理

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

### Context API

```jsx
const ThemeContext = createContext('light');

function App() {
  return (
    <ThemeContext.Provider value="dark">
      <Toolbar />
    </ThemeContext.Provider>
  );
}

function ThemedButton() {
  const theme = useContext(ThemeContext);
  return <button className={theme}>Themed</button>;
}
```

### 自訂 Hook

```jsx
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [url]);

  return { data, loading, error };
}

// 使用
function UserProfile({ userId }) {
  const { data, loading, error } = useFetch(`/api/users/${userId}`);
  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;
  return <div>{data.name}</div>;
}
```

## Reference Files

詳細文檔請參考 `references/` 目錄：

| 檔案 | 內容 |
|------|------|
| `getting_started.md` | 快速入門指南 |
| `components.md` | 元件開發指南 |
| `hooks.md` | Hooks 完整參考 |
| `state.md` | 狀態管理 |
| `api.md` | API 參考 |
| `other.md` | 進階主題 |

## Resources

- 官方文檔: https://react.dev/
- GitHub: https://github.com/facebook/react
