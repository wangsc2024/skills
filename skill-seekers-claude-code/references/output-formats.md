# 輸出格式規範

## 概述

Skill Seekers 生成的輸出遵循標準化格式，確保與 Claude AI 的 Skill 系統相容。

## 目錄結構

```
output/[skill-name]/
├── SKILL.md              # 主要技能說明（必要）
├── metadata.json         # 中繼資料（建議）
└── references/           # 分類參考文件
    ├── index.md          # 目錄索引
    ├── getting_started.md
    ├── core_concepts.md
    ├── api_reference.md
    ├── guides.md
    ├── examples.md
    ├── configuration.md
    ├── troubleshooting.md
    └── advanced.md
```

## SKILL.md 格式

### 必要區塊

```markdown
# [Framework Name] Skill

## Description
一句話描述這個 skill 的用途和適用場景。

## When to Use
- 使用場景 1
- 使用場景 2
- 使用場景 3

## Core Concepts
### 概念名稱
解釋文字和程式碼範例。

## Quick Reference
### 常用 API
| API | 用途 | 範例 |
|-----|------|------|

## Code Examples
### Example 1: 標題
完整可執行的程式碼。

## Related Resources
- 連結列表
```

### 完整範本

```markdown
# React Skill

## Description

React 18+ 現代前端開發完整知識庫，涵蓋 Hooks、Server Components、狀態管理與效能優化最佳實踐。

## When to Use

- 建立互動式 Web 應用程式
- 需要元件化、可重用的 UI 架構
- 使用 Next.js、Remix 等 React 生態系框架
- 需要高效能的單頁應用程式 (SPA)

## Statistics

| 項目 | 數值 |
|------|------|
| 總頁數 | 150 |
| 程式碼區塊 | 320 |
| 偵測語言 | JavaScript, TypeScript, JSX |

## Categories

- **API Reference**: 45 頁
- **Guides**: 35 頁
- **Getting Started**: 20 頁
- **Examples**: 18 頁

## Core Concepts

### Components

React 的基本建構單元，使用函數元件搭配 Hooks。

```jsx
function Welcome({ name }) {
  return <h1>Hello, {name}</h1>;
}
```

### Hooks

讓函數元件擁有狀態和生命週期功能。

| Hook | 用途 |
|------|------|
| `useState` | 管理狀態 |
| `useEffect` | 處理副作用 |
| `useContext` | 跨元件共享資料 |

## Quick Reference

### 狀態更新模式

```jsx
// 基本更新
setState(newValue);

// 基於前值更新
setState(prev => prev + 1);

// 物件更新
setState(prev => ({ ...prev, key: value }));
```

## Code Examples

### Example 1: Counter

```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <button onClick={() => setCount(c => c + 1)}>
      Clicked {count} times
    </button>
  );
}
```

### Example 2: Data Fetching

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

  if (loading) return <div>Loading...</div>;
  return <div>{user.name}</div>;
}
```

## Common Pitfalls

- ❌ 直接修改 state 物件
  ```jsx
  state.items.push(newItem); // 錯誤
  ```
  ✅ 使用展開運算子
  ```jsx
  setItems([...items, newItem]); // 正確
  ```

## Related Resources

- [React 官方文件](https://react.dev)
- [GitHub](https://github.com/facebook/react)
```

## metadata.json 格式

```json
{
  "skill_name": "react",
  "display_name": "React",
  "version": "1.0.0",
  "generated_at": "2025-01-29T12:00:00Z",
  "source_url": "https://react.dev",
  "source_type": "documentation",
  "language": "zh-TW",
  "statistics": {
    "total_pages": 150,
    "total_code_blocks": 320,
    "languages_detected": ["javascript", "typescript", "jsx"]
  },
  "categories": [
    {
      "name": "api_reference",
      "display_name": "API Reference",
      "file": "references/api_reference.md",
      "page_count": 45
    },
    {
      "name": "guides",
      "display_name": "Guides",
      "file": "references/guides.md",
      "page_count": 35
    }
  ]
}
```

## 分類參考文件格式

### references/index.md

```markdown
# 參考文件索引

## 文件列表

| 類別 | 檔案 | 頁數 |
|------|------|------|
| 入門指南 | [getting_started.md](./getting_started.md) | 20 |
| 核心概念 | [core_concepts.md](./core_concepts.md) | 15 |
| API 參考 | [api_reference.md](./api_reference.md) | 45 |
```

### 各分類文件

```markdown
# API Reference

## useState

React Hook，用於在函數元件中添加狀態。

### 語法

```jsx
const [state, setState] = useState(initialValue);
```

### 參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `initialValue` | `any` | 初始狀態值 |

### 回傳值

回傳一個陣列：`[currentState, setterFunction]`

### 範例

```jsx
const [count, setCount] = useState(0);
const [user, setUser] = useState({ name: '', email: '' });
```

---

## useEffect

React Hook，用於處理副作用。

[... 更多內容 ...]
```

## 品質檢查清單

生成輸出後，確認以下項目：

- [ ] SKILL.md 包含所有必要區塊
- [ ] 所有程式碼區塊都有標記語言
- [ ] 至少 5 個實用程式碼範例
- [ ] 範例程式碼可直接執行
- [ ] metadata.json 格式正確
- [ ] 分類文件都已生成
- [ ] 無拼寫或格式錯誤

## 檔案大小建議

| 檔案類型 | 建議大小 |
|----------|----------|
| SKILL.md | 10-50 KB |
| 單一分類文件 | 20-100 KB |
| 整體 Skill | < 5 MB |

過大的檔案可能影響 Claude 的處理效率，建議進行適當的精簡或分割。
