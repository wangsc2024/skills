# SKILL.md 輸出模板

## 標準結構

每個生成的知識庫應包含以下區塊：

```markdown
# [Framework/Tool Name] Skill

## Description

[一句話描述這個知識庫的用途和適用場景]

## When to Use

- [使用場景 1]
- [使用場景 2]
- [使用場景 3]

## Core Concepts

### [概念 1 名稱]

[概念解釋]

```[language]
[程式碼範例]
```

### [概念 2 名稱]

[概念解釋]

## Quick Reference

### 常用 API

| API/方法 | 用途 | 範例 |
|----------|------|------|
| `method1()` | 描述 | `code` |
| `method2()` | 描述 | `code` |

### 常見模式

```[language]
// 模式名稱
[程式碼範例]
```

## Code Examples

### Example 1: [基礎用法]

```[language]
[完整可執行的程式碼]
```

### Example 2: [進階用法]

```[language]
[完整可執行的程式碼]
```

## Common Pitfalls

- ❌ [錯誤做法]
  ```[language]
  [錯誤程式碼]
  ```
  ✅ [正確做法]
  ```[language]
  [正確程式碼]
  ```

## Related Resources

- [資源名稱](URL)
- [資源名稱](URL)
```

## 格式規範

### 標題層級

- `#` - Skill 名稱（僅一個）
- `##` - 主要區塊（Description, When to Use 等）
- `###` - 子區塊（概念名稱、範例標題）
- `####` - 細項（極少使用）

### 程式碼區塊

始終標記語言類型：

```markdown
```python
def hello():
    print("Hello")
```
```

常用語言標記：
- `python`
- `javascript` / `js`
- `typescript` / `ts`
- `jsx` / `tsx`
- `bash` / `shell`
- `json`
- `yaml`
- `sql`
- `html`
- `css`

### 表格

用於 API 參考和對照：

```markdown
| 欄位 1 | 欄位 2 | 欄位 3 |
|--------|--------|--------|
| 值 1   | 值 2   | 值 3   |
```

### 列表

用於使用場景和要點：

```markdown
- 項目 1
- 項目 2
  - 子項目 2.1
  - 子項目 2.2
```

## 品質檢查

生成後確認：

- [ ] 包含所有必要區塊
- [ ] 程式碼區塊都有語言標記
- [ ] 至少 5 個實用範例
- [ ] 程式碼可直接執行
- [ ] 無格式錯誤
