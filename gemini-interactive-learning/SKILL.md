---
name: gemini-interactive-learning
description: |
  Gemini AI 互動式學習網站開發工具。使用 Next.js 14+、Tailwind CSS、Shadcn/ui 建立教學網站。
  Use when: 建立 Gemini 教學網站、設計互動式學習元件、開發 AI 教學前端功能，or when user mentions Gemini 學習, 互動式教學, learning website.
  Triggers: "Gemini 學習", "互動式教學", "learning website", "教學網站", "Gemini tutorial", "AI 教學網站"
version: 2.0.0
---

# Gemini AI 互動式學習網站開發 Skill

使用 Next.js 14+、Tailwind CSS、Shadcn/ui 建立 Gemini AI 互動式教學網站。

## When to Use This Skill

- 建立 Gemini 教學網站相關功能
- 設計互動式學習元件
- 實作動態檢視或版面配置功能
- 開發 AI 教學相關的前端功能

## 技術棧

| 類別 | 技術 |
|------|------|
| 框架 | Next.js 14+ (App Router) |
| 樣式 | Tailwind CSS |
| UI 元件 | Shadcn/ui |
| 動畫 | Framer Motion |
| 狀態管理 | Zustand |
| AI 整合 | Google Gemini API |

## 執行指令集

### /gemini-learn:init

**初始化專案**

執行流程：
1. 建立 Next.js 14+ 專案結構
2. 安裝依賴：Tailwind CSS, Shadcn/ui, Framer Motion, Zustand
3. 設定 Gemini API 整合
4. 建立基礎 Layout 與導航

### /gemini-learn:component [name]

**建立學習元件**

支援的元件類型：
- `quiz` - 互動式測驗
- `code-playground` - 程式碼練習區
- `concept-card` - 概念卡片
- `progress-tracker` - 學習進度追蹤

## 專案結構

```
teach-gemini/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   └── lessons/
├── components/
│   ├── ui/
│   └── learning/
├── lib/
│   └── gemini.ts
└── stores/
    └── progress.ts
```
