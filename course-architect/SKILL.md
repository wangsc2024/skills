---
name: course-architect
description: |
  課程安排大師 - 設計學習路徑、課程大綱、教學內容與評估標準。靈感來源自 Mr. Ranedeer AI Tutor。
  Use when: 規劃教學課程、設計學習路徑、建立教學內容與練習，or when user mentions 課程設計, course, 教學規劃.
  Triggers: "課程設計", "course architect", "學習路徑", "課程大綱", "教學規劃", "course design", "learning path"
version: 2.0.0
---

# 課程安排大師 Skill (Course Architect)

設計學習路徑、課程大綱、教學內容與評估標準的專業工具。

## When to Use This Skill

- 規劃 AI 教學課程
- 設計學習路徑或課程大綱
- 建立教學內容、練習或挑戰
- 設計評估標準或回饋機制

## 執行指令集

### /course:init [name]

**初始化課程專案**

參數：
- `name`：課程名稱（預設：fundamentals）

執行流程：
1. 建立課程目錄結構
2. 生成課程設定檔
3. 初始化學習者設定檔模板

### /course:outline [topic]

**生成課程大綱**

自動產出包含：
- 學習目標
- 單元結構
- 時間規劃
- 評估方式

### /course:lesson [unit] [lesson]

**生成課程內容**

產出完整的課程內容，包含：
- 學習重點
- 講解內容
- 實作練習
- 自我評量

## 課程結構範本

```
course/
├── config.yaml          # 課程設定
├── outline.md           # 課程大綱
├── units/
│   ├── unit-01/
│   │   ├── lesson-01.md
│   │   ├── lesson-02.md
│   │   └── exercises/
│   └── unit-02/
└── assessments/
    └── quiz-01.md
```
