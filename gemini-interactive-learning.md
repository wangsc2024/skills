# Gemini AI 互動式學習網站開發 Skill
# 版本: 2.0
# 適用專案: teach-gemini

---

## Skill 觸發條件

當使用者請求以下類型任務時，自動啟用此 Skill：
- 建立 Gemini 教學網站相關功能
- 設計互動式學習元件
- 實作動態檢視或版面配置功能
- 開發 AI 教學相關的前端功能

---

## 執行指令集

### /gemini-learn:init
**初始化專案**

執行流程：
1. 建立 Next.js 14+ 專案結構
2. 安裝依賴：Tailwind CSS, Shadcn/ui, Framer Motion, Zustand
3. 設定 Gemini API 整合
4. 建立基礎 Layout 與導航

**輸出檔案：**
```
teach-gemini/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── playground/
│   │   ├── courses/
│   │   └── settings/
│   ├── components/
│   │   ├── ui/              # Shadcn/ui 元件
│   │   ├── editor/          # Prompt 編輯器
│   │   ├── viewer/          # 回應檢視器
│   │   └── layout/          # 版面配置
│   ├── lib/
│   │   ├── gemini.ts        # Gemini API 封裝
│   │   └── store.ts         # Zustand 狀態管理
│   └── styles/
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

---

### /gemini-learn:component [name]
**建立互動元件**

參數：
- `name`：元件名稱（prompt-editor | response-viewer | parameter-panel | token-counter）

執行流程：
1. 根據元件類型生成對應的 React 元件
2. 包含 TypeScript 型別定義
3. 整合 Framer Motion 動畫
4. 加入無障礙支援

**範例輸出（prompt-editor）：**
```typescript
// src/components/editor/PromptEditor.tsx
'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { useGeminiStore } from '@/lib/store';

interface PromptEditorProps {
  mode: 'guided' | 'freeform';
  hints?: boolean;
  templateSlots?: TemplateSlot[];
  onSubmit: (prompt: string) => void;
}

export function PromptEditor({
  mode = 'freeform',
  hints = true,
  templateSlots = [],
  onSubmit
}: PromptEditorProps) {
  const [prompt, setPrompt] = useState('');
  const { tokenCount, updateTokenCount } = useGeminiStore();

  const handleChange = useCallback((value: string) => {
    setPrompt(value);
    updateTokenCount(value);
  }, [updateTokenCount]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full space-y-4"
    >
      <Textarea
        value={prompt}
        onChange={(e) => handleChange(e.target.value)}
        placeholder={mode === 'guided'
          ? '依照模板填入內容...'
          : '輸入你的 Prompt...'}
        className="min-h-[200px] resize-y"
        aria-label="Prompt 輸入區"
      />

      {hints && (
        <div className="text-sm text-muted-foreground">
          Token 數量：{tokenCount}
        </div>
      )}

      <Button
        onClick={() => onSubmit(prompt)}
        disabled={!prompt.trim()}
      >
        送出
      </Button>
    </motion.div>
  );
}
```

---

### /gemini-learn:preview [mode]
**設定動態檢視系統**

參數：
- `mode`：檢視模式（split | overlay | compare | timeline）

執行流程：
1. 建立對應的檢視容器元件
2. 設定即時同步機制
3. 整合串流回應處理
4. 加入狀態監控面板

**生成的核心元件：**

```typescript
// src/components/viewer/LivePreviewContainer.tsx
'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ResizablePanel,
  ResizablePanelGroup,
  ResizableHandle
} from '@/components/ui/resizable';

interface LivePreviewContainerProps {
  mode: 'split' | 'overlay' | 'compare' | 'timeline';
  children: {
    editor: React.ReactNode;
    viewer: React.ReactNode;
  };
  syncOptions?: {
    debounceMs?: number;
    autoSync?: boolean;
  };
}

export function LivePreviewContainer({
  mode,
  children,
  syncOptions = { debounceMs: 300, autoSync: true }
}: LivePreviewContainerProps) {
  const renderLayout = () => {
    switch (mode) {
      case 'split':
        return (
          <ResizablePanelGroup direction="horizontal">
            <ResizablePanel defaultSize={50} minSize={30}>
              {children.editor}
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={50} minSize={30}>
              {children.viewer}
            </ResizablePanel>
          </ResizablePanelGroup>
        );

      case 'compare':
        return (
          <div className="grid grid-cols-2 gap-4">
            {children.viewer}
          </div>
        );

      case 'overlay':
        return (
          <div className="relative">
            {children.editor}
            <motion.div
              className="absolute inset-0 bg-background/95"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {children.viewer}
            </motion.div>
          </div>
        );

      default:
        return children.editor;
    }
  };

  return (
    <div className="w-full h-full">
      {renderLayout()}
    </div>
  );
}
```

---

### /gemini-learn:layout [template]
**套用版面配置**

參數：
- `template`：模板名稱（beginner | power-user | comparison | focus）

執行流程：
1. 載入預設佈局配置
2. 生成對應的 Widget 組合
3. 設定拖放功能
4. 加入配置持久化

**版面配置器核心：**

```typescript
// src/components/layout/LayoutConfigurator.tsx
'use client';

import { useState, useCallback } from 'react';
import {
  DndContext,
  closestCenter,
  DragEndEvent
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  rectSortingStrategy
} from '@dnd-kit/sortable';

// 預設佈局模板
const layoutTemplates = {
  beginner: {
    name: '新手友善',
    widgets: [
      { id: 'hint', type: 'hint-panel', size: { w: 12, h: 1 } },
      { id: 'editor', type: 'prompt-editor', size: { w: 12, h: 3 } },
      { id: 'viewer', type: 'response-viewer', size: { w: 12, h: 4 } },
    ]
  },
  'power-user': {
    name: '進階使用者',
    widgets: [
      { id: 'editor', type: 'prompt-editor', size: { w: 6, h: 4 } },
      { id: 'viewer', type: 'response-viewer', size: { w: 6, h: 4 } },
      { id: 'params', type: 'parameter-panel', size: { w: 4, h: 3 } },
      { id: 'history', type: 'history-list', size: { w: 4, h: 3 } },
      { id: 'tokens', type: 'token-counter', size: { w: 4, h: 3 } },
    ]
  },
  comparison: {
    name: '對比模式',
    widgets: [
      { id: 'editor', type: 'prompt-editor', size: { w: 12, h: 2 } },
      { id: 'viewer-a', type: 'response-viewer', size: { w: 6, h: 5 } },
      { id: 'viewer-b', type: 'response-viewer', size: { w: 6, h: 5 } },
    ]
  },
  focus: {
    name: '專注模式',
    widgets: [
      { id: 'editor', type: 'prompt-editor', size: { w: 8, h: 3 } },
      { id: 'viewer', type: 'response-viewer', size: { w: 8, h: 5 } },
    ]
  }
};

export function LayoutConfigurator({
  template = 'beginner'
}: { template: keyof typeof layoutTemplates }) {
  const [widgets, setWidgets] = useState(
    layoutTemplates[template].widgets
  );

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      setWidgets((items) => {
        const oldIndex = items.findIndex(i => i.id === active.id);
        const newIndex = items.findIndex(i => i.id === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  }, []);

  // 儲存到 localStorage
  const saveLayout = useCallback(() => {
    localStorage.setItem('gemini-layout', JSON.stringify(widgets));
  }, [widgets]);

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
    >
      <SortableContext items={widgets} strategy={rectSortingStrategy}>
        <div className="grid grid-cols-12 gap-4 p-4">
          {widgets.map((widget) => (
            <SortableWidget key={widget.id} {...widget} />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  );
}
```

---

### /gemini-learn:streaming
**設定串流回應**

執行流程：
1. 建立 SSE 連線處理
2. 實作打字機效果
3. 加入暫停/停止控制
4. 設定 Token 即時計數

**串流處理核心：**

```typescript
// src/lib/gemini.ts
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

export async function* streamGeminiResponse(prompt: string) {
  const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

  const result = await model.generateContentStream(prompt);

  for await (const chunk of result.stream) {
    const text = chunk.text();
    yield {
      text,
      tokenCount: chunk.usageMetadata?.totalTokenCount ?? 0
    };
  }
}

// React Hook
export function useGeminiStream() {
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [tokenCount, setTokenCount] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(async (prompt: string) => {
    setIsStreaming(true);
    setResponse('');
    abortRef.current = new AbortController();

    try {
      for await (const chunk of streamGeminiResponse(prompt)) {
        if (abortRef.current?.signal.aborted) break;
        setResponse(prev => prev + chunk.text);
        setTokenCount(chunk.tokenCount);
      }
    } finally {
      setIsStreaming(false);
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { response, isStreaming, tokenCount, generate, stop };
}
```

---

### /gemini-learn:theme [name]
**設定主題樣式**

參數：
- `name`：主題名稱（light | dark | high-contrast）

執行流程：
1. 生成 CSS 變數配置
2. 設定主題切換邏輯
3. 整合系統偏好偵測

**主題配置：**

```typescript
// src/lib/themes.ts
export const themes = {
  light: {
    '--background': '0 0% 100%',
    '--foreground': '222.2 84% 4.9%',
    '--primary': '221.2 83.2% 53.3%',      // Google Blue
    '--secondary': '142.1 70.6% 45.3%',    // Success Green
    '--accent': '43 96% 56%',              // Warning Yellow
    '--destructive': '0 84.2% 60.2%',      // Error Red
    '--muted': '210 40% 96.1%',
    '--card': '0 0% 100%',
    '--border': '214.3 31.8% 91.4%',
  },
  dark: {
    '--background': '222.2 84% 4.9%',
    '--foreground': '210 40% 98%',
    '--primary': '217.2 91.2% 59.8%',
    '--secondary': '142.1 70.6% 45.3%',
    '--accent': '43 96% 56%',
    '--destructive': '0 62.8% 30.6%',
    '--muted': '217.2 32.6% 17.5%',
    '--card': '222.2 84% 4.9%',
    '--border': '217.2 32.6% 17.5%',
  },
  'high-contrast': {
    '--background': '0 0% 0%',
    '--foreground': '0 0% 100%',
    '--primary': '210 100% 50%',
    '--secondary': '120 100% 40%',
    '--accent': '60 100% 50%',
    '--destructive': '0 100% 50%',
    '--muted': '0 0% 20%',
    '--card': '0 0% 10%',
    '--border': '0 0% 100%',
  }
};
```

---

### /gemini-learn:lesson [id]
**建立課程內容**

參數：
- `id`：課程單元 ID（例：3.2）

執行流程：
1. 根據課程大師 Skill 的結構生成 MDX 內容
2. 加入互動元件區塊
3. 設定練習與挑戰

**MDX 範例輸出：**

```mdx
---
title: "角色設定的魔力"
unit: "3.2"
duration: "15-20 分鐘"
objectives:
  - 理解角色設定對回應品質的影響
  - 掌握有效的角色設定技巧
prerequisites:
  - "2.3"
  - "3.1"
---

import { PromptEditor } from '@/components/editor/PromptEditor';
import { ComparisonViewer } from '@/components/viewer/ComparisonViewer';
import { Challenge } from '@/components/interactive/Challenge';

# 角色設定的魔力

## 為什麼需要角色設定？

想像你走進一間餐廳，對服務生說：「給我一份好吃的」...

<ComparisonViewer
  before={{
    prompt: "幫我寫一封請假信",
    response: "...(籠統的回應)"
  }}
  after={{
    prompt: "你是一位專業的行政秘書...",
    response: "...(專業的回應)"
  }}
/>

## 動手試試

<PromptEditor
  mode="guided"
  template="你是一位 [專業領域] 的 [角色]，擅長 [技能]..."
  hints={[
    "考慮任務需要什麼專業",
    "加入具體技能描述"
  ]}
/>

## 挑戰時間

<Challenge
  goal="設計一個能寫出專業產品說明的角色"
  criteria={[
    "包含專業背景",
    "明確技能描述",
    "語氣設定"
  ]}
  maxAttempts={3}
/>
```

---

## 核心特色功能

### 功能一：動態檢視系統 (Live Preview System)

動態檢視系統讓使用者能即時看到 AI 回應的效果，無需刷新頁面。

#### 系統架構
```
┌─────────────────────────────────────────────────────┐
│                  動態檢視系統                        │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │ 輸入面板    │ →  │ 處理引擎    │ →  │ 輸出面板 │ │
│  │ (Editor)    │    │ (Processor) │    │ (Viewer) │ │
│  └─────────────┘    └─────────────┘    └──────────┘ │
│         ↑                  ↓                  ↓     │
│  ┌─────────────────────────────────────────────────┐│
│  │              狀態同步層 (State Sync)            ││
│  │   - Zustand 狀態管理                            ││
│  │   - Optimistic Update                           ││
│  │   - 串流回應處理                                ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

#### 檢視模式
| 模式 | 說明 | 使用場景 |
|------|------|----------|
| split | 左右分割，可調整比例 | 一般使用 |
| overlay | 覆蓋切換，熱鍵控制 | 小螢幕 |
| compare | 多版本並排 | 參數對比 |
| timeline | 時間軸歷史 | 版本追蹤 |

---

### 功能二：視覺版面配置器 (Visual Layout Configurator)

提供直覺的拖放介面，讓使用者自訂學習介面佈局。

#### 預設佈局模板
| 模板 | 特色 | 適合對象 |
|------|------|----------|
| beginner | 大面積操作區、清晰指引 | 初學者 |
| power-user | 多面板、高資訊密度 | 進階使用者 |
| comparison | 左右對比效果 | 參數調校 |
| focus | 極簡介面 | 專注學習 |

#### Widget 清單
- `prompt-editor` - Prompt 編輯器
- `response-viewer` - 回應檢視器
- `parameter-panel` - 參數面板
- `history-list` - 歷史記錄
- `token-counter` - Token 計數器
- `diff-viewer` - 差異檢視器
- `hint-panel` - 提示面板
- `progress-tracker` - 進度追蹤

---

## 技術棧

```yaml
framework: Next.js 14+ (App Router)
styling: Tailwind CSS + Shadcn/ui
animation: Framer Motion
state: Zustand
drag-drop: @dnd-kit/core
forms: React Hook Form + Zod
gemini: @google/generative-ai
```

---

## 開發檢查清單

### Phase 1: 基礎建設
- [ ] `/gemini-learn:init` 初始化專案
- [ ] 設定 Gemini API Key
- [ ] 建立基礎 Layout

### Phase 2: 核心元件
- [ ] `/gemini-learn:component prompt-editor`
- [ ] `/gemini-learn:component response-viewer`
- [ ] `/gemini-learn:streaming` 串流處理

### Phase 3: 動態檢視
- [ ] `/gemini-learn:preview split`
- [ ] `/gemini-learn:layout beginner`

### Phase 4: 課程內容
- [ ] `/gemini-learn:lesson 1.1`
- [ ] 整合課程安排大師 Skill

---

## 參考資源

- [Gemini API 文件](https://ai.google.dev/docs)
- [Next.js 文件](https://nextjs.org/docs)
- [Shadcn/ui 元件庫](https://ui.shadcn.com)
- [dnd-kit 拖放庫](https://dndkit.com)
