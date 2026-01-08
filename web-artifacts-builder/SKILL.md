---
name: web-artifacts-builder
description: |
  Build sophisticated multi-component frontend applications using React, TypeScript, Tailwind CSS, and shadcn/ui. Creates bundled single-file HTML artifacts.
  Use when: building complex web applications, interactive prototypes, multi-component React projects, or when user mentions shadcn, 複雜應用, web app, 前端專案, React專案, 打包HTML.
  Triggers: "shadcn", "web app", "複雜應用", "前端專案", "React專案", "打包HTML", "multi-component"
---

# Web Artifacts Builder

Build production-quality frontend applications with modern web technologies.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type safety |
| **Vite** | Development server |
| **Tailwind CSS** | Styling |
| **shadcn/ui** | Component library |
| **Parcel** | Bundling to single HTML |

## Setup

### Initialize Project

```bash
# Create new project
npm create vite@latest my-app -- --template react-ts
cd my-app

# Install dependencies
npm install

# Add Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Add shadcn/ui
npx shadcn@latest init
```

### Tailwind Configuration

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
      },
    },
  },
  plugins: [],
}
```

## shadcn/ui Components

### Install Components

```bash
# Install commonly used components
npx shadcn@latest add button
npx shadcn@latest add card
npx shadcn@latest add input
npx shadcn@latest add dialog
npx shadcn@latest add dropdown-menu
npx shadcn@latest add table
npx shadcn@latest add tabs
npx shadcn@latest add toast
npx shadcn@latest add form
```

### Available Components (40+)

```markdown
Layout: Card, Separator, Scroll Area, Resizable
Forms: Input, Textarea, Select, Checkbox, Radio, Switch, Slider
Feedback: Alert, Toast, Progress, Skeleton
Overlay: Dialog, Drawer, Popover, Tooltip, Sheet
Navigation: Tabs, Accordion, Navigation Menu, Breadcrumb
Data: Table, Data Table, Calendar, Date Picker
```

## Project Structure

```
my-app/
├── public/
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── ui/           # shadcn components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   └── ...
│   │   └── custom/       # Your components
│   │       ├── Header.tsx
│   │       └── Dashboard.tsx
│   ├── hooks/
│   │   └── useLocalStorage.ts
│   ├── lib/
│   │   └── utils.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Building Components

### Basic Component

```tsx
import { FC } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface DashboardCardProps {
  title: string;
  value: string;
  trend: 'up' | 'down' | 'neutral';
}

export const DashboardCard: FC<DashboardCardProps> = ({ title, value, trend }) => {
  const trendColors = {
    up: 'text-green-500',
    down: 'text-red-500',
    neutral: 'text-gray-500',
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className={`text-xs ${trendColors[trend]}`}>
          {trend === 'up' && '↑'}
          {trend === 'down' && '↓'}
          {trend === 'neutral' && '→'}
        </p>
      </CardContent>
    </Card>
  );
};
```

### With State Management

```tsx
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export const TaskManager = () => {
  const [tasks, setTasks] = useState<string[]>([]);
  const [newTask, setNewTask] = useState('');
  const [open, setOpen] = useState(false);

  const addTask = () => {
    if (newTask.trim()) {
      setTasks([...tasks, newTask.trim()]);
      setNewTask('');
      setOpen(false);
    }
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Tasks</h2>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button>Add Task</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>New Task</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <Input
                value={newTask}
                onChange={(e) => setNewTask(e.target.value)}
                placeholder="Enter task..."
              />
              <Button onClick={addTask} className="w-full">
                Add
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <ul className="space-y-2">
        {tasks.map((task, i) => (
          <li key={i} className="p-3 bg-secondary rounded-lg">
            {task}
          </li>
        ))}
      </ul>
    </div>
  );
};
```

## Routing (React Router)

```bash
npm install react-router-dom
```

```tsx
// App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Home } from './pages/Home';
import { Dashboard } from './pages/Dashboard';
import { Settings } from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  );
}
```

## Bundling to Single HTML

### Using Parcel

```bash
npm install -D parcel
```

```json
// package.json
{
  "scripts": {
    "build:bundle": "parcel build src/index.html --no-optimize"
  }
}
```

### Alternative: Inline Script

```javascript
// scripts/bundle.js
const fs = require('fs');
const path = require('path');

// Read built files
const html = fs.readFileSync('dist/index.html', 'utf8');
const js = fs.readFileSync('dist/assets/index.js', 'utf8');
const css = fs.readFileSync('dist/assets/index.css', 'utf8');

// Inline everything
const bundled = html
  .replace(/<link[^>]+href="[^"]+\.css"[^>]*>/, `<style>${css}</style>`)
  .replace(/<script[^>]+src="[^"]+\.js"[^>]*><\/script>/, `<script>${js}</script>`);

fs.writeFileSync('bundle.html', bundled);
console.log('✓ Created bundle.html');
```

## Design Guidelines

### Avoid "AI Slop"

```markdown
❌ DON'T:
- Excessive centered layouts
- Purple-to-blue gradients
- Uniform rounded corners on everything
- Inter font for everything
- Generic "tech startup" aesthetic
- Identical card shadows

✅ DO:
- Asymmetric, purposeful layouts
- Intentional color choices
- Varied corner radii based on context
- Distinctive typography
- Context-appropriate design
- Depth through varied shadows
```

### Good Patterns

```tsx
// Varied layout
<div className="grid grid-cols-12 gap-4">
  <div className="col-span-8">
    <MainContent />
  </div>
  <div className="col-span-4">
    <Sidebar />
  </div>
</div>

// Intentional shadows
<div className="shadow-sm hover:shadow-md transition-shadow">
  {/* Subtle to medium on hover */}
</div>

// Varied spacing
<section className="py-16 md:py-24">
  {/* Generous, responsive spacing */}
</section>
```

## Performance Tips

```tsx
// Lazy loading
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Dashboard />
    </Suspense>
  );
}

// Memoization
import { memo, useMemo, useCallback } from 'react';

const ExpensiveComponent = memo(({ data }) => {
  const processed = useMemo(() => processData(data), [data]);
  return <div>{processed}</div>;
});
```

## Common Recipes

### Data Table with Sorting

```tsx
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

interface DataTableProps<T> {
  data: T[];
  columns: { key: keyof T; label: string }[];
}

export function DataTable<T>({ data, columns }: DataTableProps<T>) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          {columns.map((col) => (
            <TableHead key={String(col.key)}>{col.label}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.map((row, i) => (
          <TableRow key={i}>
            {columns.map((col) => (
              <TableCell key={String(col.key)}>
                {String(row[col.key])}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
```

### Toast Notifications

```tsx
import { useToast } from '@/components/ui/use-toast';
import { Button } from '@/components/ui/button';

export function ToastDemo() {
  const { toast } = useToast();

  return (
    <Button
      onClick={() => {
        toast({
          title: 'Success!',
          description: 'Your changes have been saved.',
        });
      }}
    >
      Save Changes
    </Button>
  );
}
```

## Checklist

### Before Development
- [ ] Vite project initialized
- [ ] Tailwind CSS configured
- [ ] shadcn/ui initialized
- [ ] Required components installed

### During Development
- [ ] TypeScript types defined
- [ ] Components are reusable
- [ ] Responsive design implemented
- [ ] Accessibility considered

### Before Bundling
- [ ] No console errors
- [ ] Performance optimized
- [ ] Design is intentional (not generic)
- [ ] All features tested
