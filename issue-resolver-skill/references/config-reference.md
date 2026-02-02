# Issue Tracker Configuration Reference

## Default Configuration

```javascript
const CONFIG = {
    relayUrl: 'https://relay-o.oopdoo.org.ua/gun',
    nodePrefix: 'issue-tracker',
    timeout: 5000
};
```

## Environment Variables

```bash
ISSUE_RELAY_URL=https://relay-o.oopdoo.org.ua/gun
ISSUE_NODE_PREFIX=issue-tracker
TARGET_SYSTEM_PATH=/path/to/project
```

## Issue Data Structure

```typescript
interface Issue {
  id: string;              // "issue-{timestamp}-{random}"
  title: string;           // Required
  description: string;     
  group: GroupId;          // Category
  priority: PriorityId;    
  status: StatusId;        
  reporter: string;        
  contact: string;         
  device: string;          
  browser: string;         
  createdAt: number;       // Unix timestamp (ms)
  updatedAt: number;       
  resolvedAt: number | null;
}
```

## Groups

| ID | Name | Icon |
|----|------|------|
| `system` | 系統問題 | ⚙️ |
| `ui` | 介面問題 | 🎨 |
| `account` | 帳號問題 | 👤 |
| `data` | 資料問題 | 💾 |
| `performance` | 效能問題 | ⚡ |
| `feature` | 功能建議 | 💡 |
| `other` | 其他 | 📋 |

## Priorities

| ID | Name | Icon | Weight |
|----|------|------|--------|
| `critical` | 緊急 | 🔴 | 4 |
| `high` | 高 | 🟠 | 3 |
| `medium` | 中 | 🟡 | 2 |
| `low` | 低 | 🟢 | 1 |

## Statuses

| ID | Name | Icon |
|----|------|------|
| `open` | 待處理 | 📬 |
| `in-progress` | 處理中 | 🔧 |
| `resolved` | 已解決 | ✅ |
| `closed` | 已關閉 | 🔒 |

## CLI Usage Examples

```bash
# Install dependencies
npm install

# Fetch all pending issues
npm run fetch

# Fetch critical issues as Markdown
node scripts/fetch-issues.js --priority critical,high --format markdown

# Process single issue
node scripts/fetch-issues.js --limit 1 | node scripts/process-issue.js

# Batch process
node scripts/batch-process.js --priority critical,high --output-dir ./reports

# Update status
node scripts/update-status.js --id issue-xxx --status resolved

# Dry run (analyze only)
node scripts/batch-process.js --dry-run
```

## Gun.js Node Structure

```
{nodePrefix}-issues/
├── issue-1234567890-abc/
│   ├── id
│   ├── title
│   ├── description
│   ├── group
│   ├── priority
│   ├── status
│   └── ...
└── issue-1234567891-def/
    └── ...
```
