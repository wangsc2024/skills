# 伺服器架構與實作細節

## 核心架構

```
server.js (616 行)
├── Configuration (行 14-33)
├── Logging System (行 38-52)
├── File System Checks (行 57-79)
├── Disk Space Monitor (行 84-109)
├── Global Error Handlers (行 114-128)
├── Peer Connection Tracking (行 131-221)
├── Express Middleware & Routes (行 224-422)
├── Server Initialization (行 427-586)
└── Graceful Shutdown (行 591-615)
```

## 環境變數

### 完整列表

```bash
# 伺服器配置
PORT=8765                        # 監聽端口 (預設 8765)
NODE_ENV=production              # 執行環境

# WebSocket 和連接
WS_HEARTBEAT_INTERVAL=25000       # 心跳間隔 (毫秒)
CONNECTION_TIMEOUT=30000          # 連接超時 (毫秒)
KEEP_ALIVE_TIMEOUT=65000          # HTTP Keep-Alive 超時
HEADERS_TIMEOUT=66000             # HTTP Headers 超時

# 性能和限制
PEER_CLEANUP_INTERVAL=60000        # 清理間隔 (毫秒)
MAX_PEERS_PER_IP=10                # 每 IP 最大連接數
HEALTH_CHECK_INTERVAL=60000        # 健康檢查間隔

# 日誌
LOG_LEVEL=INFO                     # INFO|DEBUG|WARN|ERROR
```

## iOS 優化

### 心跳機制

iOS 會在 30 秒無活動後關閉背景 WebSocket 連接。我們的解決方案：

| 參數 | 值 | 說明 |
|------|-----|------|
| WS_HEARTBEAT_INTERVAL | 25000 | 每 25 秒發送心跳 |
| CONNECTION_TIMEOUT | 30000 | 30 秒檢測斷線 |
| SSE /events | 25000 | 備用心跳機制 |

### 平台檢測

```javascript
function detectPlatform(userAgent) {
  if (!userAgent) return 'unknown';
  const ua = userAgent.toLowerCase();
  if (ua.includes('iphone') || ua.includes('ipad')) return 'ios';
  if (ua.includes('android')) return 'android';
  if (ua.includes('macintosh') || ua.includes('mac os')) return 'macos';
  if (ua.includes('windows')) return 'windows';
  if (ua.includes('linux')) return 'linux';
  return 'other';
}
```

## 日誌系統

### 日誌級別

```javascript
const LOG_LEVELS = {
  ERROR: 0,    // 系統錯誤、異常
  WARN: 1,     // 警告（如低磁盤空間）
  INFO: 2,     // 普通信息（預設）
  DEBUG: 3     // 調試信息
};
```

### 日誌格式

```
[2025-02-01T12:00:00.000Z] [INFO] Peer connected {"peerId":"abc123","platform":"ios","totalPeers":42}
```

## 連接追蹤

### 資料結構

```javascript
// peerId -> { lastSeen, ip, userAgent, platform, connectedAt }
const peerConnections = new Map();

// ip -> count
const ipConnectionCount = new Map();
```

### 核心函數

| 函數 | 功能 |
|------|------|
| `trackPeerConnection()` | 記錄連接，檢查 IP 限制 |
| `untrackPeerConnection()` | 移除連接記錄 |
| `updatePeerActivity()` | 更新最後活動時間 |
| `cleanupStalePeers()` | 清理超時連接 |

## Gun.js 初始化

```javascript
const gunInstance = Gun({
  web: httpServer,           // Express 伺服器
  file: DATA_DIR,            // 持久化位置
  radisk: true,              // 啟用磁盤存儲
  localStorage: false,       // 禁用本地存儲
  multicast: false,          // 禁用多播發現
  axe: false,                // 禁用 AXE

  ws: {
    path: '/gun',            // WebSocket 路徑
    pingTimeout: 30000,      // 心跳超時
    pingInterval: 25000      // 心跳間隔
  }
});
```

## 定期任務

| 任務 | 間隔 | 說明 |
|------|------|------|
| 健康檢查 | 60 秒 | 檢查磁盤、記憶體、連接 |
| 清理過期連接 | 60 秒 | 刪除超時連接 |
| 記憶體監控 | 2 分鐘 | 檢測高使用量 |

## 優雅關閉

```javascript
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// 關閉流程:
// 1. 記錄信號
// 2. 關閉 HTTP 伺服器
// 3. 等待 2 秒
// 4. 退出進程
```

## 錯誤處理

### 全局錯誤

```javascript
process.on('uncaughtException', (error) => {
  log('ERROR', 'Uncaught Exception', { message, stack });
  setTimeout(() => process.exit(1), 1000);
});

process.on('unhandledRejection', (reason) => {
  log('ERROR', 'Unhandled Promise Rejection', { reason });
});
```

### HTTP 錯誤

```javascript
httpServer.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    log('ERROR', `Port ${port} is already in use`);
    process.exit(1);
  }
});
```

## 文件系統

### 數據目錄

```javascript
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');

function ensureDataDirectory() {
  // 1. 確保目錄存在
  // 2. 測試寫入權限
  // 3. 失敗則退出
}
```

### 磁盤監控

```javascript
function checkDiskSpace() {
  const stats = fs.statfsSync(DATA_DIR);
  const freeGB = (stats.bfree * stats.bsize) / (1024 ** 3);
  const freePercent = (stats.bfree / stats.blocks) * 100;

  if (freePercent < 10) {
    log('WARN', 'Low disk space warning');
  }

  return { freeGB, freePercent, warning: freePercent < 10 };
}
```
