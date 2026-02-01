# 核心組件說明

## 組件架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gun Relay Server                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │   CONFIG    │   │  LOG_LEVELS │   │   Global Handlers   │    │
│  │ (環境配置)   │   │  (日誌系統)  │   │   (錯誤處理)        │    │
│  └─────────────┘   └─────────────┘   └─────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                File System Module                        │    │
│  │  ┌───────────────────┐   ┌─────────────────────┐        │    │
│  │  │ ensureDataDir()   │   │ checkDiskSpace()    │        │    │
│  │  └───────────────────┘   └─────────────────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Peer Connection Tracking                    │    │
│  │  ┌─────────────────┐   ┌──────────────────────────┐     │    │
│  │  │ peerConnections │   │ trackPeerConnection()    │     │    │
│  │  │ (Map)           │   │ untrackPeerConnection()  │     │    │
│  │  └─────────────────┘   │ updatePeerActivity()     │     │    │
│  │  ┌─────────────────┐   │ cleanupStalePeers()      │     │    │
│  │  │ ipConnectionCnt │   │ detectPlatform()         │     │    │
│  │  │ (Map)           │   └──────────────────────────┘     │    │
│  │  └─────────────────┘                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                Express Middleware                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │    │
│  │  │  CORS    │→ │ Logging  │→ │  JSON    │              │    │
│  │  └──────────┘  └──────────┘  └──────────┘              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   REST Routes                            │    │
│  │  GET /  │  GET /ping  │  GET /status  │  GET /events   │    │
│  │  GET /debug/write-test  │  GET /debug/connections       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Gun.js Core                           │    │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────────┐      │    │
│  │  │ gunInst   │   │ httpServer│   │ WebSocket /gun│      │    │
│  │  └───────────┘   └───────────┘   └───────────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Periodic Tasks                           │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │    │
│  │  │HealthCheck│  │ PeerCleanup│  │ MemoryMon  │        │    │
│  │  │ (60s)     │  │ (60s)      │  │ (120s)     │        │    │
│  │  └────────────┘  └────────────┘  └────────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Graceful Shutdown                           │    │
│  │  SIGTERM / SIGINT → gracefulShutdown()                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 1. 配置模組 (CONFIG)

### 位置
`server.js` 行 14-33

### 說明
集中管理所有可配置參數，支援環境變數覆寫。

### 程式碼

```javascript
const CONFIG = {
  // WebSocket 心跳間隔 (iOS Safari 需要頻繁 ping)
  WS_HEARTBEAT_INTERVAL: parseInt(process.env.WS_HEARTBEAT_INTERVAL) || 25000,

  // 連接超時 (iOS 可能在 30 秒後關閉閒置連接)
  CONNECTION_TIMEOUT: parseInt(process.env.CONNECTION_TIMEOUT) || 30000,

  // HTTP Keep-Alive 設定
  KEEP_ALIVE_TIMEOUT: parseInt(process.env.KEEP_ALIVE_TIMEOUT) || 65000,
  HEADERS_TIMEOUT: parseInt(process.env.HEADERS_TIMEOUT) || 66000,

  // 對等清理間隔
  PEER_CLEANUP_INTERVAL: parseInt(process.env.PEER_CLEANUP_INTERVAL) || 60000,

  // 每 IP 最大連接數 (防止連接洪水)
  MAX_PEERS_PER_IP: parseInt(process.env.MAX_PEERS_PER_IP) || 10,

  // 健康檢查間隔
  HEALTH_CHECK_INTERVAL: parseInt(process.env.HEALTH_CHECK_INTERVAL) || 60000
};
```

### 設計決策

| 參數 | 預設值 | 設計原因 |
|------|--------|----------|
| WS_HEARTBEAT_INTERVAL | 25000 | iOS 會在 30 秒後關閉閒置 WebSocket |
| CONNECTION_TIMEOUT | 30000 | 配合 iOS 超時機制 |
| KEEP_ALIVE_TIMEOUT | 65000 | 大於 60 秒避免代理超時 |
| MAX_PEERS_PER_IP | 10 | 防止單一 IP 佔用過多資源 |

---

## 2. 日誌系統 (Logging System)

### 位置
`server.js` 行 38-52

### 說明
提供分級日誌輸出，支援 JSON 格式化。

### 程式碼

```javascript
const LOG_LEVELS = { ERROR: 0, WARN: 1, INFO: 2, DEBUG: 3 };
const currentLogLevel = LOG_LEVELS[process.env.LOG_LEVEL] ?? LOG_LEVELS.INFO;

function log(level, message, data = null) {
  if (LOG_LEVELS[level] > currentLogLevel) return;

  const timestamp = new Date().toISOString();
  const prefix = `[${timestamp}] [${level}]`;

  if (data) {
    console.log(`${prefix} ${message}`, JSON.stringify(data, null, 2));
  } else {
    console.log(`${prefix} ${message}`);
  }
}
```

### 日誌級別

| 級別 | 數值 | 用途 |
|------|------|------|
| ERROR | 0 | 系統錯誤、異常、致命問題 |
| WARN | 1 | 警告（低磁盤、高記憶體、過多連接） |
| INFO | 2 | 正常操作日誌（預設級別） |
| DEBUG | 3 | 詳細調試信息（開發用） |

### 輸出格式

```
[2025-02-01T12:00:00.000Z] [INFO] Peer connected {
  "peerId": "abc123",
  "platform": "ios",
  "totalPeers": 42
}
```

---

## 3. 文件系統模組 (File System Module)

### 位置
`server.js` 行 57-109

### 組件

#### 3.1 ensureDataDirectory()

確保數據目錄存在且可寫入。

```javascript
function ensureDataDirectory() {
  try {
    // 1. 創建目錄（如不存在）
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }

    // 2. 測試寫入權限
    const testFile = path.join(DATA_DIR, '.write-test');
    fs.writeFileSync(testFile, 'test');
    fs.unlinkSync(testFile);

    return true;
  } catch (error) {
    log('ERROR', 'Failed to setup data directory', { error: error.message });
    return false;
  }
}
```

#### 3.2 checkDiskSpace()

監控磁盤空間使用情況。

```javascript
function checkDiskSpace() {
  const stats = fs.statfsSync(DATA_DIR);
  const freeBytes = stats.bfree * stats.bsize;
  const totalBytes = stats.blocks * stats.bsize;
  const freePercent = ((freeBytes / totalBytes) * 100).toFixed(2);
  const freeGB = (freeBytes / (1024 * 1024 * 1024)).toFixed(2);

  return {
    freeGB: `${freeGB} GB`,
    freePercent: `${freePercent}%`,
    warning: freePercent < 10  // 低於 10% 發出警告
  };
}
```

---

## 4. 全局錯誤處理 (Global Error Handlers)

### 位置
`server.js` 行 114-128

### 說明
捕獲未處理的異常和 Promise 拒絕，防止進程崩潰。

### 程式碼

```javascript
// 未捕獲的同步異常
process.on('uncaughtException', (error) => {
  log('ERROR', 'Uncaught Exception', {
    message: error.message,
    stack: error.stack
  });
  // 給日誌刷新時間
  setTimeout(() => process.exit(1), 1000);
});

// 未處理的 Promise 拒絕
process.on('unhandledRejection', (reason, promise) => {
  log('ERROR', 'Unhandled Promise Rejection', {
    reason: reason?.toString?.() || reason,
    stack: reason?.stack
  });
});
```

### 行為

| 事件 | 處理方式 |
|------|----------|
| uncaughtException | 記錄錯誤，1 秒後退出進程 |
| unhandledRejection | 記錄錯誤，不退出（允許恢復） |

---

## 5. 對等連接追蹤 (Peer Connection Tracking)

### 位置
`server.js` 行 131-221

### 資料結構

```javascript
// peerId -> { lastSeen, ip, userAgent, platform, connectedAt }
const peerConnections = new Map();

// ip -> count (每 IP 連接計數)
const ipConnectionCount = new Map();
```

### 核心函數

#### 5.1 detectPlatform()

從 User-Agent 識別客戶端平台。

```javascript
function detectPlatform(userAgent) {
  if (!userAgent) return 'unknown';
  const ua = userAgent.toLowerCase();
  if (ua.includes('iphone') || ua.includes('ipad') || ua.includes('ipod')) return 'ios';
  if (ua.includes('android')) return 'android';
  if (ua.includes('mac')) return 'macos';
  if (ua.includes('windows')) return 'windows';
  if (ua.includes('linux')) return 'linux';
  return 'other';
}
```

#### 5.2 trackPeerConnection()

記錄新連接並檢查 IP 限制。

```javascript
function trackPeerConnection(peerId, ip, userAgent) {
  const platform = detectPlatform(userAgent);

  peerConnections.set(peerId, {
    lastSeen: Date.now(),
    ip: ip,
    userAgent: userAgent,
    platform: platform,
    connectedAt: Date.now()
  });

  // 追蹤每 IP 連接數
  const currentCount = ipConnectionCount.get(ip) || 0;
  ipConnectionCount.set(ip, currentCount + 1);

  return {
    platform,
    allowed: currentCount < CONFIG.MAX_PEERS_PER_IP
  };
}
```

#### 5.3 untrackPeerConnection()

移除連接記錄並更新 IP 計數。

#### 5.4 updatePeerActivity()

更新對等的最後活動時間。

#### 5.5 cleanupStalePeers()

清理超時連接（超過 2 倍 CONNECTION_TIMEOUT 未活動）。

```javascript
function cleanupStalePeers() {
  const now = Date.now();
  const staleTimeout = CONFIG.CONNECTION_TIMEOUT * 2;  // 60 秒

  for (const [peerId, peer] of peerConnections) {
    if (now - peer.lastSeen > staleTimeout) {
      untrackPeerConnection(peerId);
    }
  }
}
```

---

## 6. Express 中間件 (Express Middleware)

### 位置
`server.js` 行 224-261

### 中間件鏈

```
請求 → CORS → 日誌 → JSON 解析 → 路由處理
```

#### 6.1 CORS 中間件

iOS 優化的跨域設定：

```javascript
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', origin || '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, PATCH');
  res.header('Access-Control-Allow-Headers', '...');
  res.header('Access-Control-Allow-Credentials', 'true');
  res.header('Access-Control-Max-Age', '86400');  // 24 小時快取

  // iOS Safari 特定
  res.header('Cache-Control', 'no-cache, no-store, must-revalidate');
  res.header('Pragma', 'no-cache');
  res.header('Expires', '0');

  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});
```

#### 6.2 請求日誌中間件

```javascript
app.use((req, res, next) => {
  const platform = detectPlatform(req.get('User-Agent'));
  log('DEBUG', `${req.method} ${req.path}`, { ip, platform });
  next();
});
```

---

## 7. REST 路由 (REST Routes)

### 位置
`server.js` 行 264-409

### 端點列表

| 路由 | 功能 |
|------|------|
| GET / | 健康檢查與系統狀態 |
| GET /ping | 心跳測試 |
| GET /status | 連接狀態與 iOS 建議參數 |
| GET /events | Server-Sent Events 保活 |
| GET /debug/write-test | 測試 Gun 寫入功能 |
| GET /debug/connections | 連接詳情列表 |

---

## 8. Gun.js 核心 (Gun.js Core)

### 位置
`server.js` 行 478-540

### 初始化參數

```javascript
gunInstance = Gun({
  web: httpServer,           // 綁定 HTTP 伺服器
  file: DATA_DIR,            // 數據持久化目錄
  radisk: true,              // 啟用磁盤存儲
  localStorage: false,       // 禁用 localStorage
  multicast: false,          // 禁用多播發現
  axe: false,                // 禁用 AXE（簡化中繼）

  ws: {
    path: '/gun',                              // WebSocket 路徑
    pingTimeout: CONFIG.CONNECTION_TIMEOUT,    // 30 秒
    pingInterval: CONFIG.WS_HEARTBEAT_INTERVAL // 25 秒
  }
});
```

### 事件監聽

```javascript
// 新連接
gunInstance.on('hi', (peer) => {
  const peerId = peer.id || peer.wire?.id;
  const ip = peer.wire?.upgradeReq?.socket?.remoteAddress;
  const userAgent = peer.wire?.upgradeReq?.headers?.['user-agent'];

  trackPeerConnection(peerId, ip, userAgent);
});

// 連接斷開
gunInstance.on('bye', (peer) => {
  untrackPeerConnection(peer.id);
});
```

---

## 9. 定期任務 (Periodic Tasks)

### 位置
`server.js` 行 544-583

### 任務列表

| 任務 | 間隔 | 功能 |
|------|------|------|
| 健康檢查 | 60 秒 | 檢查磁盤、數據目錄、連接統計 |
| 過期清理 | 60 秒 | 清理超時連接 |
| 記憶體監控 | 120 秒 | 檢測高使用量，觸發 GC |

### 健康檢查

```javascript
setInterval(() => {
  checkDiskSpace();

  if (!fs.existsSync(DATA_DIR)) {
    log('ERROR', 'Data directory no longer exists!');
  }

  // 平台統計
  const platforms = {};
  for (const [, peer] of peerConnections) {
    platforms[peer.platform] = (platforms[peer.platform] || 0) + 1;
  }

  log('DEBUG', 'Health check', { uptime, peers, platforms, memory });
}, CONFIG.HEALTH_CHECK_INTERVAL);
```

### 記憶體監控

```javascript
setInterval(() => {
  const heapUsedMB = Math.round(memUsage.heapUsed / 1024 / 1024);

  if (heapUsedMB > 500) {
    log('WARN', 'High memory usage', { heapUsedMB });

    // 強制垃圾回收（如可用）
    if (global.gc) {
      global.gc();
    }
  }
}, 120000);
```

---

## 10. 優雅關閉 (Graceful Shutdown)

### 位置
`server.js` 行 591-609

### 程式碼

```javascript
async function gracefulShutdown(signal) {
  log('INFO', `Received ${signal}, shutting down gracefully...`);

  // 關閉 HTTP 伺服器（停止接受新連接）
  if (httpServer) {
    httpServer.close(() => {
      log('INFO', 'HTTP server closed');
    });
  }

  // 給現有連接時間完成
  setTimeout(() => {
    log('INFO', 'Shutdown complete');
    process.exit(0);
  }, 2000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
```

### 關閉流程

```
1. 接收信號 (SIGTERM/SIGINT)
2. 停止接受新連接
3. 等待 2 秒讓現有連接完成
4. 退出進程
```

---

## 組件依賴關係

```
startServer()
    │
    ├── ensureDataDirectory()     ← 失敗則退出
    │
    ├── checkDiskSpace()          ← 警告但繼續
    │
    ├── http.createServer(app)
    │       │
    │       └── Express Middleware Chain
    │               │
    │               └── REST Routes
    │
    ├── Gun({ web: httpServer })
    │       │
    │       ├── on('hi')  → trackPeerConnection()
    │       │
    │       └── on('bye') → untrackPeerConnection()
    │
    └── Periodic Tasks
            │
            ├── Health Check (60s)
            │
            ├── cleanupStalePeers() (60s)
            │
            └── Memory Monitor (120s)
```
