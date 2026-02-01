# REST API 與 WebSocket 協議

## REST API

### 健康檢查 - GET /

主要健康檢查端點，返回伺服器狀態和統計信息。

```bash
curl http://localhost:8765/
```

**回應：**
```json
{
  "status": "ok",
  "service": "Gun.js Relay Server",
  "message": "Counter Sync Relay is running!",
  "version": "2.0.0-ios-optimized",
  "uptime": 3600,
  "memory": {
    "heapUsed": "125 MB",
    "heapTotal": "512 MB"
  },
  "disk": {
    "freeGB": "50.25 GB",
    "freePercent": "75.50%",
    "warning": false
  },
  "dataDirectory": {
    "path": "/app/data",
    "exists": true
  },
  "connections": {
    "total": 42,
    "byPlatform": {
      "ios": 15,
      "android": 10,
      "windows": 8,
      "macos": 5,
      "linux": 3,
      "other": 1
    }
  },
  "config": {
    "heartbeatInterval": 25000,
    "connectionTimeout": 30000
  }
}
```

### 心跳測試 - GET /ping

簡單的 ping/pong 機制，用於連接檢測和延遲測量。

```bash
curl http://localhost:8765/ping
```

**回應：**
```json
{
  "pong": true,
  "timestamp": 1735735200000,
  "serverTime": "2025-02-01T12:00:00.000Z"
}
```

### 連接狀態 - GET /status

返回連接狀態和 iOS 專用的建議參數。

```bash
curl http://localhost:8765/status \
  -H "User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
```

**回應：**
```json
{
  "connected": true,
  "platform": "ios",
  "serverTime": "2025-02-01T12:00:00.000Z",
  "peers": 42,
  "recommendations": {
    "pingInterval": 25000,
    "reconnectDelay": 1000,
    "maxReconnectDelay": 30000
  }
}
```

**平台專屬建議：**

| 平台 | pingInterval | maxReconnectDelay |
|------|-------------|-------------------|
| iOS | 25000 | 30000 |
| 其他 | 30000 | 60000 |

### Server-Sent Events - GET /events

iOS Safari 保活機制，每 25 秒發送心跳事件。

```bash
curl -N http://localhost:8765/events
```

**事件流：**
```
data: {"type":"connected","platform":"ios","timestamp":1735735200000}

data: {"type":"heartbeat","timestamp":1735735225000}

data: {"type":"heartbeat","timestamp":1735735250000}
```

**特性：**
- 首次連接事件
- 每 25 秒心跳
- 禁用 nginx 緩衝 (`X-Accel-Buffering: no`)
- 適合 iOS Safari 保活

### 寫入測試 - GET /debug/write-test

測試 Gun 寫入功能是否正常。

```bash
curl http://localhost:8765/debug/write-test
```

**成功回應 (200)：**
```json
{
  "success": true,
  "message": "Write test passed",
  "key": "test-1735735200000",
  "value": {
    "timestamp": "2025-02-01T12:00:00.000Z",
    "test": true
  }
}
```

**失敗回應 (500)：**
```json
{
  "success": false,
  "error": "Write timeout after 5 seconds"
}
```

### 連接詳情 - GET /debug/connections

返回所有連接的詳細信息。

```bash
curl http://localhost:8765/debug/connections
```

**回應：**
```json
{
  "total": 42,
  "connections": [
    {
      "id": "abc12345...",
      "platform": "ios",
      "connectedFor": "125s",
      "lastActivity": "2s ago"
    }
  ],
  "byIP": {
    "192.168.1.100": 3,
    "10.0.0.50": 2
  }
}
```

## WebSocket 協議

### 連接 URL

```
ws://localhost:8765/gun
wss://your-domain.com/gun
```

### Gun 協議訊號

#### PUT (寫入)

```json
{
  "#": "message-id",
  "put": {
    "user/alice": {
      "name": {
        "value": "Alice",
        ">": 1735735200
      }
    }
  }
}
```

#### GET (讀取)

```json
{
  "#": "message-id",
  "get": {
    "#": "user/alice",
    ".": "name"
  }
}
```

#### ACK (確認)

```json
{
  "#": "message-id",
  "@": "original-message-id",
  "ok": true
}
```

#### 心跳

```json
{ "ping": 1735735200000 }
{ "pong": 1735735200000 }
```

### 協議特性

| 特性 | 說明 |
|------|------|
| Soul | 圖形節點唯一標識符 |
| Key | 節點中的屬性名 |
| Value | 實際數據值 |
| State (>) | 版本戳記（時間戳） |
| CRDT | Last-Write-Wins 衝突解決 |
| Broadcast | 自動向其他連接廣播更新 |

### 寫入流程

```
1. 客戶端發送 PUT 訊號

2. 伺服器合併到記憶體圖形
   - 使用 Last-Write-Wins 策略
   - 比較時間戳記

3. 廣播給其他連接的客戶端

4. 持久化到磁盤 (/data/*.yap)

5. 發送 ACK
```

### 讀取流程

```
1. 客戶端發送 GET 訊號

2. 伺服器查詢記憶體圖形

3. 返回數據 PUT
```

## CORS 配置

### 預設配置

```javascript
res.header('Access-Control-Allow-Origin', origin || '*');
res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
res.header('Access-Control-Allow-Credentials', 'true');
res.header('Access-Control-Max-Age', '86400');  // 24 小時

// iOS Safari 特定頭部
res.header('Cache-Control', 'no-store, no-cache, must-revalidate');
```

### 生產環境建議

```javascript
const allowedOrigins = [
  'https://yourdomain.com',
  'https://app.yourdomain.com'
];

res.header('Access-Control-Allow-Origin',
  allowedOrigins.includes(origin) ? origin : '');
```

## 錯誤回應

### 500 Internal Server Error

```json
{
  "error": "Internal server error",
  "message": "Error description"
}
```

### 404 Not Found

```json
{
  "error": "Not found",
  "message": "Resource not found"
}
```
