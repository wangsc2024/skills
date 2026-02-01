---
name: wsc-relay
description: |
  Gun.js 中繼伺服器指南。指導如何部署、配置和使用 Gun relay server
  進行分散式實時數據同步。針對 iOS 優化，支援多種部署方式。
triggers:
  - "Gun"
  - "gun"
  - "relay"
  - "中繼"
  - "實時同步"
  - "分散式"
  - "WebSocket"
  - "數據同步"
  - "iOS sync"
---

# WSC Relay - Gun.js 中繼伺服器指南

專業的 Gun.js 中繼伺服器，針對 iOS 優化，支援多種部署方式。

## 什麼是 Gun Relay Server？

Gun.js 是一個分散式圖形資料庫，Relay Server 負責：
- **數據中繼**：在多個客戶端之間同步數據
- **持久化存儲**：將數據保存到磁盤
- **衝突解決**：使用 Last-Write-Wins (LWW) 策略
- **WebSocket 連接**：提供實時雙向通訊

## 核心組件

伺服器由 10 個核心組件構成，詳細說明請參考 [components.md](references/components.md)。

### 組件架構

```
┌─────────────────────────────────────────────────────────┐
│                    Gun Relay Server                      │
├─────────────────────────────────────────────────────────┤
│  CONFIG (環境配置)  │  Logging (日誌)  │  Error Handlers │
├─────────────────────────────────────────────────────────┤
│  File System Module   │   Peer Connection Tracking      │
│  - ensureDataDir()    │   - peerConnections (Map)       │
│  - checkDiskSpace()   │   - trackPeer/untrackPeer()     │
├─────────────────────────────────────────────────────────┤
│  Express Middleware:  CORS → Logging → JSON → Routes    │
├─────────────────────────────────────────────────────────┤
│  REST Routes: / │ /ping │ /status │ /events │ /debug/*  │
├─────────────────────────────────────────────────────────┤
│  Gun.js Core: gunInstance │ httpServer │ WebSocket /gun │
├─────────────────────────────────────────────────────────┤
│  Periodic Tasks: HealthCheck(60s) │ Cleanup(60s) │ Mem  │
├─────────────────────────────────────────────────────────┤
│  Graceful Shutdown: SIGTERM/SIGINT → gracefulShutdown() │
└─────────────────────────────────────────────────────────┘
```

### 組件列表

| 組件 | 說明 |
|------|------|
| **CONFIG** | 環境配置模組，支援環境變數覆寫 |
| **Logging System** | 分級日誌系統 (ERROR/WARN/INFO/DEBUG) |
| **File System Module** | 數據目錄驗證與磁盤空間監控 |
| **Global Error Handlers** | 捕獲未處理異常和 Promise 拒絕 |
| **Peer Connection Tracking** | 連接追蹤與 IP 限制管理 |
| **Express Middleware** | CORS、日誌、JSON 解析中間件鏈 |
| **REST Routes** | 健康檢查、心跳、狀態、調試端點 |
| **Gun.js Core** | Gun 實例初始化與事件監聽 |
| **Periodic Tasks** | 健康檢查、過期清理、記憶體監控 |
| **Graceful Shutdown** | 優雅關閉處理 |

### 關鍵設計決策

| 決策 | 原因 |
|------|------|
| 心跳 25 秒 | iOS 會在 30 秒後關閉閒置 WebSocket |
| 超時 30 秒 | 配合 iOS 超時機制 |
| Keep-Alive 65 秒 | 大於 60 秒避免代理超時 |
| 每 IP 限 10 連接 | 防止單一 IP 佔用過多資源 |
| 禁用 AXE | 簡化中繼邏輯，提高穩定性 |

## 快速開始

### 本地啟動

```bash
# 安裝依賴
npm install

# 啟動伺服器
npm start

# 驗證
curl http://localhost:8765/
curl http://localhost:8765/ping
```

### Docker 部署

```bash
# 構建鏡像
docker build -t gun-relay:latest .

# 運行容器
docker run -d \
  -p 8765:8765 \
  -e LOG_LEVEL=INFO \
  -v gun-data:/app/data \
  --name gun-relay \
  gun-relay:latest
```

## 配置選項

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `PORT` | 8765 | 監聽端口 |
| `WS_HEARTBEAT_INTERVAL` | 25000 | WebSocket 心跳間隔 (ms) - iOS 優化 |
| `CONNECTION_TIMEOUT` | 30000 | 連接超時 (ms) |
| `KEEP_ALIVE_TIMEOUT` | 65000 | HTTP Keep-Alive 超時 (ms) |
| `HEADERS_TIMEOUT` | 66000 | HTTP Headers 超時 (ms) |
| `PEER_CLEANUP_INTERVAL` | 60000 | 清理過期連接間隔 (ms) |
| `MAX_PEERS_PER_IP` | 10 | 每 IP 最大連接數 |
| `HEALTH_CHECK_INTERVAL` | 60000 | 健康檢查間隔 (ms) |
| `LOG_LEVEL` | INFO | 日誌級別 (DEBUG/INFO/WARN/ERROR) |

### iOS 優化說明

心跳間隔設為 25 秒是因為：
- iOS 會在 30 秒無活動後關閉 WebSocket 連接
- 25 秒心跳確保連接不會被系統殺死
- 搭配 30 秒連接超時，確保及時檢測斷線

## REST API 端點

### 健康檢查 - GET /

```bash
curl http://localhost:8765/
```

**回應：**
```json
{
  "status": "ok",
  "service": "Gun.js Relay Server",
  "version": "2.0.0-ios-optimized",
  "uptime": 3600,
  "memory": { "heapUsed": "125 MB", "heapTotal": "512 MB" },
  "disk": { "freeGB": "50.25 GB", "freePercent": "75.50%", "warning": false },
  "connections": {
    "total": 42,
    "byPlatform": { "ios": 15, "android": 10, "windows": 8 }
  }
}
```

### 心跳測試 - GET /ping

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

```bash
curl http://localhost:8765/status
```

**回應：**
```json
{
  "connected": true,
  "platform": "ios",
  "peers": 42,
  "recommendations": {
    "pingInterval": 25000,
    "reconnectDelay": 1000,
    "maxReconnectDelay": 30000
  }
}
```

### Server-Sent Events - GET /events

iOS Safari 保活機制，每 25 秒發送心跳事件：

```bash
curl -N http://localhost:8765/events
```

### 調試端點

```bash
# 寫入測試
curl http://localhost:8765/debug/write-test

# 連接詳情
curl http://localhost:8765/debug/connections
```

## WebSocket 連接

### 連接 URL

```
ws://localhost:8765/gun
wss://your-domain.com/gun
```

### Gun.js 客戶端配置

```javascript
const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  retry: 1000,            // 重連延遲 (ms)
  until: 2000,            // 等待超時
  chunk: 1024 * 50,       // 分塊大小
  axe: false,             // 禁用 AXE
  localStorage: false,    // 禁用 localStorage
  multicast: false        // 禁用多播
});
```

### iOS 優化配置

```javascript
const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  retry: 1000,
  until: 2000,
  timeout: 30000,
  localStorage: false,
  multicast: false,
  axe: false
});

// iOS 保活機制
let isConnected = false;

gun.on('hi', peer => {
  isConnected = true;
  console.log('Connected:', peer.id);
});

gun.on('bye', peer => {
  isConnected = false;
  console.log('Disconnected:', peer.id);
});

// 每 25 秒檢查連接狀態
setInterval(() => {
  if (!isConnected) {
    gun.opt({ peers: ['https://your-relay.com/gun'] });
  }
}, 25000);
```

### React Native 配置

```javascript
import Gun from 'gun';
import AsyncStorage from '@react-native-async-storage/async-storage';

const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  localStorage: AsyncStorage,  // 使用 AsyncStorage
  radisk: false,
  timeout: 30000,
  retry: 1000
});
```

## 部署方式

### 1. Render 平台

使用 render.yaml 自動部署：

```yaml
services:
  - type: web
    name: gun-relay
    env: node
    plan: free
    buildCommand: npm install
    startCommand: npm start
    healthCheckPath: /
```

### 2. Cloudflare Tunnel

安全暴露本地服務：

```bash
# 登入 Cloudflare
cloudflared tunnel login

# 創建 Tunnel
cloudflared tunnel create gun-relay

# 配置 DNS
cloudflared tunnel route dns gun-relay gun-relay.yourdomain.com

# 啟動 Tunnel
cloudflared tunnel run gun-relay
```

### 3. Cloudflare Workers

邊緣計算方案，全球最低延遲：

```bash
cd cloudflare-workers
wrangler login
wrangler deploy
```

## 數據操作

### 寫入數據

```javascript
gun.get('users').get('alice').put({
  name: 'Alice',
  email: 'alice@example.com'
}, ack => {
  if (ack.err) {
    console.error('Write failed:', ack.err);
  } else {
    console.log('Write success');
  }
});
```

### 讀取數據

```javascript
// 一次性讀取
gun.get('users').get('alice').once(data => {
  console.log('User:', data);
});

// 實時監聽
gun.get('users').get('alice').on(data => {
  console.log('User updated:', data);
});
```

### 嵌套數據

```javascript
gun.get('posts').get('post-1').put({
  title: 'Hello World',
  content: 'This is my first post',
  author: gun.get('users').get('alice')  // 引用其他節點
});
```

## 監控與故障排除

### 監控指標

```bash
# 健康狀態
curl http://localhost:8765/ | jq '.status'

# 記憶體使用
curl http://localhost:8765/ | jq '.memory'

# 磁盤空間
curl http://localhost:8765/ | jq '.disk'

# 連接統計
curl http://localhost:8765/ | jq '.connections'
```

### 常見問題

#### iOS 連接頻繁斷開

1. 確認心跳間隔 ≤ 25 秒
2. 確認連接超時 ≥ 30 秒
3. 使用 /events SSE 端點作為備用保活

```bash
# 調整配置
WS_HEARTBEAT_INTERVAL=15000
CONNECTION_TIMEOUT=45000
```

#### 寫入失敗

```bash
# 測試寫入功能
curl http://localhost:8765/debug/write-test

# 檢查磁盤空間
curl http://localhost:8765/ | jq '.disk'
```

#### 高記憶體使用

```bash
# 監控記憶體
watch -n 5 'curl -s http://localhost:8765/ | jq .memory'

# 降低連接限制
MAX_PEERS_PER_IP=5
```

### 日誌查詢

```bash
# Docker 日誌
docker logs -f gun-relay

# 查看連接事件
docker logs gun-relay | grep "Peer connected\|Peer disconnected"

# 查看錯誤
docker logs gun-relay | grep "\[ERROR\]"
```

## 安全建議

### 生產環境

1. **禁用調試端點** 或限制 IP 訪問
2. **使用 HTTPS/WSS** 加密連接
3. **配置防火牆** 限制端口訪問
4. **設置速率限制** 防止濫用
5. **定期備份** /data 目錄

### CORS 配置

```javascript
// 限制允許的來源
const allowedOrigins = [
  'https://yourdomain.com',
  'https://app.yourdomain.com'
];
```

## 專案結構

```
gun-relay/
├── server.js              # 主伺服器
├── Dockerfile             # Docker 配置
├── render.yaml            # Render 部署配置
├── package.json           # 依賴配置
├── data/                  # Gun 數據目錄
├── cloudflare-tunnel/     # Cloudflare Tunnel 配置
│   ├── config.yml
│   └── docker-compose.yml
└── cloudflare-workers/    # Workers 方案
    ├── src/index.js
    └── wrangler.toml
```

## 版本資訊

- **版本**: 2.0.0-ios-optimized
- **Node 版本**: ≥18.0.0
- **Gun 版本**: ^0.2020.1240
- **Express 版本**: ^4.18.2

## 相關資源

- [Gun.js 官方文檔](https://gun.eco/docs/)
- [Cloudflare Tunnel 文檔](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [Render 部署指南](https://render.com/docs)

---

**Generated by WSC** | Gun Relay Server Skill
