# 部署方式與配置

## 部署方案比較

| 特性 | Express Server | Docker | Cloudflare Tunnel | Cloudflare Workers |
|------|---------------|--------|-------------------|-------------------|
| 需要伺服器 | 是 | 是 | 是 | 否 |
| 全球部署 | 否 | 否 | 否 | 是（邊緣） |
| 成本 | 按伺服器 | 按伺服器 | 免費 | $5/月起 |
| 延遲 | 中等 | 中等 | 中等 | 極低 |
| WebSocket | 原生 | 原生 | 原生 | 原生 |

## 方案 1：本地 Express 伺服器

### 快速啟動

```bash
# 安裝依賴
npm install

# 設置環境變數（可選）
export PORT=8765
export LOG_LEVEL=INFO
export WS_HEARTBEAT_INTERVAL=25000

# 啟動伺服器
npm start

# 驗證
curl http://localhost:8765/
curl http://localhost:8765/ping
```

### 系統要求

- Node.js ≥ 18.0.0
- 可用磁盤空間
- 開放端口 8765

## 方案 2：Docker 部署

### Dockerfile

```dockerfile
FROM node:20-alpine
RUN apk add --no-cache curl
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY server.js ./
RUN mkdir -p /app/data
ENV PORT=8765
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8765/ || exit 1
RUN adduser -S nodejs
USER nodejs
CMD ["node", "server.js"]
```

### 構建和運行

```bash
# 構建鏡像
docker build -t gun-relay:latest .

# 運行容器
docker run -d \
  -p 8765:8765 \
  -e LOG_LEVEL=INFO \
  -e WS_HEARTBEAT_INTERVAL=25000 \
  -v gun-data:/app/data \
  --name gun-relay \
  gun-relay:latest

# 驗證
docker exec gun-relay curl http://localhost:8765/

# 查看日誌
docker logs -f gun-relay
```

### Docker Compose

```yaml
version: '3.8'
services:
  gun-relay:
    build: .
    ports:
      - "8765:8765"
    environment:
      - LOG_LEVEL=INFO
      - WS_HEARTBEAT_INTERVAL=25000
      - CONNECTION_TIMEOUT=30000
    volumes:
      - gun-data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  gun-data:
```

## 方案 3：Render 平台

### render.yaml

```yaml
services:
  - type: web
    name: gun-relay
    env: node
    plan: free
    buildCommand: npm install
    startCommand: npm start
    healthCheckPath: /
    envVars:
      - key: NODE_ENV
        value: production
      - key: LOG_LEVEL
        value: INFO
```

### 部署步驟

1. 連接 GitHub 帳戶到 Render
2. 創建新的 Web Service
3. 選擇 gun-relay 倉庫
4. Render 自動檢測 render.yaml
5. 點擊 Deploy

### 自動化特性

- 自動構建和部署
- 健康檢查自動監控
- 錯誤時自動重啟
- 環境變數管理

## 方案 4：Cloudflare Tunnel

### 架構

```
Internet → Cloudflare Edge → Tunnel → Local Server (8765)
```

### 安裝 cloudflared

```bash
# macOS
brew install cloudflare/cloudflare/cloudflared

# Linux
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# Windows
choco install cloudflared
```

### 設置 Tunnel

```bash
# 登入 Cloudflare
cloudflared tunnel login

# 創建 Tunnel
cloudflared tunnel create gun-relay
# 記下 Tunnel ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# 配置 DNS
cloudflared tunnel route dns gun-relay gun-relay.yourdomain.com
```

### config.yml

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /root/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: gun-relay.yourdomain.com
    service: http://localhost:8765
    originRequest:
      connectTimeout: 30s
      tcpKeepAlive: 30s
      keepAliveConnections: 100
      keepAliveTimeout: 90s
      http2Origin: true

  - hostname: gun-health.yourdomain.com
    service: http://localhost:8765

  - service: http_status:404

loglevel: info
metrics: localhost:8766
protocol: quic
retries: 5
grace-period: 30s
```

### 啟動

```bash
# 啟動本地伺服器
npm start &

# 啟動 Tunnel
cloudflared tunnel --config cloudflare-tunnel/config.yml run
```

### Docker Compose Tunnel

```yaml
version: '3.8'
services:
  gun-relay:
    build: ..
    ports:
      - "8765:8765"
    volumes:
      - gun-data:/app/data

  cloudflared:
    image: cloudflare/cloudflared:latest
    command: tunnel --config /etc/cloudflared/config.yml run
    volumes:
      - ./config.yml:/etc/cloudflared/config.yml
      - ./creds:/etc/cloudflared/creds
    depends_on:
      - gun-relay

volumes:
  gun-data:
```

## 方案 5：Cloudflare Workers

### 架構

```
Internet → Cloudflare Workers (邊緣計算) → Durable Objects (狀態持久化)
```

### wrangler.toml

```toml
name = "gun-relay"
main = "src/index.js"
compatibility_date = "2024-01-01"
compatibility_flags = ["nodejs_compat"]

workers_dev = true
route = { pattern = "gun-relay.yourdomain.com/*", zone_name = "yourdomain.com" }

[durable_objects]
bindings = [
  { name = "GUN_RELAY", class_name = "GunRelayDO" }
]

[[migrations]]
tag = "v1"
new_classes = ["GunRelayDO"]

[[kv_namespaces]]
binding = "GUN_CACHE"
id = "your-kv-namespace-id"

[vars]
ENVIRONMENT = "production"
WS_HEARTBEAT_INTERVAL = "25000"
CONNECTION_TIMEOUT = "30000"
```

### 部署

```bash
cd cloudflare-workers

# 登入 Wrangler
wrangler login

# 創建 KV 命名空間
wrangler kv:namespace create GUN_CACHE

# 開發測試
wrangler dev

# 部署
wrangler deploy
```

### Workers 特性

| 特性 | 值 |
|------|-----|
| 請求超時 | 30 秒 |
| 請求體大小 | 100 MB |
| Durable Objects 費用 | $0.15/百萬請求 |
| 存儲費用 | $0.20/GB-月 |
| 全球數據中心 | 200+ |
| 延遲 | 1-10 ms |

## 監控命令

```bash
# 健康狀態
curl http://localhost:8765/ | jq '.status'

# 記憶體使用
curl http://localhost:8765/ | jq '.memory'

# 磁盤空間
curl http://localhost:8765/ | jq '.disk'

# 連接統計
curl http://localhost:8765/ | jq '.connections'

# 連接詳情
curl http://localhost:8765/debug/connections

# 驗證寫入
curl http://localhost:8765/debug/write-test
```

## 日誌查詢

```bash
# Docker 日誌
docker logs -f gun-relay

# 查看連接事件
docker logs gun-relay | grep "Peer connected\|Peer disconnected"

# 查看錯誤
docker logs gun-relay | grep "\[ERROR\]"

# 實時跟蹤警告和錯誤
docker logs -f gun-relay | grep -E "\[ERROR\]|\[WARN\]"
```
