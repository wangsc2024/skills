# WSC Relay 參考文檔索引

## 概述

WSC Relay 是一個針對 iOS 優化的 Gun.js 中繼伺服器，提供分散式實時數據同步功能。

## 文檔結構

| 文檔 | 說明 |
|------|------|
| [components.md](components.md) | 核心組件說明（架構圖、模組詳解） |
| [server.md](server.md) | 伺服器架構與實作細節 |
| [api.md](api.md) | REST API 與 WebSocket 協議 |
| [deployment.md](deployment.md) | 部署方式與配置 |
| [client.md](client.md) | 客戶端連接指南 |

## 快速連結

### 配置

- 環境變數列表 → [server.md#環境變數](server.md#環境變數)
- iOS 優化參數 → [server.md#ios-優化](server.md#ios-優化)

### API

- REST 端點 → [api.md#rest-api](api.md#rest-api)
- WebSocket 協議 → [api.md#websocket](api.md#websocket)

### 部署

- Docker → [deployment.md#docker](deployment.md#docker)
- Cloudflare Tunnel → [deployment.md#cloudflare-tunnel](deployment.md#cloudflare-tunnel)
- Cloudflare Workers → [deployment.md#cloudflare-workers](deployment.md#cloudflare-workers)

### 客戶端

- Web 客戶端 → [client.md#web](client.md#web)
- React Native → [client.md#react-native](client.md#react-native)
- iOS 優化 → [client.md#ios](client.md#ios)
