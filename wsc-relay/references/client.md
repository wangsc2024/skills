# 客戶端連接指南

## Gun.js 基礎配置

### 基本連接

```javascript
const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  retry: 1000,            // 重連延遲 (ms)
  until: 2000,            // 等待超時
  chunk: 1024 * 50,       // 分塊大小
  axe: false              // 禁用 AXE
});
```

### 完整配置選項

```javascript
const gun = Gun({
  peers: ['https://relay1.com/gun', 'https://relay2.com/gun'],

  // 連接參數
  retry: 1000,              // 首次重連延遲 (ms)
  until: 2000,              // 請求超時
  timeout: 30000,           // 連接超時
  chunk: 1024 * 50,         // 分塊大小

  // 功能開關
  localStorage: false,      // 禁用 localStorage
  radisk: false,            // 禁用本地磁盤存儲
  multicast: false,         // 禁用多播發現
  axe: false,               // 禁用 AXE
  file: false               // 禁用檔案存儲
});
```

## iOS 優化配置

iOS 會在 30 秒無活動後關閉背景 WebSocket 連接，需要特別處理。

### 完整 iOS 配置

```javascript
const gun = Gun({
  peers: ['https://your-relay.com/gun'],

  // iOS 優化參數
  retry: 1000,
  until: 2000,
  timeout: 30000,
  localStorage: false,
  multicast: false,
  axe: false
});

// 連接狀態追蹤
let isConnected = false;
let reconnectAttempts = 0;
const MAX_RECONNECT_DELAY = 30000;

// 監聽連接事件
gun.on('hi', peer => {
  isConnected = true;
  reconnectAttempts = 0;
  console.log('Connected to relay:', peer.id);
});

gun.on('bye', peer => {
  isConnected = false;
  console.log('Disconnected from relay:', peer.id);
});

// 保活機制 (每 25 秒)
setInterval(() => {
  if (!isConnected) {
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), MAX_RECONNECT_DELAY);
    reconnectAttempts++;

    setTimeout(() => {
      gun.opt({ peers: ['https://your-relay.com/gun'] });
    }, delay);
  }
}, 25000);
```

### iOS Safari SSE 保活

如果 WebSocket 不穩定，可以使用 Server-Sent Events 作為備用保活機制：

```javascript
// SSE 保活連接
const eventSource = new EventSource('https://your-relay.com/events');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('SSE heartbeat:', data);
};

eventSource.onerror = () => {
  console.log('SSE disconnected, reconnecting...');
  // EventSource 會自動重連
};
```

## React Native 配置

### 基本設置

```javascript
import Gun from 'gun';

const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  localStorage: false,    // React Native 無傳統 localStorage
  radisk: false,          // 使用 AsyncStorage 代替
  timeout: 30000,
  retry: 1000,
  until: 2000
});
```

### 使用 AsyncStorage

```javascript
import Gun from 'gun';
import AsyncStorage from '@react-native-async-storage/async-storage';

// 自定義 AsyncStorage 適配器
const asyncStorageAdapter = {
  setItem: async (key, value) => {
    await AsyncStorage.setItem(key, value);
  },
  getItem: async (key) => {
    return await AsyncStorage.getItem(key);
  },
  removeItem: async (key) => {
    await AsyncStorage.removeItem(key);
  }
};

const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  localStorage: asyncStorageAdapter
});
```

### React Native 連接監控

```javascript
import { AppState } from 'react-native';

// 監聽 App 狀態
AppState.addEventListener('change', (nextAppState) => {
  if (nextAppState === 'active') {
    // App 回到前台，重新連接
    gun.opt({ peers: ['https://your-relay.com/gun'] });
  }
});
```

## Web 客戶端

### HTML 範例

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/gun/gun.js"></script>
</head>
<body>
  <script>
    const gun = Gun({
      peers: ['https://your-relay.com/gun']
    });

    // 寫入數據
    gun.get('users').get('alice').put({
      name: 'Alice',
      email: 'alice@example.com'
    }, (ack) => {
      if (ack.err) {
        console.error('Write failed:', ack.err);
      } else {
        console.log('Write success');
      }
    });

    // 實時監聽
    gun.get('users').get('alice').on((data) => {
      console.log('User updated:', data);
    });
  </script>
</body>
</html>
```

### ES6 模組

```javascript
import Gun from 'gun';

const gun = Gun({
  peers: ['https://your-relay.com/gun']
});

export default gun;
```

## 多伺服器配置 (高可用)

```javascript
const gun = Gun({
  peers: [
    'https://relay1.yourdomain.com/gun',
    'https://relay2.yourdomain.com/gun',
    'https://relay3.yourdomain.com/gun'
  ]
});

// Gun 會自動連接所有 peers 並同步數據
```

## 數據操作

### 寫入數據

```javascript
// 簡單寫入
gun.get('users').get('alice').put({
  name: 'Alice',
  email: 'alice@example.com'
});

// 帶回調的寫入
gun.get('users').get('alice').put({
  name: 'Alice',
  email: 'alice@example.com'
}, (ack) => {
  if (ack.err) {
    console.error('Write failed:', ack.err);
  } else {
    console.log('Write acknowledged');
  }
});
```

### 讀取數據

```javascript
// 一次性讀取
gun.get('users').get('alice').once((data) => {
  console.log('User:', data);
});

// 實時監聽（數據變化時觸發）
gun.get('users').get('alice').on((data) => {
  console.log('User updated:', data);
});

// 取消監聯
const listener = gun.get('users').get('alice').on((data) => {
  console.log('User:', data);
});
listener.off();  // 取消監聽
```

### 嵌套數據

```javascript
// 創建引用
const alice = gun.get('users').get('alice');
const post = gun.get('posts').get('post-1');

post.put({
  title: 'Hello World',
  content: 'This is my first post',
  author: alice  // 引用其他節點
});

// 解析引用
post.get('author').once((author) => {
  console.log('Author:', author);
});
```

### 集合操作

```javascript
// 添加到集合
gun.get('posts').set({
  title: 'New Post',
  timestamp: Date.now()
});

// 遍歷集合
gun.get('posts').map().once((post, id) => {
  console.log(id, post);
});
```

## 連接監控

```javascript
const connectionStats = {
  connected: false,
  peersConnected: 0,
  lastActivity: Date.now()
};

// 監聽連接事件
gun.on('hi', (peer) => {
  connectionStats.connected = true;
  connectionStats.peersConnected++;
  connectionStats.lastActivity = Date.now();
  console.log(`Connected to peer: ${peer.id}`);
});

gun.on('bye', (peer) => {
  connectionStats.peersConnected--;
  if (connectionStats.peersConnected === 0) {
    connectionStats.connected = false;
  }
  console.log(`Disconnected from peer: ${peer.id}`);
});

// 定期監控
setInterval(() => {
  const timeSinceLastActivity = Date.now() - connectionStats.lastActivity;
  if (timeSinceLastActivity > 60000 && connectionStats.connected) {
    console.warn('No activity for 60 seconds');
  }
}, 10000);
```

## 錯誤處理

```javascript
// 全局錯誤處理
gun.on('out', (msg) => {
  if (msg.err) {
    console.error('Gun error:', msg.err);
  }
});

// 寫入錯誤處理
gun.get('data').put({ value: 'test' }, (ack) => {
  if (ack.err) {
    console.error('Write error:', ack.err);
    // 重試邏輯
    setTimeout(() => {
      gun.get('data').put({ value: 'test' });
    }, 1000);
  }
});
```

## 離線支援

```javascript
// 啟用本地存儲（離線優先）
const gun = Gun({
  peers: ['https://your-relay.com/gun'],
  localStorage: true,      // 啟用 localStorage
  radisk: true             // 啟用 IndexedDB
});

// 離線時數據會存儲在本地
// 恢復連接後自動同步到伺服器
```
