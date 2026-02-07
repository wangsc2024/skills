---
name: nanoclaw-skills
version: 1.0.0
description: |
  Nanoclaw AI Agent 技能集合，包含知識管理、市場分析、軟體工程、瀏覽器自動化和日常例行程序管理。
  已針對資訊安全與個資保護進行強化，符合台灣個資法與 GDPR 基本要求。
  Use when: 需要 AI Agent 執行自動化任務、知識管理、市場監控、開發輔助，or when user mentions nanoclaw, agent, 自動化代理.
  Triggers: "nanoclaw", "agent skills", "知識助手", "市場分析", "瀏覽器自動化", "日常例行"
---

# Nanoclaw Skills 安全強化版

> **版本**: 1.0.0
> **安全等級**: 強化版（資安 + 個資保護）
> **合規標準**: 台灣個資法、GDPR 基本原則

---

## 目錄

1. [安全總則](#安全總則)
2. [Knowledge Assistant（知識助手）](#knowledge-assistant知識助手)
3. [Market Analysis（市場分析）](#market-analysis市場分析)
4. [Software Engineer（軟體工程）](#software-engineer軟體工程)
5. [Agent Browser（瀏覽器自動化）](#agent-browser瀏覽器自動化)
6. [Daily Routine（日常例行）](#daily-routine日常例行)

---

## 安全總則

### 資料分級制度

| 等級 | 定義 | 範例 | 處理要求 |
|------|------|------|----------|
| **L4 極機密** | 法律規範之特種個資 | 醫療紀錄、犯罪前科、性生活 | **禁止儲存**，僅限即時處理 |
| **L3 機密** | 一般個人資料 | 姓名、地址、電話、身分證號 | 加密儲存 + 存取日誌 |
| **L2 內部** | 敏感商業資訊 | 財務數據、交易紀錄、密碼 | 加密儲存 |
| **L1 公開** | 非敏感資訊 | 公開新聞、技術文檔 | 一般儲存 |

### 強制安全規則

```yaml
全域規則:
  - 禁止明文儲存密碼、API Key、Token
  - 禁止在日誌中記錄完整個資
  - 所有外部 API 呼叫必須使用 HTTPS
  - 錯誤訊息禁止暴露系統內部資訊
  - 群組間資料完全隔離，禁止跨群組存取

個資處理原則（台灣個資法）:
  - 蒐集前告知目的與利用範圍
  - 最小化原則：僅蒐集必要資料
  - 當事人有權請求刪除
  - 禁止未經授權的國際傳輸
```

### 資料保留政策

| 資料類型 | 保留期限 | 刪除方式 |
|----------|----------|----------|
| 對話日誌 | 30 天 | 自動清除 |
| 知識庫內容 | 用戶決定 | 安全覆寫 |
| 認證狀態 | 24 小時 | 自動過期 + 刪除 |
| 市場快照 | 7 天 | 自動歸檔 |

---

## Knowledge Assistant（知識助手）

### 功能描述

個人知識管理助手，用於資訊保存、檢索、研究、摘要與概念連接。

### 安全強化項目

#### 1. 敏感資料加密

```python
# 必須對 L2/L3 等級資料加密後儲存
from cryptography.fernet import Fernet

class SecureKnowledgeStore:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)

    def save_sensitive(self, data: str, data_level: str) -> bytes:
        """儲存敏感資料，L2/L3 必須加密"""
        if data_level in ['L2', 'L3']:
            return self.cipher.encrypt(data.encode())
        return data.encode()

    def load_sensitive(self, encrypted_data: bytes, data_level: str) -> str:
        """讀取敏感資料"""
        if data_level in ['L2', 'L3']:
            return self.cipher.decrypt(encrypted_data).decode()
        return encrypted_data.decode()
```

#### 2. 個資識別與標記

```yaml
個資識別模式:
  身分證號: '/[A-Z][12][0-9]{8}/'
  手機號碼: '/09[0-9]{8}/'
  Email: '/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/'
  信用卡: '/[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}/'

自動處理:
  - 識別到個資時自動標記等級
  - 儲存前警告用戶
  - L3 以上需用戶明確同意
```

#### 3. 存取控制與審計

```yaml
存取日誌格式:
  timestamp: ISO8601
  action: read | write | delete
  data_level: L1 | L2 | L3 | L4
  user_id: string
  data_hash: SHA256（不記錄原始內容）

日誌保留: 90 天
```

### 儲存結構（安全版）

```
/workspace/group/
├── CLAUDE.md              # 核心知識（僅 L1 資料）
├── .secrets/              # 加密儲存區（L2/L3）
│   ├── keyfile.enc        # 加密金鑰（由主金鑰保護）
│   └── sensitive.enc      # 加密內容
├── memory/
│   ├── MEMORY.md          # 長期知識（已脫敏）
│   └── YYYY-MM-DD.md      # 每日筆記
├── knowledge/
│   ├── topics/
│   ├── bookmarks/
│   └── notes/
└── audit/
    └── access_log.jsonl   # 存取日誌
```

### 禁止事項

- **禁止**儲存 L4 等級資料（醫療、犯罪紀錄）
- **禁止**未加密儲存 L3 資料
- **禁止**在摘要中引用完整個資
- **禁止**跨群組分享知識

---

## Market Analysis（市場分析）

### 功能描述

24/7 金融市場監控與分析，提供股票、加密貨幣、外匯等市場資訊。

### 安全強化項目

#### 1. 強制免責聲明

```markdown
## 重要聲明

本分析僅供參考，不構成任何投資建議、招攬或推薦。

1. **非投資建議**：本內容不構成購買、出售或持有任何金融商品的建議
2. **風險警示**：投資涉及風險，過去績效不代表未來表現
3. **獨立決策**：用戶應自行評估並諮詢專業顧問後做出投資決策
4. **免責條款**：對因使用本資訊而產生的任何損失，概不負責

依據台灣《證券投資信託及顧問法》，未經主管機關核准，
不得從事證券投資顧問業務。本系統不提供個別投資建議。
```

#### 2. 禁止行為清單

```yaml
絕對禁止:
  - 提供具體買入/賣出建議（如「建議買進 XXX 股票」）
  - 預測特定價格目標（如「目標價 150 元」）
  - 推薦特定投資組合配置
  - 存取或處理用戶個人金融帳戶資料
  - 執行任何交易操作

允許行為:
  - 報導公開市場數據
  - 解釋技術指標定義
  - 彙整公開財經新聞
  - 教育性質的投資概念說明
```

#### 3. 資料來源限制

```yaml
允許資料來源:
  - 公開財經新聞（Reuters、Bloomberg、CNBC）
  - 交易所公開數據
  - 公司公開財報
  - 政府經濟統計

禁止資料來源:
  - 用戶個人交易紀錄
  - 非公開內線資訊
  - 未經驗證的社群媒體傳言
```

### 輸出格式（安全版）

```markdown
## 市場摘要 - [日期]

> **免責聲明**：本資訊僅供參考，不構成投資建議。投資有風險，請謹慎評估。

### 主要指數
- S&P 500: [數值] ([變動%])
- NASDAQ: [數值] ([變動%])

### 市場觀察
[客觀描述市場狀況，不包含買賣建議]

### 新聞摘要
[引用公開新聞來源]

---
資料來源：公開市場數據 | 更新時間：[時間戳]
```

---

## Software Engineer（軟體工程）

### 功能描述

全端開發助手，支援程式碼生成、審查、除錯、架構設計和 DevOps 任務。

### 安全強化項目

#### 1. OWASP Top 10 防護清單

| 漏洞 | 防護措施 | 檢查項目 |
|------|----------|----------|
| **A01 存取控制失效** | 實作 RBAC、最小權限 | 每個端點都有權限檢查 |
| **A02 加密失效** | 使用 TLS 1.3、AES-256 | 禁止明文傳輸敏感資料 |
| **A03 注入攻擊** | 參數化查詢、輸入驗證 | 禁止字串拼接 SQL |
| **A04 不安全設計** | 威脅建模、安全需求 | 設計階段考慮安全 |
| **A05 安全設定缺陷** | 安全預設、最小化部署 | 禁止預設密碼 |
| **A06 易受攻擊元件** | 依賴掃描、及時更新 | 定期執行 npm audit |
| **A07 認證失敗** | MFA、安全 Session | 禁止弱密碼 |
| **A08 軟體完整性失效** | 程式碼簽章、CI/CD 安全 | 驗證依賴來源 |
| **A09 日誌監控失效** | 結構化日誌、異常偵測 | 記錄安全事件 |
| **A10 SSRF** | 白名單、URL 驗證 | 禁止任意 URL 請求 |

#### 2. 秘密管理規範

```yaml
禁止:
  - 程式碼中硬編碼任何秘密（API Key、密碼、Token）
  - .env 檔案加入版本控制
  - 日誌中記錄秘密
  - 錯誤訊息中暴露秘密

必須:
  - 使用環境變數或秘密管理服務
  - 秘密輪換機制
  - 開發/測試/生產環境分離

檢查工具:
  - git-secrets (pre-commit hook)
  - truffleHog (秘密掃描)
  - detect-secrets (靜態分析)
```

#### 3. 安全程式碼範本

```typescript
// 安全的 API 端點範本
import { z } from 'zod';
import { rateLimit } from 'express-rate-limit';

// 輸入驗證 Schema
const UserInputSchema = z.object({
  email: z.string().email().max(255),
  name: z.string().min(1).max(100).regex(/^[a-zA-Z\s]+$/),
});

// 速率限制
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 分鐘
  max: 100, // 每個 IP 最多 100 次請求
  standardHeaders: true,
  legacyHeaders: false,
});

// 安全的錯誤處理
function handleError(error: unknown): ApiResponse {
  console.error('Internal error:', error); // 記錄完整錯誤
  return {
    success: false,
    message: 'An error occurred', // 不暴露內部細節
  };
}
```

#### 4. 程式碼審查安全清單

```yaml
審查必檢項目:
  輸入驗證:
    - [ ] 所有用戶輸入都經過驗證
    - [ ] 使用白名單而非黑名單
    - [ ] 驗證輸入長度和格式

  輸出編碼:
    - [ ] HTML 輸出經過轉義
    - [ ] JSON 輸出正確編碼
    - [ ] SQL 使用參數化查詢

  認證授權:
    - [ ] 敏感操作需要認證
    - [ ] 權限檢查在後端執行
    - [ ] Session 正確管理

  秘密處理:
    - [ ] 無硬編碼秘密
    - [ ] 日誌不包含敏感資料
    - [ ] 錯誤訊息不暴露內部
```

#### 5. 禁止的危險模式

```yaml
絕對禁止使用:
  - 動態程式碼執行（避免執行任意字串作為程式碼）
  - 不安全的反序列化
  - Shell 命令拼接（使用參數陣列）
  - 直接嵌入用戶輸入到 SQL/HTML

安全替代方案:
  資料解析: 使用 JSON.parse() 搭配驗證
  動態功能: 使用策略模式或工廠模式
  Shell 執行: 使用 execFile() 搭配參數陣列
```

---

## Agent Browser（瀏覽器自動化）

### 功能描述

網頁自動化工具，用於研究、填表、截圖、資料擷取。

### 安全強化項目

#### 1. 認證狀態安全

```yaml
認證狀態儲存規則:
  - 認證狀態檔案必須加密
  - 有效期限: 最長 24 小時
  - 自動過期後安全刪除
  - 禁止跨群組共享認證狀態

加密儲存:
  檔案: auth.json.enc
  演算法: AES-256-GCM
  金鑰: 每群組獨立
```

#### 2. URL 白名單機制

```yaml
URL 存取控制:
  預設政策: deny-all

  白名單範例:
    - "*.google.com"
    - "*.github.com"
    - "用戶明確授權的網域"

  絕對禁止:
    - 內網 IP (10.*, 172.16-31.*, 192.168.*)
    - localhost, 127.0.0.1
    - 敏感金融網站（除非明確授權）
    - 非 HTTPS 網站（含敏感資料時）
```

#### 3. XSS 與注入防護

```yaml
JavaScript 執行限制:
  - 禁止執行來自不可信來源的腳本
  - 禁止透過 JS 存取本地檔案系統
  - 所有執行的腳本必須經過審查

表單填充安全:
  - 填充前確認目標網站合法性
  - 禁止自動填充密碼到非 HTTPS 網站
  - 禁止填充信用卡資訊到未驗證網站
```

#### 4. 敏感操作警示

```yaml
需要用戶確認的操作:
  - 儲存認證狀態
  - 存取金融相關網站
  - 上傳檔案
  - 執行 JavaScript
  - 存取非白名單網域
```

### 安全使用範例

```bash
# 安全的認證狀態管理
agent-browser state save auth.json.enc --encrypt --expire 24h

# 載入前驗證
agent-browser state verify auth.json.enc
agent-browser state load auth.json.enc

# 安全清理
agent-browser state delete auth.json.enc --secure-wipe
agent-browser cookies clear
agent-browser storage clear
```

---

## Daily Routine（日常例行）

### 功能描述

個人日常例行程序與生產力管理助手，包含晨間簡報、任務管理、習慣追蹤。

### 安全強化項目

#### 1. 隱私保護規則

```yaml
對話摘要規則:
  允許:
    - 任務完成狀態（「完成 3 項任務」）
    - 習慣達成情況（「運動習慣連續 5 天」）
    - 時間統計（「工作 6 小時」）

  禁止:
    - 引用完整對話內容
    - 記錄對話對象名稱
    - 保存對話中的個人資訊
    - 記錄具體工作內容細節

摘要脫敏範例:
  原始: "與張先生討論了 A 公司的合併案"
  脫敏: "進行了 1 場商務討論"
```

#### 2. 日曆整合安全

```yaml
日曆存取原則:
  - 僅讀取用戶明確授權的日曆
  - 不儲存完整行事曆內容
  - 僅保留必要資訊（時間、標題）
  - 敏感會議標題自動脫敏

禁止行為:
  - 自動存取未授權的外部日曆
  - 分享日曆資訊給第三方
  - 永久儲存日曆詳細內容
```

#### 3. 通知安全

```yaml
通知內容規則:
  - 通知不包含完整任務內容
  - 使用通用描述（「您有 3 項待辦」而非具體內容）
  - 透過安全通道發送（HTTPS）
  - 通知日誌保留 7 天後刪除
```

### 安全日誌格式

```yaml
# 每日記錄（脫敏版）
date: 2026-02-07
summary:
  tasks_completed: 5
  tasks_pending: 2
  focus_hours: 4.5
  habits:
    - name: exercise
      completed: true
      streak: 5
notes: "高效的一天"  # 不包含具體內容
```

---

## 合規檢查清單

### 台灣個資法合規

- [ ] 蒐集前告知當事人蒐集目的
- [ ] 取得當事人同意（敏感資料需書面）
- [ ] 提供當事人查詢、更正、刪除權利
- [ ] 資料處理符合最小化原則
- [ ] 禁止未經授權的國際傳輸
- [ ] 發生資安事件 72 小時內通報

### GDPR 基本原則

- [ ] 合法性、公平性、透明性
- [ ] 目的限制
- [ ] 資料最小化
- [ ] 準確性
- [ ] 儲存限制
- [ ] 完整性與機密性
- [ ] 當責性

---

## 版本歷史

| 版本 | 日期 | 更新內容 |
|------|------|----------|
| 1.0.0 | 2026-02-07 | 安全強化版初版，整合 5 個 nanoclaw skills |
