# NICS 共通規範完整指引

國家資通安全研究院（NICS）依據 NIST 網路安全框架（Cybersecurity Framework, CSF）發布的共通規範完整清單。

## 框架對照

NIST CSF 2.0 六大功能：
1. **治理（Govern）** - 組織資安治理與風險管理
2. **識別（Identify）** - 資產與風險識別
3. **保護（Protect）** - 存取控制與資料保護
4. **偵測（Detect）** - 持續監控與異常偵測
5. **應變（Respond）** - 事件應變與溝通
6. **復原（Recover）** - 營運持續與改善

---

## 一、治理類（Govern）

### 1.1 政府資訊作業委外資安參考指引 v6.5

**適用對象**：辦理資訊作業委外之公務機關

**主要內容**：
- 委外前準備：需求規劃、風險評估、契約規範
- 廠商選擇：資安能力評估、第三方驗證要求
- 履約管理：監督機制、查核作業、資安事件處理
- 專案結案：資料移交、保密義務、智財權處理

**關鍵要求**：
- 受託者須具備完善資安管理措施或通過 ISO 27001 驗證
- 客製化系統開發須提供安全性檢測證明
- 委託金額達1,000萬以上須由第三方辦理安全性檢測

---

### 1.2 公務機關IT資安治理成熟度評估參考指引 v1.4

**評估框架**：ISG（Information Security Governance）

**評估構面**：
1. 治理架構（Governance Structure）
2. 風險管理（Risk Management）
3. 資源管理（Resource Management）
4. 績效管理（Performance Management）
5. 價值傳遞（Value Delivery）

**成熟度等級**：
- Level 0：不存在
- Level 1：初始
- Level 2：可重複
- Level 3：已定義
- Level 4：已管理
- Level 5：最佳化

---

### 1.3 關鍵基礎設施提供者IT資安治理成熟度評估參考指引 v1.0

**適用對象**：關鍵基礎設施提供者（Critical Infrastructure Provider）

**八大關鍵基礎設施領域**：
1. 能源（電力、石油、天然氣）
2. 水資源
3. 通訊傳播
4. 交通
5. 金融
6. 緊急救援與醫院
7. 政府機關
8. 科學園區與工業區

---

## 二、識別類（Identify）

### 2.1 紅隊演練作業參考指引 v1.0

**演練類型**：
- **紅隊**：攻擊方，模擬真實攻擊者
- **藍隊**：防守方，機關資安團隊
- **紫隊**：協調方，促進紅藍互動

**演練階段**：
1. 規劃準備：範圍界定、規則訂定、授權取得
2. 情蒐偵查：公開資訊蒐集、弱點探測
3. 初始入侵：社交工程、漏洞利用
4. 橫向移動：權限提升、內網滲透
5. 目標達成：機敏資料存取、系統控制
6. 報告撰寫：弱點清單、改善建議

**注意事項**：
- 須取得書面授權
- 不得影響正常營運
- 發現重大弱點須即時通報

---

### 2.2 資通系統風險評鑑參考指引 v4.1

**風險評鑑流程**：

```
資產識別 → 威脅識別 → 弱點識別 → 風險分析 → 風險評估 → 風險處理
```

**資產分類**：
- 資訊資產（資料、文件）
- 軟體資產（應用系統、作業系統）
- 實體資產（伺服器、網路設備）
- 服務資產（雲端服務、委外服務）
- 人員資產（員工、廠商）

**風險計算**：
```
風險值 = 資產價值 × 威脅可能性 × 弱點嚴重性
```

**風險處理策略**：
- 風險規避（Risk Avoidance）
- 風險降低（Risk Reduction）
- 風險轉移（Risk Transfer）
- 風險接受（Risk Acceptance）

---

## 三、保護類（Protect）

### 3.1 行動裝置資安參考指引

**BYOD 管理要求**：
- 裝置註冊與認證
- MDM（行動裝置管理）部署
- 資料容器化（Containerization）
- 遠端抹除功能

**安全設定**：
- 強制密碼/生物辨識
- 加密儲存
- 自動鎖定
- 禁用 USB 偵錯

---

### 3.2 網頁應用程式安全參考指引

**OWASP Top 10 對應措施**：

| 風險 | 防護措施 |
|-----|---------|
| 注入攻擊 | 參數化查詢、輸入驗證 |
| 認證失效 | MFA、安全的 Session 管理 |
| 敏感資料外洩 | 加密傳輸、加密儲存 |
| XXE | 禁用外部實體 |
| 存取控制失效 | RBAC、最小權限 |
| 安全配置錯誤 | 安全基線、定期審查 |
| XSS | 輸出編碼、CSP |
| 不安全的反序列化 | 驗證輸入、限制反序列化 |
| 使用已知弱點元件 | 弱點掃描、定期更新 |
| 不足的日誌與監控 | 集中日誌、異常偵測 |

**安全開發生命週期（SSDLC）**：
1. 需求階段：安全需求分析
2. 設計階段：威脅模型建立
3. 開發階段：安全編碼規範
4. 測試階段：安全性測試
5. 部署階段：安全配置
6. 維護階段：弱點管理

---

### 3.3 身分鑑別與存取控制參考指引

**身分鑑別強度**：
- Level 1：單因子（密碼）
- Level 2：雙因子（密碼 + OTP）
- Level 3：多因子（密碼 + 硬體金鑰 + 生物特徵）

**存取控制模型**：
- DAC（自主存取控制）
- MAC（強制存取控制）
- RBAC（角色存取控制）
- ABAC（屬性存取控制）

**零信任實施**：
- 持續驗證身分
- 設備健康檢查
- 最小權限原則
- 微分段（Micro-segmentation）

---

### 3.4 政府機關雲端服務應用資安參考指引

**雲端服務模式**：
- IaaS（基礎設施即服務）
- PaaS（平台即服務）
- SaaS（軟體即服務）

**雲端安全共同責任**：

| 項目 | IaaS | PaaS | SaaS |
|-----|------|------|------|
| 資料 | 用戶 | 用戶 | 用戶 |
| 應用程式 | 用戶 | 用戶 | 供應商 |
| 作業系統 | 用戶 | 供應商 | 供應商 |
| 網路 | 供應商 | 供應商 | 供應商 |
| 實體設施 | 供應商 | 供應商 | 供應商 |

**禁止事項**：
- 禁止使用大陸地區雲端服務
- 資料不得存放於大陸地區
- 不得跨大陸地區傳輸資料

---

### 3.5 資料保護參考指引

**資料分類**：
- 機密（Confidential）
- 敏感（Sensitive）
- 內部（Internal）
- 公開（Public）

**保護措施**：
- 傳輸加密：TLS 1.2+
- 儲存加密：AES-256
- 金鑰管理：HSM、KMS
- 資料遮罩：去識別化

**DLP（資料外洩防護）**：
- 端點 DLP
- 網路 DLP
- 雲端 DLP

---

### 3.6 電子郵件安全參考指引

**郵件安全機制**：
- SPF（寄件者政策框架）
- DKIM（網域金鑰識別郵件）
- DMARC（網域訊息驗證報告）

**防護措施**：
- 郵件閘道過濾
- 附件沙箱分析
- URL 重寫與檢查
- 釣魚郵件訓練

---

### 3.7 網路架構安全參考指引

**網路分段原則**：
- DMZ（非軍事區）
- 內部網路區隔
- 伺服器農場隔離
- 管理網路分離

**防火牆部署**：
- 邊界防火牆
- 內部防火牆
- Web 應用防火牆（WAF）
- 次世代防火牆（NGFW）

---

### 3.8 VPN 安全參考指引

**VPN 類型**：
- Site-to-Site VPN
- Remote Access VPN
- SSL VPN

**安全要求**：
- 強加密（AES-256、ChaCha20）
- 完美前向保密（PFS）
- 多因子認證
- Split Tunneling 管控

---

## 四、偵測類（Detect）

### 4.1 入侵偵測與防禦系統建置資安參考指引 v2.0

**IDS/IPS 類型**：
- NIDS（網路型入侵偵測）
- HIDS（主機型入侵偵測）
- NIPS（網路型入侵防禦）
- HIPS（主機型入侵防禦）

**偵測方法**：
- 簽章比對（Signature-based）
- 異常偵測（Anomaly-based）
- 啟發式分析（Heuristic）
- 行為分析（Behavioral）

**部署位置**：
- 網路邊界
- 核心交換器
- 重要伺服器前端
- 雲端環境

---

### 4.2 領域SOC實務建置指引 v1.0

**SOC 功能**：
- 7×24 監控
- 事件分析與調查
- 威脅獵捕（Threat Hunting）
- 情資整合與分享

**SOC 成熟度模型**：
- Level 1：被動反應
- Level 2：主動監控
- Level 3：威脅獵捕
- Level 4：情資驅動
- Level 5：自動化應變

**關鍵工具**：
- SIEM（安全資訊與事件管理）
- SOAR（安全協調、自動化與回應）
- EDR（端點偵測與回應）
- NDR（網路偵測與回應）
- TIP（威脅情資平台）

---

## 五、應變類（Respond）

### 5.1 領域CERT實務建置指引 v1.0

**CERT 職能**：
- 資安事件接收與分類
- 事件分析與調查
- 協調與溝通
- 技術支援與復原協助
- 弱點通報與追蹤

**事件處理流程**：
1. 準備（Preparation）
2. 偵測與分析（Detection & Analysis）
3. 圍堵與根除（Containment & Eradication）
4. 復原（Recovery）
5. 事後活動（Post-Incident Activity）

---

### 5.2 領域ISAC實務建置指引 v1.0

**ISAC 功能**：
- 情資蒐集與分析
- 情資分享與發布
- 早期預警
- 態勢感知

**情資分享等級**：
- TLP:RED - 僅限指定對象
- TLP:AMBER - 限組織內部
- TLP:GREEN - 限社群內部
- TLP:WHITE - 無限制

**情資格式**：
- STIX（結構化威脅資訊表達）
- TAXII（可信自動化情資交換）
- OpenIOC（開放入侵指標）

---

## 六、復原類（Recover）

### 6.1 營運持續管理參考指引 v2.0

**BCM 生命週期**：
1. 政策與方案管理
2. 風險評鑑與營運衝擊分析（BIA）
3. 營運持續策略
4. 營運持續計畫（BCP）
5. 演練與測試
6. 維護與審查

**關鍵指標**：
- **RTO**（Recovery Time Objective）：最大可容忍停機時間
- **RPO**（Recovery Point Objective）：最大可容忍資料遺失
- **MTPD**（Maximum Tolerable Period of Disruption）：最大可容忍中斷期

**備份策略**：
- 3-2-1 原則：3份備份、2種媒體、1份異地
- 定期驗證備份可還原性
- 異地備援距離 ≥ 30公里

**參考標準**：
- ISO 22301（營運持續管理系統）
- ISO/IEC 24762（ICT 災難復原服務指南）
- NIST SP 800-34（IT 系統緊急計畫指南）
- TIA-942（資料中心標準）

---

## 七、基礎文件

### 7.1 資通系統防護基準驗證實務 v1.1

**防護基準構面**：
1. 存取控制（Access Control）
2. 識別與鑑別（Identification & Authentication）
3. 稽核與可歸責性（Audit & Accountability）
4. 組態管理（Configuration Management）
5. 事故回應（Incident Response）
6. 維護（Maintenance）
7. 媒體保護（Media Protection）
8. 實體與環境保護（Physical & Environmental Protection）
9. 人員安全（Personnel Security）
10. 風險評鑑（Risk Assessment）
11. 系統與服務獲得（System & Services Acquisition）
12. 系統與通訊保護（System & Communications Protection）
13. 系統與資訊完整性（System & Information Integrity）

---

### 7.2 安全控制措施參考指引 v4.1

**控制措施對照**：
- 對照 CNS 27001/ISO 27001
- 對照 NIST SP 800-53
- 對照 CIS Controls

---

## 資源連結

- [NICS 共通規範專區](https://www.nics.nat.gov.tw/cybersecurity_resources/reference_guide/Common_Standards/)
- [NICS 技術報告](https://www.nics.nat.gov.tw/cybersecurity_resources/publications/)
- [GCB 專區](https://www.nics.nat.gov.tw/core_business/protection/)

---

**更新日期**：2026-01-13
**資料來源**：國家資通安全研究院
