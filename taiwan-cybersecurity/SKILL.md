---
name: taiwan-cybersecurity
description: |
  Expert knowledge about Taiwan's cybersecurity landscape including laws, regulations,
  organizations, threat intelligence, industry practices, and defensive strategies.
  Use this skill when discussing Taiwan's information security policies, compliance
  requirements, critical infrastructure protection, APT threats, or local cybersecurity resources.
triggers:
  - "taiwan cybersecurity"
  - "台灣資安"
  - "台灣資通安全"
  - "資通安全管理法"
  - "數位發展部"
  - "NICS"
  - "TWCERT"
  - "taiwan information security"
  - "taiwan cyber threat"
  - "台灣資安法規"
  - "關鍵基礎設施"
  - "critical infrastructure taiwan"
---

# 台灣資通訊安全專家技能

## Overview

本技能涵蓋台灣資通訊安全的全方位知識體系，包括法規框架、政府機構、威脅情報、產業生態、技術標準、訓練資源等面向，適用於需要了解或遵循台灣資安規範的專業人士。

**最後更新**: 2026-01 (資通安全管理法 2025年9月修正版)

## 核心知識領域

### 1. 法規與合規 (Laws & Compliance)

#### 資通安全管理法 (Cyber Security Management Act)
- **制定**: 2018年6月6日公布，2019年1月1日施行
- **最新修正**: 2025年9月24日總統公布修正案，同年12月1日實施
- **主管機關**: 數位發展部（原行政院國家資通安全會報）
- **適用對象**:
  - 公務機關
  - 特定非公務機關（關鍵基礎設施提供者）
  - 政府捐助之財團法人

**2025年修正重點**:
1. **主管機關變更**: 從行政院改為數位發展部
2. **禁用危害產品**: 公務機關禁止使用危害國家資安的產品（提升至法律位階）
3. **稽核權擴大**: 數位發展部可稽核公務機關資安維護計畫實施情形
4. **專責人力**: 特定非公務機關須設置專責資安人員及資安長
5. **罰則加重**:
   - 未通報資安事件最高罰款 NT$1,000萬
   - 違反維護計畫要求最高罰款 NT$500萬
6. **調查權**: 中央目的事業主管機關可調查特定非公務機關重大資安事件

**官方來源**:
- 全國法規資料庫: https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0030297
- 數位發展部法規系統: https://law.moda.gov.tw/LawContent.aspx?id=FL088622

#### 資通安全責任等級分級辦法
- **依據**: 資通安全管理法第7條第1項
- **分級制度**: A、B、C、D、E 五級（A級最高）
- **分級考量**: 機關性質、業務特性、資通系統重要性
- **官方來源**: https://law.moda.gov.tw/LawContent.aspx?id=FL089966

#### 資通安全事件通報及應變辦法
- **目的**: 規範資安事件通報流程與應變機制
- **通報機制**: 24/7 全天候通報服務
- **應變協調**: 由資通安全署及國家資通安全研究院協調處理
- **官方來源**: https://law.moda.gov.tw/LawContent.aspx?id=FL089967

#### 關鍵基礎設施防護
**涵蓋領域**:
- 能源（電力、油氣）
- 水資源
- 通訊傳播
- 交通運輸
- 金融
- 緊急救援與醫院
- 政府機關
- 科學園區與工業區
- 食品

**防護框架**:
- 《關鍵資訊基礎設施安全防護指引》
- 各部會訂定行業別資安防護標準
- 中央目的事業主管機關稽核制度

### 2. 政府組織與機構

#### 數位發展部資通安全署 (ACS, MODA)
- **成立**: 2022年8月（數位發展部成立時）
- **前身**: 行政院國家資通安全會報技術服務中心
- **職責**:
  - 國家資安政策規劃與執行
  - 資安事件通報應變協調
  - 關鍵基礎設施防護
  - 資安法規制定與推動
  - 資安產業發展
- **官網**: https://moda.gov.tw/ACS/
- **聯絡**: 02-2380-8500

#### 國家資通安全研究院 (NICS)
- **成立**: 2023年1月（行政法人）
- **監督機關**: 數位發展部
- **核心任務**:
  - 資安技術研發
  - 技術應用與移轉
  - 協助政府機關與關鍵基礎設施資安防護
  - 人才培育
  - 營運 TWCERT/CC
  - 國際資安合作
- **官網**: https://www.nics.nat.gov.tw/
- **聯絡**: 02-6631-1881

**主要計畫**:
1. **NICS 台灣資安計畫**: 協助中小企業與非營利組織強化資安
2. **政府組態基準 (GCB)**: 提供系統安全設定標準
3. **共通規範**: 資安防護控制措施參考
4. **數位韌性教材**: 高可用性、可維護性、易用性指引
5. **資通安全弱點通報系統 (VANS)**: https://vans.nat.gov.tw/

**GitHub 開源專案**: https://github.com/nics-tw
- resilience-material: 數位韌性共通建議教材
- 其他技術研究專案

#### TWCERT/CC (台灣電腦網路危機處理暨協調中心)
- **營運**: 國家資通安全研究院（2024年1月起）
- **服務對象**: 民間企業
- **核心功能**:
  - 24/7 資安事件通報與應變
  - 威脅情資分享
  - 產品安全漏洞通報
  - 惡意檔案檢測服務
  - 國際 CERT 合作
  - 資安意識推廣
- **官網**: https://www.twcert.org.tw/
- **年度活動**: 台灣資安通報應變年會

**弱點通報平台**:
- HITCON ZeroDay: https://zeroday.hitcon.org/
- Taiwan Vulnerability Disclosure Network (TVN)

### 3. 威脅情報與現況

#### 台灣面臨的主要威脅 (2025-2026)

**攻擊強度**:
- 2025年Q2平均每週 **4,055次攻擊**，居亞太地區之首
- 攻擊頻率為全球平均的兩倍以上

**主要攻擊目標產業**:
1. 硬體供應商
2. 政府與軍事機構
3. 製造業
4. 電信業
5. 醫療產業
6. 科技業

#### APT 組織

**UAT-5918**:
- 目標: 台灣關鍵基礎設施（2023年起）
- 攻擊領域: 電信、醫療、資訊技術
- 手法: 利用 N-day 漏洞進行初始入侵

**其他活躍組織**:
- Crazyhunter
- Nightspire
- 中國背景 APT 組織持續針對台灣

#### 勒索軟體威脅

**Crazyhunter**:
- 目標: 台灣醫療機構（2025年3月起多次攻擊）
- 特點: 結合「超快速攻擊手法」與 AI 增強攻擊
- 影響: 系統癱瘓、資料外洩

**Nightspire**:
- 手法: 暗網入口 + 心理恐嚇雙重勒索
- 受害: 台灣與香港企業

**Clop**:
- 2025年2月全球最活躍勒索軟體組織（332個受害案例）
- 台灣亦有受害組織

**全球趨勢** (2025年2月):
- 全球勒索攻擊達 **956件**，較1月增加 87%

#### 2026年資安趨勢預測
1. **量子運算威脅**: 現有加密演算法面臨挑戰
2. **AI代理攻擊**: AI工業化帶動自動化攻擊
3. **多雲與供應鏈**: 持續成為駭客主戰場
4. **容器安全**: 容器化環境成為新攻擊向量
5. **AI防禦**: 以行為分析與機器學習主動偵測異常

### 4. 技術標準與指引

#### 政府組態基準 (Government Configuration Baseline, GCB)
- **發布機關**: 國家資通安全研究院
- **目的**: 提供系統安全設定標準
- **涵蓋系統**: Windows Server、Linux、Exchange Server、資料庫等
- **範例**: TWGCB-04-001 Exchange Server 2013 政府組態基準 v1.3
- **官網**: https://www.nics.nat.gov.tw/core_business/cybersecurity_defense/GCB/

#### 共通規範 (Common Standards)
- **來源**: 依據《政府資安標準整體發展藍圖》
- **內容**: 從政策、管理、技術面說明相關控制措施
- **用途**: 供政府機關參考，強化資安防護，確保機密性、完整性、可用性
- **官網**: https://www.nics.nat.gov.tw/cybersecurity_resources/reference_guide/Common_Standards/

#### 數位韌性 (Digital Resilience)
**三大核心面向**:

1. **高可用性 (High Availability)**
   - 機房管理: 電力、冷卻系統
   - 網路設定: DNS、CDN、本地網路韌性

2. **可維護性 (Maintainability)**
   - 廠商原始碼獨立驗證
   - 漏洞通報自動驗證
   - 自動化部署工作流程
   - 元件掃描工具

3. **易用性 (Usability)**
   - 隱私權政策一致性
   - 無廣告嵌入式服務

**教材**: https://github.com/nics-tw/resilience-material

#### 資通系統防護基準驗證
- 標準文件: 《資通系統防護基準驗證實務》v1.5 (2022年9月)
- 執行單位: 行政院國家資通安全會報技術服務中心（現為資通安全署）

### 5. 產業生態

#### 台灣資安公司

**TeamT5 (杜浦數位安全)**
- **成立**: 2017年（由5位資安專家創立）
- **專長**: 網路間諜威脅情資研究、專業威脅鑑識
- **經驗**: 20年以上惡意程式與 APT 研究經驗
- **投資**: 2022年獲日本 JAFCO、伊藤忠商事、MACNICA 投資
- **國際**: 設立日本子公司
- **官網**: https://teamt5.org/

**CyCraft (奧義智慧科技)**
- **技術**: 機器學習、深度學習、對抗網路
- **產品**: AI 驅動資安平台
- **特色**: 自動化威脅調查與事件應變
- **功能**: 自動蒐集數位證據、生成報告

**TXOne Networks (睿控網安)**
- **成立**: 2019年（由趨勢科技與四零四科技 MOXA 共同成立）
- **定位**: 工業控制系統 (ICS) / 營運技術 (OT) 資安
- **解決方案**: 端點防護、網路防禦、安全檢測
- **募資**: A輪 NT$3.6億（JAFCO Asia 領投），累積 NT$6.6億 / USD $144.9M
- **市場**: 工業資安藍海，預估2026年市場規模達 USD $120億（2019年為 USD $20億）

#### 其他重要廠商
- 趨勢科技 (Trend Micro)
- 安碁資訊 (Aeoris)
- 奧義智慧 (CyCraft)
- 中華資安國際 (CHTSEC)

#### 重點產業應用
- **半導體**: 資產生命週期防護
- **智慧製造**: OT 環境資安
- **金融**: 法遵與風險管理
- **醫療**: 病患資料保護與系統可用性
- **政府**: 機敏資料防護

### 6. 訓練與社群資源

#### 官方訓練計畫

**AIS3 (Advanced Information Security Summer School)**
- **AIS3 Junior**: 高中生資安研習營
- **AIS3 Main**: 進階資安人才培訓
- **主辦**: 教育部、數位發展部支持
- **內容**: Pwn、Web、Reverse、Crypto、Forensics等

**其他官方計畫**:
- 台灣好厲駭 (Taiwan Holy Hacker Program)
- SCIST (南臺灣學生資訊社群)
- SCAICT (中部高中電資聯合會議)
- TeamT5 Security Camp
- Global Cybersecurity Camp (GCC)
- TDOH 資安功德院
- 國家資通安全研究院課程

#### CTF 競賽與練習平台

**台灣本土競賽**:
- MyFirstCTF (初學者友善)
- AIS3 Pre-exam & EOF
- BambooFox CTF
- THJCC (台灣高中職資安技能金盾獎)
- TSC CTF (Taiwan Security Club)
- TSJ CTF
- Balsn CTF
- HITCON CTF (國際級)
- ACSC (Asian Cyber Security Challenge)
- CGGC 網路守護者挑戰賽
- Aegis Shield CTF
- GiCS 女性資安
- No Hack No CTF

**Wargame 練習平台**:
- pwnable.tw (國際知名)
- HackMe CTF
- NCKU CTF
- BambooFox Wargame
- SCIST CTF
- LoTuX CTF

**國際平台**:
- picoCTF
- CTFtime (競賽聚合器)
- The Flare-On Challenge (逆向工程)
- pwn.college
- HackTheBox
- TryHackMe
- PortSwigger Web Security Academy

#### 大學社團與組織

**頂尖 CTF 隊伍**:
- **Balsn** (國立台灣大學) - 世界級強隊
- **BambooFox** (國立陽明交通大學)
- **217** (台北科技大學 - is1lab)

**其他大學組織**:
- NTUST ISC (台灣科技大學)
- NCtfU (中央大學)
- HackerSir (逢甲大學)
- NSYSU ISC (中山大學)
- NISRA (輔仁大學)
- 東海 Hacker Club
- NCKUCTF (成功大學)
- NTTU Security Club (台東大學)
- NUTN ISC (台南大學)

#### 社群與組織

**區域社群**:
- SCIST (南臺灣學生資訊社群)
- SCAICT (中部高中電資聯合會議)
- CURA (Cybersecurity Unions Research Association)
- HITCON GIRLS (女性專屬)
- TDOH Hacker
- UCCU Hacker
- CHROOT
- B33F 50UP
- StarHack Academy
- Deep Hacking Study Group

**產業組織**:
- HITCON (Taiwan Hacker Conference)
- DEVCORE (戴夫寇爾)
- TeamT5
- TWCERT/CC

#### 活動與會議

- **HITCON** (台灣駭客年會) - 亞洲最大資安會議
- **CYBERSEC** (台灣資安大會)
- **DEVCORE CONFERENCE**
- **TeamT5 威脅分析師高峰會**
- **TWCERT/CC 資安通報應變年會**
- **CraftCon**
- **/dev/meet** (DEVCORE 資安交流會)

#### 獎學金與補助

- **DEVCORE 資安獎學金**
- **HITCON 社群補助計畫**
- **DEVCORE 教育活動贊助**
- **安碁智慧科技贊助**

#### 實習計畫

- DEVCORE 實習
- 安碁智慧研究實習
- TeamT5 職缺
- TrapaSecurity 實習
- TXOne Networks 實習
- 趨勢科技實習
- 教育部 ISIP 就業平台

#### 專家資源

**知名研究者** (筆名/別名):
- **Orange**: 資安研究員，簡報與部落格
- **Angel Boy**: Exploit 技術，簡報集
- **Hao's Arsenal**: 技術 writeup
- **Yu-Awn**: Linux kernel exploitation, fuzzing
- **Huli**: Web 安全，前端安全
- **Maple**: 密碼學部落格
- **Kazma**: 逆向工程課程
- **u1f383**: Fuzzing, 軟體安全教材
- **NiNi**: 資安職涯指引，技術演講
- **Zeze**: Windows 安全

#### 學習資源整理

**GitHub 資源庫**:
- **TW-Security-and-CTF-Resource**: https://github.com/Ice1187/TW-Security-and-CTF-Resource
  完整的台灣資安與 CTF 學習資源整理（741 stars）
- **Awesome-Taiwan-Security-Course**: https://github.com/fei3363/Awesome-Taiwan-Security-Course
  台灣大學與社群資安課程列表
- **NICS Resilience Material**: https://github.com/nics-tw/resilience-material
  數位韌性共通建議教材

**其他平台**:
- **資安解壓縮**: 資安學習資源包
- **Got Your PW** (gotyour.pw): 台灣資安學習資源平台，含「抓周」功能幫助初學者找方向

### 7. 國際資安合作

#### 國際組織參與
- FIRST (Forum of Incident Response and Security Teams)
- APCERT (Asia Pacific Computer Emergency Response Team)
- 與美國、日本、歐洲 CERT 組織合作

#### 國際資安政策觀測
- NICS 發布《國際資安政策法制觀測週報》
- 追蹤各國資安政策與法規發展

### 8. 實務應用指引

#### 公務機關資安合規
1. **確認責任等級**: 依《資通安全責任等級分級辦法》確認等級
2. **制定維護計畫**: 依等級訂定資安維護計畫
3. **配置人力**: 設置資安長及專責人員
4. **定期稽核**: 接受主管機關稽核
5. **事件通報**: 依《資安事件通報及應變辦法》通報
6. **採購管理**: 避免使用危害國家資安之產品

#### 關鍵基礎設施業者
1. **行業標準**: 遵循目的事業主管機關訂定之行業別標準
2. **防護指引**: 參考《關鍵資訊基礎設施安全防護指引》
3. **專責組織**: 建立資安專責單位
4. **演練計畫**: 定期執行資安演練
5. **情資分享**: 參與 TWCERT/CC 或 N-ISAC 情資分享

#### 中小企業資安強化
1. **NICS 台灣資安計畫**: 申請免費資安諮詢與輔導
2. **基礎防護**:
   - 防火牆與入侵防禦系統
   - 端點防護 (EDR/XDR)
   - 定期備份與災難復原演練
   - 員工資安意識訓練
3. **弱點管理**: 定期進行弱點掃描與滲透測試
4. **供應鏈安全**: 評估供應商資安能力

#### 資安事件應變流程
1. **偵測與確認**: 發現可疑活動
2. **通報**: 依法向 NCERT (公務機關) 或 TWCERT/CC (企業) 通報
3. **隔離與控制**: 阻止攻擊擴散
4. **證據保全**: 保存數位證據
5. **根因分析**: 調查攻擊手法與影響範圍
6. **修復與復原**: 系統重建與服務恢復
7. **事後檢討**: 改善防護機制

## Quick Reference

### 緊急聯絡資訊

**國家資通安全通報應變網站 (NCERT)**:
- 網站: https://www.ncert.nat.gov.tw/
- 電話: 提供 24/7 通報服務

**TWCERT/CC**:
- 網站: https://www.twcert.org.tw/
- 服務: 24/7 企業資安事件通報

**數位發展部資通安全署**:
- 電話: 02-2380-8500
- 網站: https://moda.gov.tw/ACS/

**國家資通安全研究院**:
- 電話: 02-6631-1881
- 網站: https://www.nics.nat.gov.tw/

### 重要法規連結

| 法規名稱 | 連結 |
|---------|------|
| 資通安全管理法 | https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0030297 |
| 資通安全管理法施行細則 | https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0030303 |
| 資通安全責任等級分級辦法 | https://law.moda.gov.tw/LawContent.aspx?id=FL089966 |
| 資通安全事件通報及應變辦法 | https://law.moda.gov.tw/LawContent.aspx?id=FL089967 |
| NICS 法規彙編 (PDF) | download.nics.nat.gov.tw/UploadFile/attachfilelaw/資通安全法規彙編【112年編印】v1.0_1121201.pdf |

### 技術資源連結

| 資源 | 連結 |
|------|------|
| 政府組態基準 (GCB) | https://www.nics.nat.gov.tw/core_business/cybersecurity_defense/GCB/ |
| 共通規範 | https://www.nics.nat.gov.tw/cybersecurity_resources/reference_guide/Common_Standards/ |
| 弱點通報系統 (VANS) | https://vans.nat.gov.tw/ |
| HITCON ZeroDay | https://zeroday.hitcon.org/ |
| NICS GitHub | https://github.com/nics-tw |
| 數位韌性教材 | https://github.com/nics-tw/resilience-material |

### 學習資源連結

| 資源 | 連結 |
|------|------|
| TW Security & CTF Resource | https://github.com/Ice1187/TW-Security-and-CTF-Resource |
| Awesome Taiwan Security Course | https://github.com/fei3363/Awesome-Taiwan-Security-Course |
| pwnable.tw | https://pwnable.tw/ |
| CTFtime | https://ctftime.org/ |

## Best Practices

### 資安防護建議

1. **深度防禦**: 多層次防護機制
2. **最小權限**: 僅授予必要權限
3. **定期更新**: 系統與軟體保持最新
4. **備份策略**: 3-2-1 備份原則（3份備份、2種媒體、1份異地）
5. **零信任架構**: 持續驗證，永不信任
6. **情資整合**: 善用威脅情資提升防護
7. **人員訓練**: 定期資安意識教育
8. **演練驗證**: 定期進行紅藍隊演練

### 應對 APT 威脅

1. **威脅狩獵**: 主動搜尋潛在威脅
2. **行為分析**: 偵測異常行為模式
3. **EDR/XDR**: 部署端點偵測與回應方案
4. **網路分段**: 限制橫向移動
5. **日誌管理**: 集中式日誌收集與分析
6. **情資交換**: 參與 TWCERT/CC 情資分享

### 勒索軟體防護

1. **郵件過濾**: 阻擋釣魚郵件
2. **端點防護**: 部署 EDR 偵測勒索行為
3. **備份驗證**: 定期測試備份還原
4. **特權管理**: 限制管理權限使用
5. **離線備份**: 保留離線不可變備份
6. **網路隔離**: 重要系統網路隔離

## Compliance Checklist

### 資通安全管理法合規檢核表

- [ ] 確認適用對象身分（公務機關/特定非公務機關）
- [ ] 確認責任等級（A/B/C/D/E）
- [ ] 設置資安長及專責人員
- [ ] 制定資安維護計畫
- [ ] 建立資安事件通報機制
- [ ] 定期辦理資安教育訓練
- [ ] 執行資安稽核
- [ ] 建立應變機制與演練
- [ ] 採購符合資安要求之產品
- [ ] 依規定通報資安事件

## 資料來源與參考

### 官方文件來源
- 數位發展部資通安全署: https://moda.gov.tw/ACS/
- 國家資通安全研究院: https://www.nics.nat.gov.tw/
- TWCERT/CC: https://www.twcert.org.tw/
- 全國法規資料庫: https://law.moj.gov.tw/
- 數位發展部法規系統: https://law.moda.gov.tw/

### 威脅情報來源
- TWCERT/CC 威脅情資
- TeamT5 ThreatVision
- 資安人科技網: https://www.informationsecurity.com.tw/
- iThome CYBERSEC: https://cybersec.ithome.com.tw/

### 社群資源
- HITCON: https://hitcon.org/
- DEVCORE: https://devco.re/
- GitHub: Ice1187/TW-Security-and-CTF-Resource
- GitHub: nics-tw

### 法規參考
- 資通安全管理法 (2025年修正)
- 資通安全責任等級分級辦法
- 資通安全事件通報及應變辦法
- 關鍵資訊基礎設施安全防護指引

---

**Note**: 本技能內容基於 2026年1月的公開資訊整理，法規以官方公告為準。建議定期查閱官方網站以獲取最新資訊。

**Maintained by**: Skill Seekers
**Version**: 1.0.0
**Last Updated**: 2026-01-12
