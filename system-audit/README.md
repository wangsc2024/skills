# System Audit Skill

通用系統審查評分工具 — 以 7 個維度（資訊安全、系統架構、系統品質、系統工作流、技術棧、系統文件、系統完成度）對目標系統進行全面評估，產出量化報告與改善建議。

## 快速開始

### 觸發方式
向 Agent 說出以下任一觸發語句：
- 「對這個專案做系統審查」
- 「幫我評估這個系統的品質」
- 「系統健檢」
- 「system audit」

### 評分範圍
- **加權總分**：0-100 分
- **7 個維度**：資訊安全（20%）、系統架構（18%）、系統品質（18%）、系統工作流（15%）、技術棧（10%）、系統文件（10%）、系統完成度（9%）
- **38 個子項**：每個維度包含 5-6 個子項，各別評分 0-100

### 等級定義
| 等級 | 分數範圍 | 說明 |
|------|---------|------|
| S 卓越 | 90-100 | 業界標竿，極少數系統能達到 |
| A 優秀 | 80-89 | 超越多數同類系統 |
| B 良好 | 70-79 | 具備專業水準 |
| C 及格 | 55-69 | 基本可用，有明顯改善空間 |
| D 待改善 | 40-54 | 多項缺陷需優先處理 |
| F 不及格 | 0-39 | 嚴重不足，需全面檢討 |

## 目錄結構

```
system-audit/
├── SKILL.md                    # 完整操作指引（556 行）
├── README.md                   # 本檔案
├── config/
│   └── audit-scoring.yaml      # 評分規則、權重模型、校準規則
└── templates/
    └── audit-report.md         # 報告模板
```

## 使用方式

### 方式 1：互動式手動審查
直接對 Agent 說「系統審查」，Agent 會：
1. 讀取 `config/audit-scoring.yaml` 取得評分規則
2. 對目標系統進行 7 維度評估
3. 產出報告到 `reports/audit-{system}-{date}.md`

### 方式 2：自動排程（需額外設定）
如果要設定自動排程審查（如每日 00:40），需要：
1. 建立 team mode prompts（4 個 Phase 1 + 1 個 Phase 2）
2. 建立 PowerShell 執行腳本（`run-system-audit-team.ps1`）
3. 設定 Windows Task Scheduler 排程

**參考實作**：[daily-digest-prompt](https://github.com/your-org/daily-digest-prompt) 專案的 `prompts/team/fetch-audit-*.md` 和 `run-system-audit-team.ps1`

## 配置文件

### audit-scoring.yaml
包含：
- **權重模型**：`balanced`（預設）、`security_first`（金融/醫療）、`startup`（新創/MVP）
- **校準規則**：防止虛假高分（如：無測試→品質上限 50）
- **等級定義**：S/A/B/C/D/F 分數門檻

### audit-report.md
結構化報告模板，包含：
- 基本資訊（系統名稱、路徑、審查日期）
- 總覽表（7 個維度分數與等級）
- 維度詳情（含證據與建議）
- TOP 5 改善建議

## 核心特性

✅ **證據導向**：每個子項必須有具體證據（檔案路徑、Grep 結果、指令輸出）
✅ **防虛高分**：校準規則自動套用硬性上限
✅ **TOP 5 建議**：自動篩選最弱 5 項，含改善建議和難度評估
✅ **多權重模型**：可根據系統類型選擇不同評分重點
✅ **通用性**：可審查任何系統（不限特定語言或框架）

## 適用情境

- 新專案啟動前的基準評估
- 定期品質檢查（建議每月一次）
- 重大變更後的完整性驗證
- 交付前的品質閘門
- 技術債務盤點

## 版本歷史

- **v1.0.0** (2026-02-16)：首次發布
  - 7 個維度、38 個子項完整評估
  - 3 種權重模型（balanced / security_first / startup）
  - 校準規則防止虛假高分
  - 結構化報告模板

## 授權

MIT License

## 相關資源

- [完整操作指引](./SKILL.md)（556 行，含所有子項的檢查步驟和給分標準）
- [評分規則配置](./config/audit-scoring.yaml)
- [報告模板](./templates/audit-report.md)
- 參考實作：[daily-digest-prompt 專案的自動排程審查](https://github.com/your-org/daily-digest-prompt)
