---
name: executing-plans
description: |
  Execute implementation plans through batched tasks with review checkpoints. Follows plans step-by-step, runs verifications, and pauses for feedback.
  Use when: implementing features from a written plan, following a task list, executing step-by-step instructions, or when user mentions 執行計畫, execute plan, 按照計畫, follow plan, 一步步做.
  Triggers: "execute plan", "follow plan", "執行計畫", "按照計畫", "開始實作", "run the plan"
---

# Executing Implementation Plans

批次執行實作計畫，搭配審查檢查點。

## 核心原則

> **跟隨計畫，不要自作主張**
> 遇到問題就停下來，不要繼續硬推

## 執行流程

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Load Plan → Execute Batch → Report → Get Feedback    │
│       │            │            │           │          │
│       ▼            ▼            ▼           ▼          │
│   檢視計畫    執行 3 個任務   報告結果    等待回饋     │
│       │            │            │           │          │
│       └────────────┴────────────┴───────────┘          │
│                        │                                │
│                        ▼                                │
│                   Continue / Stop                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Step 1: 載入並審查計畫

```markdown
1. 讀取計畫檔案
2. 理解整體目標
3. 識別潛在問題
4. **開始前提出疑慮**

⚠️ 如果有疑問，在執行前就要問！
```

### 宣告使用此 Skill

```markdown
我將使用 **executing-plans** skill 執行此計畫。

**計畫概要**：
- 目標：[目標]
- 任務數：[N] 個
- 批次大小：3 個任務

**執行方式**：
- 每批次完成後暫停報告
- 等待您的回饋再繼續
- 遇到阻礙立即停止

準備好開始了嗎？
```

## Step 2: 批次執行

### 標準批次大小：3 個任務

```markdown
## 批次 1 執行中...

### Task 1: [任務名稱]
狀態：⏳ 進行中

步驟：
1. ✅ 寫測試
2. ✅ 驗證失敗
3. ✅ 實作
4. ✅ 驗證通過
5. ✅ Commit

狀態：✅ 完成

---

### Task 2: [任務名稱]
狀態：⏳ 進行中
...

### Task 3: [任務名稱]
狀態：⏳ 進行中
...
```

### 執行每個步驟

```markdown
對於每個任務：

1. **讀取任務說明**
2. **執行每個步驟**（不要跳過）
3. **執行驗證命令**
4. **確認預期結果**
5. **Commit**（使用指定訊息）
```

## Step 3: 報告結果

### 批次完成報告

```markdown
## 📊 批次 1 完成報告

### 執行結果
| Task | 狀態 | 備註 |
|------|------|------|
| Task 1: User Entity | ✅ | 2 tests passed |
| Task 2: Password Service | ✅ | 3 tests passed |
| Task 3: User Repository | ✅ | 4 tests passed |

### 測試結果
```bash
$ pytest tests/ -v
==================== 9 passed in 0.45s ====================
```

### 已完成的 Commits
- `feat(domain): add User entity and Email value object`
- `feat(application): add PasswordService for secure hashing`
- `feat(infrastructure): add SQLAlchemy User repository`

### 剩餘任務
- Task 4-6（共 3 個）

### 問題/疑慮
[無 / 列出發現的問題]

---

**請確認是否繼續下一批次？**
```

## Step 4: 處理阻礙

### 立即停止的情況

```markdown
🛑 **必須立即停止的情況：**

1. **缺少依賴**
   - 計畫中假設的套件不存在
   - 需要的 API 未實作

2. **測試失敗**
   - 前一個任務的測試失敗
   - 無法重現預期結果

3. **指示不清**
   - 不確定該怎麼做
   - 計畫有矛盾之處

4. **計畫有缺口**
   - 缺少必要步驟
   - 順序不正確
```

### 阻礙報告格式

```markdown
## 🛑 執行中斷報告

### 中斷位置
- 批次：1
- 任務：Task 2 - Password Service
- 步驟：2.3 實作

### 問題描述
```
ModuleNotFoundError: No module named 'bcrypt'
```

### 已嘗試
1. 確認 requirements.txt 中有 bcrypt
2. 檢查虛擬環境是否啟用

### 需要的協助
請確認：
1. 是否需要安裝 bcrypt？
2. 或是計畫應該使用其他套件？

### 狀態
⏸️ 等待指示
```

## Step 5: 完成計畫

### 全部完成報告

```markdown
## ✅ 計畫執行完成

### 總結
- **計畫**：使用者認證 API
- **總任務數**：6
- **完成任務**：6
- **總 Commits**：6

### 測試覆蓋
```bash
$ pytest tests/ -v --cov
==================== 15 passed ====================
Coverage: 87%
```

### Commits 歷史
1. `feat(domain): add User entity and Email value object`
2. `feat(application): add PasswordService for secure hashing`
3. `feat(infrastructure): add SQLAlchemy User repository`
4. `feat(application): add RegisterUser use case`
5. `feat(application): add LoginUser use case`
6. `feat(presentation): add auth API endpoints`

### 下一步建議
1. 執行整合測試
2. 更新 API 文件
3. 部署到測試環境

---

是否需要使用 `finishing-a-development-branch` skill 完成分支合併？
```

## 執行原則

### ✅ DO（應該做）

```markdown
- 完全按照計畫步驟執行
- 執行所有驗證命令
- 每個批次後暫停報告
- 遇到問題立即停止
- 使用指定的 commit 訊息
```

### ❌ DON'T（不應該做）

```markdown
- 不要跳過任何步驟
- 不要跳過驗證
- 不要自己假設解決方案
- 不要一次執行太多任務
- 不要修改計畫內容（除非被要求）
```

## 批次大小調整

| 情況 | 建議批次大小 |
|------|-------------|
| 標準任務 | 3 |
| 複雜任務 | 1-2 |
| 簡單任務 | 4-5 |
| 第一次執行 | 1（先確認流程） |

## Checklist

執行計畫時確認：

- [ ] 已宣告使用此 skill
- [ ] 已讀取並理解計畫
- [ ] 已提出任何疑慮
- [ ] 每個步驟都有執行
- [ ] 驗證命令都有執行
- [ ] 結果符合預期
- [ ] 每個批次後有報告
- [ ] 等待回饋後才繼續
- [ ] 遇到阻礙立即停止
