---
name: git-worktree
version: "1.0.0"
description: |
  Git Worktree 並行工作區管理。一個 repo、多個工作區，零 stash、零上下文切換成本。
  分析倉庫結構、設計工作樹拆分策略、建立 spec-driven 特性分支、管理工作樹生命週期。
  Use when: 同時開發多個功能, 需要不 stash 就切換上下文, 平行 code review,
  hotfix 同時進行 feature 開發, CI/CD 平行測試, bare repo 工作流, Agent Team 並行開發.
  Triggers: "git worktree", "worktree", "工作樹", "平行開發", "並行開發",
  "isolated workspace", "隔離工作區", "同時多分支", "EnterWorktree", "parallel agents",
  "context switch without stash", "bare repo worktree", "多工作區"
allowed-tools: Read, Bash, Write, Glob, Grep
cache-ttl: N/A
triggers:
  - "git worktree"
  - "worktree"
  - "工作樹"
  - "平行開發"
  - "並行開發"
  - "隔離工作區"
  - "同時多分支"
  - "EnterWorktree"
  - "parallel agents"
  - "context switch without stash"
  - "bare repo worktree"
  - "isolated workspace"
  - "多工作區"
depends-on:
  - "git-workflow"
---

# Git Worktree — 並行開發與代理協作

Git 的「多桌面」——一個 repo，多個工作區，零 stash，零上下文切換成本。

> **與 git-workflow 的分工**：git-workflow 管「單一工作區內的操作」（分支、commit、PR、衝突）；
> 本 skill 管「多個工作區的並行編排」（建立、拆分、協調、生命週期）。

---

## Decision Framework：何時用 Worktree

在選擇工具前，先判斷情境：

```
需要在另一個分支工作？
│
├─ 改動 < 5 分鐘？ ──────────────→ git stash（見 git-workflow）
│
├─ 需要同時看兩份程式碼？ ────────→ git worktree add
│
├─ PR review 同時繼續開發？ ──────→ git worktree add
│
├─ 緊急 hotfix + feature 未完成？ → git worktree add
│
├─ 多個 Agent 平行開發？ ─────────→ git worktree（每 Agent 一個）
│
└─ 完全獨立倉庫（不同 remote）？ → git clone
```

### 比較表

| 方式 | 磁碟用量 | 建立速度 | 共享歷史 | 獨立 Index | 適用場景 |
|------|---------|---------|---------|-----------|---------|
| `git stash` | 0 | 即時 | 是 | 否 | 快速暫存（< 5 分鐘）|
| `git checkout` | 0 | 即時 | 是 | 否 | 線性單分支工作 |
| `git worktree` | ~150 MB | ~20 秒 | 是 | 是 | 平行多分支（推薦）|
| `git clone` | ~1 GB | ~90 秒 | 否 | 是 | 完全獨立環境 |

> Worktree 比 clone 節省 ~85% 磁碟空間，切換速度快 4-5 倍。

---

## 核心概念：共享 vs 獨立

```
.git/                              ← 共享：objects, refs, hooks, stash, config
  │
  ├── main worktree                (project/)
  │     ├── .git (file → ../.git)
  │     ├── working files          ← 獨立
  │     └── index (staging)        ← 獨立
  │
  ├── linked worktree              (project-feature-auth/)
  │     ├── .git (file → .git/worktrees/feature-auth/)
  │     ├── working files          ← 獨立
  │     └── index (staging)        ← 獨立
  │
  └── linked worktree              (project-hotfix-login/)
        ├── .git (file → .git/worktrees/hotfix-login/)
        ├── working files          ← 獨立
        └── index (staging)        ← 獨立
```

| 共享（單一份）| 獨立（每 worktree）|
|---|---|
| Object database (.git/objects) | Working directory |
| References (.git/refs) | Index (staging area) |
| Hooks (.git/hooks) | HEAD |
| Stash list | Untracked files |
| Git config | Build artifacts (node_modules 等) |

---

## 指令速查表

| 指令 | 語法 | 範例 | 注意 |
|------|------|------|------|
| **add** | `git worktree add <path> [<branch>]` | `git worktree add ../proj-auth feature/auth` | 用 `-b` 建新分支 |
| **add -b** | `git worktree add -b <new> <path> [<start>]` | `git worktree add -b feature/pay ../proj-pay main` | 從 main 建新分支 |
| **list** | `git worktree list [--porcelain]` | `git worktree list` | `--porcelain` 供腳本解析 |
| **remove** | `git worktree remove <worktree>` | `git worktree remove ../proj-auth` | **永遠用這個，別 rm -rf** |
| **prune** | `git worktree prune` | `git worktree prune --dry-run` | 清理已刪目錄的殘留參照 |
| **move** | `git worktree move <wt> <new-path>` | `git worktree move ../proj-auth ../proj-auth-v2` | 搬遷工作樹 |
| **lock** | `git worktree lock <wt> [--reason]` | `git worktree lock ../proj-usb --reason "外接硬碟"` | 防止 prune 誤刪 |
| **unlock** | `git worktree unlock <wt>` | `git worktree unlock ../proj-usb` | 解除鎖定 |
| **repair** | `git worktree repair [<path>...]` | `git worktree repair` | 修復斷裂的路徑參照 |

---

## Worktree 生命週期

```
[1. 分析倉庫] → [2. 設計拆分] → [3. 建立 Worktree]
                                        │
                                 [4. Setup 環境]
                                        │
                                 [5. 撰寫 Feature Spec]
                                        │
                                 [6. 開發 & 測試]
                                        │
                                 [7. PR & Merge]
                                        │
                                 [8. 清理 Worktree]
```

### Phase 1-3: 建立

```bash
# 確認當前狀態
git worktree list
git status                    # 確保工作區乾淨

# 從 main 建立 feature worktree
git worktree add -b feature/auth ../myapp-auth main

# 從 main 建立 hotfix worktree
git worktree add -b hotfix/fix-login ../myapp-hotfix main
```

### Phase 4: Setup 環境

```bash
cd ../myapp-auth

# Node.js 專案
npm ci                        # 或 pnpm install

# Python 專案（Linux / macOS）
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# Python 專案（Windows Git Bash）
python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt

# 複製環境變數
cp ../myapp/.env.example .env
```

### Phase 5: 撰寫 Feature Spec（見下一節）

### Phase 6-7: 開發 & PR

在 worktree 內的開發流程與單一工作區完全相同——遵循 **git-workflow** skill 的規範：
Conventional Commits 格式、PR 建立流程、code review 標準。

### Phase 8: 清理

```bash
# 合併後清理
git worktree remove ../myapp-auth
git branch -d feature/auth        # 刪除本地分支
git worktree prune                # 清理殘留參照
git worktree list                 # 確認乾淨
```

---

## Spec-Driven 工作樹設計

### 拆分四原則

| 原則 | 說明 | 範例 |
|------|------|------|
| **功能獨立性** | 每個 worktree 處理一個內聚功能 | Auth worktree vs Payment worktree |
| **最小依賴** | 最小化跨 worktree 檔案重疊 | 共享型別在 main，功能在各 worktree |
| **合理粒度** | 一個 user story ≈ 一個 worktree | 太粗 → 衝突多；太細 → 管理成本高 |
| **語義命名** | 目錄名反映用途 | `proj-auth` 而非 `proj-wt1` |

### 命名規範

```
目錄：../<project>-<feature-short-name>
分支：feature/<descriptive-name>

範例：
  ../myapp-auth        → feature/user-authentication
  ../myapp-hotfix-123  → hotfix/fix-login-crash
  ../myapp-review-pr45 → （checkout 現有分支）
```

### Feature Spec 模板

在每個 worktree 根目錄建立 `git-worktree-spec.md`：

```markdown
# Feature Spec: [Feature Name]

## Goals
- [ ] Primary objective
- [ ] Secondary objective

## Implementation Scope
- **修改檔案:** src/auth/, src/middleware/
- **新增檔案:** src/auth/oauth.ts, tests/auth/
- **不可觸碰:** src/core/, src/database/（其他 worktree 負責）

## Acceptance Criteria
- [ ] AC-1: 使用者可透過 OAuth 登入
- [ ] AC-2: Token 過期自動刷新
- [ ] AC-3: 所有測試通過

## Technical Constraints
- 依賴：shared-types（main 分支維護）
- 破壞性變更：無

## Cross-Branch Notes
- 依賴：`../proj-shared` 的型別定義需先合併
- 被阻塞：無
```

---

## Worktrees for Claude Code Agent Teams

Git worktree 天然對應 Claude Code 的 Agent Team 模式——每個子 Agent 獲得隔離工作區，避免檔案衝突。

### Claude Code Worktree 隔離

Claude Code Agent SDK 的 `Task` 工具支援 `isolation: "worktree"` 參數，
可要求框架自動建立 git worktree 給子 Agent 使用，實現檔案系統隔離：

```
Task(subagent_type="general-purpose", isolation="worktree", prompt="...")
```

若無法使用此參數，可手動建立 worktree 後在 prompt 中指定路徑。
建議在 prompt 中明確要求 Agent 回報已修改的檔案清單。

### 模式 A：一個 Worktree 一個 Sub-Agent

```
Main Agent（調度，在 main）
  │
  ├── [唯讀] Explore Agent → 直接讀 main（不需 worktree）
  │
  ├── [worktree] Task Agent #1 → ../proj-auth    feature/auth
  ├── [worktree] Task Agent #2 → ../proj-payment  feature/payment
  └── [worktree] Task Agent #3 → ../proj-ui       feature/ui
```

適用：3+ 個獨立模組需同時實作。

### 模式 B：Contract-First 平行開發

```
Step 1: Main Agent 在 main 定義共享介面
        └── src/types/auth.d.ts, src/types/payment.d.ts

Step 2: Fan out — 每個 Agent 在各自 worktree 實作介面
        ├── Agent #1: ../proj-auth → implements AuthService
        └── Agent #2: ../proj-pay  → implements PaymentService

Step 3: Merge
  - 先合 types → 若有介面衝突需人工審查
  - 再合各實作分支（有衝突則解決後 commit）
  - 若任一 Agent 擴展了 types，先更新 main 再讓其他 worktree rebase
```

適用：功能間有共享依賴，需先定義契約再平行實作。

### 模式 C：Review + Develop 同步

```
├── code-reviewer Agent → worktree: ../proj-review-pr42（審查 PR #42）
│
└── 開發 Agent → 在 main 或另一個 worktree 繼續開發
```

適用：不想讓 code review 阻塞開發進度。

### 模式 D：Hotfix + Feature 同時進行

```
├── [worktree] Hotfix Agent → ../proj-hotfix-123（基於 main，修完立即合併）
│
└── [worktree] Feature Agent → ../proj-auth（基於 main，hotfix 合併後 rebase）
```

適用：最常見場景——緊急修復不中斷功能開發。

### 模式 E：TDD 平行（測試 + 實作分離）

```
Step 1: Main Agent 在 main 撰寫 Feature Spec + 定義 AC
Step 2: Fan out
        ├── Agent #1 → ../proj-tests   撰寫測試（紅燈）
        └── Agent #2 → ../proj-impl    撰寫實作
Step 3: Merge tests 先進 main → 再合 impl → 確認測試全綠
```

適用：搭配 TDD 流程，測試與實作由不同 Agent 平行撰寫。

### Task vs TeamCreate 選擇

| 情境 | 工具 | worktree |
|------|------|----------|
| 獨立任務，一次性，無需跨 Agent 溝通 | Task | 必要（防衝突）|
| 多 Agent 持續協作，有中間產物傳遞 | TeamCreate | 建議（避免衝突）|
| 純唯讀探索 | Task(Explore) | 不需要 |

### 合併協調

```bash
# 1. 確認所有 worktree 都已 commit 且測試通過
for wt in ../proj-auth ../proj-pay ../proj-ui; do
  echo "=== $wt ==="
  if [ -d "$wt" ]; then
    (cd "$wt" && git status)   # subshell：不影響當前目錄
  else
    echo "WARNING: $wt 不存在，跳過"
  fi
done

# 2. 逐一推送（建議用 PR 合併）
(cd ../proj-auth && git push -u origin feature/auth)
(cd ../proj-pay && git push -u origin feature/payment)

# 3. 合併後清理所有 worktree
git worktree remove ../proj-auth
git worktree remove ../proj-pay
git worktree remove ../proj-ui
git worktree prune
```

---

## 進階模式

### Bare Repo + Worktrees

大規模多分支管理時，用 bare repo 作為中央：

```bash
# 初始化
git clone --bare git@github.com:user/project.git project.git
cd project.git
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git fetch origin

# 建立工作樹
git worktree add ../project-main main
git worktree add -b feature/auth ../project-auth main
git worktree add -b feature/pay ../project-pay main
```

```
parent/
  project.git/       ← bare repo（無工作檔案）
  project-main/      ← main 分支
  project-auth/      ← feature 分支
  project-pay/       ← feature 分支
```

### Lock / Unlock（外接裝置）

```bash
git worktree lock ../proj-on-usb --reason "外接硬碟，可能斷開"
git worktree list                 # 顯示 [locked]
git worktree unlock ../proj-on-usb
```

### CI/CD 平行測試

```bash
# 在 CI pipeline 中平行測試多分支
git worktree add /tmp/test-feat-a feature-a
git worktree add /tmp/test-feat-b feature-b

npm test --prefix /tmp/test-feat-a &
npm test --prefix /tmp/test-feat-b &
wait

git worktree remove /tmp/test-feat-a
git worktree remove /tmp/test-feat-b
```

---

## 故障排除

### 路徑斷裂（手動移動目錄後）

當 worktree 目錄被手動移動（非 `git worktree move`）時，Git 會失去參照：

```bash
git worktree list                # 顯示 "prunable" 或過期路徑
git worktree repair              # 自動修復所有斷裂的 worktree
git worktree repair ../new-path  # 修復特定 worktree
```

### Windows 特定問題

```bash
# 長路徑問題（路徑 > 260 字元時 git worktree add 失敗）
git config --global core.longpaths true

# OneDrive / SharePoint 同步資料夾下建 worktree 會導致 index 損壞
# 建議：worktree 路徑選在非同步目錄（如 C:\Dev\ 而非 OneDrive 下）
```

### 常見錯誤訊息

| 錯誤 | 原因 | 解法 |
|------|------|------|
| `fatal: '<branch>' is already checked out` | 同一分支已有 worktree | `git worktree list` 找到佔用者 |
| `fatal: '<path>' already exists` | 目錄已存在 | 先刪除目錄或用其他路徑 |
| Stale worktree in `list` | 目錄被 `rm -rf` 刪除 | `git worktree prune` |

---

## Anti-Patterns

| 反模式 | 問題 | 正確做法 |
|--------|------|---------|
| `rm -rf` 刪除 worktree | 殘留 .git 參照，`list` 顯示過期條目 | `git worktree remove <path>` |
| 同一分支 checkout 到多個 worktree | Git 禁止（保護資料一致性）| 每個 worktree 用不同分支 |
| 在 main repo 內部建 worktree | 污染主目錄、gitignore 複雜化 | 用 `../proj-feat` 建在上層目錄 |
| 多 worktree 編輯相同檔案 | 合併時必定衝突 | 拆分時確認零檔案重疊 |
| 忘記安裝依賴 | 新 worktree 無 node_modules 等 | 建立後立即 `npm ci` / `pip install` |
| Agent 未定義介面就平行開發 | 合併時介面不相容 | Contract-First：先定義型別再 fan out |
| Submodule + Worktree | Git 官方標記為實驗性功能 | 避免使用，或充分測試後再用 |

---

## Checklists

### 建立 Worktree 前

- [ ] 工作區乾淨（已 commit 或 stash）
- [ ] 目標分支無其他 worktree 佔用（`git worktree list`）
- [ ] 目錄命名遵循 `../<project>-<feature>` 規範
- [ ] 若平行開發：已確認各 worktree 間零檔案重疊

### 建立 Worktree 後

- [ ] 依賴已安裝（npm ci / pip install / etc.）
- [ ] 環境變數已配置（.env）
- [ ] 基線測試通過
- [ ] Feature Spec（git-worktree-spec.md）已撰寫

### 合併 / 清理前

- [ ] 所有測試通過
- [ ] 變更已 commit（`git status` 乾淨）
- [ ] PR 已建立或分支已 push
- [ ] 使用 `git worktree remove`（非 rm -rf）
- [ ] 本地分支已刪除（`git branch -d`）
- [ ] 執行 `git worktree prune`
- [ ] `git worktree list` 確認無殘留
