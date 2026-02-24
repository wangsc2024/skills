# 帳號管理系統 v1.3.0

完整的使用者帳號管理系統，提供安全的身份驗證與授權機制，整合 Firebase Auth 進行 Email 驗證和密碼重設。

## 🌟 功能特色
###  每分鐘只能發出一封驗證信
### 核心功能
- ✅ 使用者註冊與 Email 驗證
- ✅ 安全登入/登出（JWT Token）
- ✅ 密碼管理（變更、重設）
- ✅ 個人資料維護
- ✅ 角色權限控制（Admin/User）
- ✅ 登入次數限制（防暴力破解）
- ✅ Token 黑名單機制

### 認證功能
- 🔐 Firebase Auth 整合
- 🔐 Email 驗證（Firebase）
- 🔐 密碼重設（Firebase）
- 🔐 傳統 JWT 認證（備用）
- 📧 外部 SMTP 郵件服務

## 🚀 技術棧

- **後端框架**: FastAPI 0.104+
- **資料庫**: PostgreSQL 15+ (開發環境支援 SQLite)
- **認證**: Firebase Auth + JWT (PyJWT)
- **密碼雜湊**: bcrypt (cost factor=12)
- **ORM**: SQLAlchemy 2.0 (非同步)
- **SMTP**: aiosmtplib (外部 SMTP)
- **Firebase**: firebase-admin SDK
- **測試**: pytest + pytest-asyncio + mutmut

## 📦 快速開始

### 1. 環境需求
```bash
Python 3.11+
PostgreSQL 15+ (或 SQLite for 開發)
```

### 2. 安裝依賴
```bash
uv pip install -r requirements.txt
```

### 3. 設定環境變數
```bash
cp .env.example .env
# 編輯 .env 填入正確的設定
```

#### Firebase 設定
1. 前往 [Firebase Console](https://console.firebase.google.com/)
2. 創建新專案或選擇現有專案
3. 啟用 Authentication 服務
4. 在專案設定中生成服務帳號金鑰
5. 將金鑰資訊填入 `.env` 檔案中的 Firebase 相關設定

#### 必要環境變數

> **WARNING**: 以下為範例佔位符，切勿將真實金鑰提交至版本控制。

```bash
# Firebase 設定
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_API_KEY=your-api-key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PRIVATE_KEY="<your-firebase-private-key-content>"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com
```

### 4. 資料庫遷移
```bash
alembic upgrade head
```

### 5. 啟動服務
```bash
# 開發模式
uv run python -m uvicorn src.account_management.main:app --reload
uv run python -m uvicorn src.account_management.main:app --host 0.0.0.0 --port 8000 --reload

# 生產模式
uvicorn src.account_management.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. 查看 API 文件
開啟瀏覽器訪問：

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 📖 使用指南

### API 端點

#### 認證相關
| 方法 | 端點 | 說明 | 權限 |
|------|------|------|------|
| POST | `/api/v1/auth/register` | 使用者註冊 | 公開 |
| POST | `/api/v1/auth/login` | 使用者登入 | 公開 |
| POST | `/api/v1/auth/logout` | 使用者登出 | 已登入 |
| POST | `/api/v1/auth/verify-email` | 驗證 Email | 公開 |
| POST | `/api/v1/auth/password/reset-request` | 請求重設密碼 | 公開 |
| POST | `/api/v1/auth/password/reset-confirm` | 確認重設密碼 | 公開 |

#### 使用者管理
| 方法 | 端點 | 說明 | 權限 |
|------|------|------|------|
| GET | `/api/v1/users/me` | 查詢個人資料 | 已登入 |
| PATCH | `/api/v1/users/me` | 更新個人資料 | 已登入 |
| POST | `/api/v1/users/me/password` | 變更密碼 | 已登入 |

#### 管理員功能
| 方法 | 端點 | 說明 | 權限 |
|------|------|------|------|
| GET | `/api/v1/admin/users` | 列出所有使用者 | Admin |
| PATCH | `/api/v1/admin/users/{id}/role` | 變更角色 | Admin |
| DELETE | `/api/v1/admin/users/{id}` | 停用帳號 | Admin |

## 📧 Email 功能

### 外部 SMTP 配置
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

**特點**:
- ✅ 真實發送郵件
- ✅ 適合生產環境
- ⚠️ 依賴外部服務
- ⚠️ 需要 SMTP 憑證

### 使用範例

#### 發送驗證郵件
```bash
curl -X POST -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123","full_name":"User Name"}' \
  http://localhost:8000/api/v1/auth/register
```

#### 發送密碼重設郵件
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"email":"user@example.com"}' \
  http://localhost:8000/api/v1/auth/password/reset
```

## 🧪 測試

### 執行所有測試
```bash
pytest tests/ -v --cov=src/account_management --cov-report=term-missing
```

### 執行單元測試
```bash
pytest tests/unit/ -v
```

### 執行整合測試
```bash
pytest tests/integration/ -v
```

### 產生覆蓋率報告
```bash
pytest tests/ --cov=src/account_management --cov-report=html
open htmlcov/index.html
```

### 執行變異測試
```bash
mutmut run
mutmut results
mutmut show
```

## 🐳 Docker 部署

### 使用 Docker Compose
```bash
# 啟動服務
docker-compose up -d

# 查看日誌
docker-compose logs -f app

# 停止服務
docker-compose down
```

### 環境變數
詳見 .env.example 檔案。

### 資料庫遷移（Docker 環境）
```bash
docker-compose exec app alembic upgrade head
```

## 🔒 安全性

### 密碼要求
- 長度 ≥ 8 字元
- 包含大寫字母
- 包含小寫字母
- 包含數字

### 安全機制
- ✅ bcrypt 雜湊（cost factor=12）
- ✅ JWT Token（24 小時有效期）
- ✅ Token 黑名單
- ✅ 登入失敗 5 次鎖定 15 分鐘
- ✅ Rate Limiting
- ✅ HTTPS 強制使用（生產環境）

## 📂 專案結構

```
account-management/
├── src/account_management/
│   ├── api/            # API 路由層
│   ├── core/           # 核心配置
│   ├── models/         # 資料模型
│   ├── repositories/   # 資料存取層
│   ├── schemas/        # Pydantic Schemas
│   └── services/       # 業務邏輯層
├── tests/              # 測試檔案
├── alembic/            # 資料庫遷移
├── docs/               # 文件
└── specs/              # 規格文件
```

## 🐛 疑難排解

### 常見問題

**Q: 郵件發送失敗？**
A: 檢查 SMTP 設定是否正確，確認 SMTP 憑證有效。

**Q: 如何配置 Gmail SMTP？**
A: 使用應用程式密碼，設定 `SMTP_HOST=smtp.gmail.com` 和 `SMTP_PORT=587`。

**Q: 如何測試郵件功能？**
A: 註冊新帳戶或請求密碼重設，系統會自動發送郵件。

## 📝 變更日誌

### v1.2.0 (2025-10-13)
- 移除自建 SMTP 服務
- 移除郵件佇列管理
- 簡化為外部 SMTP 直接發送
- 優化系統架構

### v1.1.0 (2025-10-01)
- 新增自建 SMTP 服務
- 新增郵件佇列管理
- 新增自動重試機制
- 新增管理員 Email API
- 支援內建/外部 SMTP 切換

### v1.0.0 (2025-10-01)
- 初始版本
- 基本帳號管理功能
- JWT 認證
- 角色權限控制

## 🤝 貢獻指南

歡迎提交 Pull Request 或回報問題！

### 開發流程
1. Fork 專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權

MIT License

## 📧 聯絡方式

- Email: support@example.com
- Issue: https://github.com/yourorg/account-management/issues

## 🙏 致謝

感謝以下開源專案：
- FastAPI
- SQLAlchemy
- aiosmtplib
- pytest

---

Built with ❤️ using FastAPI
