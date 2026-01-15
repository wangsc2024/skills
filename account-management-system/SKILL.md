---
name: account-management-system
description: 帳號管理與權限控管系統設計指引。協助建立登入登出流程、用戶 CRUD、RBAC 角色權限控制，並確保新功能納入權限管控。適用於設計認證系統、實作權限檢查、建立用戶管理介面時使用。
version: 1.3.0
---

# 帳號管理系統 (Account Management System)

完整的使用者帳號管理系統，提供安全的身份驗證與授權機制，整合 Firebase Auth 進行 Email 驗證和密碼重設。

## 使用時機

當你需要以下功能時，請使用此 skill：

- **設計認證系統**：登入、登出、JWT Token 管理
- **實作權限檢查**：RBAC 角色權限控制（Admin/User）
- **建立用戶管理介面**：CRUD 操作、搜尋過濾
- **整合 Firebase Auth**：Email 驗證、密碼重設
- **實作安全機制**：密碼強度驗證、登入次數限制、Token 黑名單

---

## 快速參考

### 技術棧

| 類別 | 技術 |
|------|------|
| 後端框架 | FastAPI 0.104+ |
| 資料庫 | PostgreSQL 15+ / SQLite (開發) |
| ORM | SQLAlchemy 2.0 (非同步) |
| 認證 | Firebase Auth + JWT (PyJWT) |
| 密碼雜湊 | bcrypt (cost factor=12) |
| 郵件服務 | aiosmtplib (外部 SMTP) |

### 專案結構

```
src/account_management/
├── api/v1/           # API 路由層
│   ├── auth.py       # 認證相關端點
│   ├── users.py      # 使用者管理端點
│   └── admin.py      # 管理員端點
├── core/             # 核心配置
│   ├── config.py     # 環境變數配置
│   ├── security.py   # JWT/密碼處理
│   ├── firebase.py   # Firebase 整合
│   ├── middleware.py # 中間件（CORS、Rate Limit）
│   └── exceptions.py # 自定義異常
├── models/           # 資料模型
│   └── user.py       # User ORM 模型
├── repositories/     # 資料存取層
│   └── user_repository.py
├── schemas/          # Pydantic Schemas
│   ├── auth.py       # 認證相關 Schema
│   ├── user.py       # 使用者 Schema
│   └── admin.py      # 管理員 Schema
├── services/         # 業務邏輯層
│   ├── auth_service.py
│   ├── user_service.py
│   └── email_service.py
└── main.py           # 應用程式進入點
```

---

## API 端點

### 認證相關 `/api/v1/auth/`

| 方法 | 端點 | 說明 | 權限 |
|------|------|------|------|
| POST | `/register` | 使用者註冊 | 公開 |
| POST | `/login` | 使用者登入 | 公開 |
| POST | `/logout` | 使用者登出 | 已登入 |
| GET | `/me` | 取得當前使用者資訊 | 已登入 |
| POST | `/verify-email` | 驗證 Email | 公開 |
| POST | `/password/forgot` | 請求重設密碼 | 公開 |
| POST | `/password/reset` | 確認重設密碼 | 公開 |
| POST | `/firebase/login` | Firebase 登入 | 公開 |
| POST | `/sync-verification-status` | 同步 Firebase 驗證狀態 | 已登入 |

### 使用者管理 `/api/v1/users/`

| 方法 | 端點 | 說明 | 權限 |
|------|------|------|------|
| GET | `/me` | 查詢個人資料 | 已登入 |
| PATCH | `/me` | 更新個人資料 | 已登入 |
| POST | `/me/password` | 變更密碼 | 已登入 |

### 管理員功能 `/api/v1/admin/`

| 方法 | 端點 | 說明 | 權限 |
|------|------|------|------|
| GET | `/users` | 列出所有使用者 | Admin |
| POST | `/users` | 創建使用者 | Admin |
| GET | `/users/{id}` | 取得使用者詳情 | Admin |
| PUT | `/users/{id}` | 更新使用者資訊 | Admin |
| PATCH | `/users/{id}/role` | 變更角色 | Admin |
| DELETE | `/users/{id}` | 刪除帳號 | Admin |
| POST | `/users/{id}/resend-verification` | 重發驗證信 | Admin |
| GET | `/stats` | 系統統計資訊 | Admin |

---

## 程式碼範例

### 1. 使用者註冊

```python
# schemas/auth.py - 註冊請求 Schema
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)

# api/v1/auth.py - 註冊端點
@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    repo = UserRepository(db)
    service = AuthService(repo)

    user = await service.register(
        email=request.email,
        password=request.password,
        full_name=request.full_name
    )

    # 發送驗證信
    email_service = get_email_service()
    verification_token = create_access_token(
        data={"sub": str(user.id), "purpose": "email_verification"},
        expires_delta=timedelta(hours=24)
    )
    await email_service.send_verification_email(user.email, verification_token)

    return RegisterResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name
    )
```

### 2. JWT Token 認證

```python
# core/security.py - Token 創建
def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

# api/dependencies.py - 取得當前使用者
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="無效的認證憑證")

    user_id = payload.get("sub")
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="使用者不存在或已停用")

    return user
```

### 3. 權限控制

```python
# api/dependencies.py - 管理員權限檢查
async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="需要管理員權限"
        )
    return current_user

# 使用範例
@router.get("/users")
async def list_users(
    current_admin: User = Depends(get_current_admin),  # 僅 Admin
    db: AsyncSession = Depends(get_db)
):
    # 只有管理員可以存取
    ...
```

### 4. 密碼強度驗證

```python
# core/password_validator.py
class PasswordValidator:
    MIN_LENGTH = 8

    @staticmethod
    def validate(password: str) -> tuple[bool, str]:
        if len(password) < PasswordValidator.MIN_LENGTH:
            return False, f"密碼長度需≥{PasswordValidator.MIN_LENGTH}字元"
        if not re.search(r'[A-Z]', password):
            return False, "密碼需包含大寫字母"
        if not re.search(r'[a-z]', password):
            return False, "密碼需包含小寫字母"
        if not re.search(r'\d', password):
            return False, "密碼需包含數字"
        return True, ""
```

### 5. Firebase 整合

```python
# core/firebase.py - 初始化
def initialize_firebase() -> bool:
    if firebase_admin._apps:
        return True

    try:
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": settings.FIREBASE_PROJECT_ID,
            "private_key": settings.FIREBASE_PRIVATE_KEY.replace('\\n', '\n'),
            "client_email": settings.FIREBASE_CLIENT_EMAIL,
        })
        firebase_admin.initialize_app(cred)
        return True
    except Exception as e:
        logger.error(f"Firebase 初始化失敗: {e}")
        return False

# 驗證 Firebase Token
async def verify_firebase_token(token: str) -> dict:
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="無效的 Firebase Token")
```

### 6. 中間件設定

```python
# core/middleware.py
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # 清理過期紀錄
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if current_time - times[-1] < 60
        }

        # 檢查速率限制
        if client_ip in self.requests:
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                raise HTTPException(status_code=429, detail="請求過於頻繁")
            self.requests[client_ip].append(current_time)
        else:
            self.requests[client_ip] = [current_time]

        return await call_next(request)

# main.py - 註冊中間件
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
```

---

## 資料模型

### User Model

```python
# models/user.py
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(Enum("admin", "user", name="role_enum"), default="user")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    firebase_uid = Column(String, unique=True, nullable=True)
    login_method = Column(String, default="traditional")  # traditional/firebase
    force_firebase_login = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
```

---

## 安全機制

### 密碼要求
- 長度 ≥ 8 字元
- 包含大寫字母
- 包含小寫字母
- 包含數字

### 安全特性
- bcrypt 雜湊（cost factor=12）
- JWT Token（24 小時有效期）
- Token 黑名單機制
- 登入失敗 5 次鎖定 15 分鐘
- Rate Limiting（每分鐘 60 請求）
- CORS 設定
- Security Headers

---

## 環境變數

```bash
# 應用程式設定
APP_NAME="帳號管理系統"
APP_VERSION="1.3.0"
DEBUG=false
SECRET_KEY=your-secret-key

# 資料庫
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname

# Firebase 設定
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_API_KEY=your-api-key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk@your-project.iam.gserviceaccount.com

# SMTP 設定
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# CORS 設定
CORS_ORIGINS=["http://localhost:3000"]
```

---

## 啟動方式

```bash
# 安裝依賴
uv pip install -r requirements.txt

# 資料庫遷移
alembic upgrade head

# 開發模式
uvicorn src.account_management.main:app --reload

# 生產模式
uvicorn src.account_management.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 權限控制設計要點

### RBAC 實作模式

1. **角色定義**：系統支援 `admin` 和 `user` 兩種角色
2. **依賴注入**：使用 FastAPI 的 `Depends` 實現權限檢查
3. **分層權限**：
   - 公開端點：無需認證
   - 已登入端點：需要有效 JWT Token
   - Admin 端點：需要 admin 角色

### 新功能納入權限管控

當新增功能時，請遵循以下步驟：

1. **定義端點權限等級**（公開/已登入/Admin）
2. **選擇適當的依賴函數**：
   - `get_current_user` - 已登入使用者
   - `get_current_admin` - 管理員
3. **在 Router 中套用**：
   ```python
   @router.get("/new-feature")
   async def new_feature(
       current_user: User = Depends(get_current_user)  # 或 get_current_admin
   ):
       ...
   ```

---

## 參考文件

- `references/README.md` - 完整 README 文件
- `references/file_structure.md` - 專案結構
- `specs/account-management/spec.md` - 功能規格書
- `specs/account-management/plan.md` - 技術方案

---

**Generated by Skill Seeker** | 基於 D:/Source/account_system
