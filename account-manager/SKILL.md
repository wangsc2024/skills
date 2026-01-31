---
name: account-manager
description: |
  RBAC 權限管控設計指引。專注於角色權限模型設計、權限命名規範、存取控制實作。
  Use when: 設計 RBAC 模型、定義角色權限、實作存取控制、權限檢查邏輯，or when user mentions 權限, RBAC, 角色, 存取控制.
  Triggers: "權限管控", "RBAC", "角色權限", "存取控制", "access control", "permission", "role", "authorization"
---

# 帳號管理與權限管控 Skill

## 核心原則

1. **最小權限原則**：用戶僅獲得執行任務所需的最低權限
2. **權限分離**：認證(Authentication)與授權(Authorization)分開處理
3. **預設拒絕**：未明確授權的操作一律拒絕
4. **可審計性**：所有權限相關操作必須留下日誌

## 權限模型架構 (RBAC)

### 層級結構

```
用戶 (User)
  └── 角色 (Role)
        └── 權限 (Permission)
              └── 資源 (Resource) + 操作 (Action)
```

### 標準角色定義

| 角色 | 代碼 | 說明 |
|------|------|------|
| 超級管理員 | `super_admin` | 完整系統存取權限 |
| 管理員 | `admin` | 管理功能存取權限 |
| 一般用戶 | `user` | 基本功能存取權限 |
| 訪客 | `guest` | 僅讀取公開內容 |

### 權限命名規範

```
<模組>:<資源>:<操作>

範例：
- user:profile:read      # 讀取用戶資料
- user:profile:update    # 更新用戶資料
- admin:users:delete     # 刪除用戶
- news:article:create    # 建立新聞
- system:settings:manage # 管理系統設定
```

### 標準操作類型

| 操作 | 代碼 | 說明 |
|------|------|------|
| 建立 | `create` | 新增資源 |
| 讀取 | `read` | 查看資源 |
| 更新 | `update` | 修改資源 |
| 刪除 | `delete` | 移除資源 |
| 管理 | `manage` | 完整 CRUD + 設定 |

## 新功能權限整合檢查清單

當新增任何功能時，必須完成以下步驟：

### 1. 權限定義

- [ ] 定義該功能所需的權限代碼
- [ ] 決定哪些角色可存取此功能
- [ ] 更新權限矩陣文件

### 2. 前端整合

- [ ] 在路由/頁面加入權限檢查
- [ ] 隱藏無權限的 UI 元素
- [ ] 顯示適當的無權限提示

### 3. 後端整合

- [ ] API 路由加入權限中介層
- [ ] 驗證 Token 有效性
- [ ] 檢查用戶角色與權限

### 4. 測試驗證

- [ ] 測試各角色的存取行為
- [ ] 測試無權限時的錯誤處理
- [ ] 測試 Token 過期情境

## 實作指引

### 前端權限檢查 (React/TypeScript)

```typescript
// hooks/usePermission.ts
export function usePermission() {
  const { user } = useAuth();

  const hasPermission = (permission: string): boolean => {
    if (!user) return false;
    return user.permissions.includes(permission) ||
           user.permissions.includes('*');
  };

  const hasRole = (role: string): boolean => {
    if (!user) return false;
    return user.roles.includes(role);
  };

  const hasAnyPermission = (permissions: string[]): boolean => {
    return permissions.some(p => hasPermission(p));
  };

  return { hasPermission, hasRole, hasAnyPermission };
}

// 元件使用範例
function AdminPanel() {
  const { hasPermission } = usePermission();

  if (!hasPermission('admin:panel:access')) {
    return <AccessDenied />;
  }

  return <AdminContent />;
}
```

### 權限守衛元件

```typescript
// components/PermissionGuard.tsx
interface PermissionGuardProps {
  permission: string | string[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function PermissionGuard({
  permission,
  children,
  fallback = null
}: PermissionGuardProps) {
  const { hasPermission, hasAnyPermission } = usePermission();

  const hasAccess = Array.isArray(permission)
    ? hasAnyPermission(permission)
    : hasPermission(permission);

  return hasAccess ? <>{children}</> : <>{fallback}</>;
}

// 使用範例
<PermissionGuard
  permission="user:delete"
  fallback={<span>無權限</span>}
>
  <DeleteButton />
</PermissionGuard>
```

### 後端權限中介層 (Node.js/Express)

```typescript
// middleware/authMiddleware.ts
export function requirePermission(permission: string) {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      const token = req.headers.authorization?.replace('Bearer ', '');
      if (!token) {
        return res.status(401).json({ error: '未提供認證 Token' });
      }

      const user = await verifyToken(token);
      if (!user) {
        return res.status(401).json({ error: 'Token 無效或已過期' });
      }

      if (!userHasPermission(user, permission)) {
        return res.status(403).json({ error: '權限不足' });
      }

      req.user = user;
      next();
    } catch (error) {
      return res.status(500).json({ error: '認證失敗' });
    }
  };
}

// 路由使用
router.delete('/users/:id',
  requirePermission('admin:users:delete'),
  deleteUserHandler
);
```

### 資料庫權限模型 (SQL)

```sql
-- 用戶表
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255),
  display_name VARCHAR(100),
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- 角色表
CREATE TABLE roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code VARCHAR(50) UNIQUE NOT NULL,
  name VARCHAR(100) NOT NULL,
  description TEXT,
  is_system BOOLEAN DEFAULT false
);

-- 權限表
CREATE TABLE permissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code VARCHAR(100) UNIQUE NOT NULL,
  name VARCHAR(100) NOT NULL,
  module VARCHAR(50) NOT NULL,
  description TEXT
);

-- 用戶角色關聯
CREATE TABLE user_roles (
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, role_id)
);

-- 角色權限關聯
CREATE TABLE role_permissions (
  role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
  permission_id UUID REFERENCES permissions(id) ON DELETE CASCADE,
  PRIMARY KEY (role_id, permission_id)
);

-- 預設角色
INSERT INTO roles (code, name, is_system) VALUES
  ('super_admin', '超級管理員', true),
  ('admin', '管理員', true),
  ('user', '一般用戶', true),
  ('guest', '訪客', true);
```

## 認證流程

### OAuth 登入流程

```
1. 用戶點擊登入按鈕
2. 重導向至 OAuth Provider (Google/GitHub)
3. 用戶授權
4. 回調至應用，取得授權碼
5. 後端用授權碼換取 Access Token
6. 取得用戶資訊，建立/更新本地用戶
7. 產生應用 JWT Token
8. 回傳 Token 給前端
9. 前端儲存 Token (httpOnly cookie 優先)
```

### Token 管理

```typescript
// JWT Payload 結構
interface TokenPayload {
  sub: string;           // 用戶 ID
  email: string;
  roles: string[];       // 角色代碼陣列
  permissions: string[]; // 權限代碼陣列
  iat: number;           // 簽發時間
  exp: number;           // 過期時間
}

// Token 設定建議
const TOKEN_CONFIG = {
  accessTokenExpiry: '15m',   // 存取 Token 15 分鐘
  refreshTokenExpiry: '7d',   // 刷新 Token 7 天
  algorithm: 'RS256'          // 使用非對稱加密
};
```

## 權限矩陣範本

建立 `docs/permission-matrix.md` 記錄權限配置：

```markdown
# 權限矩陣

| 功能模組 | 權限代碼 | super_admin | admin | user | guest |
|----------|----------|:-----------:|:-----:|:----:|:-----:|
| 用戶管理 | user:list:read | ✓ | ✓ | - | - |
| 用戶管理 | user:profile:update | ✓ | ✓ | ✓ | - |
| 用戶管理 | user:delete | ✓ | - | - | - |
| 新聞 | news:article:read | ✓ | ✓ | ✓ | ✓ |
| 新聞 | news:article:create | ✓ | ✓ | - | - |
| 系統 | system:settings:manage | ✓ | - | - | - |
```

## 常見問題處理

### 1. Token 過期處理

```typescript
// 前端 Axios 攔截器
axios.interceptors.response.use(
  response => response,
  async error => {
    if (error.response?.status === 401) {
      const refreshed = await refreshToken();
      if (refreshed) {
        return axios.request(error.config);
      }
      // 刷新失敗，登出用戶
      logout();
    }
    return Promise.reject(error);
  }
);
```

### 2. 權限快取失效

當用戶權限變更時，需要：
1. 使該用戶現有 Token 失效
2. 強制用戶重新登入
3. 或使用短期 Token + 權限即時查詢

### 3. 多租戶權限

若需支援多租戶，權限代碼需包含租戶識別：
```
tenant:<tenant_id>:<module>:<resource>:<action>
```

## 安全注意事項

1. **密碼儲存**：使用 bcrypt/argon2 雜湊，永不明文儲存
2. **Token 儲存**：優先使用 httpOnly Cookie，避免 localStorage
3. **HTTPS**：生產環境必須使用 HTTPS
4. **Rate Limiting**：登入 API 必須限流防暴力破解
5. **日誌記錄**：記錄所有認證失敗嘗試
6. **敏感操作**：重要操作要求重新驗證密碼
