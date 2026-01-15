# 帳號管理參考文件

## 認證提供者整合

### Firebase Authentication

```typescript
// config/firebase.ts
import { initializeApp } from 'firebase/app';
import {
  getAuth,
  signInWithPopup,
  GoogleAuthProvider,
  signOut,
  onAuthStateChanged
} from 'firebase/auth';

const firebaseConfig = {
  apiKey: process.env.FIREBASE_API_KEY,
  authDomain: process.env.FIREBASE_AUTH_DOMAIN,
  projectId: process.env.FIREBASE_PROJECT_ID,
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// Google OAuth 登入
export async function signInWithGoogle() {
  const provider = new GoogleAuthProvider();
  provider.addScope('email');
  provider.addScope('profile');

  try {
    const result = await signInWithPopup(auth, provider);
    const idToken = await result.user.getIdToken();
    // 將 idToken 傳送至後端驗證並取得應用 Token
    return idToken;
  } catch (error) {
    console.error('Google 登入失敗:', error);
    throw error;
  }
}

// 登出
export async function logOut() {
  await signOut(auth);
}

// 監聽認證狀態
export function onAuthChange(callback: (user: User | null) => void) {
  return onAuthStateChanged(auth, callback);
}
```

### Supabase Authentication

```typescript
// config/supabase.ts
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseKey);

// OAuth 登入
export async function signInWithOAuth(provider: 'google' | 'github') {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider,
    options: {
      redirectTo: `${window.location.origin}/auth/callback`
    }
  });

  if (error) throw error;
  return data;
}

// 登出
export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
}

// 取得目前用戶
export async function getCurrentUser() {
  const { data: { user } } = await supabase.auth.getUser();
  return user;
}

// 監聽認證狀態
export function onAuthStateChange(callback: (session: Session | null) => void) {
  return supabase.auth.onAuthStateChange((event, session) => {
    callback(session);
  });
}
```

### JWT 自建認證

```typescript
// services/authService.ts
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const JWT_SECRET = process.env.JWT_SECRET!;
const JWT_EXPIRES_IN = '15m';
const REFRESH_TOKEN_EXPIRES_IN = '7d';

interface TokenPayload {
  sub: string;
  email: string;
  roles: string[];
  permissions: string[];
}

// 產生 Token
export function generateTokens(user: User, permissions: string[]) {
  const payload: TokenPayload = {
    sub: user.id,
    email: user.email,
    roles: user.roles.map(r => r.code),
    permissions
  };

  const accessToken = jwt.sign(payload, JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN
  });

  const refreshToken = jwt.sign(
    { sub: user.id, type: 'refresh' },
    JWT_SECRET,
    { expiresIn: REFRESH_TOKEN_EXPIRES_IN }
  );

  return { accessToken, refreshToken };
}

// 驗證 Token
export function verifyToken(token: string): TokenPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as TokenPayload;
  } catch {
    return null;
  }
}

// 密碼雜湊
export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 12);
}

// 密碼驗證
export async function verifyPassword(
  password: string,
  hash: string
): Promise<boolean> {
  return bcrypt.compare(password, hash);
}
```

---

## 權限查詢函式

```typescript
// services/permissionService.ts

// 取得用戶所有權限
export async function getUserPermissions(userId: string): Promise<string[]> {
  const result = await db.query(`
    SELECT DISTINCT p.code
    FROM permissions p
    JOIN role_permissions rp ON p.id = rp.permission_id
    JOIN user_roles ur ON rp.role_id = ur.role_id
    WHERE ur.user_id = $1
  `, [userId]);

  return result.rows.map(r => r.code);
}

// 檢查用戶是否有特定權限
export async function userHasPermission(
  userId: string,
  permission: string
): Promise<boolean> {
  const permissions = await getUserPermissions(userId);

  // 檢查完整權限或萬用字元
  return permissions.some(p =>
    p === permission ||
    p === '*' ||
    (p.endsWith(':*') && permission.startsWith(p.slice(0, -1)))
  );
}

// 取得用戶角色
export async function getUserRoles(userId: string): Promise<Role[]> {
  const result = await db.query(`
    SELECT r.*
    FROM roles r
    JOIN user_roles ur ON r.id = ur.role_id
    WHERE ur.user_id = $1
  `, [userId]);

  return result.rows;
}

// 授予用戶角色
export async function assignRoleToUser(
  userId: string,
  roleCode: string
): Promise<void> {
  await db.query(`
    INSERT INTO user_roles (user_id, role_id)
    SELECT $1, id FROM roles WHERE code = $2
    ON CONFLICT DO NOTHING
  `, [userId, roleCode]);
}

// 移除用戶角色
export async function removeRoleFromUser(
  userId: string,
  roleCode: string
): Promise<void> {
  await db.query(`
    DELETE FROM user_roles
    WHERE user_id = $1
    AND role_id = (SELECT id FROM roles WHERE code = $2)
  `, [userId, roleCode]);
}
```

---

## React Context 整合

```typescript
// contexts/AuthContext.tsx
import { createContext, useContext, useEffect, useState } from 'react';

interface User {
  id: string;
  email: string;
  displayName: string;
  roles: string[];
  permissions: string[];
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (provider: string) => Promise<void>;
  logout: () => Promise<void>;
  hasPermission: (permission: string) => boolean;
  hasRole: (role: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 初始化時檢查現有 session
    checkSession();
  }, []);

  async function checkSession() {
    try {
      const response = await fetch('/api/auth/me');
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      }
    } finally {
      setLoading(false);
    }
  }

  async function login(provider: string) {
    // 實作 OAuth 登入流程
  }

  async function logout() {
    await fetch('/api/auth/logout', { method: 'POST' });
    setUser(null);
  }

  function hasPermission(permission: string): boolean {
    if (!user) return false;
    return user.permissions.includes(permission) ||
           user.permissions.includes('*');
  }

  function hasRole(role: string): boolean {
    if (!user) return false;
    return user.roles.includes(role);
  }

  return (
    <AuthContext.Provider value={{
      user,
      loading,
      login,
      logout,
      hasPermission,
      hasRole
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```

---

## 常用 SQL 查詢

### 取得用戶完整資訊（含角色與權限）

```sql
SELECT
  u.id,
  u.email,
  u.display_name,
  array_agg(DISTINCT r.code) as roles,
  array_agg(DISTINCT p.code) as permissions
FROM users u
LEFT JOIN user_roles ur ON u.id = ur.user_id
LEFT JOIN roles r ON ur.role_id = r.id
LEFT JOIN role_permissions rp ON r.id = rp.role_id
LEFT JOIN permissions p ON rp.permission_id = p.id
WHERE u.id = $1
GROUP BY u.id;
```

### 取得特定權限的所有用戶

```sql
SELECT DISTINCT u.*
FROM users u
JOIN user_roles ur ON u.id = ur.user_id
JOIN role_permissions rp ON ur.role_id = rp.role_id
JOIN permissions p ON rp.permission_id = p.id
WHERE p.code = $1;
```

### 取得角色的所有權限

```sql
SELECT p.*
FROM permissions p
JOIN role_permissions rp ON p.id = rp.permission_id
JOIN roles r ON rp.role_id = r.id
WHERE r.code = $1;
```

---

## 錯誤代碼定義

| 代碼 | HTTP Status | 說明 |
|------|-------------|------|
| AUTH_001 | 401 | 未提供認證 Token |
| AUTH_002 | 401 | Token 無效 |
| AUTH_003 | 401 | Token 已過期 |
| AUTH_004 | 403 | 權限不足 |
| AUTH_005 | 403 | 帳號已停用 |
| AUTH_006 | 400 | 登入憑證錯誤 |
| AUTH_007 | 429 | 登入嘗試次數過多 |
