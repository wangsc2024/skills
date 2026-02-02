# SKILL.md 完整範例

以下是一個高品質的 SKILL.md 範例，供 Claude Chat 生成時參考：

---

# Vue.js 3 Skill

## Description

Vue.js 3 是漸進式 JavaScript 框架，用於建構現代化的響應式 Web 用戶介面，支援 Composition API 和更好的 TypeScript 整合。

## When to Use

- 需要建構單頁應用程式 (SPA)
- 需要響應式的資料綁定
- 想要漸進式採用框架（從小元件開始）
- 需要良好的 TypeScript 支援
- 想要使用 Composition API 組織邏輯
- 建構中大型前端應用

## Core Concepts

### 響應式系統

Vue 3 使用 Proxy-based 響應式系統，自動追蹤依賴並更新 DOM：

```javascript
import { ref, reactive, computed } from 'vue'

// ref 用於基本型別
const count = ref(0)
console.log(count.value) // 0

// reactive 用於物件
const state = reactive({
  name: 'Vue',
  version: 3
})

// computed 用於衍生狀態
const doubled = computed(() => count.value * 2)
```

### Composition API

Composition API 讓你將相關邏輯組織在一起：

```javascript
import { ref, onMounted, onUnmounted } from 'vue'

// 可重用的組合函數
function useMousePosition() {
  const x = ref(0)
  const y = ref(0)
  
  function update(event) {
    x.value = event.pageX
    y.value = event.pageY
  }
  
  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))
  
  return { x, y }
}

// 在元件中使用
export default {
  setup() {
    const { x, y } = useMousePosition()
    return { x, y }
  }
}
```

### 單檔案元件 (SFC)

Vue 使用 .vue 檔案整合模板、腳本和樣式：

```vue
<script setup>
import { ref } from 'vue'

const message = ref('Hello Vue!')
</script>

<template>
  <div class="greeting">
    <h1>{{ message }}</h1>
    <button @click="message = 'Updated!'">Update</button>
  </div>
</template>

<style scoped>
.greeting {
  color: #42b883;
}
</style>
```

### 生命週期

Vue 3 提供組合式 API 的生命週期鉤子：

```javascript
import { 
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted
} from 'vue'

export default {
  setup() {
    onMounted(() => {
      console.log('元件已掛載')
    })
    
    onUnmounted(() => {
      console.log('元件已卸載')
    })
  }
}
```

## Quick Reference

### 響應式 API

| API | 用途 | 範例 |
|-----|------|------|
| `ref()` | 建立響應式基本值 | `const count = ref(0)` |
| `reactive()` | 建立響應式物件 | `const state = reactive({})` |
| `computed()` | 建立計算屬性 | `const double = computed(() => x * 2)` |
| `watch()` | 監聽變化 | `watch(source, callback)` |
| `watchEffect()` | 自動追蹤依賴 | `watchEffect(() => {})` |
| `toRef()` | 從物件建立 ref | `const name = toRef(state, 'name')` |
| `toRefs()` | 解構響應式物件 | `const { a, b } = toRefs(state)` |

### 模板語法

| 語法 | 用途 | 範例 |
|------|------|------|
| `{{ }}` | 文字插值 | `{{ message }}` |
| `v-bind:` / `:` | 屬性綁定 | `:href="url"` |
| `v-on:` / `@` | 事件綁定 | `@click="handler"` |
| `v-model` | 雙向綁定 | `v-model="text"` |
| `v-if` | 條件渲染 | `v-if="show"` |
| `v-for` | 列表渲染 | `v-for="item in items"` |
| `v-show` | 切換顯示 | `v-show="visible"` |

### 常見模式：表單處理

```vue
<script setup>
import { ref, reactive } from 'vue'

const form = reactive({
  username: '',
  email: '',
  password: ''
})

const errors = ref({})

async function handleSubmit() {
  errors.value = {}
  
  // 驗證
  if (!form.username) {
    errors.value.username = '請輸入使用者名稱'
  }
  if (!form.email.includes('@')) {
    errors.value.email = '請輸入有效的 Email'
  }
  
  if (Object.keys(errors.value).length === 0) {
    // 提交表單
    console.log('提交:', form)
  }
}
</script>

<template>
  <form @submit.prevent="handleSubmit">
    <div>
      <label>使用者名稱</label>
      <input v-model="form.username" />
      <span v-if="errors.username" class="error">{{ errors.username }}</span>
    </div>
    
    <div>
      <label>Email</label>
      <input v-model="form.email" type="email" />
      <span v-if="errors.email" class="error">{{ errors.email }}</span>
    </div>
    
    <div>
      <label>密碼</label>
      <input v-model="form.password" type="password" />
    </div>
    
    <button type="submit">送出</button>
  </form>
</template>
```

## Code Examples

### Example 1: 基礎計數器

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)

function increment() {
  count.value++
}

function decrement() {
  count.value--
}
</script>

<template>
  <div class="counter">
    <button @click="decrement">-</button>
    <span>{{ count }}</span>
    <button @click="increment">+</button>
  </div>
</template>

<style scoped>
.counter {
  display: flex;
  gap: 1rem;
  align-items: center;
}
</style>
```

### Example 2: 資料獲取

```vue
<script setup>
import { ref, onMounted } from 'vue'

const users = ref([])
const loading = ref(true)
const error = ref(null)

async function fetchUsers() {
  try {
    loading.value = true
    const response = await fetch('https://jsonplaceholder.typicode.com/users')
    if (!response.ok) throw new Error('Failed to fetch')
    users.value = await response.json()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(fetchUsers)
</script>

<template>
  <div>
    <div v-if="loading">載入中...</div>
    <div v-else-if="error" class="error">錯誤: {{ error }}</div>
    <ul v-else>
      <li v-for="user in users" :key="user.id">
        {{ user.name }} ({{ user.email }})
      </li>
    </ul>
    <button @click="fetchUsers">重新載入</button>
  </div>
</template>
```

### Example 3: 可重用的組合函數

```javascript
// composables/useFetch.js
import { ref, watchEffect, toValue } from 'vue'

export function useFetch(url) {
  const data = ref(null)
  const error = ref(null)
  const loading = ref(false)

  async function doFetch() {
    loading.value = true
    data.value = null
    error.value = null

    try {
      const res = await fetch(toValue(url))
      data.value = await res.json()
    } catch (e) {
      error.value = e
    } finally {
      loading.value = false
    }
  }

  watchEffect(() => {
    doFetch()
  })

  return { data, error, loading, retry: doFetch }
}

// 使用方式
import { useFetch } from './composables/useFetch'

const { data, loading, error } = useFetch('/api/users')
```

### Example 4: 父子元件通訊

```vue
<!-- Parent.vue -->
<script setup>
import { ref } from 'vue'
import Child from './Child.vue'

const message = ref('Hello from Parent')

function handleUpdate(newValue) {
  message.value = newValue
}
</script>

<template>
  <div>
    <p>Parent: {{ message }}</p>
    <Child :message="message" @update="handleUpdate" />
  </div>
</template>
```

```vue
<!-- Child.vue -->
<script setup>
defineProps({
  message: String
})

const emit = defineEmits(['update'])

function sendUpdate() {
  emit('update', 'Hello from Child')
}
</script>

<template>
  <div>
    <p>Child received: {{ message }}</p>
    <button @click="sendUpdate">Update Parent</button>
  </div>
</template>
```

### Example 5: Provide/Inject 跨層級傳遞

```vue
<!-- GrandParent.vue -->
<script setup>
import { provide, ref } from 'vue'

const theme = ref('dark')

provide('theme', theme)
provide('toggleTheme', () => {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
})
</script>
```

```vue
<!-- DeepChild.vue (任意深度的子元件) -->
<script setup>
import { inject } from 'vue'

const theme = inject('theme')
const toggleTheme = inject('toggleTheme')
</script>

<template>
  <div :class="theme">
    <p>Current theme: {{ theme }}</p>
    <button @click="toggleTheme">Toggle Theme</button>
  </div>
</template>
```

## Common Pitfalls

### ❌ 直接賦值給 ref 而不是 .value

```javascript
const count = ref(0)

// 錯誤：這不會觸發響應式更新
count = 1

// 也是錯誤
count = ref(1)  // 建立了新的 ref，但原本的綁定還是舊的
```

### ✅ 正確做法

```javascript
const count = ref(0)

// 正確：修改 .value
count.value = 1

// 在模板中不需要 .value
// <span>{{ count }}</span>  ← 自動解包
```

**原因**：`ref()` 返回的是一個包裝物件，實際值存在 `.value` 屬性中。

---

### ❌ 解構 reactive 物件導致失去響應性

```javascript
const state = reactive({ count: 0, name: 'Vue' })

// 錯誤：解構後失去響應性
const { count, name } = state

// count 現在是普通數字，不再響應
count++ // 這不會觸發更新
```

### ✅ 正確做法

```javascript
import { reactive, toRefs } from 'vue'

const state = reactive({ count: 0, name: 'Vue' })

// 正確：使用 toRefs 保持響應性
const { count, name } = toRefs(state)

// 現在 count 是 ref，需要用 .value
count.value++ // 這會觸發更新
```

---

### ❌ 在 setup 外使用組合式 API

```javascript
// 錯誤：在 setup 外呼叫
const count = ref(0) // 這在模組頂層，不是 setup 內

export default {
  // ...
}
```

### ✅ 正確做法

```javascript
export default {
  setup() {
    // 正確：在 setup 內呼叫
    const count = ref(0)
    
    return { count }
  }
}

// 或使用 <script setup>
```

**原因**：組合式 API 需要在元件的 setup 上下文中執行，以正確綁定到元件實例。

## Related Resources

- [Vue.js 官方文件](https://vuejs.org/)
- [Vue.js GitHub](https://github.com/vuejs/core)
- [Vue Router](https://router.vuejs.org/)
- [Pinia（狀態管理）](https://pinia.vuejs.org/)
- [VueUse（組合函數集）](https://vueuse.org/)
- [Nuxt.js（全端框架）](https://nuxt.com/)
