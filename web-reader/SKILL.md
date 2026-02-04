---
name: web-reader
version: "1.0"
description: |
  使用 Python requests 讀取網頁內容。當需要抓取網頁資料時，
  一律使用 requests 搭配適當的 headers 來讀取網頁，避免被網站封鎖。
  支援 HTML 解析、JSON API 讀取、以及自動處理編碼問題。
triggers:
  - "讀取網頁"
  - "抓取網頁"
  - "網頁內容"
  - "fetch"
  - "scrape"
  - "web reader"
  - "requests"
---

# 網頁讀取技能 (Web Reader Skill)

使用 Python requests 模組讀取網頁內容，搭配適當的 HTTP headers 以避免被網站封鎖。

## 核心原則

**一律使用 requests + headers 讀取網頁**，不使用 curl 或其他命令行工具。

---

## 基本讀取模板

### 標準 Headers 設定

```python
import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}
```

### 讀取 HTML 網頁

```python
import requests
from bs4 import BeautifulSoup

def fetch_webpage(url, timeout=30):
    """讀取網頁並返回 BeautifulSoup 物件"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # 自動偵測編碼
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f'Error fetching {url}: {e}')
        return None

# 使用範例
soup = fetch_webpage('https://example.com')
if soup:
    title = soup.find('title').get_text()
    print(f'Page title: {title}')
```

### 讀取 JSON API

```python
import requests

def fetch_json(url, timeout=30):
    """讀取 JSON API 並返回 dict"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error fetching {url}: {e}')
        return None

# 使用範例
data = fetch_json('https://api.example.com/data')
if data:
    print(data)
```

---

## 完整讀取範例

### 範例 1：讀取網頁並提取內容

```python
import requests
from bs4 import BeautifulSoup

url = 'https://example.com/page'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
}

try:
    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = 'utf-8'  # 或使用 response.apparent_encoding

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取標題
        title = soup.find('title')
        if title:
            print(f'Title: {title.get_text()}')

        # 提取主要內容
        content = soup.find('div', class_='content')
        if content:
            print(f'Content: {content.get_text(strip=True)}')
    else:
        print(f'Error: Status code {response.status_code}')

except requests.exceptions.RequestException as e:
    print(f'Request failed: {e}')
```

### 範例 2：讀取多個網頁

```python
import requests
from bs4 import BeautifulSoup
import time

urls = [
    'https://example.com/page1',
    'https://example.com/page2',
    'https://example.com/page3',
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
}

results = []

for url in urls:
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.encoding = response.apparent_encoding

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results.append({
                'url': url,
                'title': soup.find('title').get_text() if soup.find('title') else '',
                'content': soup.get_text(strip=True)[:500]
            })

        time.sleep(1)  # 禮貌性延遲，避免過於頻繁請求

    except Exception as e:
        print(f'Error fetching {url}: {e}')

print(f'Successfully fetched {len(results)} pages')
```

### 範例 3：帶有 Session 的請求

```python
import requests
from bs4 import BeautifulSoup

# 使用 Session 保持 cookies
session = requests.Session()

session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
})

# 第一次請求（可能設定 cookies）
response1 = session.get('https://example.com/login', timeout=30)

# 後續請求會帶上 cookies
response2 = session.get('https://example.com/dashboard', timeout=30)
```

---

## Headers 說明

| Header | 用途 | 建議值 |
|--------|------|--------|
| `User-Agent` | 識別瀏覽器類型 | 使用最新 Chrome UA |
| `Accept` | 接受的內容類型 | `text/html,application/xhtml+xml,...` |
| `Accept-Language` | 偏好語言 | `zh-TW,zh;q=0.9,en;q=0.8` |
| `Accept-Encoding` | 接受的壓縮格式 | `gzip, deflate, br` |
| `Referer` | 來源頁面 | 某些網站會檢查 |
| `Cookie` | 登入狀態 | 需要時手動設定 |

---

## 常見問題處理

### 1. 編碼問題

```python
response = requests.get(url, headers=headers)

# 方法 1：自動偵測
response.encoding = response.apparent_encoding

# 方法 2：強制指定
response.encoding = 'utf-8'

# 方法 3：從 Content-Type 取得
if 'charset' in response.headers.get('content-type', ''):
    # 自動處理
    pass
```

### 2. SSL 憑證問題

```python
# 忽略 SSL 驗證（不建議用於生產環境）
response = requests.get(url, headers=headers, verify=False)

# 使用自訂憑證
response = requests.get(url, headers=headers, verify='/path/to/cert.pem')
```

### 3. 超時處理

```python
try:
    response = requests.get(url, headers=headers, timeout=(5, 30))
    # (connect_timeout, read_timeout)
except requests.exceptions.Timeout:
    print('Request timed out')
except requests.exceptions.ConnectionError:
    print('Connection error')
```

### 4. 重試機制

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)

session.mount('https://', HTTPAdapter(max_retries=retries))
session.mount('http://', HTTPAdapter(max_retries=retries))

response = session.get(url, headers=headers, timeout=30)
```

---

## 快速範本

### 單頁讀取

```python
import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
}

response = requests.get('URL_HERE', headers=headers, timeout=30)
response.encoding = response.apparent_encoding
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text(strip=True)
print(text[:2000])
```

### JSON API 讀取

```python
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

response = requests.get('API_URL_HERE', headers=headers, timeout=30)
data = response.json()
print(data)
```

---

## 注意事項

1. **遵守 robots.txt**：尊重網站的爬蟲規則
2. **適當延遲**：避免過於頻繁的請求
3. **錯誤處理**：務必處理網路錯誤和超時
4. **編碼處理**：注意中文網頁的編碼問題
5. **User-Agent**：使用合理的 User-Agent

---

**Generated by Skill Seekers** | Web Reader Skill v1.0
