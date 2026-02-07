---
name: playwright
description: |
  Microsoft Playwright 瀏覽器自動化與端對端測試框架。支援 Chromium、WebKit、Firefox 跨瀏覽器測試。
  支援 TypeScript/JavaScript 與 Python 兩種語言。
  Use when: E2E 測試、瀏覽器自動化、網頁爬蟲、跨瀏覽器測試、視覺回歸測試、webapp testing，or when user mentions Playwright, E2E, 自動化測試, browser test, UI測試, 網頁測試.
  Triggers: "Playwright", "playwright", "E2E", "端對端測試", "瀏覽器自動化", "browser automation", "web testing", "webapp-testing", "UI測試"
version: 2.0.0
---

# Playwright Skill

## Overview

Playwright is a framework for Web Testing and Automation developed by Microsoft. It enables cross-browser web automation that is reliable, fast, and capable. Playwright supports all modern rendering engines including Chromium, WebKit, and Firefox.

## When to Use This Skill

Use this skill when working with:
- End-to-end (E2E) testing for web applications
- Browser automation and web scraping
- Cross-browser testing (Chrome, Firefox, Safari/WebKit)
- API testing alongside browser tests
- Visual regression testing
- Mobile web testing

## Decision Tree

```
Is the target static HTML?
├─ Yes → Read file directly, find selectors, write Playwright script
└─ No (Dynamic web app) →
    Is a dev server already running?
    ├─ Yes → Perform reconnaissance first, then execute actions
    └─ No → Use helper script to launch server, then test
```

---

## Quick Reference

### Installation

```bash
# Node.js
npm init playwright@latest

# Python
uv add playwright
playwright install chromium

# Java
# Add to pom.xml or build.gradle

# .NET
dotnet add package Microsoft.Playwright
```

---

## TypeScript/JavaScript

### Basic Test Structure

```typescript
import { test, expect } from '@playwright/test';

test('basic navigation test', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example/);
});

test('click and verify', async ({ page }) => {
  await page.goto('https://example.com');
  await page.click('text=More information');
  await expect(page).toHaveURL(/iana/);
});
```

### Locators (Recommended)

```typescript
// Preferred locator strategies (most stable)
page.getByRole('button', { name: 'Submit' })
page.getByText('Welcome')
page.getByLabel('Username')
page.getByPlaceholder('Enter email')
page.getByTestId('submit-button')

// CSS and XPath (when needed)
page.locator('css=button.primary')
page.locator('xpath=//button[@type="submit"]')
```

### Actions

```typescript
// Click actions
await page.click('button#submit');
await page.dblclick('text=Double click me');
await page.click('button', { button: 'right' });

// Input actions
await page.fill('input[name="email"]', 'user@example.com');
await page.type('input[name="search"]', 'query', { delay: 100 });

// Select and check
await page.selectOption('select#country', 'USA');
await page.check('input[type="checkbox"]');

// File upload
await page.setInputFiles('input[type="file"]', 'file.pdf');

// Keyboard and mouse
await page.keyboard.press('Enter');
await page.mouse.click(100, 200);
```

### Assertions

```typescript
// Page assertions
await expect(page).toHaveTitle('Page Title');
await expect(page).toHaveURL(/.*dashboard/);

// Locator assertions
await expect(page.locator('.status')).toBeVisible();
await expect(page.locator('.status')).toHaveText('Success');
await expect(page.locator('input')).toHaveValue('test@example.com');
await expect(page.locator('button')).toBeEnabled();
await expect(page.locator('.items')).toHaveCount(3);
```

### Page Object Model (TypeScript)

```typescript
// pages/login.page.ts
export class LoginPage {
  constructor(private page: Page) {}

  async login(email: string, password: string) {
    await this.page.fill('[name="email"]', email);
    await this.page.fill('[name="password"]', password);
    await this.page.click('button[type="submit"]');
  }
}

// tests/login.spec.ts
test('user can login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await page.goto('/login');
  await loginPage.login('user@test.com', 'password');
  await expect(page).toHaveURL('/dashboard');
});
```

### API Testing

```typescript
import { test, expect } from '@playwright/test';

test('API test', async ({ request }) => {
  const response = await request.get('/api/users');
  expect(response.ok()).toBeTruthy();

  const users = await response.json();
  expect(users.length).toBeGreaterThan(0);
});

test('POST request', async ({ request }) => {
  const response = await request.post('/api/users', {
    data: {
      name: 'John',
      email: 'john@example.com'
    }
  });
  expect(response.status()).toBe(201);
});
```

### Configuration (playwright.config.ts)

```typescript
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  retries: 2,
  workers: 4,

  use: {
    baseURL: 'https://example.com',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'on-first-retry',
  },

  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'mobile', use: { ...devices['iPhone 13'] } },
  ],
});
```

### Network Interception (TypeScript)

```typescript
await page.route('**/api/**', route => {
  route.fulfill({
    status: 200,
    body: JSON.stringify({ mocked: true })
  });
});
```

---

## Python

### Basic Test Template

```python
from playwright.sync_api import sync_playwright

def test_login_flow():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate
            page.goto("http://localhost:3000/login")
            page.wait_for_load_state("networkidle")

            # Fill form
            page.fill('[data-testid="email"]', "user@example.com")
            page.fill('[data-testid="password"]', "password123")

            # Submit
            page.click('[data-testid="submit"]')

            # Assert
            page.wait_for_url("**/dashboard")
            assert page.locator("h1").text_content() == "Dashboard"

        finally:
            browser.close()

if __name__ == "__main__":
    test_login_flow()
    print("✓ Login flow test passed")
```

### Server Management Helper

```python
# scripts/with_server.py
import subprocess
import time
import socket
import sys
from contextlib import contextmanager

@contextmanager
def run_server(command: str, port: int, startup_timeout: int = 30):
    """Start a server and wait for it to be ready."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for port to be available
    start = time.time()
    while time.time() - start < startup_timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("localhost", port))
                break
        except ConnectionRefusedError:
            time.sleep(0.5)
    else:
        process.kill()
        raise TimeoutError(f"Server did not start on port {port}")

    try:
        yield process
    finally:
        process.terminate()
        process.wait()

# Usage
if __name__ == "__main__":
    with run_server("npm run dev", port=5173):
        # Run your tests here
        subprocess.run([sys.executable, "test_app.py"])
```

### Wait Strategies (Python)

```python
# Wait for network to be idle (dynamic apps)
page.wait_for_load_state("networkidle")

# Wait for specific element
page.wait_for_selector('[data-testid="loaded"]')

# Wait for navigation
page.wait_for_url("**/success")

# Custom wait
page.wait_for_function("window.appReady === true")
```

### Selectors (Python)

```python
# Prefer data-testid (most stable)
page.click('[data-testid="submit-btn"]')

# By role (accessible)
page.get_by_role("button", name="Submit")

# By text
page.get_by_text("Click me")

# By label (forms)
page.get_by_label("Email address")

# CSS selector (fallback)
page.click(".submit-button")
```

### Screenshots & Debugging (Python)

```python
# Full page screenshot
page.screenshot(path="screenshot.png", full_page=True)

# Element screenshot
page.locator(".card").screenshot(path="card.png")

# Debug mode (opens browser)
browser = p.chromium.launch(headless=False, slow_mo=500)

# Trace for debugging
context = browser.new_context()
context.tracing.start(screenshots=True, snapshots=True)
# ... run tests ...
context.tracing.stop(path="trace.zip")
```

### Form Testing (Python)

```python
def test_form_validation():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000/register")

        # Test empty submission
        page.click('[data-testid="submit"]')
        assert page.locator(".error-email").is_visible()
        assert page.locator(".error-password").is_visible()

        # Test invalid email
        page.fill('[data-testid="email"]', "invalid-email")
        page.click('[data-testid="submit"]')
        error = page.locator(".error-email").text_content()
        assert "valid email" in error.lower()

        # Test valid submission
        page.fill('[data-testid="email"]', "user@example.com")
        page.fill('[data-testid="password"]', "SecurePass123!")
        page.click('[data-testid="submit"]')
        page.wait_for_url("**/success")

        browser.close()
```

### API Mocking (Python)

```python
def test_with_mocked_api():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()

        # Mock API response
        context.route(
            "**/api/users",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='[{"id": 1, "name": "Test User"}]'
            )
        )

        page = context.new_page()
        page.goto("http://localhost:3000/users")
        assert page.locator(".user-card").count() == 1

        browser.close()
```

### Page Object Pattern (Python)

```python
# helpers/pages.py
class LoginPage:
    def __init__(self, page):
        self.page = page
        self.email_input = page.locator('[data-testid="email"]')
        self.password_input = page.locator('[data-testid="password"]')
        self.submit_button = page.locator('[data-testid="submit"]')

    def goto(self):
        self.page.goto("http://localhost:3000/login")
        self.page.wait_for_load_state("networkidle")

    def login(self, email: str, password: str):
        self.email_input.fill(email)
        self.password_input.fill(password)
        self.submit_button.click()
```

---

## Common Patterns

### Waiting Strategies

```typescript
// Auto-waiting (built-in, preferred)
await page.click('button'); // Waits automatically

// Explicit waits when needed
await page.waitForSelector('.loaded');
await page.waitForLoadState('networkidle');
await page.waitForURL('**/success');
await page.waitForResponse('/api/data');
await page.waitForTimeout(1000); // Avoid if possible
```

### Screenshots and Videos

```typescript
// Screenshot
await page.screenshot({ path: 'screenshot.png' });
await page.screenshot({ path: 'full.png', fullPage: true });
await page.locator('.card').screenshot({ path: 'element.png' });

// Video recording (in config)
use: {
  video: 'on', // 'off', 'on', 'retain-on-failure', 'on-first-retry'
}
```

### Authentication State

```typescript
// Save auth state
await page.context().storageState({ path: 'auth.json' });

// Reuse auth state
const context = await browser.newContext({
  storageState: 'auth.json'
});
```

---

## Debugging Tools

```bash
# Run with headed browser
npx playwright test --headed

# Run in debug mode (Inspector)
npx playwright test --debug

# Run codegen (record tests)
npx playwright codegen example.com

# View trace
npx playwright show-trace trace.zip
```

---

## Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_auth.py          # Authentication tests
├── test_dashboard.py     # Dashboard tests
├── test_forms.py         # Form validation tests
└── helpers/
    ├── pages.py          # Page objects
    └── fixtures.py       # Test data
```

---

## Key Concepts

1. **Auto-waiting**: Playwright automatically waits for elements to be actionable before performing actions
2. **Locators**: Use semantic locators (getByRole, getByText) over CSS/XPath when possible
3. **Test Isolation**: Each test runs in a fresh browser context
4. **Parallel Execution**: Tests run in parallel by default for faster execution
5. **Cross-browser**: Same API works across Chromium, Firefox, and WebKit

## Best Practices

1. Use locators with explicit attributes over text-based locators
2. Prefer `getByRole`, `getByLabel`, `getByTestId` over CSS selectors
3. Use Page Object Model for maintainable tests
4. Enable tracing for debugging failures
5. Run tests in headless mode for CI/CD pipelines
6. Use fixtures for setup and teardown
7. Wait for `networkidle` on dynamic pages
8. Close browser in `finally` block
9. Handle server startup/shutdown
10. Take screenshots on failure
11. Test both happy path and error cases
12. Mock external APIs when needed

## Checklist

- [ ] Wait for `networkidle` on dynamic pages
- [ ] Use `data-testid` attributes for selectors
- [ ] Close browser in `finally` block
- [ ] Handle server startup/shutdown
- [ ] Take screenshots on failure
- [ ] Test both happy path and error cases
- [ ] Mock external APIs when needed
- [ ] Use Page Object pattern for reusability

## Reference Documentation

- Official Docs: https://playwright.dev/
- Python Docs: https://playwright.dev/python/
- API Reference: https://playwright.dev/docs/api/class-playwright
- GitHub: https://github.com/microsoft/playwright
