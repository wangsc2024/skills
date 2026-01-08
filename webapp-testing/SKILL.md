---
name: webapp-testing
description: |
  Test web applications using Playwright automation. Creates end-to-end tests, performs visual testing, and automates browser interactions.
  Use when: testing web apps, writing E2E tests, automating browser tasks, validating UI functionality, or when user mentions Playwright, E2E, 端對端測試, browser test, 自動化測試, 網頁測試.
  Triggers: "Playwright", "E2E", "端對端測試", "browser test", "自動化測試", "網頁測試", "test web", "UI測試"
---

# Web Application Testing

Automate web application testing using Python Playwright for reliable end-to-end testing.

## Decision Tree

```
Is the target static HTML?
├─ Yes → Read file directly, find selectors, write Playwright script
└─ No (Dynamic web app) →
    Is a dev server already running?
    ├─ Yes → Perform reconnaissance first, then execute actions
    └─ No → Use helper script to launch server, then test
```

## Setup

```bash
# Install Playwright
uv add playwright
playwright install chromium
```

## Basic Test Template

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

## Server Management Helper

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

## Common Patterns

### Wait Strategies

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

### Selectors

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

### Screenshots & Debugging

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

### Form Testing

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

### API Mocking

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

### Page Object Pattern

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

## Checklist

- [ ] Wait for `networkidle` on dynamic pages
- [ ] Use `data-testid` attributes for selectors
- [ ] Close browser in `finally` block
- [ ] Handle server startup/shutdown
- [ ] Take screenshots on failure
- [ ] Test both happy path and error cases
- [ ] Mock external APIs when needed
- [ ] Use Page Object pattern for reusability
