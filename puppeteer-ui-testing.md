# Puppeteer UI 測試與優化 Skill
# 版本: 1.0
# 適用專案: 前端 Web 應用程式

---

## Skill 觸發條件

當使用者請求以下類型任務時，自動啟用此 Skill：
- 使用 Puppeteer 進行前端 UI 自動化測試
- 進行視覺回歸測試 (Visual Regression Testing)
- 分析並優化前端效能
- 執行無障礙 (Accessibility) 測試
- 自動化截圖與 UI 比較
- E2E (End-to-End) 測試開發

---

## 執行指令集

### /puppeteer:init
**初始化測試專案**

執行流程：
1. 安裝 Puppeteer 及相關依賴
2. 建立測試目錄結構
3. 設定基礎配置檔案
4. 建立測試輔助工具

**輸出結構：**
```
project-root/
├── tests/
│   ├── e2e/
│   │   ├── specs/              # 測試規格檔案
│   │   │   ├── homepage.spec.ts
│   │   │   └── components/
│   │   ├── fixtures/           # 測試固定資料
│   │   ├── screenshots/        # 截圖輸出
│   │   │   ├── baseline/       # 基準截圖
│   │   │   ├── current/        # 當前截圖
│   │   │   └── diff/           # 差異截圖
│   │   └── reports/            # 測試報告
│   ├── utils/
│   │   ├── browser.ts          # 瀏覽器工具
│   │   ├── screenshot.ts       # 截圖工具
│   │   ├── visual-diff.ts      # 視覺比較
│   │   └── performance.ts      # 效能分析
│   └── config/
│       ├── puppeteer.config.ts # Puppeteer 配置
│       └── viewports.ts        # 視窗尺寸定義
├── package.json
└── jest.config.js              # 或 vitest.config.ts
```

**安裝命令：**
```bash
npm install puppeteer puppeteer-core @types/puppeteer --save-dev
npm install pixelmatch pngjs sharp --save-dev
npm install jest @types/jest ts-jest --save-dev
# 或使用 vitest
npm install vitest --save-dev
```

**基礎配置範例：**
```typescript
// tests/config/puppeteer.config.ts
import { LaunchOptions } from 'puppeteer';

export const puppeteerConfig: LaunchOptions = {
  headless: true,
  args: [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--window-size=1920,1080'
  ],
  defaultViewport: {
    width: 1920,
    height: 1080,
    deviceScaleFactor: 1
  },
  slowMo: process.env.SLOW_MO ? parseInt(process.env.SLOW_MO) : 0
};

export const viewports = {
  desktop: { width: 1920, height: 1080 },
  laptop: { width: 1366, height: 768 },
  tablet: { width: 768, height: 1024 },
  mobile: { width: 375, height: 667 }
};

export const testConfig = {
  baseUrl: process.env.BASE_URL || 'http://localhost:3000',
  screenshotDir: './tests/e2e/screenshots',
  timeout: 30000,
  retries: 2
};
```

---

### /puppeteer:test [type]
**建立測試檔案**

參數：
- `type`：測試類型（visual | functional | performance | a11y | e2e）

執行流程：
1. 根據類型生成對應的測試模板
2. 包含常用斷言與測試模式
3. 加入錯誤處理與重試機制

**視覺回歸測試範例：**
```typescript
// tests/e2e/specs/visual-regression.spec.ts
import puppeteer, { Browser, Page } from 'puppeteer';
import { puppeteerConfig, viewports, testConfig } from '../config/puppeteer.config';
import { takeScreenshot, compareScreenshots } from '../utils/screenshot';

describe('Visual Regression Tests', () => {
  let browser: Browser;
  let page: Page;

  beforeAll(async () => {
    browser = await puppeteer.launch(puppeteerConfig);
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    page = await browser.newPage();
    await page.setViewport(viewports.desktop);
  });

  afterEach(async () => {
    await page.close();
  });

  describe('Homepage Visual Tests', () => {
    it('should match baseline screenshot on desktop', async () => {
      await page.goto(`${testConfig.baseUrl}/`, {
        waitUntil: 'networkidle0'
      });

      // 等待關鍵元素載入
      await page.waitForSelector('[data-testid="main-content"]');

      // 隱藏動態內容避免誤報
      await page.evaluate(() => {
        document.querySelectorAll('[data-dynamic]').forEach(el => {
          (el as HTMLElement).style.visibility = 'hidden';
        });
      });

      const screenshot = await takeScreenshot(page, 'homepage-desktop');
      const diff = await compareScreenshots('homepage-desktop');

      expect(diff.percentage).toBeLessThan(0.1); // 允許 0.1% 差異
    });

    it('should match baseline on mobile viewport', async () => {
      await page.setViewport(viewports.mobile);
      await page.goto(`${testConfig.baseUrl}/`, {
        waitUntil: 'networkidle0'
      });

      const screenshot = await takeScreenshot(page, 'homepage-mobile');
      const diff = await compareScreenshots('homepage-mobile');

      expect(diff.percentage).toBeLessThan(0.1);
    });
  });

  describe('Component Visual Tests', () => {
    it('should render button component correctly', async () => {
      await page.goto(`${testConfig.baseUrl}/components/button`);
      await page.waitForSelector('[data-testid="button-showcase"]');

      // 截取特定元素
      const element = await page.$('[data-testid="button-showcase"]');
      const screenshot = await element?.screenshot({
        path: `${testConfig.screenshotDir}/current/button-component.png`
      });

      const diff = await compareScreenshots('button-component');
      expect(diff.percentage).toBeLessThan(0.05);
    });
  });
});
```

**功能測試範例：**
```typescript
// tests/e2e/specs/functional.spec.ts
import puppeteer, { Browser, Page } from 'puppeteer';
import { puppeteerConfig, testConfig } from '../config/puppeteer.config';

describe('Functional UI Tests', () => {
  let browser: Browser;
  let page: Page;

  beforeAll(async () => {
    browser = await puppeteer.launch(puppeteerConfig);
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    page = await browser.newPage();
  });

  afterEach(async () => {
    await page.close();
  });

  describe('Form Interactions', () => {
    it('should submit form with valid data', async () => {
      await page.goto(`${testConfig.baseUrl}/contact`);

      // 填寫表單
      await page.type('[data-testid="name-input"]', 'Test User');
      await page.type('[data-testid="email-input"]', 'test@example.com');
      await page.type('[data-testid="message-input"]', 'Hello, this is a test message.');

      // 點擊提交
      await Promise.all([
        page.waitForNavigation({ waitUntil: 'networkidle0' }),
        page.click('[data-testid="submit-button"]')
      ]);

      // 驗證成功訊息
      const successMessage = await page.$eval(
        '[data-testid="success-message"]',
        el => el.textContent
      );
      expect(successMessage).toContain('成功');
    });

    it('should show validation errors for invalid input', async () => {
      await page.goto(`${testConfig.baseUrl}/contact`);

      // 提交空表單
      await page.click('[data-testid="submit-button"]');

      // 檢查錯誤訊息
      await page.waitForSelector('[data-testid="error-message"]');
      const errors = await page.$$('[data-testid="error-message"]');
      expect(errors.length).toBeGreaterThan(0);
    });
  });

  describe('Navigation', () => {
    it('should navigate through menu items', async () => {
      await page.goto(testConfig.baseUrl);

      const menuItems = ['about', 'services', 'contact'];

      for (const item of menuItems) {
        await page.click(`[data-testid="nav-${item}"]`);
        await page.waitForNavigation({ waitUntil: 'networkidle0' });

        const currentUrl = page.url();
        expect(currentUrl).toContain(`/${item}`);
      }
    });
  });

  describe('Modal Interactions', () => {
    it('should open and close modal', async () => {
      await page.goto(testConfig.baseUrl);

      // 開啟 Modal
      await page.click('[data-testid="open-modal-button"]');
      await page.waitForSelector('[data-testid="modal"]', { visible: true });

      // 驗證 Modal 可見
      const modalVisible = await page.$eval(
        '[data-testid="modal"]',
        el => getComputedStyle(el).display !== 'none'
      );
      expect(modalVisible).toBe(true);

      // 關閉 Modal
      await page.click('[data-testid="close-modal-button"]');
      await page.waitForSelector('[data-testid="modal"]', { hidden: true });
    });

    it('should close modal on backdrop click', async () => {
      await page.goto(testConfig.baseUrl);

      await page.click('[data-testid="open-modal-button"]');
      await page.waitForSelector('[data-testid="modal"]', { visible: true });

      // 點擊背景關閉
      await page.click('[data-testid="modal-backdrop"]');
      await page.waitForSelector('[data-testid="modal"]', { hidden: true });
    });
  });
});
```

---

### /puppeteer:screenshot [mode]
**截圖與比較工具**

參數：
- `mode`：模式（capture | compare | update-baseline）

執行流程：
1. 根據模式執行對應操作
2. 生成截圖或比較報告
3. 輸出差異視覺化

**截圖工具實作：**
```typescript
// tests/utils/screenshot.ts
import puppeteer, { Page } from 'puppeteer';
import fs from 'fs/promises';
import path from 'path';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';
import sharp from 'sharp';

export interface ScreenshotOptions {
  fullPage?: boolean;
  clip?: { x: number; y: number; width: number; height: number };
  omitBackground?: boolean;
  hideSelectors?: string[];
  waitForSelector?: string;
  delay?: number;
}

export interface ComparisonResult {
  match: boolean;
  percentage: number;
  diffPixels: number;
  totalPixels: number;
  diffImagePath?: string;
}

const SCREENSHOT_DIR = './tests/e2e/screenshots';

/**
 * 擷取頁面截圖
 */
export async function takeScreenshot(
  page: Page,
  name: string,
  options: ScreenshotOptions = {}
): Promise<string> {
  const {
    fullPage = true,
    clip,
    omitBackground = false,
    hideSelectors = [],
    waitForSelector,
    delay = 0
  } = options;

  // 等待指定元素
  if (waitForSelector) {
    await page.waitForSelector(waitForSelector);
  }

  // 額外延遲（用於動畫完成等）
  if (delay > 0) {
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  // 隱藏指定元素（如動態內容）
  if (hideSelectors.length > 0) {
    await page.evaluate((selectors) => {
      selectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
          (el as HTMLElement).style.visibility = 'hidden';
        });
      });
    }, hideSelectors);
  }

  const screenshotPath = path.join(SCREENSHOT_DIR, 'current', `${name}.png`);

  await fs.mkdir(path.dirname(screenshotPath), { recursive: true });

  await page.screenshot({
    path: screenshotPath,
    fullPage,
    clip,
    omitBackground
  });

  return screenshotPath;
}

/**
 * 比較截圖與基準圖片
 */
export async function compareScreenshots(
  name: string,
  threshold: number = 0.1
): Promise<ComparisonResult> {
  const baselinePath = path.join(SCREENSHOT_DIR, 'baseline', `${name}.png`);
  const currentPath = path.join(SCREENSHOT_DIR, 'current', `${name}.png`);
  const diffPath = path.join(SCREENSHOT_DIR, 'diff', `${name}-diff.png`);

  // 檢查基準圖片是否存在
  try {
    await fs.access(baselinePath);
  } catch {
    // 如果沒有基準圖片，複製當前截圖作為基準
    await fs.mkdir(path.dirname(baselinePath), { recursive: true });
    await fs.copyFile(currentPath, baselinePath);
    console.log(`Created baseline for: ${name}`);
    return {
      match: true,
      percentage: 0,
      diffPixels: 0,
      totalPixels: 0
    };
  }

  // 讀取圖片
  const [baselineBuffer, currentBuffer] = await Promise.all([
    fs.readFile(baselinePath),
    fs.readFile(currentPath)
  ]);

  const baselineImg = PNG.sync.read(baselineBuffer);
  const currentImg = PNG.sync.read(currentBuffer);

  // 確保尺寸一致
  if (baselineImg.width !== currentImg.width || baselineImg.height !== currentImg.height) {
    // 調整尺寸以匹配
    const targetWidth = Math.max(baselineImg.width, currentImg.width);
    const targetHeight = Math.max(baselineImg.height, currentImg.height);

    const [resizedBaseline, resizedCurrent] = await Promise.all([
      sharp(baselineBuffer).resize(targetWidth, targetHeight, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } }).png().toBuffer(),
      sharp(currentBuffer).resize(targetWidth, targetHeight, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } }).png().toBuffer()
    ]);

    const resizedBaselineImg = PNG.sync.read(resizedBaseline);
    const resizedCurrentImg = PNG.sync.read(resizedCurrent);

    return performComparison(resizedBaselineImg, resizedCurrentImg, diffPath, threshold);
  }

  return performComparison(baselineImg, currentImg, diffPath, threshold);
}

async function performComparison(
  baseline: PNG,
  current: PNG,
  diffPath: string,
  threshold: number
): Promise<ComparisonResult> {
  const { width, height } = baseline;
  const diff = new PNG({ width, height });

  const diffPixels = pixelmatch(
    baseline.data,
    current.data,
    diff.data,
    width,
    height,
    { threshold }
  );

  const totalPixels = width * height;
  const percentage = (diffPixels / totalPixels) * 100;

  // 儲存差異圖片
  await fs.mkdir(path.dirname(diffPath), { recursive: true });
  await fs.writeFile(diffPath, PNG.sync.write(diff));

  return {
    match: diffPixels === 0,
    percentage,
    diffPixels,
    totalPixels,
    diffImagePath: diffPath
  };
}

/**
 * 更新基準圖片
 */
export async function updateBaseline(name: string): Promise<void> {
  const currentPath = path.join(SCREENSHOT_DIR, 'current', `${name}.png`);
  const baselinePath = path.join(SCREENSHOT_DIR, 'baseline', `${name}.png`);

  await fs.mkdir(path.dirname(baselinePath), { recursive: true });
  await fs.copyFile(currentPath, baselinePath);
  console.log(`Updated baseline for: ${name}`);
}

/**
 * 批次更新所有基準圖片
 */
export async function updateAllBaselines(): Promise<void> {
  const currentDir = path.join(SCREENSHOT_DIR, 'current');
  const files = await fs.readdir(currentDir);

  for (const file of files) {
    if (file.endsWith('.png')) {
      const name = file.replace('.png', '');
      await updateBaseline(name);
    }
  }
}
```

---

### /puppeteer:performance
**效能分析與優化**

執行流程：
1. 收集 Core Web Vitals 指標
2. 分析頁面載入效能
3. 生成效能報告與優化建議

**效能分析工具：**
```typescript
// tests/utils/performance.ts
import puppeteer, { Browser, Page, CDPSession } from 'puppeteer';
import { puppeteerConfig, testConfig } from '../config/puppeteer.config';

export interface PerformanceMetrics {
  // Core Web Vitals
  LCP: number;  // Largest Contentful Paint
  FID: number;  // First Input Delay (模擬)
  CLS: number;  // Cumulative Layout Shift

  // 其他重要指標
  FCP: number;  // First Contentful Paint
  TTFB: number; // Time to First Byte
  TTI: number;  // Time to Interactive
  TBT: number;  // Total Blocking Time

  // 資源指標
  totalResources: number;
  totalSize: number;
  jsSize: number;
  cssSize: number;
  imageSize: number;
  fontSize: number;

  // 網路請求
  requestCount: number;
  failedRequests: number;
}

export interface PerformanceReport {
  url: string;
  timestamp: Date;
  metrics: PerformanceMetrics;
  recommendations: string[];
  score: number;
}

/**
 * 分析頁面效能
 */
export async function analyzePerformance(url: string): Promise<PerformanceReport> {
  const browser = await puppeteer.launch(puppeteerConfig);
  const page = await browser.newPage();

  // 啟用效能追蹤
  const client = await page.createCDPSession();
  await client.send('Performance.enable');

  // 追蹤資源
  const resources: Array<{ type: string; size: number; url: string }> = [];
  let failedRequests = 0;

  page.on('response', async (response) => {
    try {
      const headers = response.headers();
      const contentLength = parseInt(headers['content-length'] || '0');
      const contentType = headers['content-type'] || '';
      const resourceUrl = response.url();

      let type = 'other';
      if (contentType.includes('javascript')) type = 'js';
      else if (contentType.includes('css')) type = 'css';
      else if (contentType.includes('image')) type = 'image';
      else if (contentType.includes('font')) type = 'font';
      else if (contentType.includes('html')) type = 'html';

      resources.push({ type, size: contentLength, url: resourceUrl });
    } catch {
      // 忽略無法解析的回應
    }
  });

  page.on('requestfailed', () => {
    failedRequests++;
  });

  // 收集 Web Vitals
  await page.evaluateOnNewDocument(() => {
    (window as any).__webVitals = {
      LCP: 0,
      CLS: 0,
      FCP: 0
    };

    // 觀察 LCP
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      (window as any).__webVitals.LCP = lastEntry.startTime;
    }).observe({ type: 'largest-contentful-paint', buffered: true });

    // 觀察 CLS
    let clsValue = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!(entry as any).hadRecentInput) {
          clsValue += (entry as any).value;
        }
      }
      (window as any).__webVitals.CLS = clsValue;
    }).observe({ type: 'layout-shift', buffered: true });

    // 觀察 FCP
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          (window as any).__webVitals.FCP = entry.startTime;
        }
      }
    }).observe({ type: 'paint', buffered: true });
  });

  const startTime = Date.now();
  await page.goto(url, { waitUntil: 'networkidle0' });
  const loadTime = Date.now() - startTime;

  // 等待指標穩定
  await new Promise(resolve => setTimeout(resolve, 2000));

  // 收集指標
  const webVitals = await page.evaluate(() => (window as any).__webVitals);
  const performanceMetrics = await client.send('Performance.getMetrics');

  const metricsMap = new Map(
    performanceMetrics.metrics.map(m => [m.name, m.value])
  );

  // 計算資源大小
  const resourceSizes = {
    js: resources.filter(r => r.type === 'js').reduce((sum, r) => sum + r.size, 0),
    css: resources.filter(r => r.type === 'css').reduce((sum, r) => sum + r.size, 0),
    image: resources.filter(r => r.type === 'image').reduce((sum, r) => sum + r.size, 0),
    font: resources.filter(r => r.type === 'font').reduce((sum, r) => sum + r.size, 0),
    total: resources.reduce((sum, r) => sum + r.size, 0)
  };

  const metrics: PerformanceMetrics = {
    LCP: webVitals.LCP || 0,
    FID: 0, // 需要真實用戶互動，這裡用 TBT 估算
    CLS: webVitals.CLS || 0,
    FCP: webVitals.FCP || 0,
    TTFB: (metricsMap.get('NavigationStart') || 0) * 1000,
    TTI: loadTime,
    TBT: (metricsMap.get('TaskDuration') || 0) * 1000,
    totalResources: resources.length,
    totalSize: resourceSizes.total,
    jsSize: resourceSizes.js,
    cssSize: resourceSizes.css,
    imageSize: resourceSizes.image,
    fontSize: resourceSizes.font,
    requestCount: resources.length,
    failedRequests
  };

  await browser.close();

  // 生成建議
  const recommendations = generateRecommendations(metrics);
  const score = calculateScore(metrics);

  return {
    url,
    timestamp: new Date(),
    metrics,
    recommendations,
    score
  };
}

/**
 * 生成效能優化建議
 */
function generateRecommendations(metrics: PerformanceMetrics): string[] {
  const recommendations: string[] = [];

  // LCP 建議
  if (metrics.LCP > 2500) {
    recommendations.push('LCP 超過 2.5 秒，建議：');
    recommendations.push('  - 優化伺服器回應時間');
    recommendations.push('  - 使用 CDN 加速靜態資源');
    recommendations.push('  - 預載入關鍵資源 (rel="preload")');
    recommendations.push('  - 優化圖片大小和格式 (WebP/AVIF)');
  }

  // CLS 建議
  if (metrics.CLS > 0.1) {
    recommendations.push('CLS 超過 0.1，建議：');
    recommendations.push('  - 為圖片和嵌入元素設定明確尺寸');
    recommendations.push('  - 避免在現有內容上方插入新內容');
    recommendations.push('  - 使用 CSS transform 動畫取代影響佈局的屬性');
  }

  // FCP 建議
  if (metrics.FCP > 1800) {
    recommendations.push('FCP 超過 1.8 秒，建議：');
    recommendations.push('  - 消除阻塞渲染的資源');
    recommendations.push('  - 內聯關鍵 CSS');
    recommendations.push('  - 延遲載入非關鍵 JavaScript');
  }

  // JS 大小建議
  if (metrics.jsSize > 500 * 1024) {
    recommendations.push(`JavaScript 總大小 ${(metrics.jsSize / 1024).toFixed(0)}KB，建議：`);
    recommendations.push('  - 使用代碼分割 (Code Splitting)');
    recommendations.push('  - 移除未使用的程式碼 (Tree Shaking)');
    recommendations.push('  - 壓縮並最小化 JavaScript');
  }

  // 圖片大小建議
  if (metrics.imageSize > 1024 * 1024) {
    recommendations.push(`圖片總大小 ${(metrics.imageSize / 1024 / 1024).toFixed(2)}MB，建議：`);
    recommendations.push('  - 使用現代圖片格式 (WebP, AVIF)');
    recommendations.push('  - 實施響應式圖片 (srcset)');
    recommendations.push('  - 延遲載入非首屏圖片 (loading="lazy")');
  }

  // 請求數建議
  if (metrics.requestCount > 50) {
    recommendations.push(`請求數量 ${metrics.requestCount} 個，建議：`);
    recommendations.push('  - 合併小型資源文件');
    recommendations.push('  - 使用 HTTP/2 多工傳輸');
    recommendations.push('  - 實施資源快取策略');
  }

  if (recommendations.length === 0) {
    recommendations.push('效能指標良好，繼續保持！');
  }

  return recommendations;
}

/**
 * 計算效能分數
 */
function calculateScore(metrics: PerformanceMetrics): number {
  let score = 100;

  // LCP 評分 (目標 < 2.5s)
  if (metrics.LCP > 4000) score -= 25;
  else if (metrics.LCP > 2500) score -= 15;
  else if (metrics.LCP > 1800) score -= 5;

  // CLS 評分 (目標 < 0.1)
  if (metrics.CLS > 0.25) score -= 25;
  else if (metrics.CLS > 0.1) score -= 15;
  else if (metrics.CLS > 0.05) score -= 5;

  // FCP 評分 (目標 < 1.8s)
  if (metrics.FCP > 3000) score -= 20;
  else if (metrics.FCP > 1800) score -= 10;
  else if (metrics.FCP > 1000) score -= 3;

  // 資源大小評分
  const totalMB = metrics.totalSize / 1024 / 1024;
  if (totalMB > 5) score -= 15;
  else if (totalMB > 2) score -= 8;
  else if (totalMB > 1) score -= 3;

  return Math.max(0, score);
}

/**
 * 效能測試規格
 */
export async function runPerformanceTests(urls: string[]): Promise<PerformanceReport[]> {
  const reports: PerformanceReport[] = [];

  for (const url of urls) {
    console.log(`Analyzing: ${url}`);
    const report = await analyzePerformance(url);
    reports.push(report);

    console.log(`Score: ${report.score}/100`);
    console.log(`LCP: ${report.metrics.LCP}ms`);
    console.log(`CLS: ${report.metrics.CLS}`);
    console.log('---');
  }

  return reports;
}
```

**效能測試範例：**
```typescript
// tests/e2e/specs/performance.spec.ts
import { analyzePerformance, PerformanceReport } from '../utils/performance';
import { testConfig } from '../config/puppeteer.config';

describe('Performance Tests', () => {
  let report: PerformanceReport;

  beforeAll(async () => {
    report = await analyzePerformance(testConfig.baseUrl);
  }, 60000);

  describe('Core Web Vitals', () => {
    it('LCP should be under 2.5 seconds', () => {
      expect(report.metrics.LCP).toBeLessThan(2500);
    });

    it('CLS should be under 0.1', () => {
      expect(report.metrics.CLS).toBeLessThan(0.1);
    });

    it('FCP should be under 1.8 seconds', () => {
      expect(report.metrics.FCP).toBeLessThan(1800);
    });
  });

  describe('Resource Optimization', () => {
    it('JavaScript bundle should be under 500KB', () => {
      expect(report.metrics.jsSize).toBeLessThan(500 * 1024);
    });

    it('Total page size should be under 2MB', () => {
      expect(report.metrics.totalSize).toBeLessThan(2 * 1024 * 1024);
    });

    it('Should have no failed requests', () => {
      expect(report.metrics.failedRequests).toBe(0);
    });
  });

  describe('Performance Score', () => {
    it('Should have performance score above 80', () => {
      expect(report.score).toBeGreaterThanOrEqual(80);
    });
  });
});
```

---

### /puppeteer:a11y
**無障礙測試**

執行流程：
1. 使用 axe-core 進行無障礙掃描
2. 檢查 WCAG 2.1 合規性
3. 生成無障礙報告

**安裝依賴：**
```bash
npm install @axe-core/puppeteer axe-core --save-dev
```

**無障礙測試工具：**
```typescript
// tests/utils/accessibility.ts
import puppeteer, { Browser, Page } from 'puppeteer';
import { AxePuppeteer } from '@axe-core/puppeteer';
import { Result, NodeResult } from 'axe-core';
import { puppeteerConfig, testConfig } from '../config/puppeteer.config';

export interface A11yViolation {
  id: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
  help: string;
  helpUrl: string;
  nodes: Array<{
    target: string[];
    html: string;
    failureSummary: string;
  }>;
}

export interface A11yReport {
  url: string;
  timestamp: Date;
  violations: A11yViolation[];
  passes: number;
  incomplete: number;
  score: number;
  wcagLevel: 'A' | 'AA' | 'AAA';
}

/**
 * 執行無障礙掃描
 */
export async function runA11yAudit(
  url: string,
  options: {
    wcagLevel?: 'A' | 'AA' | 'AAA';
    includeIncomplete?: boolean;
  } = {}
): Promise<A11yReport> {
  const { wcagLevel = 'AA', includeIncomplete = false } = options;

  const browser = await puppeteer.launch(puppeteerConfig);
  const page = await browser.newPage();

  await page.goto(url, { waitUntil: 'networkidle0' });

  // 配置 axe-core 標籤
  const tags = [`wcag2${wcagLevel.toLowerCase()}`, 'wcag21aa', 'best-practice'];

  const results = await new AxePuppeteer(page)
    .withTags(tags)
    .analyze();

  await browser.close();

  const violations: A11yViolation[] = results.violations.map(v => ({
    id: v.id,
    impact: v.impact as A11yViolation['impact'],
    description: v.description,
    help: v.help,
    helpUrl: v.helpUrl,
    nodes: v.nodes.map(n => ({
      target: n.target as string[],
      html: n.html,
      failureSummary: n.failureSummary || ''
    }))
  }));

  // 計算分數
  const totalChecks = results.passes.length + results.violations.length;
  const score = totalChecks > 0
    ? Math.round((results.passes.length / totalChecks) * 100)
    : 100;

  return {
    url,
    timestamp: new Date(),
    violations,
    passes: results.passes.length,
    incomplete: results.incomplete.length,
    score,
    wcagLevel
  };
}

/**
 * 檢查特定元素的無障礙性
 */
export async function auditElement(
  page: Page,
  selector: string
): Promise<A11yViolation[]> {
  const results = await new AxePuppeteer(page)
    .include(selector)
    .analyze();

  return results.violations.map(v => ({
    id: v.id,
    impact: v.impact as A11yViolation['impact'],
    description: v.description,
    help: v.help,
    helpUrl: v.helpUrl,
    nodes: v.nodes.map(n => ({
      target: n.target as string[],
      html: n.html,
      failureSummary: n.failureSummary || ''
    }))
  }));
}

/**
 * 生成無障礙報告
 */
export function generateA11yReport(report: A11yReport): string {
  let output = `# 無障礙測試報告\n\n`;
  output += `**URL:** ${report.url}\n`;
  output += `**測試時間:** ${report.timestamp.toISOString()}\n`;
  output += `**WCAG 等級:** ${report.wcagLevel}\n`;
  output += `**分數:** ${report.score}/100\n\n`;

  output += `## 摘要\n\n`;
  output += `- 通過檢查: ${report.passes}\n`;
  output += `- 違規項目: ${report.violations.length}\n`;
  output += `- 待確認項目: ${report.incomplete}\n\n`;

  if (report.violations.length > 0) {
    output += `## 違規詳情\n\n`;

    // 按嚴重程度分組
    const critical = report.violations.filter(v => v.impact === 'critical');
    const serious = report.violations.filter(v => v.impact === 'serious');
    const moderate = report.violations.filter(v => v.impact === 'moderate');
    const minor = report.violations.filter(v => v.impact === 'minor');

    if (critical.length > 0) {
      output += `### 嚴重 (Critical)\n\n`;
      output += formatViolations(critical);
    }

    if (serious.length > 0) {
      output += `### 重要 (Serious)\n\n`;
      output += formatViolations(serious);
    }

    if (moderate.length > 0) {
      output += `### 中等 (Moderate)\n\n`;
      output += formatViolations(moderate);
    }

    if (minor.length > 0) {
      output += `### 輕微 (Minor)\n\n`;
      output += formatViolations(minor);
    }
  } else {
    output += `## 恭喜！未發現無障礙問題\n`;
  }

  return output;
}

function formatViolations(violations: A11yViolation[]): string {
  let output = '';

  for (const v of violations) {
    output += `#### ${v.id}\n\n`;
    output += `**說明:** ${v.description}\n\n`;
    output += `**建議:** ${v.help}\n\n`;
    output += `**參考:** [${v.helpUrl}](${v.helpUrl})\n\n`;

    output += `**受影響元素:**\n\n`;
    for (const node of v.nodes) {
      output += `- \`${node.target.join(' > ')}\`\n`;
      output += `  \`\`\`html\n  ${node.html}\n  \`\`\`\n`;
      if (node.failureSummary) {
        output += `  ${node.failureSummary}\n`;
      }
    }
    output += '\n';
  }

  return output;
}
```

**無障礙測試範例：**
```typescript
// tests/e2e/specs/accessibility.spec.ts
import { runA11yAudit, A11yReport } from '../utils/accessibility';
import { testConfig } from '../config/puppeteer.config';

describe('Accessibility Tests', () => {
  describe('Homepage A11y', () => {
    let report: A11yReport;

    beforeAll(async () => {
      report = await runA11yAudit(testConfig.baseUrl, { wcagLevel: 'AA' });
    }, 60000);

    it('should have no critical violations', () => {
      const critical = report.violations.filter(v => v.impact === 'critical');
      expect(critical).toHaveLength(0);
    });

    it('should have no serious violations', () => {
      const serious = report.violations.filter(v => v.impact === 'serious');
      expect(serious).toHaveLength(0);
    });

    it('should have accessibility score above 90', () => {
      expect(report.score).toBeGreaterThanOrEqual(90);
    });
  });

  describe('Form A11y', () => {
    let report: A11yReport;

    beforeAll(async () => {
      report = await runA11yAudit(`${testConfig.baseUrl}/contact`);
    }, 60000);

    it('should have accessible form labels', () => {
      const labelViolations = report.violations.filter(
        v => v.id.includes('label')
      );
      expect(labelViolations).toHaveLength(0);
    });

    it('should have proper focus indicators', () => {
      const focusViolations = report.violations.filter(
        v => v.id.includes('focus')
      );
      expect(focusViolations).toHaveLength(0);
    });
  });

  describe('Color Contrast', () => {
    let report: A11yReport;

    beforeAll(async () => {
      report = await runA11yAudit(testConfig.baseUrl);
    }, 60000);

    it('should meet color contrast requirements', () => {
      const contrastViolations = report.violations.filter(
        v => v.id === 'color-contrast'
      );
      expect(contrastViolations).toHaveLength(0);
    });
  });
});
```

---

### /puppeteer:responsive [breakpoints]
**響應式設計測試**

參數：
- `breakpoints`：斷點設定（預設使用常見斷點）

執行流程：
1. 在多個視窗尺寸下測試頁面
2. 擷取各尺寸截圖
3. 驗證元素可見性與佈局

**響應式測試工具：**
```typescript
// tests/utils/responsive.ts
import puppeteer, { Browser, Page, Viewport } from 'puppeteer';
import { puppeteerConfig, testConfig } from '../config/puppeteer.config';
import { takeScreenshot } from './screenshot';

export interface Breakpoint {
  name: string;
  width: number;
  height: number;
  deviceScaleFactor?: number;
}

export const defaultBreakpoints: Breakpoint[] = [
  { name: 'mobile-s', width: 320, height: 568 },
  { name: 'mobile-m', width: 375, height: 667 },
  { name: 'mobile-l', width: 425, height: 896 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'laptop', width: 1024, height: 768 },
  { name: 'laptop-l', width: 1440, height: 900 },
  { name: 'desktop', width: 1920, height: 1080 },
  { name: '4k', width: 2560, height: 1440 }
];

export interface ResponsiveTestResult {
  breakpoint: Breakpoint;
  screenshotPath: string;
  visibilityChecks: Array<{
    selector: string;
    visible: boolean;
    expected: boolean;
    passed: boolean;
  }>;
  layoutChecks: Array<{
    description: string;
    passed: boolean;
    details?: string;
  }>;
}

/**
 * 在所有斷點測試頁面
 */
export async function testResponsive(
  url: string,
  options: {
    breakpoints?: Breakpoint[];
    visibilityRules?: Array<{
      selector: string;
      visibleAt: string[]; // breakpoint names
    }>;
    layoutChecks?: Array<{
      description: string;
      check: (page: Page, breakpoint: Breakpoint) => Promise<boolean>;
    }>;
  } = {}
): Promise<ResponsiveTestResult[]> {
  const {
    breakpoints = defaultBreakpoints,
    visibilityRules = [],
    layoutChecks = []
  } = options;

  const browser = await puppeteer.launch(puppeteerConfig);
  const results: ResponsiveTestResult[] = [];

  for (const breakpoint of breakpoints) {
    const page = await browser.newPage();
    await page.setViewport({
      width: breakpoint.width,
      height: breakpoint.height,
      deviceScaleFactor: breakpoint.deviceScaleFactor || 1
    });

    await page.goto(url, { waitUntil: 'networkidle0' });

    // 截圖
    const screenshotPath = await takeScreenshot(
      page,
      `responsive-${breakpoint.name}`
    );

    // 可見性檢查
    const visibilityChecks = await Promise.all(
      visibilityRules.map(async rule => {
        const element = await page.$(rule.selector);
        const visible = element
          ? await page.evaluate(
              el => getComputedStyle(el).display !== 'none' &&
                     getComputedStyle(el).visibility !== 'hidden',
              element
            )
          : false;

        const expected = rule.visibleAt.includes(breakpoint.name);

        return {
          selector: rule.selector,
          visible,
          expected,
          passed: visible === expected
        };
      })
    );

    // 佈局檢查
    const layoutResults = await Promise.all(
      layoutChecks.map(async check => {
        try {
          const passed = await check.check(page, breakpoint);
          return {
            description: check.description,
            passed
          };
        } catch (error) {
          return {
            description: check.description,
            passed: false,
            details: String(error)
          };
        }
      })
    );

    results.push({
      breakpoint,
      screenshotPath,
      visibilityChecks,
      layoutChecks: layoutResults
    });

    await page.close();
  }

  await browser.close();
  return results;
}

/**
 * 常用的佈局檢查函數
 */
export const layoutCheckHelpers = {
  /**
   * 檢查元素是否水平堆疊（行內排列）
   */
  async isHorizontalStack(page: Page, selector: string): Promise<boolean> {
    const elements = await page.$$(selector);
    if (elements.length < 2) return true;

    const boxes = await Promise.all(
      elements.map(el => el.boundingBox())
    );

    for (let i = 1; i < boxes.length; i++) {
      if (boxes[i]!.y > boxes[i - 1]!.y + boxes[i - 1]!.height) {
        return false; // 垂直排列
      }
    }
    return true;
  },

  /**
   * 檢查元素是否垂直堆疊
   */
  async isVerticalStack(page: Page, selector: string): Promise<boolean> {
    const elements = await page.$$(selector);
    if (elements.length < 2) return true;

    const boxes = await Promise.all(
      elements.map(el => el.boundingBox())
    );

    for (let i = 1; i < boxes.length; i++) {
      if (boxes[i]!.x > boxes[i - 1]!.x + boxes[i - 1]!.width) {
        return false; // 水平排列
      }
    }
    return true;
  },

  /**
   * 檢查元素是否超出視窗
   */
  async isWithinViewport(page: Page, selector: string): Promise<boolean> {
    const viewport = page.viewport();
    if (!viewport) return false;

    const element = await page.$(selector);
    if (!element) return false;

    const box = await element.boundingBox();
    if (!box) return false;

    return (
      box.x >= 0 &&
      box.y >= 0 &&
      box.x + box.width <= viewport.width &&
      box.y + box.height <= viewport.height
    );
  },

  /**
   * 檢查無水平捲軸
   */
  async noHorizontalScroll(page: Page): Promise<boolean> {
    return page.evaluate(() => {
      return document.documentElement.scrollWidth <= window.innerWidth;
    });
  }
};
```

**響應式測試範例：**
```typescript
// tests/e2e/specs/responsive.spec.ts
import {
  testResponsive,
  defaultBreakpoints,
  layoutCheckHelpers
} from '../utils/responsive';
import { testConfig } from '../config/puppeteer.config';

describe('Responsive Design Tests', () => {
  it('should display correctly across all breakpoints', async () => {
    const results = await testResponsive(testConfig.baseUrl, {
      visibilityRules: [
        {
          selector: '[data-testid="mobile-menu"]',
          visibleAt: ['mobile-s', 'mobile-m', 'mobile-l']
        },
        {
          selector: '[data-testid="desktop-nav"]',
          visibleAt: ['tablet', 'laptop', 'laptop-l', 'desktop', '4k']
        }
      ],
      layoutChecks: [
        {
          description: 'No horizontal scroll',
          check: async (page) => layoutCheckHelpers.noHorizontalScroll(page)
        },
        {
          description: 'Navigation items stack on mobile',
          check: async (page, breakpoint) => {
            if (!['mobile-s', 'mobile-m', 'mobile-l'].includes(breakpoint.name)) {
              return true;
            }
            return layoutCheckHelpers.isVerticalStack(page, '[data-testid="nav-item"]');
          }
        }
      ]
    });

    for (const result of results) {
      // 檢查可見性規則
      for (const check of result.visibilityChecks) {
        expect(check.passed).toBe(true);
      }

      // 檢查佈局規則
      for (const check of result.layoutChecks) {
        expect(check.passed).toBe(true);
      }
    }
  }, 120000);
});
```

---

### /puppeteer:report
**生成測試報告**

執行流程：
1. 彙整所有測試結果
2. 生成 HTML 報告
3. 包含截圖、效能圖表、無障礙摘要

**報告生成器：**
```typescript
// tests/utils/report-generator.ts
import fs from 'fs/promises';
import path from 'path';
import { PerformanceReport } from './performance';
import { A11yReport } from './accessibility';
import { ComparisonResult } from './screenshot';

export interface TestReport {
  timestamp: Date;
  projectName: string;
  performance?: PerformanceReport[];
  accessibility?: A11yReport[];
  visualRegression?: Array<{
    name: string;
    result: ComparisonResult;
  }>;
  summary: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
  };
}

export async function generateHtmlReport(report: TestReport): Promise<string> {
  const html = `
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UI 測試報告 - ${report.projectName}</title>
  <style>
    :root {
      --color-pass: #22c55e;
      --color-fail: #ef4444;
      --color-warn: #f59e0b;
      --color-bg: #f8fafc;
      --color-card: #ffffff;
      --color-text: #1e293b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--color-bg);
      color: var(--color-text);
      line-height: 1.6;
      padding: 2rem;
    }

    .container { max-width: 1200px; margin: 0 auto; }

    header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
      border-radius: 1rem;
      margin-bottom: 2rem;
    }

    h1 { font-size: 2rem; margin-bottom: 0.5rem; }

    .meta { opacity: 0.9; font-size: 0.9rem; }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .summary-card {
      background: var(--color-card);
      padding: 1.5rem;
      border-radius: 0.75rem;
      text-align: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .summary-card .value {
      font-size: 2.5rem;
      font-weight: bold;
    }

    .summary-card .label { color: #64748b; font-size: 0.875rem; }

    .summary-card.pass .value { color: var(--color-pass); }
    .summary-card.fail .value { color: var(--color-fail); }

    section {
      background: var(--color-card);
      padding: 1.5rem;
      border-radius: 0.75rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    section h2 {
      font-size: 1.25rem;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid #e2e8f0;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .metric {
      padding: 1rem;
      background: #f8fafc;
      border-radius: 0.5rem;
    }

    .metric .name { font-size: 0.875rem; color: #64748b; }
    .metric .value { font-size: 1.5rem; font-weight: 600; }
    .metric.good .value { color: var(--color-pass); }
    .metric.warn .value { color: var(--color-warn); }
    .metric.bad .value { color: var(--color-fail); }

    .screenshot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1rem;
    }

    .screenshot-card {
      border: 1px solid #e2e8f0;
      border-radius: 0.5rem;
      overflow: hidden;
    }

    .screenshot-card img {
      width: 100%;
      height: auto;
      display: block;
    }

    .screenshot-card .info {
      padding: 0.75rem;
      background: #f8fafc;
    }

    .badge {
      display: inline-block;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 500;
    }

    .badge.pass { background: #dcfce7; color: #166534; }
    .badge.fail { background: #fee2e2; color: #991b1b; }

    .violation-list { list-style: none; }

    .violation-item {
      padding: 1rem;
      margin-bottom: 0.5rem;
      background: #fef2f2;
      border-left: 4px solid var(--color-fail);
      border-radius: 0 0.5rem 0.5rem 0;
    }

    .violation-item.critical { border-color: #7f1d1d; }
    .violation-item.serious { border-color: var(--color-fail); }
    .violation-item.moderate { border-color: var(--color-warn); }
    .violation-item.minor { border-color: #94a3b8; }

    code {
      background: #e2e8f0;
      padding: 0.2rem 0.4rem;
      border-radius: 0.25rem;
      font-size: 0.875rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>${report.projectName} - UI 測試報告</h1>
      <div class="meta">
        生成時間：${report.timestamp.toLocaleString('zh-TW')}
      </div>
    </header>

    <div class="summary-grid">
      <div class="summary-card">
        <div class="value">${report.summary.total}</div>
        <div class="label">總測試數</div>
      </div>
      <div class="summary-card pass">
        <div class="value">${report.summary.passed}</div>
        <div class="label">通過</div>
      </div>
      <div class="summary-card fail">
        <div class="value">${report.summary.failed}</div>
        <div class="label">失敗</div>
      </div>
      <div class="summary-card">
        <div class="value">${report.summary.skipped}</div>
        <div class="label">跳過</div>
      </div>
    </div>

    ${report.performance ? generatePerformanceSection(report.performance) : ''}
    ${report.accessibility ? generateA11ySection(report.accessibility) : ''}
    ${report.visualRegression ? generateVisualSection(report.visualRegression) : ''}
  </div>
</body>
</html>
  `;

  const reportPath = path.join('./tests/e2e/reports', `report-${Date.now()}.html`);
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, html);

  return reportPath;
}

function generatePerformanceSection(reports: PerformanceReport[]): string {
  return reports.map(report => `
    <section>
      <h2>效能分析 - ${report.url}</h2>
      <div class="metric-grid">
        <div class="metric ${report.metrics.LCP < 2500 ? 'good' : report.metrics.LCP < 4000 ? 'warn' : 'bad'}">
          <div class="name">LCP (Largest Contentful Paint)</div>
          <div class="value">${report.metrics.LCP.toFixed(0)}ms</div>
        </div>
        <div class="metric ${report.metrics.CLS < 0.1 ? 'good' : report.metrics.CLS < 0.25 ? 'warn' : 'bad'}">
          <div class="name">CLS (Cumulative Layout Shift)</div>
          <div class="value">${report.metrics.CLS.toFixed(3)}</div>
        </div>
        <div class="metric ${report.metrics.FCP < 1800 ? 'good' : report.metrics.FCP < 3000 ? 'warn' : 'bad'}">
          <div class="name">FCP (First Contentful Paint)</div>
          <div class="value">${report.metrics.FCP.toFixed(0)}ms</div>
        </div>
        <div class="metric">
          <div class="name">總分</div>
          <div class="value">${report.score}/100</div>
        </div>
      </div>
      ${report.recommendations.length > 0 ? `
        <h3 style="margin-top: 1rem;">優化建議</h3>
        <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
          ${report.recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>
      ` : ''}
    </section>
  `).join('');
}

function generateA11ySection(reports: A11yReport[]): string {
  return reports.map(report => `
    <section>
      <h2>無障礙測試 - ${report.url}</h2>
      <div class="metric-grid">
        <div class="metric ${report.score >= 90 ? 'good' : report.score >= 70 ? 'warn' : 'bad'}">
          <div class="name">無障礙分數</div>
          <div class="value">${report.score}/100</div>
        </div>
        <div class="metric">
          <div class="name">WCAG 等級</div>
          <div class="value">${report.wcagLevel}</div>
        </div>
        <div class="metric">
          <div class="name">違規項目</div>
          <div class="value">${report.violations.length}</div>
        </div>
      </div>
      ${report.violations.length > 0 ? `
        <h3 style="margin-top: 1rem;">違規詳情</h3>
        <ul class="violation-list">
          ${report.violations.map(v => `
            <li class="violation-item ${v.impact}">
              <strong>${v.id}</strong> (${v.impact})
              <p>${v.description}</p>
              <p><a href="${v.helpUrl}" target="_blank">了解更多</a></p>
            </li>
          `).join('')}
        </ul>
      ` : '<p style="margin-top: 1rem; color: #22c55e;">無發現無障礙問題</p>'}
    </section>
  `).join('');
}

function generateVisualSection(results: Array<{ name: string; result: ComparisonResult }>): string {
  return `
    <section>
      <h2>視覺回歸測試</h2>
      <div class="screenshot-grid">
        ${results.map(r => `
          <div class="screenshot-card">
            ${r.result.diffImagePath ? `<img src="${r.result.diffImagePath}" alt="${r.name} diff">` : ''}
            <div class="info">
              <strong>${r.name}</strong>
              <span class="badge ${r.result.match ? 'pass' : 'fail'}">
                ${r.result.match ? '通過' : `差異 ${r.result.percentage.toFixed(2)}%`}
              </span>
            </div>
          </div>
        `).join('')}
      </div>
    </section>
  `;
}
```

---

## CI/CD 整合

### GitHub Actions 配置

```yaml
# .github/workflows/ui-tests.yml
name: UI Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  ui-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Puppeteer browser
        run: npx puppeteer browsers install chrome

      - name: Start application
        run: npm run start &
        env:
          CI: true

      - name: Wait for application
        run: npx wait-on http://localhost:3000 -t 60000

      - name: Run visual regression tests
        run: npm run test:visual

      - name: Run performance tests
        run: npm run test:performance

      - name: Run accessibility tests
        run: npm run test:a11y

      - name: Upload test artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: |
            tests/e2e/screenshots/
            tests/e2e/reports/
          retention-days: 7

      - name: Upload baseline on main
        if: github.ref == 'refs/heads/main' && success()
        run: |
          # 更新基準截圖
          npm run test:update-baseline
          git config user.name github-actions
          git config user.email github-actions@github.com
          git add tests/e2e/screenshots/baseline/
          git commit -m "chore: update visual baselines" || exit 0
          git push
```

### package.json 腳本

```json
{
  "scripts": {
    "test:e2e": "jest --config jest.e2e.config.js",
    "test:visual": "jest --config jest.e2e.config.js --testPathPattern=visual",
    "test:performance": "jest --config jest.e2e.config.js --testPathPattern=performance",
    "test:a11y": "jest --config jest.e2e.config.js --testPathPattern=accessibility",
    "test:responsive": "jest --config jest.e2e.config.js --testPathPattern=responsive",
    "test:update-baseline": "UPDATE_BASELINE=true npm run test:visual",
    "test:report": "node scripts/generate-report.js"
  }
}
```

---

## 開發檢查清單

### Phase 1: 專案設置
- [ ] `/puppeteer:init` 初始化測試專案
- [ ] 安裝所有依賴
- [ ] 配置 Puppeteer 設定
- [ ] 建立目錄結構

### Phase 2: 核心測試
- [ ] `/puppeteer:test visual` 視覺回歸測試
- [ ] `/puppeteer:test functional` 功能測試
- [ ] `/puppeteer:screenshot capture` 截圖工具

### Phase 3: 效能與無障礙
- [ ] `/puppeteer:performance` 效能分析
- [ ] `/puppeteer:a11y` 無障礙測試
- [ ] `/puppeteer:responsive` 響應式測試

### Phase 4: CI/CD 整合
- [ ] 配置 GitHub Actions
- [ ] 設置自動化報告
- [ ] 整合測試到部署流程

---

## 最佳實踐

### 測試穩定性
1. **等待策略**：使用 `waitForSelector` 而非固定時間延遲
2. **重試機制**：對於 flaky 測試實施自動重試
3. **隔離測試**：每個測試使用獨立的瀏覽器上下文
4. **避免硬編碼**：使用 data-testid 而非 CSS 類別選擇器

### 效能優化
1. **並行執行**：使用 Jest 並行模式執行測試
2. **資源複用**：在測試套件間複用瀏覽器實例
3. **選擇性截圖**：僅對關鍵畫面進行視覺測試
4. **快取基準**：在 CI 中快取基準截圖

### 維護性
1. **Page Object Pattern**：封裝頁面操作
2. **模組化工具**：拆分可重用的測試工具函數
3. **清晰命名**：測試與截圖使用描述性名稱
4. **文檔同步**：保持測試文檔與實作同步

---

## 版本資訊

```yaml
version: "1.0"
last_updated: "2025-12-29"
changelog:
  - version: "1.0"
    date: "2025-12-29"
    changes:
      - "初始版本建立"
      - "包含視覺回歸測試"
      - "效能分析與 Core Web Vitals"
      - "無障礙測試 (axe-core)"
      - "響應式設計測試"
      - "CI/CD 整合配置"
      - "HTML 報告生成器"
```
