# Skill Seekers 參考文件索引

## 核心指南

| 文件 | 說明 |
|------|------|
| [scraping.md](./scraping.md) | 網頁抓取完整指南 |
| [github.md](./github.md) | GitHub 倉庫抓取指南 |
| [categorization.md](./categorization.md) | 內容自動分類說明 |
| [output-formats.md](./output-formats.md) | 輸出格式規範 |

## 快速導航

### 我想抓取文件網站
→ 查看 [scraping.md](./scraping.md)

### 我想分析 GitHub 倉庫
→ 查看 [github.md](./github.md)

### 我想了解分類邏輯
→ 查看 [categorization.md](./categorization.md)

### 我想自訂輸出格式
→ 查看 [output-formats.md](./output-formats.md)

## 常見問題

### Q: 抓取速度很慢？
A: 使用 `--async` 模式可提升 2-3 倍速度

### Q: 某些頁面抓取不到？
A: 可能是 JavaScript 動態載入，嘗試使用 llms.txt 或請使用者貼上內容

### Q: 分類不準確？
A: 可以手動調整 `categories` 設定，或在抓取後手動修改
