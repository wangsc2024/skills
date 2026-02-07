---
name: daily-greeting-digest
description: >
  每日時段摘要助手。當使用者以簡短問候觸發對話時（如「hi」、「hello」、「嗨」、「早安」、「午安」、「下午好」、「哈囉」），
  根據當前時間（Asia/Taipei）自動回覆對應時段的資訊摘要。
  早上(6-11)：今日待辦事項、屏東天氣、前一天屏東新聞。
  中午(11-14)：Hacker News 前三則（翻譯成中文）、00637L 滬深三百正二 ETF 股價。
  下午(14-18)：當日屏東新聞、明日屏東天氣。
  其他時段：簡單問候。觸發詞：hi, hello, 嗨, 哈囉, 早安, 午安, 下午好, 安安, hey。
---

# 每日時段摘要

使用者以簡短問候開啟對話時，根據 Asia/Taipei 時區判斷時段，回覆對應資訊。

## 時段判斷

```
早上：06:00 - 10:59
中午：11:00 - 13:59
下午：14:00 - 17:59
其他：18:00 - 05:59 → 簡單問候，不主動查資料
```

## 早上摘要（06:00 - 10:59）

依序取得並呈現：

1. **今日待辦** — 使用 todoist skill（`scripts/todoist.py list`），需先 `export TODOIST_API_TOKEN`
2. **屏東今日天氣** — 使用 `weather_fetch` 工具（屏東：lat 22.6827, lng 120.4897）
3. **前一天屏東新聞** — 使用 pingtung-news skill 的 `pingtung_news_by_date`，start/end 設為昨天日期

## 中午摘要（11:00 - 13:59）

依序取得並呈現：

1. **Hacker News 前三則**（翻譯成繁體中文）
   - 用 `web_fetch` 取得 `https://hacker-news.firebaseio.com/v0/topstories.json`，取前三個 ID
   - 對每個 ID 用 `web_fetch` 取 `https://hacker-news.firebaseio.com/v0/item/{id}.json` 取得標題與連結
   - 將標題翻譯成繁體中文，附上原文連結
2. **00637L 滬深三百正二 ETF 股價** — 用 `web_search` 搜尋「00637L 股價」

## 下午摘要（14:00 - 17:59）

依序取得並呈現：

1. **今日屏東新聞** — 使用 pingtung-news skill 的 `pingtung_news_by_date`，start/end 設為今天日期
2. **明日屏東天氣** — 使用 `weather_fetch` 工具（屏東：lat 22.6827, lng 120.4897），呈現明日預報

## 其他時段（18:00 - 05:59）

簡單友善問候，不主動查詢資料。若使用者另有需求再回應。

## 輸出格式

使用溫暖的問候開場，然後分段呈現各項資訊。範例結構：

```
早安 wangsc！☀️

📋 **今日待辦**
🔴 完成報告
🟡 回覆客戶信件

🌤️ **屏東天氣**
今天晴，28°C / 22°C，降雨機率 10%

📰 **昨日屏東新聞**
| # | 標題 |
|---|------|
| 1 | xxx |
| 2 | xxx |
```

## 注意事項

- 任一資料來源失敗時，跳過該項繼續呈現其餘內容，不要因單一失敗而中斷整個摘要
- 屏東新聞 API 可能不穩定，遵循 pingtung-news skill 的重試機制
- ETF 股價在非交易時段可能顯示前一交易日收盤價，註明資料時間
- Hacker News 標題翻譯保持簡潔，附原始連結供參考
