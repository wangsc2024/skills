# Pingtung Travel Planner Skill

## Description
以「屏東縣」為唯一範圍的旅遊規劃 Skill，整合景點、交通、活動、展覽與票券資訊，自動生成可執行的一日到多日行程，適用於自由行、親子旅遊與無車族。

## When to Use
- 規劃屏東一日遊、二日一夜、三天兩夜以上行程
- 查詢屏東近期活動、展覽、節慶
- 規劃墾丁、小琉球、屏東市區與原鄉動線
- 整合大眾運輸、觀光巴士與船班

## Core Concepts

### 區域導向規劃（Region-first Planning）
- 屏東市區
- 墾丁半島
- 小琉球
- 沿山原鄉（霧台、三地門）

### 主題導向行程（Theme-based Itinerary）
- 親子｜文化｜生態｜海洋｜溫泉｜美食

### 交通優先原則
- 台鐵 + 屏東客運
- 台灣好行（墾丁快線、185 沿山線）
- 東港 ↔ 小琉球船班

## Quick Reference

### 常用查詢 API
| API | 用途 |
|-----|------|
| get_attractions | 查詢景點 |
| get_events | 查詢活動 / 展覽 |
| get_transport | 交通建議 |
| get_tickets | 票券 / 體驗 |
| plan_itinerary | 自動排行程 |

## Code Examples

```json
{
  "days": 3,
  "regions": ["小琉球", "墾丁"],
  "theme": ["海洋", "親子"],
  "transport": "public"
}
```

## Common Pitfalls
- ❌ 同一天市區＋墾丁 → ✅ 依區域分天安排
- ❌ 小琉球當天來回 → ✅ 至少安排一晚住宿

## Related Resources
- https://www.i-pingtung.com/
- https://www.ptcg.gov.tw/traffic/
- https://www.taiwantrip.com.tw/

## Metadata
```json
{
  "skill_name": "pingtung-travel-planner",
  "version": "1.0.0",
  "source_scope": "Pingtung County, Taiwan"
}
```
