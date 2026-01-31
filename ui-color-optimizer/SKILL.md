---
name: ui-color-optimizer
description: |
  Optimize UI color schemes for accessibility, aesthetics, and brand consistency. Generates harmonious color palettes, checks WCAG contrast ratios, and provides CSS/Tailwind variables.
  Use when: designing color schemes, fixing contrast issues, creating dark mode, building design systems, or when user mentions 配色, color, 顏色, palette, dark mode, 深色模式, contrast, 對比度, WCAG.
  Triggers: "color", "配色", "顏色", "palette", "dark mode", "深色模式", "對比度", "WCAG", "調色盤"
version: 1.0.0
---

# UI Color Optimizer

Create accessible, harmonious, and visually appealing color schemes for web applications.

## Color Theory Fundamentals

### Harmony Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Monochromatic** | Single hue, varying saturation/lightness | Elegant, minimal UI |
| **Analogous** | Adjacent hues (30° apart) | Harmonious, calm feel |
| **Complementary** | Opposite hues (180° apart) | High contrast, vibrant |
| **Split-Complementary** | Base + two adjacent to complement | Balanced contrast |
| **Triadic** | Three hues (120° apart) | Dynamic, balanced |
| **Tetradic** | Four hues (rectangle) | Rich, complex palette |

### 60-30-10 Rule

```
60% - Dominant color (backgrounds, large areas)
30% - Secondary color (cards, sections)
10% - Accent color (CTAs, highlights)
```

## Accessibility (WCAG 2.1)

### Contrast Ratio Requirements

| Level | Normal Text | Large Text | UI Components |
|-------|-------------|------------|---------------|
| AA | 4.5:1 | 3:1 | 3:1 |
| AAA | 7:1 | 4.5:1 | 4.5:1 |

### Contrast Calculation

```javascript
// Calculate relative luminance
function luminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map(c => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

// Calculate contrast ratio
function contrastRatio(rgb1, rgb2) {
  const l1 = luminance(...rgb1);
  const l2 = luminance(...rgb2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

// Example
contrastRatio([255, 255, 255], [0, 0, 0]); // 21:1 (max)
contrastRatio([255, 255, 255], [118, 118, 118]); // 4.5:1 (AA pass)
```

### Common Contrast Fixes

```css
/* ❌ Fails AA (2.8:1) */
.bad {
  color: #767676;
  background: #ffffff;
}

/* ✅ Passes AA (4.5:1) */
.good {
  color: #595959;
  background: #ffffff;
}

/* ❌ Fails AA (3.1:1) */
.bad-button {
  color: #ffffff;
  background: #6fa8dc;
}

/* ✅ Passes AA (4.5:1) */
.good-button {
  color: #ffffff;
  background: #2563eb;
}
```

## Professional Color Palettes

### Neutral Grays (Most Important)

```css
:root {
  /* Warm Gray (friendly, approachable) */
  --gray-50: #fafaf9;
  --gray-100: #f5f5f4;
  --gray-200: #e7e5e4;
  --gray-300: #d6d3d1;
  --gray-400: #a8a29e;
  --gray-500: #78716c;
  --gray-600: #57534e;
  --gray-700: #44403c;
  --gray-800: #292524;
  --gray-900: #1c1917;

  /* Cool Gray (professional, tech) */
  --slate-50: #f8fafc;
  --slate-100: #f1f5f9;
  --slate-200: #e2e8f0;
  --slate-300: #cbd5e1;
  --slate-400: #94a3b8;
  --slate-500: #64748b;
  --slate-600: #475569;
  --slate-700: #334155;
  --slate-800: #1e293b;
  --slate-900: #0f172a;

  /* Neutral Gray (balanced) */
  --zinc-50: #fafafa;
  --zinc-100: #f4f4f5;
  --zinc-200: #e4e4e7;
  --zinc-300: #d4d4d8;
  --zinc-400: #a1a1aa;
  --zinc-500: #71717a;
  --zinc-600: #52525b;
  --zinc-700: #3f3f46;
  --zinc-800: #27272a;
  --zinc-900: #18181b;
}
```

### Primary Color Scales

```css
:root {
  /* Blue (trust, stability) */
  --blue-50: #eff6ff;
  --blue-100: #dbeafe;
  --blue-200: #bfdbfe;
  --blue-300: #93c5fd;
  --blue-400: #60a5fa;
  --blue-500: #3b82f6;
  --blue-600: #2563eb;  /* Primary action */
  --blue-700: #1d4ed8;
  --blue-800: #1e40af;
  --blue-900: #1e3a8a;

  /* Indigo (creative, modern) */
  --indigo-500: #6366f1;
  --indigo-600: #4f46e5;

  /* Violet (luxury, innovation) */
  --violet-500: #8b5cf6;
  --violet-600: #7c3aed;

  /* Teal (health, nature) */
  --teal-500: #14b8a6;
  --teal-600: #0d9488;
}
```

### Semantic Colors

```css
:root {
  /* Success */
  --success-light: #dcfce7;
  --success: #22c55e;
  --success-dark: #15803d;

  /* Warning */
  --warning-light: #fef3c7;
  --warning: #f59e0b;
  --warning-dark: #b45309;

  /* Error */
  --error-light: #fee2e2;
  --error: #ef4444;
  --error-dark: #b91c1c;

  /* Info */
  --info-light: #dbeafe;
  --info: #3b82f6;
  --info-dark: #1d4ed8;
}
```

## Dark Mode Strategy

### Color Mapping

```css
/* Light Mode */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f9fafb;
  --bg-tertiary: #f3f4f6;

  --text-primary: #111827;
  --text-secondary: #4b5563;
  --text-tertiary: #9ca3af;

  --border: #e5e7eb;
  --border-focus: #3b82f6;

  --surface: #ffffff;
  --surface-hover: #f9fafb;
}

/* Dark Mode */
[data-theme="dark"] {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;

  --text-primary: #f9fafb;
  --text-secondary: #cbd5e1;
  --text-tertiary: #64748b;

  --border: #334155;
  --border-focus: #60a5fa;

  --surface: #1e293b;
  --surface-hover: #334155;
}
```

### Dark Mode Best Practices

```markdown
1. **Don't invert** - Dark mode ≠ inverted colors
2. **Reduce saturation** - Bright colors are harsh on dark backgrounds
3. **Adjust elevation** - Use lighter surfaces for elevated elements
4. **Keep contrast** - Maintain WCAG compliance
5. **Test images** - Add subtle backgrounds behind transparent images
```

```css
/* Adjust primary for dark mode */
:root {
  --primary: #2563eb;  /* Blue 600 */
}

[data-theme="dark"] {
  --primary: #60a5fa;  /* Blue 400 - less saturated */
}
```

## Tailwind CSS Configuration

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          DEFAULT: '#2563eb',
        },
        surface: {
          DEFAULT: 'var(--surface)',
          hover: 'var(--surface-hover)',
        },
      },
    },
  },
};
```

## Color Palette Generator

### From Brand Color

```javascript
const chroma = require('chroma-js');

function generatePalette(baseColor) {
  const base = chroma(baseColor);
  const hue = base.get('hsl.h');

  return {
    50: chroma.hsl(hue, 0.95, 0.97).hex(),
    100: chroma.hsl(hue, 0.90, 0.92).hex(),
    200: chroma.hsl(hue, 0.85, 0.85).hex(),
    300: chroma.hsl(hue, 0.80, 0.75).hex(),
    400: chroma.hsl(hue, 0.75, 0.60).hex(),
    500: base.hex(),
    600: base.darken(0.5).hex(),
    700: base.darken(1).hex(),
    800: base.darken(1.5).hex(),
    900: base.darken(2).hex(),
  };
}

// Generate from brand blue
generatePalette('#2563eb');
```

### Complementary Accent

```javascript
function getComplementary(hex) {
  const color = chroma(hex);
  const hue = color.get('hsl.h');
  const complementHue = (hue + 180) % 360;
  return chroma.hsl(complementHue, color.get('hsl.s'), color.get('hsl.l')).hex();
}

// From blue #2563eb → orange #eb8925
getComplementary('#2563eb');
```

## Industry Color Associations

| Industry | Primary Colors | Mood |
|----------|---------------|------|
| **Finance** | Blue, Navy, Green | Trust, stability |
| **Health** | Teal, Green, Blue | Clean, calm |
| **Tech** | Blue, Purple, Cyan | Innovation, modern |
| **Food** | Red, Orange, Yellow | Appetite, energy |
| **Luxury** | Black, Gold, Purple | Premium, elegant |
| **Eco** | Green, Brown, Earth | Natural, sustainable |
| **Creative** | Purple, Pink, Multi | Artistic, bold |

## Quick Fixes Cheat Sheet

### Problem → Solution

```markdown
**"Text is hard to read"**
→ Increase contrast ratio to 4.5:1 minimum
→ Use darker gray (#374151) instead of light gray (#9ca3af)

**"Colors feel random"**
→ Use analogous colors (adjacent on color wheel)
→ Stick to one primary + one accent

**"Too many colors"**
→ Limit to 3-5 colors max
→ Use shades of same hue for variety

**"Dark mode looks washed out"**
→ Reduce saturation on bright colors
→ Add subtle colored tints to surfaces

**"UI feels flat"**
→ Add subtle shadows
→ Use slight color differences for depth

**"Buttons don't stand out"**
→ Use complementary color for CTAs
→ Ensure 3:1 contrast with surrounding elements

**"Error states unclear"**
→ Use red with adequate contrast
→ Add icon, not just color (accessibility)
```

## Checklist

### Accessibility
- [ ] Text contrast ≥ 4.5:1 (AA)
- [ ] Large text contrast ≥ 3:1
- [ ] UI component contrast ≥ 3:1
- [ ] Focus states visible
- [ ] Not relying on color alone

### Consistency
- [ ] Primary color scale defined (50-900)
- [ ] Semantic colors defined (success, error, warning, info)
- [ ] Neutral gray scale defined
- [ ] Dark mode variant created
- [ ] CSS variables used

### Harmony
- [ ] 60-30-10 rule applied
- [ ] Color harmony type identified
- [ ] Limited palette (3-5 colors)
- [ ] Brand colors incorporated

## Tools & Resources

- **Contrast Checker**: https://webaim.org/resources/contrastchecker/
- **Palette Generator**: https://coolors.co/
- **Color Blindness Simulator**: https://www.color-blindness.com/coblis-color-blindness-simulator/
- **Tailwind Colors**: https://tailwindcss.com/docs/customizing-colors
- **Realtime Colors**: https://www.realtimecolors.com/
