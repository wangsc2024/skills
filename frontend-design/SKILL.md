---
name: frontend-design
description: |
  Build distinctive, production-grade web interfaces with intentional design direction. Emphasizes unique aesthetics over generic AI patterns, covering typography, color, layout, and motion.
  Use when: creating websites, landing pages, dashboards, unique UI designs, or when user mentions 前端設計, web design, UI設計, 網頁設計, landing page, 介面設計, 獨特風格.
  Triggers: "web design", "UI design", "前端設計", "網頁設計", "landing page", "介面設計", "獨特風格", "不要太普通"
---

# Frontend Design

Create distinctive, high-quality web interfaces that avoid generic "AI slop" aesthetics.

## Design Philosophy

> "Bold commitment and refined restraint both win over muddled compromise."

Every interface should have a clear conceptual direction executed with precision.

## Before Coding

### Establish Design Direction

1. **Purpose & Audience**: Who uses this? What do they need?
2. **Tone & Style**: Minimalist? Maximalist? Brutalist? Retro? Organic?
3. **Technical Constraints**: Performance, accessibility, browser support
4. **Differentiator**: What makes this memorable?

### Style Directions

| Style | Characteristics | When to Use |
|-------|-----------------|-------------|
| **Minimalist** | White space, restraint, precision | Professional tools, luxury |
| **Maximalist** | Rich layers, bold colors, density | Creative, entertainment |
| **Brutalist** | Raw, stark, unconventional | Avant-garde, tech |
| **Retro** | Nostalgic palettes, vintage type | Playful, unique brands |
| **Organic** | Soft shapes, natural colors, flow | Wellness, sustainability |
| **Corporate** | Clean, trustworthy, structured | Enterprise, finance |

## Typography

### Avoid These (Overused)

```
❌ Inter, Roboto, Open Sans, Arial, Helvetica
❌ System fonts as primary
❌ Single font for everything
```

### Choose Distinctive Fonts

```css
/* Sophisticated pairings */

/* Modern Tech */
font-family: 'Space Grotesk', sans-serif;  /* Headings */
font-family: 'IBM Plex Mono', monospace;   /* Code/accents */

/* Editorial */
font-family: 'Playfair Display', serif;    /* Headings */
font-family: 'Source Serif Pro', serif;    /* Body */

/* Contemporary */
font-family: 'Clash Display', sans-serif;  /* Headings */
font-family: 'Satoshi', sans-serif;        /* Body */

/* Brutalist */
font-family: 'Monument Extended', sans-serif;
font-family: 'Neue Machina', sans-serif;
```

### Type Scale

```css
:root {
  /* Perfect Fourth (1.333) */
  --text-xs: 0.75rem;    /* 12px */
  --text-sm: 0.875rem;   /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg: 1.333rem;   /* 21px */
  --text-xl: 1.777rem;   /* 28px */
  --text-2xl: 2.369rem;  /* 38px */
  --text-3xl: 3.157rem;  /* 51px */
  --text-4xl: 4.209rem;  /* 67px */
}
```

## Color

### Avoid These (Clichéd)

```
❌ Purple gradients on white
❌ Blue-to-purple "tech" gradients
❌ Pastel rainbows
❌ Pure black (#000) on pure white (#fff)
```

### Build Intentional Palettes

```css
/* Sophisticated Neutral */
:root {
  --ink: #1a1a1a;
  --paper: #faf9f7;
  --stone: #8b8680;
  --accent: #e63946;
}

/* Warm Modern */
:root {
  --primary: #2d3436;
  --secondary: #636e72;
  --warm: #fab1a0;
  --cream: #ffeaa7;
  --bg: #f8f5f2;
}

/* Bold Contemporary */
:root {
  --electric: #5cff95;
  --deep: #0a0a0a;
  --mid: #2a2a2a;
  --light: #e0e0e0;
}

/* Earth Tones */
:root {
  --forest: #2d5a27;
  --clay: #c4a77d;
  --sand: #e8dcc4;
  --charcoal: #2c2c2c;
}
```

### Color Usage

```css
/* 60-30-10 Rule */
.page {
  background: var(--bg);           /* 60% - dominant */
}

.card {
  background: var(--surface);      /* 30% - secondary */
}

.cta-button {
  background: var(--accent);       /* 10% - accent */
}
```

## Layout

### Avoid These (Generic)

```
❌ Everything centered
❌ Uniform card grids
❌ Predictable hero sections
❌ Same border-radius everywhere
```

### Create Visual Interest

```css
/* Asymmetric grid */
.layout {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  gap: clamp(1rem, 3vw, 3rem);
}

/* Overlapping elements */
.overlap-container {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
}

.overlap-item-1 {
  grid-column: 1 / 8;
  grid-row: 1;
}

.overlap-item-2 {
  grid-column: 5 / 13;
  grid-row: 1;
  margin-top: 4rem;
}

/* Breaking the grid */
.breakout {
  width: 100vw;
  margin-left: calc(-50vw + 50%);
}

/* Varied spacing */
.section:nth-child(odd) {
  padding: 8rem 0;
}

.section:nth-child(even) {
  padding: 12rem 0;
}
```

### Spatial Rhythm

```css
/* Consistent but varied spacing */
:root {
  --space-3xs: clamp(0.25rem, 0.5vw, 0.375rem);
  --space-2xs: clamp(0.5rem, 1vw, 0.75rem);
  --space-xs: clamp(0.75rem, 1.5vw, 1rem);
  --space-s: clamp(1rem, 2vw, 1.5rem);
  --space-m: clamp(1.5rem, 3vw, 2rem);
  --space-l: clamp(2rem, 4vw, 3rem);
  --space-xl: clamp(3rem, 6vw, 5rem);
  --space-2xl: clamp(5rem, 10vw, 8rem);
  --space-3xl: clamp(8rem, 15vw, 12rem);
}
```

## Motion & Animation

### Avoid These

```
❌ Generic fade-in on scroll
❌ Bouncy animations everywhere
❌ Motion for motion's sake
```

### Intentional Motion

```css
/* Page load choreography */
@keyframes reveal {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.hero-title {
  animation: reveal 0.8s ease-out 0.1s both;
}

.hero-subtitle {
  animation: reveal 0.8s ease-out 0.2s both;
}

.hero-cta {
  animation: reveal 0.8s ease-out 0.3s both;
}

/* Micro-interactions */
.button {
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Smooth state changes */
.card {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
```

### Easing Functions

```css
:root {
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);
  --ease-elastic: cubic-bezier(0.68, -0.6, 0.32, 1.6);
}
```

## Visual Texture

### Add Depth

```css
/* Subtle gradients */
.surface {
  background: linear-gradient(
    135deg,
    rgba(255, 255, 255, 0.1) 0%,
    rgba(255, 255, 255, 0) 100%
  );
}

/* Noise texture */
.textured {
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.7' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  opacity: 0.03;
}

/* Layered shadows */
.elevated {
  box-shadow:
    0 1px 2px rgba(0, 0, 0, 0.04),
    0 2px 4px rgba(0, 0, 0, 0.04),
    0 4px 8px rgba(0, 0, 0, 0.04),
    0 8px 16px rgba(0, 0, 0, 0.04);
}
```

## Component Examples

### Distinctive Button

```css
/* Not this */
.button-generic {
  background: #6366f1;
  border-radius: 8px;
  padding: 12px 24px;
}

/* This - with character */
.button-distinctive {
  background: #0a0a0a;
  color: #fff;
  padding: 1rem 2rem;
  border: none;
  position: relative;
  overflow: hidden;
  font-weight: 500;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.button-distinctive::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  transform: translateX(-100%);
  transition: transform 0.5s;
}

.button-distinctive:hover::before {
  transform: translateX(100%);
}
```

### Unique Card

```css
.card-unique {
  background: #faf9f7;
  padding: 2rem;
  position: relative;
}

/* Asymmetric border */
.card-unique::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  background: #e63946;
}

/* Corner accent */
.card-unique::after {
  content: '';
  position: absolute;
  right: -1px;
  top: -1px;
  width: 40px;
  height: 40px;
  border-top: 2px solid #e63946;
  border-right: 2px solid #e63946;
}
```

## Design Checklist

### Before Starting
- [ ] Clear aesthetic direction chosen
- [ ] Target audience defined
- [ ] Distinctive elements identified
- [ ] Font pairing selected (not defaults)
- [ ] Color palette created (not gradients)

### During Development
- [ ] Avoiding centered-everything layouts
- [ ] Varied spacing creates rhythm
- [ ] Motion is intentional, not decorative
- [ ] Details are refined (shadows, borders)
- [ ] Mobile-first responsive

### Before Shipping
- [ ] Accessibility verified (contrast, keyboard)
- [ ] Performance optimized (fonts, images)
- [ ] Looks intentional, not templated
- [ ] Would be proud to show this

## Anti-Patterns to Avoid

```markdown
❌ Purple-to-blue gradients
❌ Everything uses Inter font
❌ Uniform border-radius on all elements
❌ Generic illustrations (Blush, Undraw)
❌ Centered hero with fade-in
❌ Cards with identical shadows
❌ "AI startup" aesthetic
❌ Overused Tailwind defaults
```

## Resources

- **Fonts**: fonts.google.com, fontshare.com, atipo.es
- **Colors**: coolors.co, colorhunt.co, realtimecolors.com
- **Inspiration**: awwwards.com, siteinspire.com, minimal.gallery
