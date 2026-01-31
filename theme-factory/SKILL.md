---
name: theme-factory
description: |
  Apply professional font and color themes to presentations, documents, and web artifacts. Provides 10 curated theme presets with cohesive palettes and typography.
  Use when: styling presentations, documents, web artifacts, or when user mentions 主題, theme, 風格, style, 套用主題, preset theme, 字體配色.
  Triggers: "theme", "主題", "風格", "style", "套用主題", "preset", "字體配色", "專業主題"
version: 1.0.0
---

# Theme Factory

Apply professional, cohesive themes to any visual artifact.

## Available Themes

### 1. Ocean Depths
```css
:root {
  --primary: #1a365d;
  --secondary: #2c5282;
  --accent: #63b3ed;
  --text: #1a202c;
  --background: #ebf8ff;
}
/* Fonts: Montserrat (headings), Lora (body) */
```

### 2. Sunset Boulevard
```css
:root {
  --primary: #c53030;
  --secondary: #e53e3e;
  --accent: #fbd38d;
  --text: #1a202c;
  --background: #fffaf0;
}
/* Fonts: Playfair Display (headings), Source Sans Pro (body) */
```

### 3. Forest Canopy
```css
:root {
  --primary: #276749;
  --secondary: #38a169;
  --accent: #9ae6b4;
  --text: #1a202c;
  --background: #f0fff4;
}
/* Fonts: Merriweather (headings), Open Sans (body) */
```

### 4. Modern Minimalist
```css
:root {
  --primary: #1a202c;
  --secondary: #4a5568;
  --accent: #e53e3e;
  --text: #2d3748;
  --background: #ffffff;
}
/* Fonts: Inter (headings), Inter (body) */
```

### 5. Golden Hour
```css
:root {
  --primary: #744210;
  --secondary: #b7791f;
  --accent: #f6e05e;
  --text: #1a202c;
  --background: #fffff0;
}
/* Fonts: Cormorant Garamond (headings), Nunito (body) */
```

### 6. Arctic Frost
```css
:root {
  --primary: #2b6cb0;
  --secondary: #4299e1;
  --accent: #bee3f8;
  --text: #1a202c;
  --background: #f7fafc;
}
/* Fonts: Poppins (headings), IBM Plex Sans (body) */
```

### 7. Desert Rose
```css
:root {
  --primary: #97266d;
  --secondary: #d53f8c;
  --accent: #fbb6ce;
  --text: #1a202c;
  --background: #fff5f7;
}
/* Fonts: Bodoni Moda (headings), Raleway (body) */
```

### 8. Tech Innovation
```css
:root {
  --primary: #0d0d0d;
  --secondary: #2d2d2d;
  --accent: #00d9ff;
  --text: #ffffff;
  --background: #0d0d0d;
}
/* Fonts: Space Grotesk (headings), JetBrains Mono (body) */
```

### 9. Botanical Garden
```css
:root {
  --primary: #22543d;
  --secondary: #48bb78;
  --accent: #f6ad55;
  --text: #1a202c;
  --background: #fffff0;
}
/* Fonts: DM Serif Display (headings), Work Sans (body) */
```

### 10. Midnight Galaxy
```css
:root {
  --primary: #1a1a2e;
  --secondary: #16213e;
  --accent: #e94560;
  --text: #edf2f7;
  --background: #0f0f1a;
}
/* Fonts: Orbitron (headings), Exo 2 (body) */
```

## Usage Workflow

### Step 1: Choose Theme

```markdown
Available themes:
1. Ocean Depths - Professional, trustworthy (finance, corporate)
2. Sunset Boulevard - Warm, inviting (hospitality, lifestyle)
3. Forest Canopy - Natural, sustainable (eco, wellness)
4. Modern Minimalist - Clean, focused (tech, SaaS)
5. Golden Hour - Elegant, premium (luxury, consulting)
6. Arctic Frost - Cool, professional (healthcare, legal)
7. Desert Rose - Bold, creative (fashion, beauty)
8. Tech Innovation - Cutting-edge, futuristic (startups, AI)
9. Botanical Garden - Organic, friendly (food, nature)
10. Midnight Galaxy - Dramatic, modern (gaming, entertainment)
```

### Step 2: Apply Theme

```python
# Theme configuration
THEMES = {
    'ocean_depths': {
        'colors': {
            'primary': '#1a365d',
            'secondary': '#2c5282',
            'accent': '#63b3ed',
            'text': '#1a202c',
            'background': '#ebf8ff',
        },
        'fonts': {
            'heading': 'Montserrat',
            'body': 'Lora',
        }
    },
    # ... other themes
}

def apply_theme(theme_name):
    return THEMES.get(theme_name, THEMES['modern_minimalist'])
```

### Step 3: Apply to Artifacts

#### For Presentations (PPTX)

```python
from pptx.dml.color import RGBColor

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return RGBColor(
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )

theme = apply_theme('ocean_depths')
primary_color = hex_to_rgb(theme['colors']['primary'])
```

#### For Documents (DOCX)

```python
from docx.shared import RGBColor as DocxRGB

def apply_docx_theme(doc, theme):
    colors = theme['colors']
    # Apply to headings, body text, etc.
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            for run in para.runs:
                run.font.color.rgb = DocxRGB.from_string(colors['primary'][1:])
```

#### For Web (CSS)

```css
/* Generated theme CSS */
:root {
  /* Colors */
  --color-primary: #1a365d;
  --color-secondary: #2c5282;
  --color-accent: #63b3ed;
  --color-text: #1a202c;
  --color-background: #ebf8ff;

  /* Typography */
  --font-heading: 'Montserrat', sans-serif;
  --font-body: 'Lora', serif;

  /* Spacing */
  --space-unit: 1rem;
}

body {
  font-family: var(--font-body);
  color: var(--color-text);
  background: var(--color-background);
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-heading);
  color: var(--color-primary);
}

.accent {
  color: var(--color-accent);
}
```

## Custom Theme Generation

When preset themes don't fit:

```python
def generate_custom_theme(base_color, mood='professional'):
    """Generate a custom theme from a base color."""
    from colorsys import rgb_to_hls, hls_to_rgb

    # Parse base color
    r, g, b = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
    h, l, s = rgb_to_hls(r/255, g/255, b/255)

    # Generate palette
    def hls_to_hex(h, l, s):
        r, g, b = hls_to_rgb(h, l, s)
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    return {
        'primary': base_color,
        'secondary': hls_to_hex(h, l * 1.2, s * 0.9),
        'accent': hls_to_hex((h + 0.5) % 1, l, s),  # Complementary
        'text': '#1a202c' if l > 0.5 else '#f7fafc',
        'background': hls_to_hex(h, 0.97, s * 0.3),
    }

custom = generate_custom_theme('#2563eb')
```

## Theme Selection Guide

| Use Case | Recommended Theme |
|----------|-------------------|
| Corporate presentation | Ocean Depths, Modern Minimalist |
| Startup pitch | Tech Innovation, Arctic Frost |
| Creative portfolio | Sunset Boulevard, Desert Rose |
| Environmental report | Forest Canopy, Botanical Garden |
| Financial document | Ocean Depths, Golden Hour |
| Tech documentation | Modern Minimalist, Tech Innovation |
| Event invitation | Sunset Boulevard, Midnight Galaxy |
| Healthcare materials | Arctic Frost, Forest Canopy |

## Checklist

- [ ] Theme matches content tone
- [ ] Colors have sufficient contrast
- [ ] Fonts are readable at all sizes
- [ ] Accent color used sparingly
- [ ] Consistent application throughout
- [ ] Accessible for color blindness
