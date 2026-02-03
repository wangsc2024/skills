---
name: canvas-design
description: |
  Create sophisticated visual art and design pieces including posters, artwork, and graphic compositions. Outputs museum-quality PNG and PDF files.
  Use when: creating visual designs, posters, artistic compositions, graphic artwork, or when user mentions 海報, poster, 視覺設計, visual design, 藝術作品, artwork, 圖形設計.
  Triggers: "poster", "海報", "視覺設計", "visual design", "artwork", "藝術作品", "圖形設計", "做海報"
---

# Canvas Design

Create sophisticated visual art through a two-phase design process.

## Philosophy

> "Treat each design as an art object, not a decorated document."

Work should appear as though someone at the absolute top of their field labored over every detail.

## Two-Phase Process

### Phase 1: Design Philosophy

Before any visual work, create an aesthetic manifesto (4-6 paragraphs):

```markdown
# Design Philosophy: [Project Name]

## Visual Movement
[Define the aesthetic movement - what school of design inspires this?]

## Form & Space
[How will shapes and negative space interact?]

## Color Language
[What emotions do colors convey? What is the palette's narrative?]

## Composition Principles
[How are elements arranged? What creates visual hierarchy?]

## Conceptual References
[What subtle themes are embedded? What should viewers feel?]
```

### Phase 2: Canvas Expression

Manifest the philosophy as museum-quality artwork.

**Output ratio**: 90% visual, 10% essential text

## Technical Setup

```bash
# Required libraries
uv add Pillow numpy

# For advanced graphics
uv add cairo pycairo

# For SVG output
uv add svgwrite
```

## Creating Artwork with Pillow

### Basic Canvas

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create canvas
width, height = 2400, 3200  # High resolution
canvas = Image.new('RGB', (width, height), color='#faf9f7')
draw = ImageDraw.Draw(canvas)

# Add geometric elements
draw.rectangle([100, 100, 500, 500], fill='#e63946')
draw.ellipse([600, 200, 1000, 600], fill='#1d3557')

canvas.save('artwork.png', quality=95)
```

### Geometric Composition

```python
from PIL import Image, ImageDraw
import random
import math

def create_geometric_poster(width=2400, height=3200):
    canvas = Image.new('RGB', (width, height), '#0a0a0a')
    draw = ImageDraw.Draw(canvas)

    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#f7fff7']

    # Create overlapping shapes
    for _ in range(50):
        shape_type = random.choice(['rect', 'circle', 'line'])
        color = random.choice(colors)

        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(50, 400)

        if shape_type == 'rect':
            draw.rectangle([x, y, x + size, y + size * 0.6],
                          fill=color, outline=None)
        elif shape_type == 'circle':
            draw.ellipse([x, y, x + size, y + size],
                        fill=color)
        else:
            angle = random.uniform(0, math.pi)
            x2 = x + size * math.cos(angle)
            y2 = y + size * math.sin(angle)
            draw.line([x, y, x2, y2], fill=color, width=random.randint(2, 20))

    return canvas

poster = create_geometric_poster()
poster.save('geometric_poster.png')
```

### Gradient Background

```python
from PIL import Image
import numpy as np

def create_gradient(width, height, color1, color2, direction='vertical'):
    """Create smooth gradient background."""
    # Convert hex to RGB
    c1 = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
    c2 = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))

    # Create gradient array
    if direction == 'vertical':
        gradient = np.linspace(0, 1, height)[:, np.newaxis]
    else:
        gradient = np.linspace(0, 1, width)[np.newaxis, :]

    # Interpolate colors
    r = (c1[0] * (1 - gradient) + c2[0] * gradient).astype(np.uint8)
    g = (c1[1] * (1 - gradient) + c2[1] * gradient).astype(np.uint8)
    b = (c1[2] * (1 - gradient) + c2[2] * gradient).astype(np.uint8)

    if direction == 'vertical':
        rgb = np.stack([
            np.tile(r, (1, width)),
            np.tile(g, (1, width)),
            np.tile(b, (1, width))
        ], axis=2)
    else:
        rgb = np.stack([
            np.tile(r, (height, 1)),
            np.tile(g, (height, 1)),
            np.tile(b, (height, 1))
        ], axis=2)

    return Image.fromarray(rgb.squeeze(), 'RGB')

gradient = create_gradient(2400, 3200, '#1a1a2e', '#16213e')
gradient.save('gradient_bg.png')
```

### Typography Poster

```python
from PIL import Image, ImageDraw, ImageFont

def create_type_poster(text, width=2400, height=3200):
    canvas = Image.new('RGB', (width, height), '#faf9f7')
    draw = ImageDraw.Draw(canvas)

    # Use system fonts (or download custom)
    try:
        font_large = ImageFont.truetype('arial.ttf', 300)
        font_small = ImageFont.truetype('arial.ttf', 48)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Large impact text
    draw.text((width//2, height//2), text.upper(),
              font=font_large, fill='#0a0a0a', anchor='mm')

    # Subtle accent line
    draw.rectangle([200, height - 400, width - 200, height - 395],
                   fill='#e63946')

    return canvas

poster = create_type_poster("CREATE")
poster.save('type_poster.png')
```

## Design Principles

### Visual Hierarchy

```markdown
1. **Primary Element** - Largest, most prominent (60% attention)
2. **Secondary Elements** - Supporting visuals (30% attention)
3. **Tertiary Details** - Subtle accents (10% attention)
```

### Composition Rules

```markdown
- **Rule of Thirds**: Place key elements at intersection points
- **Golden Ratio**: 1.618 proportions for natural balance
- **Asymmetric Balance**: Unequal but balanced visual weight
- **Negative Space**: Let elements breathe
```

### Color Relationships

```python
PALETTES = {
    'bold_contrast': {
        'primary': '#0a0a0a',
        'accent': '#ff3366',
        'neutral': '#f5f5f5',
    },
    'sophisticated': {
        'primary': '#2d3436',
        'secondary': '#636e72',
        'accent': '#d4a373',
        'bg': '#faf9f7',
    },
    'vibrant': {
        'red': '#e63946',
        'cream': '#f1faee',
        'light_blue': '#a8dadc',
        'blue': '#457b9d',
        'dark': '#1d3557',
    },
    'earth': {
        'forest': '#606c38',
        'olive': '#283618',
        'cream': '#fefae0',
        'tan': '#dda15e',
        'brown': '#bc6c25',
    },
}
```

## Output Formats

### PNG (Raster)

```python
# High quality PNG
canvas.save('output.png', quality=95, optimize=True)

# With transparency
canvas_rgba = Image.new('RGBA', (width, height), (0, 0, 0, 0))
canvas_rgba.save('transparent.png')
```

### PDF (Vector-friendly)

```python
from PIL import Image

# Save as PDF
canvas.save('output.pdf', 'PDF', resolution=300)

# Multi-page PDF
pages = [page1, page2, page3]
pages[0].save('multipage.pdf', save_all=True, append_images=pages[1:])
```

### SVG (True Vector)

```python
import svgwrite

dwg = svgwrite.Drawing('output.svg', size=('2400px', '3200px'))

# Add shapes
dwg.add(dwg.rect(insert=(100, 100), size=(400, 400), fill='#e63946'))
dwg.add(dwg.circle(center=(800, 400), r=200, fill='#1d3557'))
dwg.add(dwg.text('DESIGN', insert=(1200, 1600),
                 font_size='200px', font_family='Arial', fill='#0a0a0a'))

dwg.save()
```

## Quality Checklist

### Conceptual
- [ ] Design philosophy documented
- [ ] Clear visual narrative
- [ ] Subtle conceptual references embedded
- [ ] Intentional, not decorative

### Technical
- [ ] High resolution (300 DPI for print)
- [ ] Proper margins (no edge bleeding unintentionally)
- [ ] No overlapping text issues
- [ ] Color profile appropriate (sRGB for web, CMYK for print)

### Aesthetic
- [ ] Museum-quality craftsmanship
- [ ] Typography is design-forward
- [ ] Color palette feels cohesive
- [ ] Composition is balanced yet interesting
- [ ] Would display this proudly

## Anti-Patterns

```markdown
❌ Clipart or generic stock elements
❌ Text as afterthought
❌ Random decorative elements
❌ Unintentional symmetry
❌ Safe, predictable layouts
❌ Over-decorated but under-designed
```
