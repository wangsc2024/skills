---
name: algorithmic-art
description: |
  Generate algorithmic art using p5.js and SVG. Creates generative patterns, fractals, noise-based visuals, geometric compositions, and interactive sketches.
  Use when: creating visual art, generative designs, creative coding, data visualization art, or when user mentions p5.js, SVG, 生成藝術, generative art, fractal, 碎形, 演算法藝術, creative coding.
  Triggers: "p5.js", "SVG", "生成藝術", "generative art", "fractal", "碎形", "演算法藝術", "creative coding", "視覺藝術"
---

# Algorithmic Art Generator

Generate beautiful algorithmic art using p5.js for interactive sketches and SVG for vector graphics.

## Capabilities

- **Generative Patterns**: Perlin noise, flow fields, particle systems
- **Geometric Art**: Tessellations, sacred geometry, recursive shapes
- **Fractals**: Mandelbrot, Julia sets, L-systems, tree fractals
- **Data Art**: Visualize data as artistic compositions
- **Interactive Sketches**: Mouse/keyboard responsive animations

## p5.js Template

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
  <style>
    body { margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #1a1a1a; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
// === CONFIGURATION ===
const CONFIG = {
  width: 800,
  height: 800,
  background: '#1a1a1a',
  // Add more parameters here
};

function setup() {
  createCanvas(CONFIG.width, CONFIG.height);
  // Initialize your art here
}

function draw() {
  background(CONFIG.background);
  // Draw your generative art here
}
</script>
</body>
</html>
```

## SVG Generation Template

```javascript
function generateSVG(width, height, elements) {
  const header = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}">`;
  const footer = '</svg>';
  const style = `<style>
    .stroke { fill: none; stroke: currentColor; stroke-width: 1; }
    .fill { fill: currentColor; stroke: none; }
  </style>`;

  return `${header}\n${style}\n${elements.join('\n')}\n${footer}`;
}

// SVG Element Helpers
const circle = (cx, cy, r, cls = 'stroke') =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" class="${cls}"/>`;

const line = (x1, y1, x2, y2, cls = 'stroke') =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`;

const path = (d, cls = 'stroke') =>
  `<path d="${d}" class="${cls}"/>`;

const rect = (x, y, w, h, cls = 'stroke') =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${cls}"/>`;

const polygon = (points, cls = 'stroke') =>
  `<polygon points="${points}" class="${cls}"/>`;
```

## Common Algorithms

### Perlin Noise Flow Field

```javascript
function setup() {
  createCanvas(800, 800);
  background(20);
  stroke(255, 10);
  noFill();

  const scale = 0.005;
  const particles = 1000;

  for (let i = 0; i < particles; i++) {
    let x = random(width);
    let y = random(height);

    beginShape();
    for (let j = 0; j < 100; j++) {
      vertex(x, y);
      const angle = noise(x * scale, y * scale) * TWO_PI * 2;
      x += cos(angle) * 2;
      y += sin(angle) * 2;
      if (x < 0 || x > width || y < 0 || y > height) break;
    }
    endShape();
  }
}
```

### Recursive Circle Packing

```javascript
const circles = [];

function setup() {
  createCanvas(800, 800);
  for (let i = 0; i < 500; i++) {
    addCircle();
  }
  noLoop();
}

function draw() {
  background(20);
  noFill();
  circles.forEach(c => {
    stroke(map(c.r, 5, 100, 100, 255));
    ellipse(c.x, c.y, c.r * 2);
  });
}

function addCircle() {
  for (let attempts = 0; attempts < 100; attempts++) {
    const x = random(width);
    const y = random(height);
    let r = 5;
    let valid = true;

    for (const c of circles) {
      const d = dist(x, y, c.x, c.y);
      if (d < r + c.r + 2) { valid = false; break; }
      r = min(r, d - c.r - 2);
    }

    r = min(r, x, y, width - x, height - y);
    if (valid && r > 5) {
      circles.push({ x, y, r });
      return;
    }
  }
}
```

### L-System Fractal Tree

```javascript
let axiom = 'F';
let sentence = axiom;
const rules = { F: 'FF+[+F-F-F]-[-F+F+F]' };
const angle = 25;
const len = 4;

function generate() {
  let next = '';
  for (const char of sentence) {
    next += rules[char] || char;
  }
  sentence = next;
}

function setup() {
  createCanvas(800, 800);
  for (let i = 0; i < 4; i++) generate();
}

function draw() {
  background(20);
  stroke(150, 255, 150);
  translate(width / 2, height);

  for (const char of sentence) {
    switch (char) {
      case 'F': line(0, 0, 0, -len); translate(0, -len); break;
      case '+': rotate(radians(angle)); break;
      case '-': rotate(radians(-angle)); break;
      case '[': push(); break;
      case ']': pop(); break;
    }
  }
  noLoop();
}
```

## Workflow

1. **Clarify the vision**: Art style, color palette, complexity level
2. **Choose the approach**: p5.js for interactive, SVG for static/print
3. **Implement algorithm**: Start simple, add complexity iteratively
4. **Fine-tune parameters**: Adjust scale, density, colors
5. **Export**: Save as PNG, SVG, or animated GIF

## Color Palettes

```javascript
const palettes = {
  sunset: ['#FF6B6B', '#FEC89A', '#FFD93D', '#6BCB77', '#4D96FF'],
  ocean: ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8', '#03045E'],
  forest: ['#2D6A4F', '#40916C', '#52B788', '#74C69D', '#95D5B2'],
  neon: ['#FF00FF', '#00FFFF', '#FF00AA', '#AAFF00', '#00FF88'],
  mono: ['#111111', '#333333', '#555555', '#888888', '#BBBBBB'],
  warm: ['#D00000', '#DC2F02', '#E85D04', '#F48C06', '#FFBA08'],
};
```

## Tips

- Use `randomSeed()` for reproducible art
- Layer transparent shapes for depth
- Combine noise with geometry for organic feels
- Export at high resolution (2x or 3x) for printing
- Add subtle animation for mesmerizing effects
