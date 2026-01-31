---
name: pptx
description: |
  Create, edit, and analyze PowerPoint presentations (.pptx). Build slides programmatically, apply themes, add charts and images, and modify existing presentations.
  Use when: creating presentations, automating slide generation, analyzing presentation content, or when user mentions PowerPoint, pptx, 簡報, presentation, slides, 投影片.
  Triggers: "PowerPoint", "pptx", "簡報", "presentation", "slides", "投影片", "做簡報"
version: 1.0.0
---

# PowerPoint (PPTX) Processing

Create, edit, and analyze PowerPoint presentations programmatically.

## Capabilities

| Operation | Tool | Use Case |
|-----------|------|----------|
| **Create** | python-pptx | Build new presentations |
| **Edit** | python-pptx | Modify existing slides |
| **Analyze** | python-pptx | Extract text and structure |
| **Convert** | LibreOffice | Export to PDF/images |

## Setup

```bash
# Install python-pptx
uv add python-pptx

# For image handling
uv add Pillow

# For PDF conversion (optional)
# Windows: winget install LibreOffice
# macOS: brew install libreoffice
```

## Creating Presentations

### Basic Presentation

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

prs = Presentation()

# Title slide
slide_layout = prs.slide_layouts[0]  # Title Slide
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]

title.text = "Presentation Title"
subtitle.text = "Subtitle goes here"

# Content slide
slide_layout = prs.slide_layouts[1]  # Title and Content
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
body = slide.placeholders[1]

title.text = "Key Points"
tf = body.text_frame
tf.text = "First bullet point"

p = tf.add_paragraph()
p.text = "Second bullet point"
p.level = 0

p = tf.add_paragraph()
p.text = "Sub-point"
p.level = 1

prs.save('presentation.pptx')
```

### Adding Images

```python
from pptx import Presentation
from pptx.util import Inches

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank

# Add image with position and size
left = Inches(1)
top = Inches(1)
width = Inches(5)
slide.shapes.add_picture('image.png', left, top, width=width)

prs.save('with_image.pptx')
```

### Adding Tables

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])

# Create table
rows, cols = 4, 3
left = Inches(1)
top = Inches(2)
width = Inches(8)
height = Inches(2)

table = slide.shapes.add_table(rows, cols, left, top, width, height).table

# Set column widths
table.columns[0].width = Inches(2)
table.columns[1].width = Inches(3)
table.columns[2].width = Inches(3)

# Header row
headers = ['Product', 'Q1 Sales', 'Q2 Sales']
for i, header in enumerate(headers):
    cell = table.cell(0, i)
    cell.text = header
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(0, 51, 102)

# Data rows
data = [
    ['Widget A', '$10,000', '$12,000'],
    ['Widget B', '$8,000', '$9,500'],
    ['Widget C', '$15,000', '$18,000'],
]
for row_idx, row_data in enumerate(data, start=1):
    for col_idx, value in enumerate(row_data):
        table.cell(row_idx, col_idx).text = value

prs.save('with_table.pptx')
```

### Adding Charts

```python
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])

# Chart data
chart_data = CategoryChartData()
chart_data.categories = ['Q1', 'Q2', 'Q3', 'Q4']
chart_data.add_series('Sales', (100, 120, 140, 160))
chart_data.add_series('Expenses', (80, 90, 100, 110))

# Add chart
x, y, cx, cy = Inches(1), Inches(1.5), Inches(8), Inches(5)
chart = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    x, y, cx, cy,
    chart_data
).chart

chart.has_legend = True
chart.legend.include_in_layout = False

prs.save('with_chart.pptx')
```

## Editing Presentations

### Modify Existing Slides

```python
from pptx import Presentation

prs = Presentation('existing.pptx')

# Modify first slide title
slide = prs.slides[0]
for shape in slide.shapes:
    if shape.has_text_frame:
        if shape.text == "Old Title":
            shape.text = "New Title"

# Add a new slide
slide_layout = prs.slide_layouts[1]
new_slide = prs.slides.add_slide(slide_layout)
new_slide.shapes.title.text = "Added Slide"

prs.save('modified.pptx')
```

### Find and Replace Text

```python
from pptx import Presentation

def find_replace(prs, find_text, replace_text):
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        if find_text in run.text:
                            run.text = run.text.replace(find_text, replace_text)
    return prs

prs = Presentation('template.pptx')
prs = find_replace(prs, '{{COMPANY}}', 'Acme Corp')
prs = find_replace(prs, '{{DATE}}', '2024-01-15')
prs.save('customized.pptx')
```

### Delete Slides

```python
from pptx import Presentation

prs = Presentation('presentation.pptx')

# Delete slide by index (0-based)
slide_to_delete = prs.slides[2]
rId = prs.part.relate_to(slide_to_delete.part, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide')
prs.part.drop_rel(rId)
del prs.slides._sldIdLst[2]

prs.save('fewer_slides.pptx')
```

## Analyzing Presentations

### Extract All Text

```python
from pptx import Presentation

def extract_text(pptx_path):
    prs = Presentation(pptx_path)
    all_text = []

    for slide_num, slide in enumerate(prs.slides, 1):
        slide_text = [f"--- Slide {slide_num} ---"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        slide_text.append(text)
        all_text.append('\n'.join(slide_text))

    return '\n\n'.join(all_text)

text = extract_text('presentation.pptx')
print(text)
```

### Get Presentation Stats

```python
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

def analyze_presentation(pptx_path):
    prs = Presentation(pptx_path)

    stats = {
        'slides': len(prs.slides),
        'images': 0,
        'tables': 0,
        'charts': 0,
        'text_boxes': 0,
        'words': 0,
    }

    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                stats['images'] += 1
            elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                stats['tables'] += 1
            elif shape.shape_type == MSO_SHAPE_TYPE.CHART:
                stats['charts'] += 1
            elif shape.has_text_frame:
                stats['text_boxes'] += 1
                for para in shape.text_frame.paragraphs:
                    stats['words'] += len(para.text.split())

    return stats

stats = analyze_presentation('report.pptx')
print(f"Slides: {stats['slides']}")
print(f"Images: {stats['images']}")
print(f"Words: {stats['words']}")
```

## Styling

### Custom Colors and Fonts

```python
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])

# Add text box with styling
from pptx.util import Inches
textbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1))
tf = textbox.text_frame

p = tf.paragraphs[0]
p.text = "Styled Text"
p.font.name = "Arial"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = RGBColor(0, 102, 204)  # Blue

prs.save('styled.pptx')
```

### Slide Background

```python
from pptx import Presentation
from pptx.dml.color import RGBColor

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])

# Solid color background
background = slide.background
fill = background.fill
fill.solid()
fill.fore_color.rgb = RGBColor(240, 240, 240)  # Light gray

prs.save('with_background.pptx')
```

## Color Palettes

```python
# Professional color schemes
PALETTES = {
    'corporate': {
        'primary': RGBColor(0, 51, 102),    # Navy
        'secondary': RGBColor(0, 102, 153),  # Teal
        'accent': RGBColor(255, 153, 0),     # Orange
        'text': RGBColor(51, 51, 51),        # Dark gray
        'background': RGBColor(255, 255, 255),
    },
    'modern': {
        'primary': RGBColor(41, 128, 185),   # Blue
        'secondary': RGBColor(52, 73, 94),   # Dark blue-gray
        'accent': RGBColor(46, 204, 113),    # Green
        'text': RGBColor(44, 62, 80),        # Charcoal
        'background': RGBColor(236, 240, 241),
    },
    'minimal': {
        'primary': RGBColor(33, 33, 33),     # Almost black
        'secondary': RGBColor(97, 97, 97),   # Gray
        'accent': RGBColor(255, 87, 34),     # Deep orange
        'text': RGBColor(33, 33, 33),
        'background': RGBColor(250, 250, 250),
    },
}
```

## Convert to PDF

```bash
# Using LibreOffice
libreoffice --headless --convert-to pdf presentation.pptx

# Export specific slides
libreoffice --headless --convert-to pdf:writer_pdf_Export --outdir ./output presentation.pptx
```

## Template Workflow

```python
from pptx import Presentation

def create_from_template(template_path, data, output_path):
    """
    Create presentation from template with placeholder replacement.

    data format: {
        'title': 'Main Title',
        'subtitle': 'Subtitle',
        'slides': [
            {'title': 'Slide 1', 'content': ['Point 1', 'Point 2']},
            {'title': 'Slide 2', 'content': ['Point A', 'Point B']},
        ]
    }
    """
    prs = Presentation(template_path)

    # Replace placeholders in existing slides
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        for key, value in data.items():
                            if isinstance(value, str):
                                placeholder = f'{{{{{key}}}}}'
                                if placeholder in run.text:
                                    run.text = run.text.replace(placeholder, value)

    prs.save(output_path)

# Usage
data = {
    'title': 'Q4 Report',
    'company': 'Acme Corp',
    'date': 'January 2024',
}
create_from_template('template.pptx', data, 'q4_report.pptx')
```

## Checklist

- [ ] python-pptx installed
- [ ] Pillow for image handling
- [ ] LibreOffice for PDF export (optional)
- [ ] Template placeholders use {{key}} format
- [ ] Consistent color palette applied
- [ ] Font sizes appropriate for projection
- [ ] Images sized correctly (not stretched)
