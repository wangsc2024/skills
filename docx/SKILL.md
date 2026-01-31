---
name: docx
description: |
  Read, create, and edit Word documents (.docx). Extract text, handle tracked changes, add comments, and build professional documents programmatically.
  Use when: working with Word files, document automation, contract review, report generation, or when user mentions Word, docx, 文件, document, 報告, report, 合約, contract.
  Triggers: "Word", "docx", "文件", "document", "報告", "word檔", "合約", "word文件"
version: 1.0.0
---

# Word Document (DOCX) Processing

Comprehensive Word document operations through multiple specialized workflows.

## Capabilities

| Operation | Tool | Use Case |
|-----------|------|----------|
| **Read** | Pandoc, python-docx | Extract text, analyze structure |
| **Create** | python-docx | Build new documents |
| **Edit** | python-docx | Modify existing documents |
| **Track Changes** | python-docx | Redlining, review workflow |

## Setup

```bash
# Install dependencies
uv add python-docx

# For Pandoc (text extraction)
# Windows: winget install pandoc
# macOS: brew install pandoc
# Linux: apt install pandoc
```

## Reading Documents

### Extract Text with Pandoc

```bash
pandoc document.docx -t markdown -o output.md
```

### Read with Python

```python
from docx import Document

doc = Document('document.docx')

# Extract all paragraphs
for para in doc.paragraphs:
    print(para.text)

# Extract tables
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            print(cell.text)

# Get document properties
print(doc.core_properties.title)
print(doc.core_properties.author)
```

## Creating Documents

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Add title
title = doc.add_heading('Document Title', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Add paragraph with formatting
para = doc.add_paragraph()
run = para.add_run('Bold text')
run.bold = True
para.add_run(' and ')
run = para.add_run('italic text')
run.italic = True

# Add bullet list
doc.add_paragraph('First item', style='List Bullet')
doc.add_paragraph('Second item', style='List Bullet')
doc.add_paragraph('Third item', style='List Bullet')

# Add table
table = doc.add_table(rows=3, cols=3)
table.style = 'Table Grid'
for i, row in enumerate(table.rows):
    for j, cell in enumerate(row.cells):
        cell.text = f'Row {i+1}, Col {j+1}'

# Add image
doc.add_picture('image.png', width=Inches(4))

# Save
doc.save('output.docx')
```

## Editing Documents

```python
from docx import Document

doc = Document('existing.docx')

# Find and replace text
for para in doc.paragraphs:
    if 'OLD_TEXT' in para.text:
        para.text = para.text.replace('OLD_TEXT', 'NEW_TEXT')

# Add content to existing document
doc.add_paragraph('New paragraph at the end.')

# Modify specific paragraph
doc.paragraphs[0].text = 'Modified first paragraph'

doc.save('modified.docx')
```

## Working with Styles

```python
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE

doc = Document()

# Create custom style
styles = doc.styles
style = styles.add_style('CustomHeading', WD_STYLE_TYPE.PARAGRAPH)
style.font.name = 'Arial'
style.font.size = Pt(16)
style.font.bold = True
style.font.color.rgb = RGBColor(0, 51, 102)

# Apply style
doc.add_paragraph('Styled Heading', style='CustomHeading')
doc.save('styled.docx')
```

## Track Changes Workflow

### Principles for Redlining

```markdown
1. **Minimal edits**: Only mark what actually changed
2. **Preserve formatting**: Keep original attributes on unchanged text
3. **Batch changes**: Group related modifications (3-10 per batch)
4. **Clear comments**: Explain why changes were made
```

### Adding Comments

```python
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_comment(paragraph, comment_text, author="Reviewer"):
    """Add a comment to a paragraph."""
    # Create comment reference
    comment = OxmlElement('w:comment')
    comment.set(qn('w:id'), '1')
    comment.set(qn('w:author'), author)

    # Add comment text
    p = OxmlElement('w:p')
    r = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = comment_text
    r.append(t)
    p.append(r)
    comment.append(p)

    return comment
```

## Document Analysis

```python
from docx import Document
from collections import Counter

def analyze_document(path):
    doc = Document(path)

    stats = {
        'paragraphs': len(doc.paragraphs),
        'tables': len(doc.tables),
        'sections': len(doc.sections),
        'words': 0,
        'characters': 0,
    }

    # Count words and characters
    for para in doc.paragraphs:
        text = para.text
        stats['words'] += len(text.split())
        stats['characters'] += len(text)

    # Analyze styles used
    styles_used = Counter()
    for para in doc.paragraphs:
        if para.style:
            styles_used[para.style.name] += 1

    stats['styles'] = dict(styles_used)

    return stats

# Usage
stats = analyze_document('report.docx')
print(f"Words: {stats['words']}")
print(f"Tables: {stats['tables']}")
```

## Convert to Other Formats

```bash
# To PDF (requires LibreOffice)
libreoffice --headless --convert-to pdf document.docx

# To HTML
pandoc document.docx -o document.html

# To plain text
pandoc document.docx -t plain -o document.txt
```

## Common Patterns

### Template-based Generation

```python
from docx import Document

def generate_from_template(template_path, data, output_path):
    doc = Document(template_path)

    # Replace placeholders
    for para in doc.paragraphs:
        for key, value in data.items():
            placeholder = f'{{{{{key}}}}}'  # {{key}}
            if placeholder in para.text:
                para.text = para.text.replace(placeholder, str(value))

    # Also check tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for key, value in data.items():
                    placeholder = f'{{{{{key}}}}}'
                    if placeholder in cell.text:
                        cell.text = cell.text.replace(placeholder, str(value))

    doc.save(output_path)

# Usage
data = {
    'name': 'John Doe',
    'date': '2024-01-15',
    'amount': '$1,234.56'
}
generate_from_template('template.docx', data, 'invoice.docx')
```

## Checklist

- [ ] python-docx installed
- [ ] Pandoc available for text extraction
- [ ] LibreOffice for PDF conversion (optional)
- [ ] Template placeholders use consistent format
- [ ] Track changes batched appropriately
- [ ] Document properties set correctly
