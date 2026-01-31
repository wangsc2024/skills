---
name: pdf
description: |
  Comprehensive PDF manipulation toolkit for extracting text and tables, creating new PDFs, merging/splitting documents, and handling forms.
  Use when: working with PDF files, extracting data, filling forms, merging PDFs, or when user mentions PDF, 合併PDF, extract, OCR, 表單, form filling.
  Triggers: "PDF", "pdf檔", "合併PDF", "extract text", "OCR", "表單", "fill form", "分割PDF"
version: 1.0.0
---

# PDF Processing Toolkit

Extract, create, manipulate, and analyze PDF documents.

## Capabilities

| Operation | Tool | Use Case |
|-----------|------|----------|
| **Read/Extract** | pdfplumber, pypdf | Text, tables, metadata |
| **Create** | reportlab | Generate new PDFs |
| **Merge/Split** | pypdf | Combine or separate pages |
| **Forms** | pypdf | Fill form fields |
| **OCR** | pytesseract | Scanned documents |

## Setup

```bash
# Core libraries
uv add pypdf pdfplumber reportlab

# For OCR (scanned PDFs)
uv add pytesseract pillow
# Also install Tesseract: https://github.com/tesseract-ocr/tesseract

# Command-line tools (optional)
# Windows: choco install qpdf poppler
# macOS: brew install qpdf poppler
# Linux: apt install qpdf poppler-utils
```

## Reading PDFs

### Extract Text

```python
import pdfplumber

with pdfplumber.open('document.pdf') as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
```

### Extract Tables to DataFrame

```python
import pdfplumber
import pandas as pd

with pdfplumber.open('report.pdf') as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()

    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        print(df)
        df.to_excel('extracted_table.xlsx', index=False)
```

### Get Metadata

```python
from pypdf import PdfReader

reader = PdfReader('document.pdf')
meta = reader.metadata

print(f"Title: {meta.title}")
print(f"Author: {meta.author}")
print(f"Pages: {len(reader.pages)}")
print(f"Created: {meta.creation_date}")
```

## Creating PDFs

### Simple PDF with ReportLab

```python
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - inch, "Report Title")

    # Body text
    c.setFont("Helvetica", 12)
    text = c.beginText(inch, height - 2*inch)
    text.textLines("""
    This is a sample PDF document.
    Created with ReportLab.

    Key features:
    - Text formatting
    - Tables and graphics
    - Multiple pages
    """)
    c.drawText(text)

    # Draw a rectangle
    c.setStrokeColor(colors.blue)
    c.setFillColor(colors.lightblue)
    c.rect(inch, 3*inch, 4*inch, 2*inch, fill=True)

    c.save()

create_pdf('output.pdf')
```

### PDF with Tables

```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

doc = SimpleDocTemplate("table_report.pdf", pagesize=letter)
elements = []
styles = getSampleStyleSheet()

# Add title
elements.append(Paragraph("Sales Report", styles['Heading1']))

# Create table data
data = [
    ['Product', 'Q1', 'Q2', 'Q3', 'Q4'],
    ['Widget A', '100', '120', '140', '160'],
    ['Widget B', '200', '180', '220', '240'],
    ['Widget C', '150', '170', '190', '210'],
]

# Create and style table
table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

elements.append(table)
doc.build(elements)
```

## Merging & Splitting

### Merge Multiple PDFs

```python
from pypdf import PdfWriter

writer = PdfWriter()

# Add PDFs in order
pdfs = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
for pdf in pdfs:
    writer.append(pdf)

writer.write('merged.pdf')
writer.close()
```

### Split PDF

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('large_document.pdf')

# Extract specific pages
writer = PdfWriter()
writer.add_page(reader.pages[0])  # First page
writer.add_page(reader.pages[4])  # Fifth page
writer.write('selected_pages.pdf')

# Split into individual pages
for i, page in enumerate(reader.pages):
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(f'page_{i+1}.pdf')
```

### Extract Page Range

```python
from pypdf import PdfReader, PdfWriter

def extract_pages(input_pdf, output_pdf, start, end):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    for i in range(start - 1, min(end, len(reader.pages))):
        writer.add_page(reader.pages[i])

    writer.write(output_pdf)

extract_pages('document.pdf', 'pages_1_to_5.pdf', 1, 5)
```

## Form Handling

### Fill PDF Forms

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('form.pdf')
writer = PdfWriter()
writer.append(reader)

# Get form fields
fields = reader.get_fields()
print("Available fields:", list(fields.keys()))

# Fill form
writer.update_page_form_field_values(
    writer.pages[0],
    {
        'name': 'John Doe',
        'email': 'john@example.com',
        'date': '2024-01-15',
        'amount': '1234.56',
    }
)

writer.write('filled_form.pdf')
```

## OCR for Scanned PDFs

```python
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def ocr_pdf(input_pdf, output_pdf):
    # Convert PDF pages to images
    images = convert_from_path(input_pdf)

    all_text = []
    for i, image in enumerate(images):
        # OCR each page
        text = pytesseract.image_to_string(image)
        all_text.append(f"--- Page {i+1} ---\n{text}")

    return '\n\n'.join(all_text)

# Usage
text = ocr_pdf('scanned_document.pdf', 'searchable.pdf')
print(text)
```

## Page Manipulation

### Rotate Pages

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('document.pdf')
writer = PdfWriter()

for page in reader.pages:
    page.rotate(90)  # Rotate 90 degrees clockwise
    writer.add_page(page)

writer.write('rotated.pdf')
```

### Add Watermark

```python
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def create_watermark(text):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    c.setFont("Helvetica", 60)
    c.setFillColorRGB(0.5, 0.5, 0.5, 0.3)  # Gray, transparent
    c.rotate(45)
    c.drawString(200, 100, text)
    c.save()
    packet.seek(0)
    return PdfReader(packet)

def add_watermark(input_pdf, output_pdf, watermark_text):
    watermark = create_watermark(watermark_text)
    watermark_page = watermark.pages[0]

    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)

    writer.write(output_pdf)

add_watermark('document.pdf', 'watermarked.pdf', 'CONFIDENTIAL')
```

## Security

### Password Protection

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('document.pdf')
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

# Encrypt with password
writer.encrypt(
    user_password='read_password',    # Required to open
    owner_password='owner_password',  # Required to edit
)

writer.write('protected.pdf')
```

### Remove Password

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('protected.pdf')
if reader.is_encrypted:
    reader.decrypt('password')

writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)

writer.write('unprotected.pdf')
```

## Command-Line Tools

```bash
# Extract text (poppler)
pdftotext document.pdf output.txt

# Merge PDFs (qpdf)
qpdf --empty --pages doc1.pdf doc2.pdf -- merged.pdf

# Split PDF (qpdf)
qpdf document.pdf --pages . 1-5 -- first_5_pages.pdf

# Compress PDF (qpdf)
qpdf --linearize input.pdf compressed.pdf
```

## Quick Reference

| Task | Tool | Command/Function |
|------|------|------------------|
| Extract text | pdfplumber | `page.extract_text()` |
| Extract tables | pdfplumber | `page.extract_tables()` |
| Merge | pypdf | `writer.append(pdf)` |
| Split | pypdf | `writer.add_page(page)` |
| Create | reportlab | `canvas.Canvas()` |
| Fill forms | pypdf | `update_page_form_field_values()` |
| OCR | pytesseract | `image_to_string()` |
| Encrypt | pypdf | `writer.encrypt()` |

## Checklist

- [ ] pypdf installed for basic operations
- [ ] pdfplumber installed for text/table extraction
- [ ] reportlab installed for PDF creation
- [ ] Tesseract installed for OCR (if needed)
- [ ] poppler-utils for command-line tools (optional)
