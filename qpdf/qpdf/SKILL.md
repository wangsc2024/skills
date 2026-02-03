---
name: qpdf
description: |
  QPDF is a command-line tool for PDF manipulation including splitting, merging, encryption, decryption, linearization, and page operations.
  Use when: working with PDF files, splitting PDF, merging PDF, encrypting/decrypting PDF, extracting pages, rotating pages, repairing PDF, or optimizing PDF for web.
  Triggers: qpdf, PDF split, PDF merge, PDF encrypt, PDF decrypt, linearize, PDF pages, PDF repair, 分割PDF, 合併PDF, PDF加密, PDF解密, PDF處理
---

# QPDF - PDF Manipulation Tool

QPDF is a command-line program and C++ library for structural, content-preserving transformations on PDF files. It's particularly useful for splitting, merging, encrypting, and manipulating PDF documents.

## Installation

```bash
# Ubuntu/Debian
sudo apt-get install qpdf

# macOS
brew install qpdf

# Windows (Chocolatey)
choco install qpdf

# Windows (Scoop)
scoop install qpdf
```

## Core Commands Quick Reference

### Split PDF into Individual Pages

```bash
# Split into single pages (output: file-01.pdf, file-02.pdf, etc.)
qpdf --split-pages input.pdf output.pdf

# Split into groups of N pages
qpdf --split-pages=2 input.pdf output.pdf   # 2 pages per file
qpdf --split-pages=5 input.pdf output.pdf   # 5 pages per file

# Custom output naming pattern
qpdf --split-pages input.pdf page-%d.pdf    # page-01.pdf, page-02.pdf, etc.
```

### Extract Specific Pages

```bash
# Extract pages 1-5
qpdf input.pdf --pages . 1-5 -- output.pdf

# Extract specific pages (1, 3, 5, 7)
qpdf input.pdf --pages . 1,3,5,7 -- output.pdf

# Extract pages in reverse order
qpdf input.pdf --pages . z-1 -- reversed.pdf

# Extract last 3 pages
qpdf input.pdf --pages . r3-r1 -- last3.pdf

# Extract odd pages only
qpdf input.pdf --pages . 1-z:odd -- odd_pages.pdf

# Extract even pages only
qpdf input.pdf --pages . 1-z:even -- even_pages.pdf

# Extract pages 1-10 except 3 and 4
qpdf input.pdf --pages . 1-10,x3-4 -- output.pdf
```

### Merge Multiple PDFs

```bash
# Merge two PDFs
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf

# Merge all PDFs in order
qpdf --empty --pages *.pdf -- merged.pdf

# Merge specific pages from multiple files
qpdf --empty --pages file1.pdf 1-5 file2.pdf 3-7 file3.pdf -- merged.pdf

# Merge with specific page ranges
qpdf --empty --pages cover.pdf -- body.pdf 1-z appendix.pdf -- complete.pdf
```

### Collate Pages (Interleave)

```bash
# Interleave pages from two files (1 page at a time)
qpdf --collate --empty --pages file1.pdf file2.pdf -- collated.pdf

# Collate in groups (2 pages from each file)
qpdf --collate=2 --empty --pages file1.pdf file2.pdf -- collated.pdf
```

### Rotate Pages

```bash
# Rotate all pages 90 degrees clockwise
qpdf input.pdf --rotate=+90 -- output.pdf

# Rotate all pages 90 degrees counter-clockwise
qpdf input.pdf --rotate=-90 -- output.pdf

# Rotate specific pages (pages 1, 3, 5)
qpdf input.pdf --rotate=+90:1,3,5 -- output.pdf

# Rotate page range
qpdf input.pdf --rotate=+180:5-10 -- output.pdf
```

### Encryption & Decryption

```bash
# Encrypt with 256-bit AES (recommended)
qpdf --encrypt user-password owner-password 256 -- input.pdf encrypted.pdf

# Encrypt with restrictions
qpdf --encrypt user-pass owner-pass 256 \
  --print=none \
  --modify=none \
  --extract=n \
  -- input.pdf secured.pdf

# Decrypt PDF (requires password)
qpdf --password=secret --decrypt input.pdf decrypted.pdf

# Remove encryption from PDF
qpdf --decrypt encrypted.pdf decrypted.pdf
```

**Encryption Restriction Options:**
- `--print=none|low|full` - Printing permission
- `--modify=none|assembly|form|annotate|all` - Modification permission
- `--extract=y|n` - Text/image extraction
- `--annotate=y|n` - Adding annotations
- `--form=y|n` - Filling forms

### Linearize (Optimize for Web)

```bash
# Linearize for fast web viewing
qpdf --linearize input.pdf web_optimized.pdf

# Linearize + compress
qpdf --linearize --compress-streams=y input.pdf optimized.pdf
```

### Compression & Optimization

```bash
# Compress streams
qpdf --compress-streams=y input.pdf compressed.pdf

# Decompress for editing
qpdf --stream-data=uncompress input.pdf uncompressed.pdf

# Optimize images (recompress)
qpdf --optimize-images input.pdf optimized.pdf

# Normalize content (for diff-friendly output)
qpdf --normalize-content=y input.pdf normalized.pdf
```

### QDF Mode (Human-Readable)

```bash
# Convert to QDF format (human-readable, editable)
qpdf --qdf input.pdf editable.pdf

# Convert to QDF with normalized content
qpdf --qdf --normalize-content=y input.pdf debug.pdf
```

### PDF Repair

```bash
# Attempt to repair damaged PDF
qpdf input.pdf repaired.pdf

# Repair with recovery mode
qpdf --recovery-mode=y input.pdf repaired.pdf
```

### PDF Inspection

```bash
# Show PDF metadata
qpdf --show-object=trailer input.pdf

# Show encryption info
qpdf --show-encryption input.pdf

# Check PDF validity
qpdf --check input.pdf

# Show page count
qpdf --show-npages input.pdf

# List all objects
qpdf --show-object-count input.pdf
```

### Overlay and Underlay

```bash
# Add watermark (overlay)
qpdf input.pdf --overlay watermark.pdf -- output.pdf

# Add background (underlay)
qpdf input.pdf --underlay background.pdf -- output.pdf

# Overlay on specific pages
qpdf input.pdf --overlay stamp.pdf --to=1-5 -- output.pdf
```

### Remove Pages

```bash
# Keep only pages 1-5, 10-15
qpdf input.pdf --pages . 1-5,10-15 -- output.pdf

# Remove page 3 (keep all except 3)
qpdf input.pdf --pages . 1-2,4-z -- output.pdf
```

## Page Range Syntax

| Syntax | Meaning |
|--------|---------|
| `1-5` | Pages 1 through 5 |
| `5-1` | Pages 5 through 1 (reverse) |
| `1,3,5` | Pages 1, 3, and 5 |
| `z` | Last page |
| `r1` | Last page (reverse notation) |
| `r3-r1` | Last 3 pages |
| `1-z:odd` | All odd-positioned pages |
| `1-z:even` | All even-positioned pages |
| `1-10,x3-4` | Pages 1-10 except 3-4 |

## Common Workflows

### Workflow 1: Split PDF by Chapter

```bash
# Split a book into chapters
qpdf book.pdf --pages . 1-20 -- chapter1.pdf
qpdf book.pdf --pages . 21-45 -- chapter2.pdf
qpdf book.pdf --pages . 46-70 -- chapter3.pdf
```

### Workflow 2: Create Print-Ready Booklet

```bash
# Extract and reorder for booklet printing
qpdf --empty --pages input.pdf 1-z:odd -- odd.pdf
qpdf --empty --pages input.pdf 1-z:even -- even.pdf
```

### Workflow 3: Secure Document Distribution

```bash
# Create secure PDF with view-only permissions
qpdf --encrypt viewpass ownerpass 256 \
  --print=none \
  --modify=none \
  --extract=n \
  --annotate=n \
  -- document.pdf secure_document.pdf
```

### Workflow 4: Batch Processing

```bash
# Split all PDFs in directory
for f in *.pdf; do
  qpdf --split-pages "$f" "split_${f%.pdf}-%d.pdf"
done

# Compress all PDFs
for f in *.pdf; do
  qpdf --compress-streams=y "$f" "compressed_$f"
done
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Errors occurred |
| 3 | Warnings but completed |

## Related Reference Files

For detailed information, see:
- [cli.md](references/cli.md) - Complete CLI documentation
- [advanced.md](references/advanced.md) - Linearization, encryption details
- [library.md](references/library.md) - C++ library usage
- [getting_started.md](references/getting_started.md) - Installation guide

## Official Resources

- Documentation: https://qpdf.readthedocs.io
- GitHub: https://github.com/qpdf/qpdf
- Downloads: https://github.com/qpdf/qpdf/releases
