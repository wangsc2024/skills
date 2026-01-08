# Tiptap - Other

**Pages:** 1

---

## Custom page layouts in DOCX

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/custom-page-layout

**Contents:**
- Custom page layouts in DOCX
- Page size configuration
  - Supported units
  - Configuration options
  - PositiveUniversalMeasure
- Page margins configuration
  - Configuration options
  - UniversalMeasure
  - PositiveUniversalMeasure
- Complete example

The DOCX Export extension now supports custom page sizes and margins, allowing you to create documents with precise layouts that match your requirements. Whether you need A5 pages, custom paper sizes, or specific margin configurations, you can configure these settings directly in the extension.

Use the pageSize option to define custom page dimensions for your exported DOCX files. The page size configuration accepts width and height values with various measurement units.

All page size and margin measurements support the following universal units:

A PositiveUniversalMeasure is a string representing a positive length with a unit, such as "21cm", "8.5in", "148mm", "595pt", "50pc", or "800px". Supported units include centimeters (cm), millimeters (mm), inches (in), points (pt), picas (pc), and pixels (px). The value must be greater than zero. If you provide a value without a unit, centimeters (cm) will be used by default.

The pageMargins option allows you to set custom margins for all sides of your document pages. Unlike page sizes, margins can accept negative values if needed.

A UniversalMeasure is a string representing a length with a unit, such as "21cm", "8.5in", "148mm", "595pt", "50pc", or "800px". Supported units include centimeters (cm), millimeters (mm), inches (in), points (pt), picas (pc), and pixels (px). The value can be positive or negative. If you provide a value without a unit, centimeters (cm) will be used by default.

A PositiveUniversalMeasure is a string representing a positive length with a unit, such as "21cm", "8.5in", "148mm", "595pt", "50pc", or "800px". Supported units include centimeters (cm), millimeters (mm), inches (in), points (pt), picas (pc), and pixels (px). The value must be greater than zero. If you provide a value without a unit, centimeters (cm) will be used by default.

Here's a complete example showing how to configure custom page layouts with the DOCX Export extension:

Here are some common page sizes you can use as reference:

You can mix and match different units based on your preference:

While you can mix different units, it's generally recommended to use consistent units throughout your configuration for better maintainability and clarity.

You can also override page layout settings when calling the export command directly:

This allows you to create different export configurations for different use cases without reconfiguring the entire extension.

**Examples:**

Example 1 (css):
```css
ExportDocx.configure({
  onCompleteExport: (result) => {
    // Handle the exported file
  },
  pageSize: {
    width: '14.8cm',    // A5 width
    height: '21cm',     // A5 height
  },
})
```

Example 2 (css):
```css
ExportDocx.configure({
  onCompleteExport: (result) => {
    // Handle the exported file
  },
  pageSize: {
    width: '14.8cm',    // A5 width
    height: '21cm',     // A5 height
  },
})
```

Example 3 (css):
```css
ExportDocx.configure({
  onCompleteExport: (result) => {
    // Handle the exported file
  },
  pageMargins: {
    top: '2cm',
    bottom: '2cm', 
    left: '1.5cm',
    right: '1.5cm',
  },
})
```

Example 4 (css):
```css
ExportDocx.configure({
  onCompleteExport: (result) => {
    // Handle the exported file
  },
  pageMargins: {
    top: '2cm',
    bottom: '2cm', 
    left: '1.5cm',
    right: '1.5cm',
  },
})
```

---
