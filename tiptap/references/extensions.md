# Tiptap - Extensions

**Pages:** 1

---

## Extend your DOCX export with Headers & Footers

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/headers-footers

**Contents:**
- Extend your DOCX export with Headers & Footers
- Headers Configuration
- Footers Configuration
- Usage Examples
- Extension Configuration
  - Note about Header and Footer Objects
  - Using Helper Functions (Alternative Approach)
  - Important note!
- Advanced Examples
  - Dynamic Headers and Footers with Async Functions

With the @tiptap-pro/extension-export-docx-header-footer you can extend your DOCX export functionality by allowing you to customize the headers and footers of the exported document.

When adding this extension it will allow you to configure your ExportDocx with some additional properties:

The headers object allows you to customize the headers of your exported DOCX document:

The footers object allows you to customize the footers of your exported DOCX document:

Headers and footers are configured through the ExportDocx.configure() method. You can use direct Docx namespace objects like Docx.Header and Docx.Footer for full control:

The Header, Footer, Paragraph, and TextRun objects are accessed through the Docx namespace exported from @tiptap-pro/extension-export-docx. You can customize them with any valid content including paragraphs, text runs, and more if available within the standard elements that can be used within a DOCX header or footer.

For easier handling of Tiptap-style content, you can use the convertHeader and convertFooter helper functions that automatically handle mark conversion, links, and other Tiptap features.

The convertHeader and convertFooter helper functions expect a Tiptap node, which in this case it's a paragraph, and they will automatically handle:

This makes it much easier to create rich headers and footers using familiar Tiptap JSON structure.

Be aware that if you use the convertHeader and convertFooter helpers you'd need to then use an asynchronous arrow functions as this convertHeader and convertFooter utility functions internally call convertParagraph helper which is an asynchronous function due to our image resolution implementation to handle images.

You can also provide asynchronous functions that return Header or Footer objects. This is useful for dynamic content generation, such as fetching user data, formatting current dates, or processing images.

Using async functions for headers and footers enables powerful use cases like: Fetching user information from APIs, including current timestamps or dynamic dates, processing and embedding images, calculating document statistics, retrieving data from databases or external services.

The asynchronous functions provided will be called after the entire document conversion has been processed, right before constructing the document.

You can combine static headers/footers with dynamic ones for different pages:

The Page numbering functionality requires using the manual Docx namespace approach. The convertHeader and convertFooter helper functions do not currently support Docx.PageNumber features. You must use new Docx.Header() or new Docx.Footer() directly to access page numbering capabilities.

**Examples:**

Example 1 (typescript):
```typescript
import { ExportDocx, Docx } from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // ... other extensions
    ExportDocx.configure({
      onCompleteExport: result => {
        // Handle the export result
        const blob = new Blob([result], {
          type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'export.docx'
        a.click()
        URL.revokeObjectURL(url)
      },
      headers: {
        evenAndOddHeaders: true,
        default: new Docx.Header({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "Default Header",
                  bold: true,
                }),
              ],
            }),
          ],
        }),
        first: new Docx.Header({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "First Page Header",
                  size: 24,
                  bold: true,
                }),
              ],
            }),
          ],
        }),
        even: new Docx.Header({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "Even Page Header",
                }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Docx.Footer({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "Default Footer",
                }),
              ],
            }),
          ],
        }),
      },
    }),
  ],
})

// Trigger export
editor.commands.exportDocx()
```

Example 2 (typescript):
```typescript
import { ExportDocx, Docx } from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // ... other extensions
    ExportDocx.configure({
      onCompleteExport: result => {
        // Handle the export result
        const blob = new Blob([result], {
          type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'export.docx'
        a.click()
        URL.revokeObjectURL(url)
      },
      headers: {
        evenAndOddHeaders: true,
        default: new Docx.Header({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "Default Header",
                  bold: true,
                }),
              ],
            }),
          ],
        }),
        first: new Docx.Header({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "First Page Header",
                  size: 24,
                  bold: true,
                }),
              ],
            }),
          ],
        }),
        even: new Docx.Header({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "Even Page Header",
                }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Docx.Footer({
          children: [
            new Docx.Paragraph({
              children: [
                new Docx.TextRun({
                  text: "Default Footer",
                }),
              ],
            }),
          ],
        }),
      },
    }),
  ],
})

// Trigger export
editor.commands.exportDocx()
```

Example 3 (typescript):
```typescript
import { ExportDocx } from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // ... other extensions
    ExportDocx.configure({
      onCompleteExport: result => {
        // Handle the export result
        const blob = new Blob([result], {
          type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'export.docx'
        a.click()
        URL.revokeObjectURL(url)
      },
      headers: {
        evenAndOddHeaders: true,
        default: async () =>
          convertHeader({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Header',
                  marks: [{ type: 'textStyle', attrs: { color: 'red' } }],
                },
              ],
            },
          }),
        first: async () =>
          convertHeader({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'First Page Header',
                  marks: [{ type: 'bold' }],
                },
              ],
            },
          }),
        even: async () =>
          convertHeader({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Even Page Header',
                  marks: [{ type: 'textStyle', attrs: { color: 'blue' } }],
                },
              ],
            },
          }),
      },
      footers: {
        default: async () =>
          convertFooter({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Footer',
                  marks: [{ type: 'textStyle', attrs: { color: 'red' } }],
                },
              ],
            },
          }),
        first: async () =>
          convertFooter({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'First Page Footer',
                  marks: [{ type: 'bold' }],
                },
              ],
            },
          }),
        even: async () =>
          convertFooter({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Even Page Footer',
                  marks: [{ type: 'textStyle', attrs: { color: 'blue' } }],
                },
              ],
            },
          }),
      },
    }),
  ],
})

// Trigger export
editor.commands.exportDocx()
```

Example 4 (typescript):
```typescript
import { ExportDocx } from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // ... other extensions
    ExportDocx.configure({
      onCompleteExport: result => {
        // Handle the export result
        const blob = new Blob([result], {
          type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'export.docx'
        a.click()
        URL.revokeObjectURL(url)
      },
      headers: {
        evenAndOddHeaders: true,
        default: async () =>
          convertHeader({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Header',
                  marks: [{ type: 'textStyle', attrs: { color: 'red' } }],
                },
              ],
            },
          }),
        first: async () =>
          convertHeader({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'First Page Header',
                  marks: [{ type: 'bold' }],
                },
              ],
            },
          }),
        even: async () =>
          convertHeader({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Even Page Header',
                  marks: [{ type: 'textStyle', attrs: { color: 'blue' } }],
                },
              ],
            },
          }),
      },
      footers: {
        default: async () =>
          convertFooter({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Footer',
                  marks: [{ type: 'textStyle', attrs: { color: 'red' } }],
                },
              ],
            },
          }),
        first: async () =>
          convertFooter({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'First Page Footer',
                  marks: [{ type: 'bold' }],
                },
              ],
            },
          }),
        even: async () =>
          convertFooter({
            node: {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: 'Even Page Footer',
                  marks: [{ type: 'textStyle', attrs: { color: 'blue' } }],
                },
              ],
            },
          }),
      },
    }),
  ],
})

// Trigger export
editor.commands.exportDocx()
```

---
