# Tiptap - Marks

**Pages:** 1

---

## Import custom marks with .docx

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/custom-mark-conversion

**Contents:**
- Import custom marks with .docx
  - DOCX, "prosemirrorNodes" and "prosemirrorMarks"
  - Was this page helpful?

When importing a DOCX file, you can also define how custom marks should be converted back to Tiptap nodes. This is done by passing an array of custom mark definitions to the import command. You could use this feature to convert your custom stylings from Word into Titap with ease.

The latest version of the @tiptap-pro/extension-import-docx has the prosemirrorMarks configuration option available.

This option allows you to map custom nodes from the DOCX to your Tiptap schema. In the example above, we are mapping the strong and em nodes from the DOCX to the bold and italic nodes in our Tiptap schema.

By doing so, whenever the DOCX contains a strong or em node, it will be converted to a bold or italic node in Tiptap when imported.

Please note that the prosemirrorNodes and prosemirrorMarks options will only work if you're importing a .docx file. If you're importing another type of file, eg: an .odt file, the /import endpoint will be used instead of the /import-docx endpoint, and the prosemirrorNodes and prosemirrorMarks options will not be available.

**Examples:**

Example 1 (sql):
```sql
import { Import } from '@tiptap-pro/extension-import-docx'

const editor = new Editor({
  extensions: [
    // Other extensions ...
    Import.configure({
      appId: 'your-app-id',
      token: 'your-jwt',
      // ATTENTION: This is for demo purposes only
      endpoint: 'https://your-endpoint.com',
      imageUploadCallbackUrl: 'https://your-endpoint.com/image-upload',
      // ProseMirror custom mark mapping
      prosemirrorMarks: {
        strong: 'bold',
        em: 'italic',
      }
    }),
    // Other extensions ...
  ],
  // Other editor settings ...
})
```

Example 2 (sql):
```sql
import { Import } from '@tiptap-pro/extension-import-docx'

const editor = new Editor({
  extensions: [
    // Other extensions ...
    Import.configure({
      appId: 'your-app-id',
      token: 'your-jwt',
      // ATTENTION: This is for demo purposes only
      endpoint: 'https://your-endpoint.com',
      imageUploadCallbackUrl: 'https://your-endpoint.com/image-upload',
      // ProseMirror custom mark mapping
      prosemirrorMarks: {
        strong: 'bold',
        em: 'italic',
      }
    }),
    // Other extensions ...
  ],
  // Other editor settings ...
})
```

---
