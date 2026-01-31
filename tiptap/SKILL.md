---
name: tiptap
description: |
  Tiptap is a headless, framework-agnostic rich text editor based on ProseMirror. Use for building custom editors, handling nodes/marks/extensions, and implementing rich text editing features.
  Use when: building rich text editors, creating WYSIWYG interfaces, custom editor nodes, collaborative editing, or when user mentions Tiptap, ProseMirror, rich text, WYSIWYG, editor, Node Views, 富文本, 編輯器, markdown editor.
  Triggers: "Tiptap", "ProseMirror", "rich text", "WYSIWYG", "editor", "Node Views", "富文本", "編輯器", "text editor", "markdown editor", "collaborative editing", "custom nodes"
version: 1.0.0
---

# Tiptap Skill

Comprehensive assistance with Tiptap - the headless, extensible rich-text editor framework for modern web applications. Build custom WYSIWYG editors with React, Vue, Svelte, or vanilla JavaScript.

## When to Use This Skill

This skill should be triggered when:
- Building rich text editors (WYSIWYG) for web applications
- Working with Tiptap or ProseMirror-based editors
- Implementing custom nodes, marks, or extensions
- Adding features like mentions, tables, images, or code blocks
- Setting up real-time collaboration in editors
- Handling keyboard shortcuts and editor commands
- Converting content (Markdown, DOCX, HTML)

## Quick Reference

### Installation

```bash
# Core packages
npm install @tiptap/core @tiptap/pm @tiptap/starter-kit

# With React
npm install @tiptap/react

# With Vue 3
npm install @tiptap/vue-3

# With Svelte
npm install svelte-tiptap
```

### Basic Setup (React)

```jsx
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'

const Tiptap = () => {
  const editor = useEditor({
    extensions: [StarterKit],
    content: '<p>Hello World!</p>',
  })

  return <EditorContent editor={editor} />
}

export default Tiptap
```

### Basic Setup (Vue 3)

```vue
<template>
  <editor-content :editor="editor" />
</template>

<script setup>
import { useEditor, EditorContent } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'

const editor = useEditor({
  extensions: [StarterKit],
  content: '<p>Hello World!</p>',
})
</script>
```

### Basic Setup (Vanilla JavaScript)

```javascript
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  element: document.querySelector('.editor'),
  extensions: [StarterKit],
  content: '<p>Hello World!</p>',
})
```

## Common Patterns

### Using Commands

```javascript
// Make text bold
editor.commands.setBold()

// Chain multiple commands
editor.chain().focus().toggleBold().run()

// Insert content
editor.commands.insertContent('<p>New paragraph</p>')

// Set content (replace all)
editor.commands.setContent('<p>New document</p>')

// Clear content
editor.commands.clearContent()
```

### Toggle Formatting

```javascript
// Toggle marks
editor.chain().focus().toggleBold().run()
editor.chain().focus().toggleItalic().run()
editor.chain().focus().toggleStrike().run()
editor.chain().focus().toggleCode().run()
editor.chain().focus().toggleUnderline().run()

// Toggle blocks
editor.chain().focus().toggleHeading({ level: 1 }).run()
editor.chain().focus().toggleBulletList().run()
editor.chain().focus().toggleOrderedList().run()
editor.chain().focus().toggleCodeBlock().run()
editor.chain().focus().toggleBlockquote().run()
```

### Check Active State

```javascript
// Check if mark is active
editor.isActive('bold')
editor.isActive('italic')
editor.isActive('link')

// Check if node is active
editor.isActive('heading', { level: 1 })
editor.isActive('bulletList')
editor.isActive('codeBlock')
```

### Working with Links

```javascript
import Link from '@tiptap/extension-link'

// Configure
extensions: [
  Link.configure({
    openOnClick: false,
    HTMLAttributes: {
      rel: 'noopener noreferrer',
      target: '_blank',
    },
  }),
]

// Set link
editor.chain().focus().setLink({ href: 'https://example.com' }).run()

// Unset link
editor.chain().focus().unsetLink().run()

// Get link attributes
const { href } = editor.getAttributes('link')
```

### Working with Images

```javascript
import Image from '@tiptap/extension-image'

// Configure
extensions: [
  Image.configure({
    inline: true,
    allowBase64: true,
  }),
]

// Insert image
editor.chain().focus().setImage({ src: 'https://example.com/image.jpg' }).run()
```

### Tables

```javascript
import Table from '@tiptap/extension-table'
import TableRow from '@tiptap/extension-table-row'
import TableCell from '@tiptap/extension-table-cell'
import TableHeader from '@tiptap/extension-table-header'

// Insert table
editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()

// Table commands
editor.chain().focus().addColumnAfter().run()
editor.chain().focus().addRowAfter().run()
editor.chain().focus().deleteColumn().run()
editor.chain().focus().deleteRow().run()
editor.chain().focus().deleteTable().run()
editor.chain().focus().mergeCells().run()
editor.chain().focus().splitCell().run()
```

### Mentions

```javascript
import Mention from '@tiptap/extension-mention'

extensions: [
  Mention.configure({
    suggestion: {
      items: ({ query }) => {
        return users
          .filter(user => user.name.toLowerCase().includes(query.toLowerCase()))
          .slice(0, 5)
      },
      render: () => {
        // Return tippy.js popup with suggestions
      },
    },
  }),
]
```

### Custom Extension

```javascript
import { Node } from '@tiptap/core'

const CustomNode = Node.create({
  name: 'customNode',
  group: 'block',
  content: 'inline*',

  addAttributes() {
    return {
      color: {
        default: 'blue',
      },
    }
  },

  parseHTML() {
    return [{ tag: 'div[data-custom]' }]
  },

  renderHTML({ HTMLAttributes }) {
    return ['div', { 'data-custom': '', ...HTMLAttributes }, 0]
  },

  addCommands() {
    return {
      setCustomNode: (attributes) => ({ commands }) => {
        return commands.setNode(this.name, attributes)
      },
    }
  },
})
```

### Custom Mark

```javascript
import { Mark } from '@tiptap/core'

const Highlight = Mark.create({
  name: 'highlight',

  addAttributes() {
    return {
      color: {
        default: 'yellow',
      },
    }
  },

  parseHTML() {
    return [{ tag: 'mark' }]
  },

  renderHTML({ HTMLAttributes }) {
    return ['mark', HTMLAttributes, 0]
  },

  addCommands() {
    return {
      toggleHighlight: (attributes) => ({ commands }) => {
        return commands.toggleMark(this.name, attributes)
      },
    }
  },
})
```

### Event Handling

```javascript
const editor = new Editor({
  extensions: [StarterKit],
  content: '<p>Hello</p>',

  onUpdate: ({ editor }) => {
    const json = editor.getJSON()
    const html = editor.getHTML()
    console.log('Content updated:', json)
  },

  onSelectionUpdate: ({ editor }) => {
    console.log('Selection changed')
  },

  onFocus: ({ editor, event }) => {
    console.log('Editor focused')
  },

  onBlur: ({ editor, event }) => {
    console.log('Editor blurred')
  },

  onCreate: ({ editor }) => {
    console.log('Editor created')
  },

  onDestroy: () => {
    console.log('Editor destroyed')
  },
})
```

### Keyboard Shortcuts

```javascript
import { Extension } from '@tiptap/core'

const CustomKeymap = Extension.create({
  name: 'customKeymap',

  addKeyboardShortcuts() {
    return {
      'Mod-s': () => {
        // Save document
        return true
      },
      'Mod-Shift-s': () => {
        // Save as
        return true
      },
    }
  },
})
```

### Collaboration (Real-time)

```javascript
import Collaboration from '@tiptap/extension-collaboration'
import CollaborationCursor from '@tiptap/extension-collaboration-cursor'
import { TiptapCollabProvider } from '@hocuspocus/provider'
import * as Y from 'yjs'

const ydoc = new Y.Doc()

const provider = new TiptapCollabProvider({
  appId: 'your-app-id',
  name: 'document-name',
  token: 'your-jwt-token',
  document: ydoc,
})

const editor = new Editor({
  extensions: [
    StarterKit.configure({ history: false }),
    Collaboration.configure({ document: ydoc }),
    CollaborationCursor.configure({
      provider,
      user: { name: 'John Doe', color: '#ff0000' },
    }),
  ],
})
```

### Get/Set Content

```javascript
// Get content as JSON
const json = editor.getJSON()

// Get content as HTML
const html = editor.getHTML()

// Get content as text
const text = editor.getText()

// Set content from JSON
editor.commands.setContent(json)

// Set content from HTML
editor.commands.setContent('<p>Hello</p>')

// Check if empty
const isEmpty = editor.isEmpty
```

### Placeholder

```javascript
import Placeholder from '@tiptap/extension-placeholder'

extensions: [
  Placeholder.configure({
    placeholder: 'Write something...',
    // Or per-node placeholder
    placeholder: ({ node }) => {
      if (node.type.name === 'heading') {
        return 'Title'
      }
      return 'Write something...'
    },
  }),
]
```

### Character Count

```javascript
import CharacterCount from '@tiptap/extension-character-count'

extensions: [
  CharacterCount.configure({
    limit: 10000,
  }),
]

// Get counts
const characters = editor.storage.characterCount.characters()
const words = editor.storage.characterCount.words()
```

## Reference Files

| File | Description |
|------|-------------|
| **getting_started.md** | Installation for React, Vue, Svelte, vanilla JS |
| **core_concepts.md** | Schema, extensions, nodes, marks, ProseMirror |
| **api.md** | Commands, events, utilities, editor methods |
| **nodes.md** | Document, paragraph, heading, code block, table |
| **marks.md** | Bold, italic, link, code, highlight |
| **extensions.md** | Custom extensions, node views |
| **collaboration.md** | Real-time collaboration, Y.js |

## Key Concepts

### Document Structure
- **Nodes**: Block-level elements (paragraphs, headings, lists)
- **Marks**: Inline formatting (bold, italic, links)
- **Extensions**: Add features to the editor

### StarterKit Extensions
Includes: Document, Paragraph, Text, Bold, Italic, Strike, Code, Heading, BulletList, OrderedList, ListItem, Blockquote, CodeBlock, HardBreak, HorizontalRule, Dropcursor, Gapcursor, History

### ProseMirror
Tiptap is built on ProseMirror. Access ProseMirror APIs:

```javascript
// Access ProseMirror state
const { state, view } = editor

// Access transaction
editor.chain().command(({ tr }) => {
  // Manipulate transaction
  return true
}).run()
```

## Best Practices

1. **Always chain commands** with `.chain()...run()`
2. **Use `.focus()`** before commands to maintain editor focus
3. **Check `editor.can()`** before applying formatting
4. **Destroy editor** on component unmount
5. **Use StarterKit** for quick setup, customize later

## Resources

- **Official Docs**: https://tiptap.dev/docs
- **GitHub**: https://github.com/ueberdosis/tiptap
- **Examples**: https://tiptap.dev/docs/examples
- **Discord**: https://discord.com/invite/tiptap

## Supported Frameworks

| Framework | Package |
|-----------|---------|
| React | `@tiptap/react` |
| Vue 3 | `@tiptap/vue-3` |
| Vue 2 | `@tiptap/vue-2` |
| Svelte | `svelte-tiptap` |
| Next.js | `@tiptap/react` |
| Nuxt | `@tiptap/vue-3` |
| Alpine.js | `@tiptap/core` |
| Vanilla JS | `@tiptap/core` |

## Notes

- This skill was generated from 247 pages of official Tiptap documentation
- Tiptap is MIT licensed (open source core)
- Pro extensions require a subscription
- Built on ProseMirror for robust text editing

## Updating

```bash
skill-seekers scrape --config configs/tiptap.json
```
