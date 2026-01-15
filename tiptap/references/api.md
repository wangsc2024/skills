# Tiptap - Api

**Pages:** 94

---

## blur command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/blur

**Contents:**
- blur command
- Use the blur command
  - Was this page helpful?

Understand the functionality of the blur command in Tiptap, which removes focus from the editor.

**Examples:**

Example 1 (sql):
```sql
// Remove the focus from the editor
editor.commands.blur()
```

Example 2 (sql):
```sql
// Remove the focus from the editor
editor.commands.blur()
```

---

## clearContent command

**URL:** https://tiptap.dev/docs/editor/api/commands/content/clear-content

**Contents:**
- clearContent command
- Parameters
- Examples
  - Was this page helpful?

The clearContent command deletes the current document. The editor will maintain at least one empty paragraph due to schema requirements.

See also: setContent, insertContent

emitUpdate?: boolean (true) Whether to emit an update event. Defaults to true (Note: This changed from false in v2).

**Examples:**

Example 1 (sql):
```sql
// Clear content (emits update event by default)
editor.commands.clearContent()

// Clear content without emitting update
editor.commands.clearContent(false)
```

Example 2 (sql):
```sql
// Clear content (emits update event by default)
editor.commands.clearContent()

// Clear content without emitting update
editor.commands.clearContent(false)
```

---

## clearNodes command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/clear-nodes

**Contents:**
- clearNodes command
- Use the clearNodes command
  - Was this page helpful?

The clearNodes command normalizes nodes to the default node, which is the paragraph by default. It’ll even normalize all kind of lists. For advanced use cases it can come in handy, before applying a new node type.

If you wonder how you can define the default node: It depends on what’s in the content attribute of your Document, by default that’s block+ (at least one block node) and the Paragraph node has the highest priority, so it’s loaded first and is therefore the default node.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.clearNodes()
```

Example 2 (unknown):
```unknown
editor.commands.clearNodes()
```

---

## Content Editor commands

**URL:** https://tiptap.dev/docs/editor/api/commands/content

**Contents:**
- Content Editor commands
- Use Cases
- List of content commands
  - Was this page helpful?

Use these commands to dynamically insert, replace, or remove content in your editor. Initialize new documents, update existing ones, or manage user selections, these commands provide you with tools to handle content manipulation.

---

## Convert .docx via REST API

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/rest-api

**Contents:**
- Convert .docx via REST API
  - Review the postman collection
- Import DOCX
  - Example (cURL)
  - Subscription required
  - Required headers
  - Body
  - Import verbose output
  - Verbose bitmask
  - Custom node and mark mapping

The DOCX document conversion API supports conversion from and to Tiptap’s JSON format.

You can also experiment with the Document Conversion API by heading over to our Postman Collection.

POST /v2/convert/import

The /v2/convert/import endpoint converts docx files into Tiptap’s JSON format. Users can POST documents to this endpoint and use various parameters to customize how different document elements are handled during the conversion process.

This endpoint requires a valid Tiptap subscription. For more details review our pricing page.

The DOCX import extension provides a verbose configuration property to help you control the level of diagnostic output during the import process. This is especially useful for debugging or for getting more insight into what happens during conversion.

The verbose property is a bitmask number that determines which types of log messages are emitted. The extension uses the following levels:

You can combine levels by adding their values together. For example, verbose: 3 will enable both log (1) and warn (2) messages.

The verbose output will give you, along the data property, one more property called logs, which will contain info, warn, and error properties, each of them being an array with all of the information related to that specific verbosity.

You can override the default node/mark types used during import by specifying them in the body of your request within prosemirrorNodes and prosemirrorMarks respectively. You would need to provide these if your editor uses custom nodes/marks and you want the imported JSON to use those.

For example, if your schema uses a custom node type called textBlock instead of the default paragraph, you can include "{\"textBlock\":\"paragraph\"}" in the request body.

You can similarly adjust headings, lists, marks like bold or italic, and more.

POST /v2/convert/export

The /v2/convert/export endpoint converts Tiptap documents into DOCX format. Users can POST documents to this endpoint and use various parameters to customize how different document elements are handled during the conversion process.

The /v2/convert/export endpoint does not support custom node conversions as functions cannot be serialized, but it does support custom style overrides. If you wish to convert your documents on the server on your own premises to have this option available, you can follow the server side export guide.

This endpoint requires a valid Tiptap subscription. For more details review our pricing page.

The pageSize object allows you to customize the dimensions of your exported DOCX document:

Example cURL with custom page size:

The pageMargins object allows you to customize the margins of your exported DOCX document:

Example cURL with custom page margins:

The headers and footers objects allows you to configure a set of headers and footers options for your exported DOCX.

In order to be able to use the headers and footers within the REST API you must have a Team license.

The headers object allows you to customize the headers of your exported DOCX document:

Since API calls cannot properly serialize very complex objects or functions, the headers configuration is limited to plain strings that we then will convert for you, without any styling, to a standard DOCX heading.

Example cURL with custom headers:

The footers object allows you to customize the footers of your exported DOCX document:

Since API calls cannot properly serialize very complex objects or functions, the footers configuration is limited to plain strings that we then will convert for you, without any styling, to a standard DOCX footer.

Example cURL with custom footers:

**Examples:**

Example 1 (unknown):
```unknown
curl -X POST "https://api.tiptap.dev/v2/convert/import" \
    -H "Authorization: Bearer YOUR_TOKEN" \
    -H "X-App-Id: YOUR_APP_ID" \
    -F "file=@/path/to/your/file.docx" \
    -F "imageUploadCallbackUrl=https://your-image-upload-endpoint.com" \
    -F "prosemirrorNodes={\"nodeKey\":\"nodeValue\"}" \
    -F "prosemirrorMarks={\"markKey\":\"markValue\"}"
```

Example 2 (unknown):
```unknown
curl -X POST "https://api.tiptap.dev/v2/convert/import" \
    -H "Authorization: Bearer YOUR_TOKEN" \
    -H "X-App-Id: YOUR_APP_ID" \
    -F "file=@/path/to/your/file.docx" \
    -F "imageUploadCallbackUrl=https://your-image-upload-endpoint.com" \
    -F "prosemirrorNodes={\"nodeKey\":\"nodeValue\"}" \
    -F "prosemirrorMarks={\"markKey\":\"markValue\"}"
```

Example 3 (json):
```json
{
  "data": {
    "content": {
        // Tiptap JSON
    }
  },
  "logs": {
    "info": [],
    "warn": [
      {
        "message": "Image file not found in media files",
        "fileName": "image1.gif",
        "availableMediaFiles": []
      }
    ],
    "error": [
      {
        "message": "Image upload failed: General error",
        "fileName": "image1.gif",
        "url": "https://your-image-upload-endpoint.com",
        "error": "Unable to connect. Is the computer able to access the url?",
        "context": "uploadImage general error"
      }
    ]
  }
}
```

Example 4 (json):
```json
{
  "data": {
    "content": {
        // Tiptap JSON
    }
  },
  "logs": {
    "info": [],
    "warn": [
      {
        "message": "Image file not found in media files",
        "fileName": "image1.gif",
        "availableMediaFiles": []
      }
    ],
    "error": [
      {
        "message": "Image upload failed: General error",
        "fileName": "image1.gif",
        "url": "https://your-image-upload-endpoint.com",
        "error": "Unable to connect. Is the computer able to access the url?",
        "context": "uploadImage general error"
      }
    ]
  }
}
```

---

## Convert Markdown via REST API

**URL:** https://tiptap.dev/docs/conversion/import-export/markdown/rest-api

**Contents:**
- Convert Markdown via REST API
  - Use Postman
- Import API endpoint
  - Import API Headers
  - Import API Fields
- Export API endpoint
    - Export API Headers
    - Export API Fields
  - Was this page helpful?

Converts .md files (or gfm) to Tiptap JSON.

Converts Tiptap JSON to .md or .gfm.

**Examples:**

Example 1 (unknown):
```unknown
curl -X POST "https://api.tiptap.dev/v1/convert/import?format=md" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "X-App-Id: <your-app-id>" \
  -F "file=@/path/to/file.md"
```

Example 2 (unknown):
```unknown
curl -X POST "https://api.tiptap.dev/v1/convert/import?format=md" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "X-App-Id: <your-app-id>" \
  -F "file=@/path/to/file.md"
```

Example 3 (json):
```json
curl -X POST "https://api.tiptap.dev/v1/convert/export" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "X-App-Id: <your-app-id>" \
  -F 'prosemirrorJson={"type":"doc","content":[{"type":"paragraph","content":[{"type":"text","text":"Hello from Tiptap!"}]}]}' \
  -F 'to=md' \
  --output document.md
```

Example 4 (json):
```json
curl -X POST "https://api.tiptap.dev/v1/convert/export" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "X-App-Id: <your-app-id>" \
  -F 'prosemirrorJson={"type":"doc","content":[{"type":"paragraph","content":[{"type":"text","text":"Hello from Tiptap!"}]}]}' \
  -F 'to=md' \
  --output document.md
```

---

## Convert ODT files via REST API

**URL:** https://tiptap.dev/docs/conversion/import-export/odt/rest-api

**Contents:**
- Convert ODT files via REST API
  - Review the postman collection
- Import API endpoint
  - Example (cURL)
  - Required headers
  - Body
- Export API endpoint
  - Required headers
  - Body
  - Was this page helpful?

The ODT document conversion API supports conversion from and to Tiptap’s JSON format.

You can also experiment with the Document Conversion API by heading over to our Postman Collection.

The /import endpoint enables the conversion of ODT files into Tiptap’s JSON format. Users can POST documents to this endpoint.

In this example, the request uploads an ODT file

The /export endpoint converts Tiptap documents back into the docx format.

**Examples:**

Example 1 (unknown):
```unknown
curl -X POST "https://api.tiptap.dev/v1/convert/import" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "X-App-Id: <your-app-id>" \
  -F "file=@/path/to/document.odt"
```

Example 2 (unknown):
```unknown
curl -X POST "https://api.tiptap.dev/v1/convert/import" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "X-App-Id: <your-app-id>" \
  -F "file=@/path/to/document.odt"
```

---

## createParagraphNear command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/create-paragraph-near

**Contents:**
- createParagraphNear command
- Use the createParagraphNear command
  - Was this page helpful?

If a block node is currently selected, the createParagraphNear command creates an empty paragraph after the currently selected block node. If the selected block node is the first child of its parent, the new paragraph will be inserted before the current selection.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.createParagraphNear()
```

Example 2 (unknown):
```unknown
editor.commands.createParagraphNear()
```

---

## Create an Emoji Inline Node with Markdown Support

**URL:** https://tiptap.dev/docs/editor/markdown/guides/create-a-emoji-inline-block

**Contents:**
- Create an Emoji Inline Node with Markdown Support
- Step 1: Create the basic emoji node
- Step 2: Add a custom Markdown tokenizer
- Step 3: Add the parser
- Step 4: Add the renderer
- Usage
- Testing and edge cases
  - Was this page helpful?

This guide shows how to add Markdown support for a small atomic inline node that renders emoji shortcodes (for example :smile:). We'll walk through four clear steps and include a full example at each step so you always have the complete context:

Example shorthand we'll support:

Start by defining a small atomic inline node that stores a name attribute and renders an emoji in HTML.

The tokenizer recognizes :name: shortcodes in inline Markdown and returns a token with the emoji name. Below is the full extension including the tokenizer so you can see how it integrates with the base Node.

Implementation notes:

The parseMarkdown function converts the tokenizer token into a Tiptap node. For an atomic inline node, it should return a node object with the type and attrs. Here's the full extension now including tokenizer + parse.

To support serializing the editor state back to Markdown shortcodes, implement the renderMarkdown function. It receives a Tiptap node and should return a Markdown string representing that node. Below is the full extension with tokenizer, parse, and render included.

Set the editor content from Markdown that contains emoji shortcodes. Depending on your Markdown integration, pass contentType: 'markdown' or use the API your setup provides:

This will produce inline emoji nodes with corresponding name attributes, and HTML rendering will display the mapped emoji characters (via emojiMap).

**Examples:**

Example 1 (unknown):
```unknown
Hello :smile: world!
```

Example 2 (unknown):
```unknown
Hello :smile: world!
```

Example 3 (javascript):
```javascript
import { Node } from '@tiptap/core'

const emojiMap = {
  smile: '😊',
  heart: '❤️',
  thumbsup: '👍',
  fire: '🔥',
  // add more mappings as needed
}

export const Emoji = Node.create({
  name: 'emoji',

  group: 'inline',
  inline: true,
  atom: true,

  addAttributes() {
    return {
      name: { default: 'smile' },
    }
  },

  parseHTML() {
    return [{ tag: 'span[data-emoji]' }]
  },

  renderHTML({ node }) {
    const emoji = emojiMap[node.attrs.name] || node.attrs.name || 'smile'
    return ['span', { 'data-emoji': node.attrs.name }, emoji]
  },
})
```

Example 4 (javascript):
```javascript
import { Node } from '@tiptap/core'

const emojiMap = {
  smile: '😊',
  heart: '❤️',
  thumbsup: '👍',
  fire: '🔥',
  // add more mappings as needed
}

export const Emoji = Node.create({
  name: 'emoji',

  group: 'inline',
  inline: true,
  atom: true,

  addAttributes() {
    return {
      name: { default: 'smile' },
    }
  },

  parseHTML() {
    return [{ tag: 'span[data-emoji]' }]
  },

  renderHTML({ node }) {
    const emoji = emojiMap[node.attrs.name] || node.attrs.name || 'smile'
    return ['span', { 'data-emoji': node.attrs.name }, emoji]
  },
})
```

---

## Create a Admonition Block with Markdown Support

**URL:** https://tiptap.dev/docs/editor/markdown/guides/create-a-admonition-block

**Contents:**
- Create a Admonition Block with Markdown Support
- Step 1: Create the basic extension
- Step 2: Add a custom Markdown tokenizer
- Step 3: Add the parser
- Step 4: Add the renderer
- Usage
- Testing and edge cases
  - Was this page helpful?

This guide walks you through adding Markdown support for a custom "Admonition" block in Tiptap. We'll break the process down into four clear steps and for each step include a full example that contains the code from previous steps so you always have full context.

We'll use the :::type style for admonitions, for example:

Start with a minimal Node definition that describes the structure, HTML parsing/rendering and attributes. Keep Markdown integration out for now so you can focus on schema and html input and output first.

Tiptap's Markdown integration can accept a tokenizer that converts Markdown source into tokens the Markdown parser understands. The tokenizer is responsible for recognizing the :::type ... ::: block and returning a token object with any relevant metadata and nested tokens (for the content).

Below is a full example that includes the base Node plus the markdownTokenizer added. This gives you full context for how tokenizer integrates with the Node.

Implementation details:

The parseMarkdown function receives the token produced by the tokenizer and must return a Tiptap-compatible JSON representation of a node (or nodes). Use the provided helpers to parse nested tokens into child content.

Below is the full example containing the base Node, the tokenizer, and now the parseMarkdown function. This shows how the pieces fit together.

To serialize content back to Markdown, implement the renderMarkdown function. This function receives a Tiptap node and should return the Markdown string representation. Use helpers.renderChildren to serialize the node's content.

Below is the full example with the tokenizer, parser, and renderer implemented so you have a complete extension that supports Markdown input and output as well as HTML rendering.

To set editor content from Markdown that uses the admonition syntax, pass the Markdown string and ensure contentType: 'markdown' (depending on your editor integration):

This will create an admonition node with type: 'warning' and the nested content parsed as Markdown.

**Examples:**

Example 1 (julia):
```julia
:::warning
This is a warning with **bold** text.
:::
```

Example 2 (julia):
```julia
:::warning
This is a warning with **bold** text.
:::
```

Example 3 (javascript):
```javascript
import { Node } from '@tiptap/core'

export const Admonition = Node.create({
  name: 'admonition',

  group: 'block',
  content: 'block+',

  addAttributes() {
    return {
      type: {
        default: 'note',
        parseHTML: (element) => element.getAttribute('data-type'),
        renderHTML: (attributes) => ({
          'data-type': attributes.type,
        }),
      },
    }
  },

  parseHTML() {
    return [{ tag: 'div[data-admonition]' }]
  },

  renderHTML({ node, HTMLAttributes }) {
    return ['div', { 'data-admonition': '', ...HTMLAttributes }, 0]
  },
})
```

Example 4 (javascript):
```javascript
import { Node } from '@tiptap/core'

export const Admonition = Node.create({
  name: 'admonition',

  group: 'block',
  content: 'block+',

  addAttributes() {
    return {
      type: {
        default: 'note',
        parseHTML: (element) => element.getAttribute('data-type'),
        renderHTML: (attributes) => ({
          'data-type': attributes.type,
        }),
      },
    }
  },

  parseHTML() {
    return [{ tag: 'div[data-admonition]' }]
  },

  renderHTML({ node, HTMLAttributes }) {
    return ['div', { 'data-admonition': '', ...HTMLAttributes }, 0]
  },
})
```

---

## Create a Highlight Mark with Markdown Support

**URL:** https://tiptap.dev/docs/editor/markdown/guides/create-a-highlight-mark

**Contents:**
- Create a Highlight Mark with Markdown Support
- Step 1: Create the basic highlight Mark
- Step 2: Add a custom Markdown tokenizer
- Step 3: Add the parser
- Step 4: Add the renderer
- Usage
- Testing and edge cases
  - Was this page helpful?

This guide walks through adding Markdown support for a small inline highlight mark that uses the ==text== shorthand (common in some Markdown flavors) to produce a mark element in HTML.

We'll follow four clear steps and at each step include a full example so you always have the complete context:

Example shorthand we'll support:

Start with a minimal Mark definition that describes the mark name, HTML parse/render behavior and any options. Keep Markdown integration out for now so you can focus on schema and HTML input/output first.

The tokenizer is responsible for recognizing ==text== in the raw Markdown and returning a token containing the inner text and any nested inline tokens. Keep this step focused on the tokenizer so you can test recognition independently.

Implementation notes:

The parseMarkdown function converts the token produced by the tokenizer into a Tiptap representation. For marks, you'll typically parse the inner tokens into inline nodes and then apply the mark to that content.

To support serializing editor state back to Markdown, implement the renderMarkdown function. It receives a Tiptap node (or mark node structure) and should return the Markdown string. Use helpers.renderChildren to serialize nested content.

Add the extension to your editor and set content from Markdown:

**Examples:**

Example 1 (unknown):
```unknown
This is ==highlighted== text.
```

Example 2 (unknown):
```unknown
This is ==highlighted== text.
```

Example 3 (javascript):
```javascript
import { Mark } from '@tiptap/core'

export const Highlight = Mark.create({
  name: 'highlight',

  addOptions() {
    return {
      HTMLAttributes: {},
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
      toggleHighlight:
        () =>
        ({ commands }) => {
          return commands.toggleMark(this.name)
        },
    }
  },
})
```

Example 4 (javascript):
```javascript
import { Mark } from '@tiptap/core'

export const Highlight = Mark.create({
  name: 'highlight',

  addOptions() {
    return {
      HTMLAttributes: {},
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
      toggleHighlight:
        () =>
        ({ commands }) => {
          return commands.toggleMark(this.name)
        },
    }
  },
})
```

---

## Custom Markdown Parsing

**URL:** https://tiptap.dev/docs/editor/markdown/advanced-usage/custom-parsing

**Contents:**
- Custom Markdown Parsing
- Creating and understanding a parse handler
- Parse Helper Functions
  - Parse inline-level child token with helpers.parseInline(tokens)
  - Parse block-level child token with helpers.parseChildren(tokens)
  - Parsing Marks with helpers.parseInline() and helpers.applyMark()
- HTML Parsing in Markdown
- Defining the Markdown token name
- Fallback Parsing
- Debug Parsing

This guide will walk you through the process of implementing custom Markdown parsing in Tiptap Editor. By the end of this tutorial, you'll be able to extract Tiptap JSON from Tokens.

Extensions can provide custom parsing logic to handle specific Markdown tokens. This is done through the markdown.parse handler.

A parse handler receives a Markdown token from MarkedJS and returns Tiptap JSON content that can be consumed by the editor. In addition to the token, the parse function also receives a helpers object with utility functions to assist in parsing.

These can be helpful for creating nodes, marks or parsing child MarkedJS tokens from token.tokens.

In this example the parse handler processes a heading token which is passed through by MarkedJS to our Markdown manager. This token is picked up in this example and transformed into a Tiptap node with the node type heading.

The appropriate level attribute is extracted from the token and it's inline content (as headlines can only contain marks or inline text) are parsed using the helpers.parseInline() function.

Important: Attributes on tokens can vary depending on how the Tokenizer is configured.

As described in the section above, the helpers object provides utility functions for parsing child tokens or creating nodes and marks. Let us go through each helper and see how they can be used.

This helper takes a list of tokens and tries to parse them as inline content (text nodes with marks). It will not verify if the tokens are actually inline tokens so make sure to only pass inline tokens here.

The function returns TiptapJSON[] that can be used as the content of a Tiptap Node.

Similar to parseInline(), but parses tokens as block-level content (e.g., list items, blockquotes, code blocks and more). It will not verify if the tokens are actually block-level tokens so make sure to only pass block-level tokens here.

The function returns TiptapJSON[] that can be used as the content of a Tiptap Node.

Use helpers.applyMark() to apply a mark to content:

When Markdown contains HTML, it's parsed using your extensions' existing parseHTML methods.

When parsing tokens to nodes or marks, it can happen that tokens may not map one-to-one to your node or mark names. In that case, you can use markdownTokenName to specify which token names to parse and to your nodes or marks type name.

If no extension handles a specific token type, the MarkdownManager provides fallback parsing for common tokens:

You can override this by providing your own handler for these token types.

Log tokens to understand what MarkedJS produces:

For large documents, consider parsing on demand:

Instead of re-parsing the entire document on each change, update specific sections:

Let's build a custom heading parser for a customHeading extension that will extract the heading level and also generate a unique ID for each heading.

Let's create a custom parser for a youtube token that turns the token into a youtubeEmbed node with the appropriate embed attributes.

**Examples:**

Example 1 (javascript):
```javascript
const MyHeading = Node.create({
  name: 'customHeading',
  // ...

  markdownTokenName: 'heading', // Token type to handle (optional, default is the extension name)
  parseMarkdown: (token, helpers) => {
    return {
      type: 'heading',
      attrs: { level: token.depth },
      content: helpers.parseInline(token.tokens || []),
    }
  },
})
```

Example 2 (javascript):
```javascript
const MyHeading = Node.create({
  name: 'customHeading',
  // ...

  markdownTokenName: 'heading', // Token type to handle (optional, default is the extension name)
  parseMarkdown: (token, helpers) => {
    return {
      type: 'heading',
      attrs: { level: token.depth },
      content: helpers.parseInline(token.tokens || []),
    }
  },
})
```

Example 3 (yaml):
```yaml
parse: (token, helpers) => {
  const content = helpers.parseInline(token.tokens || [])

  return {
    type: 'paragraph',
    content,
  }
}
```

Example 4 (yaml):
```yaml
parse: (token, helpers) => {
  const content = helpers.parseInline(token.tokens || [])

  return {
    type: 'paragraph',
    content,
  }
}
```

---

## Custom Markdown Serializing

**URL:** https://tiptap.dev/docs/editor/markdown/advanced-usage/custom-serializing

**Contents:**
- Custom Markdown Serializing
- Understanding Render Handlers
- Render Helper Functions
  - helpers.renderChildren(nodes, separator)
  - helpers.indent(content)
  - helpers.wrapInBlock(prefix, content)
- Serializing Marks
  - Marks with Attributes
- Render Context
- Rendering with Indentation

This guide will walk you through the process of implementing custom Markdown serialization in Tiptap Editor. By the end of this tutorial, you'll be able to serialize Tiptap JSON to Markdown content.

Serializing Tiptap JSON content to Markdown is done through the markdown.render handler.

The process of turning Tiptap JSON nodes into Markdown strings is handled by the renderMarkdown function defined in the config of an extension.

This function can range in complexity from a simple string return to more complex logic that takes into account the node's attributes, its children, nesting and the context in which it appears.

The following parameters are passed to the render function:

The returned value of the render function should be a string representing the Markdown equivalent of the node that will be concatenated with other strings to form the complete Markdown document.

As described above, the helpers object provides utility functions for rendering child nodes and formatting content. Let us go through each helper and see how they can be used.

The helpers.renderChildren function will take a list of Tiptap JSON nodes and render each of them to a Markdown string using their respective render handlers.

It accepts an optional separator parameter that can be used to join the rendered child nodes with a specific string (defaults to '').

or with a custom separator:

The helpers.indent(content) function will add indentation to each line of the provided content string based on the current context level and the configured indentation style (spaces or tabs).

This is helpful for example when rendering nested structures like lists.

The helpers.wrapInBlock function will wrap content with a prefix on each line which is useful for block-level elements like blockquotes or code blocks.

Marks are handled differently because they wrap inline content and need to be applied to the text within a node.

When rendering marks, you typically want to render the children of the node first and then wrap that content with the appropriate Markdown syntax for the mark.

The third parameter to render handlers is a context object that holds information about the current node's position in the document tree, like index, level, parent type and custom metadata.

The isIndenting flag tells the MarkdownManager that a node increases the nesting level:

This is important for proper indentation of nested lists and code blocks.

You can implement custom indentation:

Log the JSON structure before serialization:

Cache Markdown serialization results:

In this scenario we want to serialize a youtubeEmbed node back to a Markdown string that could be parsed by our custom YouTube tokenizer.

For this example, the syntax will be ![youtube](videoId?start=60&width=800&height=450)

For this example we want to render a custom list node that contains list items, each list items syntax should look like => item.

Because this is very verbose, Tiptap exports a renderNestedMarkdownContent() helper from the @tiptap/markdown package that can be used to simplify this:

Read more on our utilities page.

**Examples:**

Example 1 (javascript):
```javascript
const CustomHeading = Node.create({
  name: 'customHeading',

  // ...

  renderMarkdown: (node, helpers) => {
    const content = helpers.renderChildren(node.content)
    return `# ${content}\n\n`
  },
})
```

Example 2 (javascript):
```javascript
const CustomHeading = Node.create({
  name: 'customHeading',

  // ...

  renderMarkdown: (node, helpers) => {
    const content = helpers.renderChildren(node.content)
    return `# ${content}\n\n`
  },
})
```

Example 3 (javascript):
```javascript
render: (node, helpers) => {
  // Render all children
  const content = helpers.renderChildren(node.content || [])

  return `> ${content}\n\n`
}
```

Example 4 (javascript):
```javascript
render: (node, helpers) => {
  // Render all children
  const content = helpers.renderChildren(node.content || [])

  return `> ${content}\n\n`
}
```

---

## Custom Markdown Tokenizers

**URL:** https://tiptap.dev/docs/editor/markdown/advanced-usage/custom-tokenizer

**Contents:**
- Custom Markdown Tokenizers
- What are Tokenizers?
  - The Tokenization Flow
- When to Use Custom Tokenizers
- Tokenizer Structure
  - Properties Explained
    - name (required)
    - level (optional)
    - start (optional)
    - tokenize (required)

Custom tokenizers extend the Markdown parser to support non-standard or custom syntax. This guide explains how tokenizers work and how to create your own.

Tip: For standard patterns like Pandoc blocks or shortcodes, check the Utility Functions first—they provide ready-made tokenizers.

Tokenizers are functions that identify and parse custom Markdown syntax into tokens. They're registered with MarkedJS and run during the lexing phase, before Tiptap's parse handlers process the tokens.

Note: Want to learn more about Tokenizers? Check out the Glossary.

Use custom tokenizers when you want to support:

A tokenizer is an object with these properties:

A unique identifier for your token type:

This name will be used when registering parse handlers.

Whether this tokenizer operates at block or inline level:

A function that returns the index where your token might start in the source string. This is an optimization to avoid unnecessary parsing attempts:

This optimization helps MarkedJS skip irrelevant parts of the text. If omitted, MarkedJS will try your tokenizer at every position.

The main parsing function that identifies and tokenizes your syntax:

The function receives:

So as described above the flow of your Markdown content will be:

And from Tiptap JSON back to Markdown:

Let's create a tokenizer for highlight syntax (==text==).

Let's create a tokenizer for admonition blocks:

Let's create a tokenizer that supports nested inline parsing:

The lexer parameter provides helper functions to parse nested content:

Parse inline content (for inline-level tokenizers):

Parse block-level content (for block-level tokenizers):

Always anchor your regex to the start of the string:

Use +? or *? instead of + or * for better control:

Test your regex with:

Test your tokenizer independently:

Verify your tokenizer is registered:

Always return undefined when your syntax doesn't match:

Always include the full matched string in raw:

Make sure level matches your tokenizer's purpose:

Be careful not to consume content beyond your syntax:

For complex syntax, maintain state across tokenization:

**Examples:**

Example 1 (unknown):
```unknown
Markdown String
      ↓
Custom Tokenizers (identify custom syntax)
      ↓
Standard MarkedJS Lexer
      ↓
Markdown Tokens
      ↓
Extension Parse Handlers
      ↓
Tiptap JSON
```

Example 2 (unknown):
```unknown
Markdown String
      ↓
Custom Tokenizers (identify custom syntax)
      ↓
Standard MarkedJS Lexer
      ↓
Markdown Tokens
      ↓
Extension Parse Handlers
      ↓
Tiptap JSON
```

Example 3 (typescript):
```typescript
type MarkdownTokenizer = {
  name: string // Token name (must be unique)
  level?: 'block' | 'inline' // Level: block or inline
  start?: (src: string) => number // Where the token starts
  tokenize: (src, tokens, lexer) => MarkdownToken | undefined
}
```

Example 4 (typescript):
```typescript
type MarkdownTokenizer = {
  name: string // Token name (must be unique)
  level?: 'block' | 'inline' // Level: block or inline
  start?: (src: string) => number // Where the token starts
  tokenize: (src, tokens, lexer) => MarkdownToken | undefined
}
```

---

## cut command

**URL:** https://tiptap.dev/docs/editor/api/commands/content/cut

**Contents:**
- cut command
- Use the cut command
  - Was this page helpful?

This command cuts out content and places it into the given position.

**Examples:**

Example 1 (sql):
```sql
const from = editor.state.selection.from
const to = editor.state.selection.to

const endPos = editor.state.doc.nodeSize - 2

// Cut out content from range and put it at the end of the document
editor.commands.cut({ from, to }, endPos)
```

Example 2 (sql):
```sql
const from = editor.state.selection.from
const to = editor.state.selection.to

const endPos = editor.state.doc.nodeSize - 2

// Cut out content from range and put it at the end of the document
editor.commands.cut({ from, to }, endPos)
```

---

## deleteNode command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/delete-node

**Contents:**
- deleteNode command
- Parameters
- Use the deleteNode command
  - Was this page helpful?

The deleteNode command deletes a node inside the current selection. It requires a typeOrName argument, which can be a string or a NodeType to find the node that needs to be deleted. After deleting the node, the view will automatically scroll to the cursors position.

typeOrName: string | NodeType

**Examples:**

Example 1 (unknown):
```unknown
// deletes a paragraph node
editor.commands.deleteNode('paragraph')

// or

// deletes a custom node
editor.commands.deleteNode(MyCustomNode)
```

Example 2 (unknown):
```unknown
// deletes a paragraph node
editor.commands.deleteNode('paragraph')

// or

// deletes a custom node
editor.commands.deleteNode(MyCustomNode)
```

---

## deleteRange commands

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/delete-range

**Contents:**
- deleteRange commands
- Parameters
- Use the deleteRange command
  - Was this page helpful?

The deleteRange command deletes everything in a given range. It requires a range attribute of type Range.

**Examples:**

Example 1 (css):
```css
editor.commands.deleteRange({ from: 0, to: 12 })
```

Example 2 (css):
```css
editor.commands.deleteRange({ from: 0, to: 12 })
```

---

## deleteSelection command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/delete-selection

**Contents:**
- deleteSelection command
- Use the deleteSelection command
  - Was this page helpful?

The deleteSelection command in Tiptap targets and removes any nodes or content that are currently selected within the editor.

The deleteSelection command deletes the currently selected nodes. If no selection exists, nothing will be deleted.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.deleteSelection()
```

Example 2 (unknown):
```unknown
editor.commands.deleteSelection()
```

---

## Document management API

**URL:** https://tiptap.dev/docs/collaboration/documents/rest-api

**Contents:**
- Document management API
- Rate limits
  - Default rate limits (per source IP):
- Access the API
  - Authentication
  - Document identifiers
- API endpoints overview
- Create a document
- Batch import documents
- Get a document

The Collaboration Management API provides a suite of RESTful endpoints for managing documents. This API can be used for document creation, listing, retrieval, updates, deletion, and duplication.

You can experiment with the REST API by visiting our Postman Collection.

To maintain system integrity and protect from misconfigured clients, our infrastructure—including the management API and websocket connections through the TiptapCollabProvider—is subject to rate limits.

If you encounter these limits under normal operation, please email us.

The REST API is exposed directly from your Document server at your custom URL:

Authenticate your API requests by including your API secret in the Authorization header. You can find your API secret in the settings of your Tiptap Cloud dashboard.

If your document identifier contains a slash (/), encode it as %2F, e.g., using encodeURIComponent.

Access the Collaboration Management API to manage your documents efficiently. For a comprehensive view of all endpoints across Tiptap products, explore our Postman Collection, which includes detailed examples and configurations.

Take a look at the metrics and statistics endpoints as well!

This call lets you create a document using binary Yjs or JSON format (default: yjs). It can be used to seed documents before a user connects to the Tiptap Collaboration server.

The endpoint returns HTTP status 204 if the document is created successfully, or 409 if the document already exists. To overwrite an existing document, you must delete it first.

This call lets you import multiple documents in bulk using a predefined JSON structure. Each document must include its metadata (such as created_at, name, and version) and its content in the Tiptap JSON format.

The endpoint returns HTTP status 204 if the documents are imported successfully, or 400 if the request contains invalid data.

This call lets you export the specified document with all fragments in JSON or Yjs format. If the document is currently open on your server, we will return the in-memory version; otherwise, we read from the database.

format supports either yjs, base64, text, or json (default: json). If you choose the yjs format, you'll get the binary Yjs update message created with Y.encodeStateAsUpdate.

fragment can be an array (e.g., fragment=a&fragment=b) or a single fragment you want to export. By default, we only export the default fragment. This parameter is only applicable when using the json or textformat; with yjs, you'll always get the entire Yjs document.

When using axios, you need to specify responseType: arraybuffer in the request options.

When using node-fetch, you need to use .arrayBuffer() and create a Buffer from it:

This call returns a paginated list of all documents in storage. By default, we return the first 100 documents. Pass take and skip parameters to adjust pagination.

This call lets you copy or duplicate a document. First, retrieve the document using the GET endpoint and then create a new one with the POST call. Here's an example in typescript:

Note that the new document will not have the versions of the source document. If you want to preserve versions, you can use the import/export endpoint (see the postman collection)

This call lets you encrypt a document with the specified identifier using Base64 encryption.

The endpoint returns HTTP status 204 if the document is successfully encrypted, or 404 if the document does not exist.

This call lets you revert a document to a specific previous version by applying an update that corresponds to a prior state of the document. You must specify the version to revert to in the request body.

The endpoint returns HTTP status 204 if the document is successfully reverted, or 404 if the document or version is not found.

This call accepts a Yjs update message and applies it to the existing document on the server.

The endpoint returns the HTTP status 204 if the document was updated successfully, 404 if the document does not exist, or 422 if the payload is invalid or the update cannot be applied.

The API endpoint also supports JSON document updates, document history for tracking changes without replacing the entire document, and node-specific updates.

For more detailed information on manipulating documents using JSON instead of Yjs, refer to our Content injection page.

This call deletes a document from the server after closing any open connection to the document.

It returns either HTTP status 204 if the document was deleted successfully, or 404 if the document was not found.

If the endpoint returns 204 but the document still exists, make sure that no user is re-creating the document from the provider. We close all connections before deleting a document, but your error handling might recreate the provider, thus creating the document again.

When Tiptap Semantic Search is enabled, you can perform contextually aware searches across all your documents.

Please handle the search requests in your backend to keep your API key secret. Consider enforcing rate limits in your application as necessary.

You can use the following query parameters to adjust the search results:

**Examples:**

Example 1 (yaml):
```yaml
https://YOUR_APP_ID.collab.tiptap.cloud/
```

Example 2 (yaml):
```yaml
https://YOUR_APP_ID.collab.tiptap.cloud/
```

Example 3 (unknown):
```unknown
POST /api/documents/:identifier
```

Example 4 (unknown):
```unknown
POST /api/documents/:identifier
```

---

## Editor

**URL:** https://tiptap.dev/docs/editor/markdown/api/editor

**Contents:**
- Editor
- Methods
  - Editor.getMarkdown()
- Properties
  - Editor.markdown
    - Example
- Options
  - Editor.content
  - Editor.contentType
- Command Options

The Markdown package does not export a new Editor class but extends the existing Tiptap Editor class. This means you can use all the standard methods and options of Tiptap's Editor, along with the additional functionality provided by the Markdown package.

Get the current content of the editor as Markdown.

Access the MarkdownManager instance.

Editor content supports HTML, Markdown or Tiptap JSON as a value.

Note: For Markdown support editor.contentAsMarkdown must be set to true.

Defines what type of content is passed to the editor. Defaults to json. When an invalid combination is set - for example content that is a JSON object, but the contentType is set to markdown, the editor will automatically fall back to json and vice versa.

Set editor content from Markdown.

boolean - Whether the command succeeded

Insert Markdown content at the current selection.

boolean - Whether the command succeeded

Insert Markdown content at a specific position.

boolean - Whether the command succeeded

The extension spec also gets extended with the following options:

The name of the token used for Markdown parsing.

A function to customize how Markdown tokens are parsed from Markdown token into ProseMirror nodes.

A function to customize how ProseMirror nodes are rendered to Markdown tokens.

A tokenizer configuration object that creates a custom tokenizer to turn Markdown string into tokens.

A optional object to pass additional options to the Markdown parser and serializer.

**Examples:**

Example 1 (javascript):
```javascript
const markdown = editor.getMarkdown()
```

Example 2 (javascript):
```javascript
const markdown = editor.getMarkdown()
```

Example 3 (unknown):
```unknown
editor.markdown: MarkdownManager
```

Example 4 (unknown):
```unknown
editor.markdown: MarkdownManager
```

---

## Editor commands

**URL:** https://tiptap.dev/docs/editor/api/commands

**Contents:**
- Editor commands
- Execute a command
  - Chain commands
  - Transaction mapping
    - Chain inside custom commands
  - Inline commands
  - Dry run commands
  - Try commands
- List of commands
  - Content

The editor provides a ton of commands to programmatically add or change content or alter the selection. If you want to build your own editor you definitely want to learn more about them.

All available commands are accessible through an editor instance. Let’s say you want to make text bold when a user clicks on a button. That’s how that would look like:

While that’s perfectly fine and does make the selected bold, you’d likely want to chain multiple commands in one run. Let’s have a look at how that works.

Most commands can be combined to one call. That’s shorter than separate function calls in most cases. Here is an example to make the selected text bold:

The .chain() is required to start a new chain and the .run() is needed to actually execute all the commands in between.

In the example above two different commands are executed at once. When a user clicks on a button outside of the content, the editor isn’t in focus anymore. That’s why you probably want to add a .focus() call to most of your commands. It brings back the focus to the editor, so the user can continue to type.

All chained commands are kind of queued up. They are combined to one single transaction. That means, the content is only updated once, also the update event is only triggered once.

By default Prosemirror does not support chaining which means that you need to update the positions between chained commands via Transaction mapping.

For example you want to chain a delete and insert command in one chain, you need to keep track of the position inside your chain commands. Here is an example:

Now you can do the following without insert inserting the content into the wrong position:

When chaining a command, the transaction is held back. If you want to chain commands inside your custom commands, you’ll need to use said transaction and add to it. Here is how you would do that:

In some cases, it’s helpful to put some more logic in a command. That’s why you can execute commands in commands. I know, that sounds crazy, but let’s look at an example:

Sometimes, you don’t want to actually run the commands, but only know if it would be possible to run commands, for example to show or hide buttons in a menu. That’s what we added .can() for. Everything coming after this method will be executed, without applying the changes to the document:

And you can use it together with .chain(), too. Here is an example which checks if it’s possible to apply all the commands:

Both calls would return true if it’s possible to apply the commands, and false in case it’s not.

In order to make that work with your custom commands, don’t forget to return true or false.

For some of your own commands, you probably want to work with the raw transaction. To make them work with .can() you should check if the transaction should be dispatched. Here is how you can create a simple .insertText() command:

If you’re just wrapping another Tiptap command, you don’t need to check that, we’ll do it for you.

If you’re just wrapping a plain ProseMirror command, you’ll need to pass dispatch anyway. Then there’s also no need to check it:

If you want to run a list of commands, but want only the first successful command to be applied, you can do this with the .first() method. This method runs one command after the other and stops at the first which returns true.

For example, the backspace key tries to undo an input rule first. If that was successful, it stops there. If no input rule has been applied and thus can’t be reverted, it runs the next command and deletes the selection, if there is one. Here is the simplified example:

Inside of commands you can do the same thing:

Have a look at all of the core commands listed below. They should give you a good first impression of what’s possible.

All extensions can add additional commands (and most do), check out the specific documentation for the provided nodes, marks, and functionality to learn more about those. And of course, you can add your custom extensions with custom commands as well. But how do you write those commands? There’s a little bit to learn about that.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.setBold()
```

Example 2 (unknown):
```unknown
editor.commands.setBold()
```

Example 3 (unknown):
```unknown
editor.chain().focus().toggleBold().run()
```

Example 4 (unknown):
```unknown
editor.chain().focus().toggleBold().run()
```

---

## Editor Instance API

**URL:** https://tiptap.dev/docs/editor/api/editor

**Contents:**
- Editor Instance API
- Settings
  - element
  - extensions
  - content
  - editable
  - textDirection
  - autofocus
  - enableInputRules
  - enablePasteRules

The editor instance is a central building block of Tiptap. It does most of the heavy lifting of creating a working ProseMirror editor such as creating the EditorView, setting the initial EditorState and so on.

The Editor class accepts a bunch of settings. Here is a list of all available settings:

The element specifies the HTML element the editor will be binded to. The following code will integrate Tiptap with an element with the .element class:

You can even initiate your editor before mounting it to an element. This is useful when your DOM is not yet available or in a server-side rendering environment. Just initialize the editor with null and mount it later.

It’s required to pass a list of extensions to the extensions property, even if you only want to allow paragraphs.

With the content property you can provide the initial content for the editor. This can be HTML or JSON.

The editable property determines if users can write into the editor.

The textDirection property sets the text direction for all content in the editor. This is useful for right-to-left (RTL) languages like Arabic and Hebrew, or for bidirectional text content.

You can also override direction for specific nodes using the setTextDirection and unsetTextDirection commands. See the commands documentation for more details.

With autofocus you can force the cursor to jump in the editor on initialization.

By default, Tiptap enables all input rules. With enableInputRules you can control that.

Alternatively you can allow only specific input rules.

By default, Tiptap enables all paste rules. With enablePasteRules you can control that.

Alternatively you can allow only specific paste rules.

By default, Tiptap injects a little bit of CSS. With injectCSS you can disable that.

When you use a Content-Security-Policy with nonce, you can specify a nonce to be added to dynamically created elements. Here is an example:

For advanced use cases, you can pass editorProps which will be handled by ProseMirror. You can use it to override various editor events or change editor DOM element attributes, for example to add some Tailwind classes. Here is an example:

You can use that to hook into event handlers and pass - for example - a custom paste handler, too.

Passed content is parsed by ProseMirror. To hook into the parsing, you can pass parseOptions which are then handled by ProseMirror.

The editor instance will provide a bunch of public methods. Methods are regular functions and can return anything. They’ll help you to work with the editor.

Don’t confuse methods with commands. Commands are used to change the state of editor (content, selection, and so on) and only return true or false.

Check if a command or a command chain can be executed – without actually executing it. Can be very helpful to enable/disable or show/hide buttons.

Create a command chain to call multiple commands at once.

Stops the editor instance and unbinds all events.

Returns the current editor document as HTML

Returns the current editor document as JSON.

Returns the current editor document as plain text.

Get attributes of the currently selected node or mark.

Returns if the currently selected node or mark is active.

Mount the editor to an element. This is useful when you want to mount the editor to an element that is not yet available in the DOM.

Unmount the editor from an element. This is useful when you want to unmount the editor from an element, but later want to re-mount it to another element.

Register a ProseMirror plugin.

Update editor options.

Update editable state of the editor.

Unregister a ProseMirror plugin.

See the NodePos class.

Returns whether the editor is editable or read-only.

Check if there is content.

Check if the editor is focused.

Check if the editor is destroyed.

Check if the editor is capturing a transaction.

**Examples:**

Example 1 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  element: document.querySelector('.element'),
  extensions: [StarterKit],
})
```

Example 2 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  element: document.querySelector('.element'),
  extensions: [StarterKit],
})
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  element: null,
  extensions: [StarterKit],
})

// Later in your code
editor.mount(document.querySelector('.element'))
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  element: null,
  extensions: [StarterKit],
})

// Later in your code
editor.mount(document.querySelector('.element'))
```

---

## enter command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/enter

**Contents:**
- enter command
- Use the enter command
  - Was this page helpful?

The enter command triggers an enter programmatically.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.enter()
```

Example 2 (unknown):
```unknown
editor.commands.enter()
```

---

## Events in Tiptap

**URL:** https://tiptap.dev/docs/editor/api/events

**Contents:**
- Events in Tiptap
- List of available events
- Register event listeners
  - Option 1: Configuration
  - Option 2: Binding
    - Bind event listeners
    - Unbind event listeners
  - Option 3: Extensions
  - Was this page helpful?

The editor fires a few different events that you can hook into. Let’s have a look at all the available events first.

There are three ways to register event listeners.

You can define your event listeners on a new editor instance right-away:

Or you can register your event listeners on a running editor instance:

If you need to unbind those event listeners at some point, you should register your event listeners with .on() and unbind them with .off() then.

Moving your event listeners to custom extensions (or nodes, or marks) is also possible. Here’s how that would look like:

**Examples:**

Example 1 (sql):
```sql
const editor = new Editor({
  onBeforeCreate({ editor }) {
    // Before the view is created.
  },
  onCreate({ editor }) {
    // The editor is ready.
  },
  onUpdate({ editor }) {
    // The content has changed.
  },
  onSelectionUpdate({ editor }) {
    // The selection has changed.
  },
  onTransaction({ editor, transaction }) {
    // The editor state has changed.
  },
  onFocus({ editor, event }) {
    // The editor is focused.
  },
  onBlur({ editor, event }) {
    // The editor isn’t focused anymore.
  },
  onDestroy() {
    // The editor is being destroyed.
  },
  onPaste(event: ClipboardEvent, slice: Slice) {
    // The editor is being pasted into.
  },
  onDrop(event: DragEvent, slice: Slice, moved: boolean) {
    // The editor is being pasted into.
  },
  onDelete({ type, deletedRange, newRange, partial, node, mark, from, to, newFrom, newTo }) {
    // Content was deleted from the editor (either a node or mark).
  },
  onContentError({ editor, error, disableCollaboration }) {
    // The editor content does not match the schema.
  },
})
```

Example 2 (sql):
```sql
const editor = new Editor({
  onBeforeCreate({ editor }) {
    // Before the view is created.
  },
  onCreate({ editor }) {
    // The editor is ready.
  },
  onUpdate({ editor }) {
    // The content has changed.
  },
  onSelectionUpdate({ editor }) {
    // The selection has changed.
  },
  onTransaction({ editor, transaction }) {
    // The editor state has changed.
  },
  onFocus({ editor, event }) {
    // The editor is focused.
  },
  onBlur({ editor, event }) {
    // The editor isn’t focused anymore.
  },
  onDestroy() {
    // The editor is being destroyed.
  },
  onPaste(event: ClipboardEvent, slice: Slice) {
    // The editor is being pasted into.
  },
  onDrop(event: DragEvent, slice: Slice, moved: boolean) {
    // The editor is being pasted into.
  },
  onDelete({ type, deletedRange, newRange, partial, node, mark, from, to, newFrom, newTo }) {
    // Content was deleted from the editor (either a node or mark).
  },
  onContentError({ editor, error, disableCollaboration }) {
    // The editor content does not match the schema.
  },
})
```

Example 3 (sql):
```sql
editor.on('beforeCreate', ({ editor }) => {
  // Before the view is created.
})

editor.on('create', ({ editor }) => {
  // The editor is ready.
})

editor.on('update', ({ editor }) => {
  // The content has changed.
})

editor.on('selectionUpdate', ({ editor }) => {
  // The selection has changed.
})

editor.on('transaction', ({ editor, transaction }) => {
  // The editor state has changed.
})

editor.on('focus', ({ editor, event }) => {
  // The editor is focused.
})

editor.on('blur', ({ editor, event }) => {
  // The editor isn’t focused anymore.
})

editor.on('destroy', () => {
  // The editor is being destroyed.
})

editor.on('paste', ({ event, slice, editor }) => {
  // The editor is being pasted into.
})

editor.on('drop', ({ editor, event, slice, moved }) => {
  // The editor is being destroyed.
})

editor.on('delete', ({ type, deletedRange, newRange, partial, node, mark }) => {
  // Content was deleted from the editor (either a node or mark).
})

editor.on('contentError', ({ editor, error, disableCollaboration }) => {
  // The editor content does not match the schema.
})
```

Example 4 (sql):
```sql
editor.on('beforeCreate', ({ editor }) => {
  // Before the view is created.
})

editor.on('create', ({ editor }) => {
  // The editor is ready.
})

editor.on('update', ({ editor }) => {
  // The content has changed.
})

editor.on('selectionUpdate', ({ editor }) => {
  // The selection has changed.
})

editor.on('transaction', ({ editor, transaction }) => {
  // The editor state has changed.
})

editor.on('focus', ({ editor, event }) => {
  // The editor is focused.
})

editor.on('blur', ({ editor, event }) => {
  // The editor isn’t focused anymore.
})

editor.on('destroy', () => {
  // The editor is being destroyed.
})

editor.on('paste', ({ event, slice, editor }) => {
  // The editor is being pasted into.
})

editor.on('drop', ({ editor, event, slice, moved }) => {
  // The editor is being destroyed.
})

editor.on('delete', ({ type, deletedRange, newRange, partial, node, mark }) => {
  // Content was deleted from the editor (either a node or mark).
})

editor.on('contentError', ({ editor, error, disableCollaboration }) => {
  // The editor content does not match the schema.
})
```

---

## exitCode command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/exit-code

**Contents:**
- exitCode command
- Use the exitCode command
  - Was this page helpful?

The exitCode command will create a default block after the current selection if the selection is a code element and move the cursor to the new block.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.exitCode()
```

Example 2 (unknown):
```unknown
editor.commands.exitCode()
```

---

## Export .docx from your editor

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/editor-export

**Contents:**
- Export .docx from your editor
- Install the DOCX Export extension
- Configuring the extension
- Export a DOCX file
  - How it works
- Server-side export
- Support & Limitations
  - Was this page helpful?

Use Tiptap’s @tiptap-pro/extension-export-docx to export the editor's content as a .docx file. This extension integrates DOCX export functionality into your editor.

You can use this extension under any JavaScript environment, including server-side applications due to the isomorphic nature of the exportDocx function.

You can also use the REST API instead if you'd prefer to handle the conversion on our end.

By default, the extension maps Tiptap nodes to DOCX elements. If your content includes custom nodes, configure their export behavior to ensure they’re properly converted.

The Conversion extensions are published in Tiptap’s private npm registry. Integrate the extensions by following the private registry guide.

Once done you can install and import the Export DOCX extension package

Using the export extension does not require any Tiptap Conversion credentials, since the conversion is handled right away in the extension.

The ExportDocx extension can be configured with an ExportDocxOptions (object) as an argument to the configure method with the following properties:

With the extension installed, you can convert your editor’s content to .docx.

Before diving into an example, let's take a look into the signature of the exportDocx method available in your editor's commands:

The exportDocx method takes an optional ExportDocxOptions (object) as an argument with the following properties that you can use to override the ones that you have configured with the ExportDocx.configure method:

The above example runs entirely in the browser, generating a DOCX Blob via the ExportDocx extension since it's the default value for the exportType as we haven't override it. We then programmatically download the file. You can adjust this logic, for instance, to send the blob to a server instead of downloading.

For applications requiring complex document generation or to reduce client-side bundle size, you can export .docx files in your the server.

In order to do so, you'd need to import the exportDocx function from the @tiptap-pro/extension-export-docx package, pass it your Tiptap JSON content, and return the resulting conversion to the client.

Let's first take a look into the exportDocx function signature:

The exportDocx function will return a docx document ready and converted to any format that .

Here you have a simple example using Express and @tiptap-pro/extension-export-docx on the server-side:

Exporting .docx files from Tiptap JSON provides a way to handle most standard Word content, but it’s not a one-to-one mapping due to inherent differences between DOCX formatting and Tiptap’s CSS-based styles.

Currently supported features and known limitations:

**Examples:**

Example 1 (python):
```python
npm i @tiptap-pro/extension-export-docx
```

Example 2 (python):
```python
npm i @tiptap-pro/extension-export-docx
```

Example 3 (sql):
```sql
import { ExportDocx } from '@tiptap-pro/extension-export-docx'
```

Example 4 (sql):
```sql
import { ExportDocx } from '@tiptap-pro/extension-export-docx'
```

---

## extendMarkRange command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/extend-mark-range

**Contents:**
- extendMarkRange command
- Parameters
- Use the extendMarkRange command
  - Was this page helpful?

The extendMarkRange command expands the current selection to encompass the current mark. If the current selection doesn’t have the specified mark, nothing changes.

typeOrName: string | MarkType

Name or type of the mark.

attributes?: Record<string, any>

Optionally, you can specify attributes that the extended mark must contain.

**Examples:**

Example 1 (sql):
```sql
// Expand selection to link marks
editor.commands.extendMarkRange('link')

// Expand selection to link marks with specific attributes
editor.commands.extendMarkRange('link', { href: 'https://google.com' })

// Expand selection to link mark and update attributes
editor
  .chain()
  .extendMarkRange('link')
  .updateAttributes('link', {
    href: 'https://duckduckgo.com',
  })
  .run()
```

Example 2 (sql):
```sql
// Expand selection to link marks
editor.commands.extendMarkRange('link')

// Expand selection to link marks with specific attributes
editor.commands.extendMarkRange('link', { href: 'https://google.com' })

// Expand selection to link mark and update attributes
editor
  .chain()
  .extendMarkRange('link')
  .updateAttributes('link', {
    href: 'https://duckduckgo.com',
  })
  .run()
```

---

## focus command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/focus

**Contents:**
- focus command
- Parameters
- Use the focus command
  - Was this page helpful?

This command sets the focus back to the editor.

When a user clicks on a button outside the editor, the browser sets the focus to that button. In most scenarios you want to focus the editor then again. That’s why you’ll see that in basically every demo here.

See also: setTextSelection, blur

position: 'start' | 'end' | 'all' | number | boolean | null (false)

By default, it’s restoring the cursor position (and text selection). Pass a position to move the cursor to.

options: { scrollIntoView: boolean }

Defines whether to scroll to the cursor when focusing. Defaults to true.

**Examples:**

Example 1 (julia):
```julia
// Set the focus to the editor
editor.commands.focus()

// Set the cursor to the first position
editor.commands.focus('start')

// Set the cursor to the last position
editor.commands.focus('end')

// Selects the whole document
editor.commands.focus('all')

// Set the cursor to position 10
editor.commands.focus(10)
```

Example 2 (julia):
```julia
// Set the focus to the editor
editor.commands.focus()

// Set the cursor to the first position
editor.commands.focus('start')

// Set the cursor to the last position
editor.commands.focus('end')

// Selects the whole document
editor.commands.focus('all')

// Set the cursor to position 10
editor.commands.focus(10)
```

---

## forEach command

**URL:** https://tiptap.dev/docs/editor/api/commands/for-each

**Contents:**
- forEach command
- Parameters
- Use the forEach command
  - Was this page helpful?

Loop through an array of items.

fn: (item: any, props: CommandProps & { index: number }) => boolean

A function to do anything with your item.

**Examples:**

Example 1 (javascript):
```javascript
const items = ['foo', 'bar', 'baz']

editor.commands.forEach(items, (item, { commands }) => {
  return commands.insertContent(item)
})
```

Example 2 (javascript):
```javascript
const items = ['foo', 'bar', 'baz']

editor.commands.forEach(items, (item, { commands }) => {
  return commands.insertContent(item)
})
```

---

## HTML Utility

**URL:** https://tiptap.dev/docs/editor/api/utilities/html

**Contents:**
- HTML Utility
- Generating HTML from JSON
  - Caution
- Generating JSON from HTML
  - Caution
- Source code
  - Was this page helpful?

The HTML Utility helps render JSON content as HTML and generate JSON from HTML without an editor instance, suitable for server-side operations. All it needs is JSON or a HTML string, and a list of extensions.

Given a JSON object, representing a ProseMirror document, the generateHTML function will return a string object representing the JSON content. The function takes two arguments: the JSON object and a list of extensions.

There are two exports available: generateHTML from @tiptap/core and from @tiptap/html. The former is only for use within the browser, the latter can be used on either the server or the browser. Make sure to use the correct one for your use case. On the server, a virtual DOM is used to generate the HTML. So using @tiptap/core can ship less code if you don't need the server-side functionality.

Given an HTML string, the generateJSON function will return a JSON object representing the HTML content as a ProseMirror document. The function takes two arguments: the HTML string and a list of extensions.

There are two exports available: generateJSON from @tiptap/core and from @tiptap/html. The former is only for use within the browser, the latter can be used on either the server or the browser. Make sure to use the correct one for your use case. On the server, a virtual DOM is used to generate the HTML. So using @tiptap/core can ship less code if you don't need the server-side functionality.

**Examples:**

Example 1 (json):
```json
/* IN BROWSER ONLY - See below for server-side compatible package */
import { generateHTML } from '@tiptap/core'

// Generate HTML from JSON
generateHTML(
  {
    type: 'doc',
    content: [{ type: 'paragraph', content: [{ type: 'text', text: 'On the browser only' }] }],
  },
  [
    Document,
    Paragraph,
    Text,
    Bold,
    // other extensions …
  ],
)
// `<p>On the browser only</p>`

/* ON SERVER OR BROWSER - See above for browser only compatible package (ships less JS) */
import { generateHTML } from '@tiptap/html'

// Generate HTML from JSON
generateHTML(
  {
    type: 'doc',
    content: [
      { type: 'paragraph', content: [{ type: 'text', text: 'On the server, or the browser' }] },
    ],
  },
  [
    Document,
    Paragraph,
    Text,
    Bold,
    // other extensions …
  ],
)
// `<p>On the server, or the browser</p>`
```

Example 2 (json):
```json
/* IN BROWSER ONLY - See below for server-side compatible package */
import { generateHTML } from '@tiptap/core'

// Generate HTML from JSON
generateHTML(
  {
    type: 'doc',
    content: [{ type: 'paragraph', content: [{ type: 'text', text: 'On the browser only' }] }],
  },
  [
    Document,
    Paragraph,
    Text,
    Bold,
    // other extensions …
  ],
)
// `<p>On the browser only</p>`

/* ON SERVER OR BROWSER - See above for browser only compatible package (ships less JS) */
import { generateHTML } from '@tiptap/html'

// Generate HTML from JSON
generateHTML(
  {
    type: 'doc',
    content: [
      { type: 'paragraph', content: [{ type: 'text', text: 'On the server, or the browser' }] },
    ],
  },
  [
    Document,
    Paragraph,
    Text,
    Bold,
    // other extensions …
  ],
)
// `<p>On the server, or the browser</p>`
```

Example 3 (sql):
```sql
/* IN BROWSER ONLY - See below for server-side compatible package */
import { generateJSON } from '@tiptap/core'

// Generate JSON from HTML
generateJSON(`<p>On the browser only</p>`, [
  Document,
  Paragraph,
  Text,
  Bold,
  // other extensions …
])
// { type: 'doc', content: [{ type: 'paragraph', content: [{ type: 'text', text: 'On the browser only' }] }] }

/* ON SERVER OR BROWSER - See above for browser only compatible package (ships less JS) */
import { generateJSON } from '@tiptap/html'

// Generate JSON from HTML
generateJSON(`<p>On the server, or the browser</p>`, [
  Document,
  Paragraph,
  Text,
  Bold,
  // other extensions …
])
// { type: 'doc', content: [{ type: 'paragraph', content: [{ type: 'text', text: 'On the server, or the browser' }] }] }
```

Example 4 (sql):
```sql
/* IN BROWSER ONLY - See below for server-side compatible package */
import { generateJSON } from '@tiptap/core'

// Generate JSON from HTML
generateJSON(`<p>On the browser only</p>`, [
  Document,
  Paragraph,
  Text,
  Bold,
  // other extensions …
])
// { type: 'doc', content: [{ type: 'paragraph', content: [{ type: 'text', text: 'On the browser only' }] }] }

/* ON SERVER OR BROWSER - See above for browser only compatible package (ships less JS) */
import { generateJSON } from '@tiptap/html'

// Generate JSON from HTML
generateJSON(`<p>On the server, or the browser</p>`, [
  Document,
  Paragraph,
  Text,
  Bold,
  // other extensions …
])
// { type: 'doc', content: [{ type: 'paragraph', content: [{ type: 'text', text: 'On the server, or the browser' }] }] }
```

---

## Import .docx in your editor

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/editor-import

**Contents:**
- Import .docx in your editor
- Install the DOCX Import extension
- Required extensions
- Configure
- Import a DOCX file
  - Basic import
  - Import handling
- Verbose output
  - Verbose bitmask
- Support & Limitations

Converting .docx files to Tiptap JSON is simple with the @tiptap-pro/extension-import-docx editor extension, which integrates directly into your Tiptap Editor.

If you need to import .docx content outside the Editor, use the REST API.

The Conversion extensions are published in Tiptap’s private npm registry. Integrate the extensions by following the private registry guide.

Install the Tiptap Import extension package:

Ensure your editor includes all necessary Tiptap extensions to handle content from DOCX. For example, include the Image extension for inline images, and the Table extension for tables.

In order to fully map DOCX content (e.g. images, tables, styled text) onto Tiptap’s schema, you must include the relevant Tiptap extensions. Without these extensions, certain DOCX elements may not be recognized or properly rendered by the editor.

Add the Import extension to your editor setup.

Once the extension is configured, you can import a DOCX file selected by the user.

The simplest approach is to pass the file directly to the import command. Here it replaces the current editor content with the converted content and focuses the editor:

In most cases, this one-liner is all you need to let users import .docx files. The extension handles sending the file to the conversion endpoint, retrieving the converted Tiptap JSON, and inserting it into the editor.

In order to have more control after the import process have finished, you would use the onImport callback to handle the conversion result. This callback provides the converted content, any errors that occurred, and a function called setEditorContent to insert the content from context.content into the editor. If you don't provide an onImport callback, the extension will automatically insert the content into the editor but you won't be able to handle anything else like errors or loading states.

Operations that we have controlled in the example above:

The DOCX import extension provides a verbose configuration property to help you control the level of diagnostic output during the import process. This is especially useful for debugging or for getting more insight into what happens during conversion.

The verbose property is a bitmask number that determines which types of log messages are emitted. The extension uses the following levels:

You can combine levels by adding their values together. For example, verbose: 3 will enable both log (1) and warn (2) messages.

The verbose output will give you, along the data property, one more property called logs, which will contain info, warn, and error properties, each of them being an array with all of the information related to that specific verbosity.

Importing .docx files into Tiptap provides a way to handle most standard Word content, but it’s not a one-to-one mapping due to inherent differences between DOCX formatting and Tiptap’s CSS-based styles.

Currently supported features and known limitations:

**Examples:**

Example 1 (python):
```python
npm i @tiptap-pro/extension-import-docx
```

Example 2 (python):
```python
npm i @tiptap-pro/extension-import-docx
```

Example 3 (python):
```python
import StarterKit from '@tiptap/starter-kit'
import Color from '@tiptap/extension-color'
import FontFamily from '@tiptap/extension-font-family'
import Highlight from '@tiptap/extension-highlight'
import { Image } from '@tiptap/extension-image'
import Paragraph from '@tiptap/extension-paragraph'
import { TableKit } from '@tiptap/extension-table'
import TextAlign from '@tiptap/extension-text-align'
import { TextStyle } from '@tiptap/extension-text-style'
```

Example 4 (python):
```python
import StarterKit from '@tiptap/starter-kit'
import Color from '@tiptap/extension-color'
import FontFamily from '@tiptap/extension-font-family'
import Highlight from '@tiptap/extension-highlight'
import { Image } from '@tiptap/extension-image'
import Paragraph from '@tiptap/extension-paragraph'
import { TableKit } from '@tiptap/extension-table'
import TextAlign from '@tiptap/extension-text-align'
import { TextStyle } from '@tiptap/extension-text-style'
```

---

## Import & export DOCX with Tiptap

**URL:** https://tiptap.dev/docs/conversion/import-export/docx

**Contents:**
- Import & export DOCX with Tiptap
  - Information related to custom nodes
  - Was this page helpful?

Integrating DOCX (Microsoft Word) conversion with Tiptap can be done in several ways, from adding in-editor “Import/Export DOCX” buttons to using the REST API for server-side workflows.

The REST API can’t handle custom node conversions. For that, implement the Editor Export extension on your own server or on the client so you can define how those nodes convert.

---

## Inject content REST API

**URL:** https://tiptap.dev/docs/collaboration/documents/content-injection

**Contents:**
- Inject content REST API
  - Use cases
- Update a document
- Update via JSON
  - Update only node attrs
- Create a document
  - Was this page helpful?

To inject content into documents server-side, use the PATCH endpoint described in this document. This feature supports version history, tracking changes as well as content added through this endpoint.

The update document endpoint also allows JSON updates to modify documents on your Collaboration server, both On-Premises and Cloud:

The content injection REST API enables a couple of handy but sophisticated use cases:

To update an existing document on the Collaboration server, you can use the PATCH method with the following API endpoint:

This endpoint accepts a Yjs update message and applies it to the specified document. The format query parameter specifies the format of the update and can be one of the following:

Upon successful update, the server will return HTTP status 204. If the document does not exist, it will return 404, and if the payload is invalid or the update cannot be applied, it will return 422.

Example: curl command to update a document

When updating via JSON, the server computes the difference between the current document state and the provided JSON, then internally calculates the required Yjs update to reach the target state.

To ensure precise updates, especially for node-specific changes, it is recommended to use the nodeAttributeName and nodeAttributeValue parameters. These can be generated by Tiptap's UniqueID Extension or a custom implementation. Note that this only works for top level nodes.

You can use ?mode=append to append nodes to the document's JSON representation without altering existing nodes.

Omitting these parameters may result in overwriting any updates made between fetching the document and issuing the update call. The get document call returns a header x-${fragmentName}-checksum which can be used to detect conflicts by passing it to the update call as ?checksum=${checksum}. If the document has been updated since the last fetch, the update will fail with a 409 Checksum mismatch. status.

Example: Updating a document using JSON

If you want to only update attributes of a node, you can use the ?mode=attrs query parameter. This will only update the attributes of the node and not its content. In this mode, the nodeAttributeName and nodeAttributeValue parameters work for any (not just top level) nodes.

Note that we're deleting all attrs on that node and then setting only the ones specified in the payload of the request. Not specifying a node filter (nodeAttributeName, nodeAttributeValue) will result in all nodes being updated.

To seed a new document on the Tiptap Collab server, use the POST method with the following endpoint:

The server will return HTTP status 204 for successful creation, 409 if the document already exists (you must delete it first to overwrite), and 422 if the action failed.

The format parameter accepts the same values as the update endpoint (binary, base64, or json).

**Examples:**

Example 1 (unknown):
```unknown
PATCH /api/documents/:identifier?format=:format
```

Example 2 (unknown):
```unknown
PATCH /api/documents/:identifier?format=:format
```

Example 3 (python):
```python
curl --location --request PATCH 'https://YOUR_APP_ID.collab.tiptap.cloud/api/documents/DOCUMENT_NAME' \\
--header 'Authorization: YOUR_SECRET_FROM_SETTINGS_AREA' \\
--data '@yjsUpdate.binary'
```

Example 4 (python):
```python
curl --location --request PATCH 'https://YOUR_APP_ID.collab.tiptap.cloud/api/documents/DOCUMENT_NAME' \\
--header 'Authorization: YOUR_SECRET_FROM_SETTINGS_AREA' \\
--data '@yjsUpdate.binary'
```

---

## Input Rules

**URL:** https://tiptap.dev/docs/editor/api/input-rules

**Contents:**
- Input Rules
- What are input rules?
- How do input rules work in Tiptap?
- Understanding markInputRule and nodeInputRule
  - Extracting information with getAttributes
    - Example: Using getAttributes in a node input rule
    - Example: Using getAttributes in a mark input rule
  - Tips
  - Was this page helpful?

Input rules are a powerful feature in Tiptap that allow you to automatically transform text as you type. They can be used to create shortcuts for formatting, inserting content, or triggering commands based on specific patterns in the text.

Input rules are pattern-based triggers that watch for specific text input and automatically transform it into something else. For example, typing **bold** can automatically turn the text into bold formatting, or typing 1. at the start of a line can create an ordered list. Input rules are especially useful for implementing Markdown-like shortcuts and improving the user experience.

Tiptap uses input rules under the hood to provide many of its default shortcuts (like lists, blockquotes, and marks). Input rules are defined as regular expressions that match user input. When the pattern is detected, the rule executes a transformation—such as applying a mark, inserting a node, or running a command.

Input rules are typically registered inside extensions (nodes, marks, or generic extensions) using the addInputRules() method. Tiptap provides helper functions like markInputRule and nodeInputRule to simplify the creation of input rules for marks and nodes.

Tiptap provides two helper functions to simplify the creation of input rules:

Both functions accept a configuration object with at least these properties:

The getAttributes function allows you to extract data from the matched input and pass it as attributes to the node or mark. This is especially useful for nodes like images or figures, where you want to capture values like src, alt, or title from the user's input.

In this example, when a user types something like ![Alt text](image.png "Optional title"), the input rule extracts the alt, src, and title from the match and passes them as attributes to the node.

For more details, see the ProseMirror input rules docs.

**Examples:**

Example 1 (typescript):
```typescript
addInputRules() {
  return [
    nodeInputRule({
      find: /!\[(.*?)\]\((.*?)(?:\s+"(.*?)")?\)$/, // Matches ![alt](src "title")
      type: this.type,
      getAttributes: match => {
        const [, alt, src, title] = match
        return { src, alt, title }
      },
    }),
  ]
},
```

Example 2 (typescript):
```typescript
addInputRules() {
  return [
    nodeInputRule({
      find: /!\[(.*?)\]\((.*?)(?:\s+"(.*?)")?\)$/, // Matches ![alt](src "title")
      type: this.type,
      getAttributes: match => {
        const [, alt, src, title] = match
        return { src, alt, title }
      },
    }),
  ]
},
```

Example 3 (typescript):
```typescript
addInputRules() {
  return [
    markInputRule({
      find: /\*\*([^*]+)\*\*$/, // Matches **bold**
      type: this.type,
      getAttributes: match => {
        // You can extract custom attributes here if needed
        return {}
      },
    }),
  ]
},
```

Example 4 (typescript):
```typescript
addInputRules() {
  return [
    markInputRule({
      find: /\*\*([^*]+)\*\*$/, // Matches **bold**
      type: this.type,
      getAttributes: match => {
        // You can extract custom attributes here if needed
        return {}
      },
    }),
  ]
},
```

---

## insertContentAt command

**URL:** https://tiptap.dev/docs/editor/api/commands/content/insert-content-at

**Contents:**
- insertContentAt command
- Parameters
- Use the insertContentAt command
  - Was this page helpful?

The insertContentAt will insert an HTML string or a node at a given position or range. If a range is given, the new content will replace the content in the given range with the new content.

position: number | Range

The position or range the content will be inserted in.

The content to be inserted. Can be plain text, an HTML string or JSON node(s).

options: Record<string, any>

**Examples:**

Example 1 (json):
```json
// Plain text
editor.commands.insertContentAt(12, 'Example Text')

// Plain text, replacing a range
editor.commands.insertContentAt({ from: 12, to: 16 }, 'Example Text')

// HTML
editor.commands.insertContentAt(12, '<h1>Example Text</h1>')

// HTML with trim white space
editor.commands.insertContentAt(12, '<p>Hello world</p>', {
  updateSelection: true,
  parseOptions: {
    preserveWhitespace: 'full',
  },
})

// JSON/Nodes
editor.commands.insertContentAt(12, {
  type: 'heading',
  attrs: {
    level: 1,
  },
  content: [
    {
      type: 'text',
      text: 'Example Text',
    },
  ],
})

// Multiple nodes at once
editor.commands.insertContentAt(12, [
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'First paragraph',
      },
    ],
  },
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'Second paragraph',
      },
    ],
  },
])
```

Example 2 (json):
```json
// Plain text
editor.commands.insertContentAt(12, 'Example Text')

// Plain text, replacing a range
editor.commands.insertContentAt({ from: 12, to: 16 }, 'Example Text')

// HTML
editor.commands.insertContentAt(12, '<h1>Example Text</h1>')

// HTML with trim white space
editor.commands.insertContentAt(12, '<p>Hello world</p>', {
  updateSelection: true,
  parseOptions: {
    preserveWhitespace: 'full',
  },
})

// JSON/Nodes
editor.commands.insertContentAt(12, {
  type: 'heading',
  attrs: {
    level: 1,
  },
  content: [
    {
      type: 'text',
      text: 'Example Text',
    },
  ],
})

// Multiple nodes at once
editor.commands.insertContentAt(12, [
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'First paragraph',
      },
    ],
  },
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'Second paragraph',
      },
    ],
  },
])
```

---

## insertContent command

**URL:** https://tiptap.dev/docs/editor/api/commands/content/insert-content

**Contents:**
- insertContent command
- Parameters
- Use the insertContent command
  - Was this page helpful?

The insertContent command adds the passed value to the document.

See also: setContent, clearContent

The command is pretty flexible and takes plain text, HTML or even JSON as a value.

**Examples:**

Example 1 (json):
```json
// Plain text
editor.commands.insertContent('Example Text')

// HTML
editor.commands.insertContent('<h1>Example Text</h1>')

// HTML with trim white space
editor.commands.insertContent('<h1>Example Text</h1>', {
  parseOptions: {
    preserveWhitespace: false,
  },
})

// JSON/Nodes
editor.commands.insertContent({
  type: 'heading',
  attrs: {
    level: 1,
  },
  content: [
    {
      type: 'text',
      text: 'Example Text',
    },
  ],
})

// Multiple nodes at once
editor.commands.insertContent([
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'First paragraph',
      },
    ],
  },
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'Second paragraph',
      },
    ],
  },
])
```

Example 2 (json):
```json
// Plain text
editor.commands.insertContent('Example Text')

// HTML
editor.commands.insertContent('<h1>Example Text</h1>')

// HTML with trim white space
editor.commands.insertContent('<h1>Example Text</h1>', {
  parseOptions: {
    preserveWhitespace: false,
  },
})

// JSON/Nodes
editor.commands.insertContent({
  type: 'heading',
  attrs: {
    level: 1,
  },
  content: [
    {
      type: 'text',
      text: 'Example Text',
    },
  ],
})

// Multiple nodes at once
editor.commands.insertContent([
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'First paragraph',
      },
    ],
  },
  {
    type: 'paragraph',
    content: [
      {
        type: 'text',
        text: 'Second paragraph',
      },
    ],
  },
])
```

---

## joinBackward command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/join-backward

**Contents:**
- joinBackward command
- Use the joinBackward command
  - Was this page helpful?

The joinBackward command joins two nodes backwards from the current selection. If the selection is empty and at the start of a textblock, joinBackward will try to reduce the distance between that block and the block before it. See also

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.joinBackward()
```

Example 2 (unknown):
```unknown
editor.commands.joinBackward()
```

---

## joinDown command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/join-down

**Contents:**
- joinDown command
- Use the joinDown command
  - Was this page helpful?

The joinDown command joins the selected block, or if there is a text selection, the closest ancestor block of the selection that can be joined, with the sibling below it. See also

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.joinDown()
```

Example 2 (unknown):
```unknown
editor.commands.joinDown()
```

---

## joinForward command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/join-forward

**Contents:**
- joinForward command
- Use the joinForward command
  - Was this page helpful?

The joinForward command joins two nodes forwards from the current selection. If the selection is empty and at the end of a textblock, joinForward will try to reduce the distance between that block and the block after it. See also

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.joinForward()
```

Example 2 (unknown):
```unknown
editor.commands.joinForward()
```

---

## joinTextblockBackward command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/join-textblock-backward

**Contents:**
- joinTextblockBackward command
- Using the joinTextblockBackward command
  - Was this page helpful?

A more limited form of joinBackward that only tries to join the current textblock to the one before it, if the cursor is at the start of a textblock. See also

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.joinTextblockBackward()
```

Example 2 (unknown):
```unknown
editor.commands.joinTextblockBackward()
```

---

## joinTextblockForward command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/join-textblock-forward

**Contents:**
- joinTextblockForward command
- Using the joinTextblockForward command
  - Was this page helpful?

A more limited form of joinForward that only tries to join the current textblock to the one after it, if the cursor is at the end of a textblock. See also

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.joinTextblockForward()
```

Example 2 (unknown):
```unknown
editor.commands.joinTextblockForward()
```

---

## joinUp command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/join-up

**Contents:**
- joinUp command
- Use the joinUp command
  - Was this page helpful?

The joinUp command joins the selected block, or if there is a text selection, the closest ancestor block of the selection that can be joined, with the sibling above it. See also

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.joinUp()
```

Example 2 (unknown):
```unknown
editor.commands.joinUp()
```

---

## JSX

**URL:** https://tiptap.dev/docs/editor/api/utilities/jsx

**Contents:**
- JSX
- Using JSX in your extension
- Writing JSX in the renderHTML function
  - Was this page helpful?

When creating custom extensions, you often have to define how they should be rendered to HTML. Usually, this is done by defining a renderHTML function that returns Prosemirror render array including the tag name, attributes and content holes.

With the Tiptap JSX renderer, you can use JSX to define how your extensions should be rendered.

To use JSX in your extension, you will need a bundler that can handle JSX or TSX files like Vite or Webpack. Most frameworks like Next.js, Remix or Nuxt already should be able to handle this.

By default, the JSX runtime of React is used if not configured otherwise. This will cause issues if you are trying to use JSX in a non-React environment like a Tiptap extension.

To handle this, you can add a comment to the top of your file to specify which JSX runtime the bundler should use. Tiptap comes with it's own bundler from @tiptap/core.

Now that the bundler should be able to handle JSX for Tiptap, you can use JSX in your renderHTML function.

The <slot /> tag is used to define a content hole for Prosemirror via JSX. This is the position your editable content will be rendered into.

Note that this is not using any component library like React or Vue under the hood and won't support hooks, states or other library specific features.

**Examples:**

Example 1 (python):
```python
/** @jsxImportSource @tiptap/core */

// your code here
```

Example 2 (python):
```python
/** @jsxImportSource @tiptap/core */

// your code here
```

Example 3 (jsx):
```jsx
/** @jsxImportSource @tiptap/core */

import { Node } from '@tiptap/core'

const MyNode = Node.create({
  // ... your node configuration

  renderHTML({ HTMLAttributes }) {
    return (
      <div {...HTMLAttributes}>
        <p>This is your custom node. And here is your content hole:</p>
        <slot />
      </div>
    )
  }
})
```

Example 4 (jsx):
```jsx
/** @jsxImportSource @tiptap/core */

import { Node } from '@tiptap/core'

const MyNode = Node.create({
  // ... your node configuration

  renderHTML({ HTMLAttributes }) {
    return (
      <div {...HTMLAttributes}>
        <p>This is your custom node. And here is your content hole:</p>
        <slot />
      </div>
    )
  }
})
```

---

## keyboardShortcut command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/keyboard-shortcut

**Contents:**
- keyboardShortcut command
- Parameters
- Use the keyboardShortcut command
  - Was this page helpful?

The keyboardShortcut command will try to trigger a ShortcutEvent with a given name.

The name of the shortcut to trigger.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.keyboardShortcut('undo')
```

Example 2 (unknown):
```unknown
editor.commands.keyboardShortcut('undo')
```

---

## liftEmptyBlock command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/lift-empty-block

**Contents:**
- liftEmptyBlock command
- Using the liftEmptyBlock command
  - Was this page helpful?

If the currently selected block is an empty textblock, lift it if possible. Lifting means, that the block will be moved to the parent of the block it is currently in.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.liftEmptyBlock()
```

Example 2 (unknown):
```unknown
editor.commands.liftEmptyBlock()
```

---

## liftListItem command

**URL:** https://tiptap.dev/docs/editor/api/commands/lists/lift-list-item

**Contents:**
- liftListItem command
- Using the liftListItem command
  - Was this page helpful?

The liftListItem will try to lift the list item around the current selection up into a wrapping parent list.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.liftListItem()
```

Example 2 (unknown):
```unknown
editor.commands.liftListItem()
```

---

## lift command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/lift

**Contents:**
- lift command
- Parameters
- Use the lift command
  - Was this page helpful?

The lift command lifts a given node up into it’s parent node. Lifting means, that the block will be moved to the parent of the block it is currently in.

typeOrName: String | NodeType

The node that should be lifted. If the node is not found in the current selection, ignore the command.

attributes: Record<string, any>

The attributes the node should have to be lifted. This is optional.

**Examples:**

Example 1 (css):
```css
// lift any headline
editor.commands.lift('headline')

// lift only h2
editor.commands.lift('headline', { level: 2 })
```

Example 2 (css):
```css
// lift any headline
editor.commands.lift('headline')

// lift only h2
editor.commands.lift('headline', { level: 2 })
```

---

## List commands

**URL:** https://tiptap.dev/docs/editor/api/commands/lists

**Contents:**
- List commands
- Use Cases
- List Keymap Extension
- Here’s a list of… list commands
  - Was this page helpful?

Lists are a crucial part of structuring content in your Tiptap editor. Tiptap provides commands to manipulate list structures easily. Here’s an overview of the essential commands that help you create, update, and manage your lists.

You might also want to include the List Keymap extension, which adds extra keymap handlers to change the default backspace and delete behavior for lists. It modifies the default behavior so that pressing backspace at the start of a list item lifts the content into the list item above.

---

## MarkdownManager API

**URL:** https://tiptap.dev/docs/editor/markdown/api/markdown-manager

**Contents:**
- MarkdownManager API
- Methods
  - Constructor
  - MarkdownManager.hasMarked()
  - MarkdownManager.registerExtension()
  - MarkdownManager.parse()
  - MarkdownManager.serialize()
  - MarkdownManager.renderNodeToMarkdown()
  - MarkdownManager.renderNodes()
- Properties

The MarkdownManager class is a stand-alone class that provides support for parsing and serializing Markdown content into Tiptap's document model.

Returns true or false depending on whether the marked library is available.

Registers a Tiptap extension to be used for parsing and serializing Markdown content.

Parses a Markdown string into a Tiptap document.

Serializes a Tiptap document or JSON content into a Markdown string.

Renders a single ProseMirror node to its Markdown representation.

Renders an array of ProseMirror nodes to their combined Markdown representation.

The MarkedJS instance used for parsing Markdown content.

The character used for indentation in lists. Defaults to a space (' ').

The string used for indentation in lists. Defaults to two spaces (' ').

**Examples:**

Example 1 (json):
```json
new MarkdownManager(options?: {
  marked?: typeof marked,
  markedOptions?: MarkedOptions,
  indentation?: {
    style?: 'space' | 'tab',
    size?: number,
  },
})
```

Example 2 (json):
```json
new MarkdownManager(options?: {
  marked?: typeof marked,
  markedOptions?: MarkedOptions,
  indentation?: {
    style?: 'space' | 'tab',
    size?: number,
  },
})
```

Example 3 (javascript):
```javascript
const manager = new MarkdownManager()
manager.hasMarked() // true or false
```

Example 4 (javascript):
```javascript
const manager = new MarkdownManager()
manager.hasMarked() // true or false
```

---

## Markdown Examples

**URL:** https://tiptap.dev/docs/editor/markdown/examples

**Contents:**
- Markdown Examples
- Basic Examples
  - Read and Write Markdown
  - Paste Markdown Detection
- Custom Tokenizers
  - Subscript and Superscript
- Integration Examples
  - Real-Time Markdown Preview
  - Saving and Loading Workflow
- Server-Side Rendering

Real-world examples and recipes for common use cases with the Markdown extension.

This example demonstrates the most common Markdown operations:

Automatically detect and parse pasted Markdown:

Support ~subscript~ and ^superscript^:

You can create a real-time Markdown preview by listening to editor updates:

Store content as Markdown and load it when needed:

Render Markdown on the server:

Load large documents progressively:

**Examples:**

Example 1 (javascript):
```javascript
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from '@tiptap/markdown'

const editor = new Editor({
  element: document.querySelector('#editor'),
  extensions: [StarterKit, Markdown],
  content: '# Hello World\n\nStart typing...',
  contentType: 'markdown', // parse initial content as Markdown
})

// Read: serialize current editor content to Markdown
console.log(editor.getMarkdown())

// Write: set editor content from a Markdown string
editor.commands.setContent('# New title\n\nSome *Markdown* content', { contentType: 'markdown' })
```

Example 2 (javascript):
```javascript
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from '@tiptap/markdown'

const editor = new Editor({
  element: document.querySelector('#editor'),
  extensions: [StarterKit, Markdown],
  content: '# Hello World\n\nStart typing...',
  contentType: 'markdown', // parse initial content as Markdown
})

// Read: serialize current editor content to Markdown
console.log(editor.getMarkdown())

// Write: set editor content from a Markdown string
editor.commands.setContent('# New title\n\nSome *Markdown* content', { contentType: 'markdown' })
```

Example 3 (julia):
```julia
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from '@tiptap/markdown'
import { Plugin } from '@tiptap/pm/state'

const PasteMarkdown = Extension.create({
  name: 'pasteMarkdown',

  addProseMirrorPlugins() {
    return [
      new Plugin({
        props: {
          handlePaste(view, event, slice) {
            const text = event.clipboardData?.getData('text/plain')

            if (!text) {
              return false
            }

            // Check if text looks like Markdown
            if (looksLikeMarkdown(text)) {
              const { state, dispatch } = view
              // Parse the Markdown text to Tiptap JSON using the Markdown manager
              const json = editor.markdown.parse(text)

              // Insert the parsed JSON content at cursor position
              editor.commands.insertContent(json)
              return true
            }

            return false
          },
        },
      }),
    ]
  },
})

function looksLikeMarkdown(text: string): boolean {
  // Simple heuristic: check for Markdown syntax
  return (
    /^#{1,6}\s/.test(text) || // Headings
    /\*\*[^*]+\*\*/.test(text) || // Bold
    /\[.+\]\(.+\)/.test(text) || // Links
    /^[-*+]\s/.test(text)
  ) // Lists
}

const editor = new Editor({
  extensions: [StarterKit, Markdown, PasteMarkdown],
})
```

Example 4 (julia):
```julia
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from '@tiptap/markdown'
import { Plugin } from '@tiptap/pm/state'

const PasteMarkdown = Extension.create({
  name: 'pasteMarkdown',

  addProseMirrorPlugins() {
    return [
      new Plugin({
        props: {
          handlePaste(view, event, slice) {
            const text = event.clipboardData?.getData('text/plain')

            if (!text) {
              return false
            }

            // Check if text looks like Markdown
            if (looksLikeMarkdown(text)) {
              const { state, dispatch } = view
              // Parse the Markdown text to Tiptap JSON using the Markdown manager
              const json = editor.markdown.parse(text)

              // Insert the parsed JSON content at cursor position
              editor.commands.insertContent(json)
              return true
            }

            return false
          },
        },
      }),
    ]
  },
})

function looksLikeMarkdown(text: string): boolean {
  // Simple heuristic: check for Markdown syntax
  return (
    /^#{1,6}\s/.test(text) || // Headings
    /\*\*[^*]+\*\*/.test(text) || // Bold
    /\[.+\]\(.+\)/.test(text) || // Links
    /^[-*+]\s/.test(text)
  ) // Lists
}

const editor = new Editor({
  extensions: [StarterKit, Markdown, PasteMarkdown],
})
```

---

## Markdown Extension API

**URL:** https://tiptap.dev/docs/editor/markdown/api/extension

**Contents:**
- Markdown Extension API
- Extension Configuration
  - Markdown.configure(options)
    - Parameters
    - Example
  - Was this page helpful?

Configure the Markdown extension with custom options.

indentation (optional)

markedOptions (optional)

**Examples:**

Example 1 (json):
```json
Markdown.configure({
  indentation?: {
    style?: 'space' | 'tab'
    size?: number
  },
  marked?: typeof marked,
  markedOptions?: MarkedOptions,
})
```

Example 2 (json):
```json
Markdown.configure({
  indentation?: {
    style?: 'space' | 'tab'
    size?: number
  },
  marked?: typeof marked,
  markedOptions?: MarkedOptions,
})
```

Example 3 (sql):
```sql
import { Markdown } from '@tiptap/markdown'

const markdown = Markdown.configure({
  indentation: {
    style: 'space',
    size: 4,
  },
  markedOptions: {
    gfm: true,
    breaks: false,
  },
})
```

Example 4 (sql):
```sql
import { Markdown } from '@tiptap/markdown'

const markdown = Markdown.configure({
  indentation: {
    style: 'space',
    size: 4,
  },
  markedOptions: {
    gfm: true,
    breaks: false,
  },
})
```

---

## Markdown Glossary for Tiptap

**URL:** https://tiptap.dev/docs/editor/markdown/glossary

**Contents:**
- Markdown Glossary for Tiptap
- Token
- Tiptap JSON
- Tokenizer
- Lexer
  - Was this page helpful?

Before we dive into the details, here are some key terms we'll be using throughout this guide:

A plain JavaScript object that represents a piece of the parsed Markdown. For example, a heading token might look like { type: "heading", depth: 2, text: "Hello" }. Tokens are the “lego bricks” that describe the document’s structure.

Note: MarkedJS comes with built-in tokenizers for standard Markdown syntax, but you can extend or replace these by providing custom tokenizers to the MarkdownManager.

You can find the list of default tokens in the MarkedJS types.

Now that we understand the difference between a Token and Tiptap JSON, let's dive into how to parse tokens and serialize Tiptap content.

The set of functions (or rules) that scan the raw Markdown text and decide how to turn chunks of it into tokens. For example, it recognizes ## Heading and produces a heading token. You can customize or override tokenizers to change how Markdown is interpreted.

You can find out how to create custom tokenizers in the Custom Tokenizers guide.

The orchestrator that runs through the entire Markdown string, applies the tokenizers in sequence, and produces the full list of tokens. Think of it as the machine that repeatedly feeds text into the tokenizers until the whole input is tokenized.

You don't need to touch the lexer directly, because Tiptap is already creating a lexer instance that will be reused for the lifetime of your editor as part of the MarkedJS instance.

This lexer instance will automatically register all tokenizers from your extensions.

---

## Markdown Types API

**URL:** https://tiptap.dev/docs/editor/markdown/api/types

**Contents:**
- Markdown Types API
- Types
  - MarkdownExtensionOptions
  - MarkdownExtensionSpec
  - MarkdownToken
  - MarkdownParseHelpers
  - MarkdownRendererHelpers
  - RenderContext
  - MarkdownTokenizer
  - MarkdownLexerConfiguration

Options for configuring the Markdown extension.

Configuration for Markdown support in extensions.

Token structure from MarkedJS.

Helpers passed to parse handlers.

Helpers passed to render handlers.

Context information passed to render handlers.

Custom tokenizer for MarkedJS.

Lexer helpers for custom tokenizers.

Result type for parse handlers.

Markdown configuration in extensions.

**Examples:**

Example 1 (typescript):
```typescript
type MarkdownExtensionOptions = {
  indentation?: {
    style?: 'space' | 'tab'
    size?: number
  }
  marked?: typeof marked
  markedOptions?: MarkedOptions
}
```

Example 2 (typescript):
```typescript
type MarkdownExtensionOptions = {
  indentation?: {
    style?: 'space' | 'tab'
    size?: number
  }
  marked?: typeof marked
  markedOptions?: MarkedOptions
}
```

Example 3 (typescript):
```typescript
type MarkdownExtensionSpec = {
  parseName?: string
  renderName?: string
  markdownName?: string // Legacy
  parseMarkdown?: (token: MarkdownToken, helpers: MarkdownParseHelpers) => MarkdownParseResult
  renderMarkdown?: (node: JSONContent, helpers: MarkdownRendererHelpers, context: RenderContext) => string
  isIndenting?: boolean
  tokenizer?: MarkdownTokenizer
}
```

Example 4 (typescript):
```typescript
type MarkdownExtensionSpec = {
  parseName?: string
  renderName?: string
  markdownName?: string // Legacy
  parseMarkdown?: (token: MarkdownToken, helpers: MarkdownParseHelpers) => MarkdownParseResult
  renderMarkdown?: (node: JSONContent, helpers: MarkdownRendererHelpers, context: RenderContext) => string
  isIndenting?: boolean
  tokenizer?: MarkdownTokenizer
}
```

---

## Markdown Utilities

**URL:** https://tiptap.dev/docs/editor/markdown/api/utilities

**Contents:**
- Markdown Utilities
- Block Utilities
  - createBlockMarkdownSpec
    - Syntax
    - Usage
    - Options
    - Example Markdown
  - createAtomBlockMarkdownSpec
    - Syntax
    - Usage

Creates a complete Markdown specification for block-level nodes using Pandoc-style syntax (:::blockName).

This utility can be be imported from @tiptap/core.

Creates a Markdown specification for atomic (self-closing) block nodes using Pandoc syntax.

This utility can be be imported from @tiptap/core.

No closing tag, no content. Perfect for embeds, images, horizontal rules, etc.

Creates a Markdown specification for inline nodes using shortcode syntax ([nodeName]).

This utility can be be imported from @tiptap/core.

Helpers provided to extension parse handlers.

Parse inline tokens (bold, italic, links, etc.).

Parse block-level child tokens.

Create a text node with optional marks.

Create a node with type, attributes, and content.

Apply a mark to content (for inline formatting).

Helpers provided to extension render handlers.

Render child nodes to Markdown.

Add indentation to content.

Wrap content with a prefix on each line.

The parseAttributes utility is mostly used internally for building attribute objects for Pandoc-style strings. You most likely won't use it except you want to build a custom syntax that requires similar syntax to the Pandoc attribute style.

This utility can be be imported from @tiptap/core.

The serializeAttributes utility is mostly used internally for converting attribute objects back to Pandoc-style strings. You most likely won't use it except you want to build a custom syntax that requires similar syntax to the Pandoc attribute style.

This utility can be be imported from @tiptap/core.

Advanced utility for parsing hierarchical indented blocks (lists, task lists, etc.).

This utility can be be imported from @tiptap/core.

Use this when you need to parse Markdown with:

Utility for rendering nodes with nested content, properly indenting child elements.

This utility can be be imported from @tiptap/core.

Use this when rendering:

Pro Tip: Start with utilities for standard patterns, then move to custom implementations only when you need specific behavior that utilities don't provide.

**Examples:**

Example 1 (julia):
```julia
:::blockName {attributes}

Content goes here
Can be **multiple** paragraphs

:::
```

Example 2 (julia):
```julia
:::blockName {attributes}

Content goes here
Can be **multiple** paragraphs

:::
```

Example 3 (sql):
```sql
import { Node } from '@tiptap/core'
import { createBlockMarkdownSpec } from '@tiptap/core'

const Callout = Node.create({
  name: 'callout',

  group: 'block',
  content: 'block+',

  addAttributes() {
    return {
      type: { default: 'info' },
      title: { default: null },
    }
  },

  parseHTML() {
    return [{ tag: 'div[data-callout]' }]
  },

  renderHTML({ node }) {
    return ['div', { 'data-callout': node.attrs.type }, 0]
  },

  // Use the utility to generate Markdown support
  ...createBlockMarkdownSpec({
    nodeName: 'callout',
    defaultAttributes: { type: 'info' },
    allowedAttributes: ['type', 'title'],
    content: 'block', // Allow nested block content
  }),
})
```

Example 4 (sql):
```sql
import { Node } from '@tiptap/core'
import { createBlockMarkdownSpec } from '@tiptap/core'

const Callout = Node.create({
  name: 'callout',

  group: 'block',
  content: 'block+',

  addAttributes() {
    return {
      type: { default: 'info' },
      title: { default: null },
    }
  },

  parseHTML() {
    return [{ tag: 'div[data-callout]' }]
  },

  renderHTML({ node }) {
    return ['div', { 'data-callout': node.attrs.type }, 0]
  },

  // Use the utility to generate Markdown support
  ...createBlockMarkdownSpec({
    nodeName: 'callout',
    defaultAttributes: { type: 'info' },
    allowedAttributes: ['type', 'title'],
    content: 'block', // Allow nested block content
  }),
})
```

---

## newlineInCode command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/newline-in-code

**Contents:**
- newlineInCode command
- Use the newlineInCode command
  - Was this page helpful?

newlineInCode inserts a new line in the current code block. If a selection is set, the selection will be replaced with a newline character.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.newlineInCode()
```

Example 2 (unknown):
```unknown
editor.commands.newlineInCode()
```

---

## Nodes and marks commands

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks

**Contents:**
- Nodes and marks commands
- Use Cases
- List of nodes and marks commands
  - Was this page helpful?

Tiptap provides commands to manipulate nodes and marks easily.

Nodes and marks are the building blocks of your Tiptap editor. Nodes represent content elements like paragraphs, headings, or images, while marks provide inline formatting, such as bold, italic, or links.

---

## Node Positions

**URL:** https://tiptap.dev/docs/editor/api/node-positions

**Contents:**
- Node Positions
- Use Node Positions
- What can I do with a NodePos?
- API
  - NodePos
    - Methods
      - constructor
      - closest
      - querySelector
      - querySelectorAll

Node Positions (NodePos) describe the specific position of a node, its children, and its parent, providing easy navigation between them. Node Positions are heavily inspired by the DOM and are based on ProseMirror's ResolvedPos implementation.

The easiest way to create a new Node Position is by using the helper functions in the Editor instance. This way you always use the correct editor instance and have direct access to the API.

You can also create your own NodePos instances:

NodePos lets you traverse the document similarly to the document DOM in your browser. You can access parent nodes, child nodes, and sibling nodes.

Example: Get and update the content of a codeBlock node

If you are familiar with the DOM, this example will look familiar:

Example: Select list items and insert a new item in a bullet list

The NodePos class is the main class you will work with. It describes a specific position of a node, its children, its parent and easy ways to navigate between them. They are heavily inspired by the DOM and are based on ProseMirror's ResolvedPos implementation.

The closest NodePos instance of your NodePosition going up the depth. If there is no matching NodePos, it will return null.

Returns NodePos | null

The first matching NodePos instance of your NodePosition going down the depth. If there is no matching NodePos, it will return null.

You can also filter by attributes via the second attribute.

Returns NodePos | null

All matching NodePos instances of your NodePosition going down the depth. If there is no matching NodePos, it will return an empty array.

You can also filter by attributes via the second attribute.

Returns Array<NodePos>

Set attributes on the current NodePos.

The ProseMirror Node at the current Node Position.

The DOM element at the current Node Position.

The content of your NodePosition. You can set this to a new value to update the content of the node.

The attributes of your NodePosition.

The text content of your NodePosition.

The depth of your NodePosition.

The position of your NodePosition.

The size of your NodePosition.

The from position of your NodePosition.

The to position of your NodePosition.

The range of your NodePosition.

The parent NodePos of your NodePosition.

The NodePos before your NodePosition. If there is no NodePos before, it will return null.

Returns NodePos | null

The NodePos after your NodePosition. If there is no NodePos after, it will return null.

Returns NodePos | null

The child NodePos instances of your NodePosition.

Returns Array<NodePos>

The first child NodePos instance of your NodePosition. If there is no child, it will return null.

Returns NodePos | null

The last child NodePos instance of your NodePosition. If there is no child, it will return null.

Returns NodePos | null

**Examples:**

Example 1 (css):
```css
// set up your editor somewhere up here

// The NodePosition for the outermost document node
const $doc = editor.$doc

// Get all nodes of type 'heading' in the document
const $headings = editor.$nodes('heading')

// Filter by attributes
const $h1 = editor.$nodes('heading', { level: 1 })

// Pick nodes directly
const $firstHeading = editor.$node('heading', { level: 1 })

// Create a new NodePos via the $pos method when the type is unknown
const $myCustomPos = editor.$pos(30)
```

Example 2 (css):
```css
// set up your editor somewhere up here

// The NodePosition for the outermost document node
const $doc = editor.$doc

// Get all nodes of type 'heading' in the document
const $headings = editor.$nodes('heading')

// Filter by attributes
const $h1 = editor.$nodes('heading', { level: 1 })

// Pick nodes directly
const $firstHeading = editor.$node('heading', { level: 1 })

// Create a new NodePos via the $pos method when the type is unknown
const $myCustomPos = editor.$pos(30)
```

Example 3 (javascript):
```javascript
// You need to have an editor instance
// and a position you want to map to
const myNodePos = new NodePos(100, editor)
```

Example 4 (javascript):
```javascript
// You need to have an editor instance
// and a position you want to map to
const myNodePos = new NodePos(100, editor)
```

---

## Paste Rules

**URL:** https://tiptap.dev/docs/editor/api/paste-rules

**Contents:**
- Paste Rules
- What are paste rules?
- How do paste rules work in Tiptap?
- Creating a paste rule in an extension
  - Example: Creating a highlight mark with a paste rule
  - Example: Creating a custom figure node with a paste rule
- Understanding markPasteRule and nodePasteRule
  - Extracting information with getAttributes
    - Example: Using getAttributes in a node paste rule
    - Example: Using getAttributes in a mark paste rule

Paste rules are a powerful feature in Tiptap that allow you to automatically transform content as it is pasted into the editor. They can be used to create shortcuts for formatting, inserting content, or triggering commands based on specific patterns in pasted text.

Paste rules are pattern-based triggers that watch for specific text or content when it is pasted into the editor, and automatically transform it into something else. For example, pasting **bold** can automatically turn the text into bold formatting, or pasting an image URL can create an image node. Paste rules are especially useful for implementing Markdown-like shortcuts and improving the user experience when pasting content from external sources.

Tiptap uses paste rules to provide many of its default shortcuts and behaviors for pasted content. Paste rules are defined as regular expressions or custom matchers that match pasted text. When the pattern is detected, the rule executes a transformation—such as applying a mark, inserting a node, or running a command.

Paste rules are typically registered inside extensions (nodes, marks, or generic extensions) using the addPasteRules() method. Tiptap provides helper functions like markPasteRule and nodePasteRule to simplify the creation of paste rules for marks and nodes.

To add a custom paste rule, define the addPasteRules() method in your extension. This method should return an array of paste rules.

Tiptap provides two helper functions to simplify the creation of paste rules:

Both functions accept a configuration object with at least these properties:

The getAttributes function allows you to extract data from the matched input and pass it as attributes to the node or mark. This is especially useful for nodes like images or figures, where you want to capture values like src, alt, or title from the pasted content.

In this example, when a user pastes something like ![Alt text](image.png "Optional title"), the paste rule extracts the alt, src, and title from the match and passes them as attributes to the node.

**Examples:**

Example 1 (go):
```go
import { Mark, markPasteRule } from '@tiptap/core'

const HighlightMark = Mark.create({
  name: 'highlight',

  addPasteRules() {
    return [
      markPasteRule({
        find: /(?:==)((?:[^=]+))(?:==)/g, // Matches ==highlight==
        type: this.type,
      }),
    ]
  },
})
```

Example 2 (go):
```go
import { Mark, markPasteRule } from '@tiptap/core'

const HighlightMark = Mark.create({
  name: 'highlight',

  addPasteRules() {
    return [
      markPasteRule({
        find: /(?:==)((?:[^=]+))(?:==)/g, // Matches ==highlight==
        type: this.type,
      }),
    ]
  },
})
```

Example 3 (typescript):
```typescript
import { Node, nodePasteRule } from '@tiptap/core'

const FigureNode = Node.create({
  name: 'figure',

  addPasteRules() {
    return [
      nodePasteRule({
        find: /!\[(.*?)\]\((.*?)(?:\s+"(.*?)")?\)/g, // Matches ![alt](src "title")
        type: this.type,
        getAttributes: match => {
          const [, alt, src, title] = match
          return { src, alt, title }
        },
      }),
    ]
  },
})
```

Example 4 (typescript):
```typescript
import { Node, nodePasteRule } from '@tiptap/core'

const FigureNode = Node.create({
  name: 'figure',

  addPasteRules() {
    return [
      nodePasteRule({
        find: /!\[(.*?)\]\((.*?)(?:\s+"(.*?)")?\)/g, // Matches ![alt](src "title")
        type: this.type,
        getAttributes: match => {
          const [, alt, src, title] = match
          return { src, alt, title }
        },
      }),
    ]
  },
})
```

---

## Position Utilities

**URL:** https://tiptap.dev/docs/editor/api/utilities/position

**Contents:**
- Position Utilities
- Creating a position
- Updating a position
- Limitations
- API reference
  - MappablePosition
    - Properties
    - Methods
      - toJSON
      - fromJSON

Position utilities help you track and update positions in your editor as the document changes.

Use createMappablePosition to create a position instance.

The createMappablePosition method returns a MappablePosition instance by default. If the Collaboration extension is active, it will return a CollaborationMappablePosition instance.

Use getUpdatedPosition to update a position after a transaction. The position is updated to reflect changes in the document.

The getUpdatedPosition method returns an object with the following properties:

When a collaborative transaction breaks a paragraph in two, Y.js sees it as the deletion of the second half of the paragraph, followed by the insertion of a new paragraph. Thus, if the original position is after the line break, the updated position will not be placed in the second paragraph, but at the end of the first paragraph. This is a limitation of Y.js relative positions and we're working on a solution.

The base MappablePosition class tracks a position in the document.

Serializes the position to a JSON object for storage.

Deserializes a position from a JSON object.

CollaborationMappablePosition extends MappablePosition with Y.js relative position support for collaborative editing. It's automatically used when the Collaboration extension is active.

Serializes the collaboration position to a JSON object, including Y.js relative position data.

Deserializes a collaboration position from a JSON object.

**Examples:**

Example 1 (javascript):
```javascript
// Create a position at offset 10
const position = editor.utils.createMappablePosition(10)
```

Example 2 (javascript):
```javascript
// Create a position at offset 10
const position = editor.utils.createMappablePosition(10)
```

Example 3 (css):
```css
// Get the updated position after a transaction
const { position: updatedPosition, mapResult } = editor.utils.getUpdatedPosition(
  position,
  transaction,
)

// The updated position reflects the new location after the transaction
const newOffset = updatedPosition.position
```

Example 4 (css):
```css
// Get the updated position after a transaction
const { position: updatedPosition, mapResult } = editor.utils.getUpdatedPosition(
  position,
  transaction,
)

// The updated position reflects the new location after the transaction
const newOffset = updatedPosition.position
```

---

## resetAttributes command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/reset-attributes

**Contents:**
- resetAttributes command
- Parameters
- Use the resetAttributes command
  - Was this page helpful?

resetAttributes resets some of the nodes attributes back to it's default attributes.

typeOrName: string | Node

The node that should be resetted. Can be a string or a Node.

attributes: string | string[]

A string or an array of strings that defines which attributes should be reset.

**Examples:**

Example 1 (php):
```php
// reset the style and class attributes on the currently selected paragraph nodes
editor.commands.resetAttributes('paragraph', ['style', 'class'])
```

Example 2 (php):
```php
// reset the style and class attributes on the currently selected paragraph nodes
editor.commands.resetAttributes('paragraph', ['style', 'class'])
```

---

## Resizable Node Views

**URL:** https://tiptap.dev/docs/editor/api/resizable-nodeviews

**Contents:**
- Resizable Node Views
- What is a Resizable Node View?
- Options
  - options properties:
- Callbacks
- Usage example
  - Notes:
- Behavior details & edge cases
- Examples
  - Resizable Images

A small, framework-agnostic NodeView that wraps any HTMLElement (image, iframe, video…) and adds configurable resize handles. It manages user interaction, applies min/max constraints, optionally preserves aspect ratio, and exposes callbacks for live updates and commits.

ResizableNodeView is a ProseMirror/Tiptap-compatible NodeView that:

directions?: ResizableNodeViewDirection[] Default: ['bottom-left', 'bottom-right', 'top-left', 'top-right'] Allowed: 'top' | 'right' | 'bottom' | 'left' | 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left'

min?: Partial<{ width: number; height: number; }> Default: { width: 8, height: 8 } (pixels)

max?: Partial<{ width: number; height: number; }> Default: undefined (no max)

preserveAspectRatio?: boolean Default: false When true always preserves aspect ratio. When false, pressing Shift while dragging temporarily preserves aspect ratio.

className?: { container?: string; wrapper?: string; handle?: string; resizing?: string } Optional class names applied to container, wrapper, each handle, and a class added while actively resizing.

Example pattern to persist sizes inside onCommit:

Minimal image extension node view:

**Examples:**

Example 1 (javascript):
```javascript
const pos = getPos()
if (pos !== undefined) {
  editor.commands.updateAttributes('image', { width, height })
}
```

Example 2 (javascript):
```javascript
const pos = getPos()
if (pos !== undefined) {
  editor.commands.updateAttributes('image', { width, height })
}
```

Example 3 (javascript):
```javascript
// inside addNodeView()
return ({ node, getPos, HTMLAttributes }) => {
  const img = document.createElement('img')
  img.src = HTMLAttributes.src

  // copy non-size attributes to element
  Object.entries(HTMLAttributes).forEach(([key, value]) => {
    if (value == null) return
    if (key === 'width' || key === 'height') return
    img.setAttribute(key, String(value))
  })

  // instantiate ResizableNodeView
  return new ResizableNodeView({
    element: img,
    node,
    getPos,
    onResize: (w, h) => {
      img.style.width = `${w}px`
      img.style.height = `${h}px`
    },
    onCommit: (w, h) => {
      const pos = getPos()
      if (pos === undefined) return
      // persist new size to the node
      editor.commands.updateAttributes('image', { width: w, height: h })
    },
    onUpdate: (updatedNode) => {
      if (updatedNode.type !== node.type) return false
      return true
    },
    options: {
      directions: ['bottom-right', 'bottom-left', 'top-right', 'top-left'],
      min: { width: 50, height: 50 },
      preserveAspectRatio: false, // hold Shift to lock aspect ratio
      className: {
        container: 'my-resize-container',
        wrapper: 'my-resize-wrapper',
        handle: 'my-resize-handle',
        resizing: 'is-resizing',
      },
    },
  })
}
```

Example 4 (javascript):
```javascript
// inside addNodeView()
return ({ node, getPos, HTMLAttributes }) => {
  const img = document.createElement('img')
  img.src = HTMLAttributes.src

  // copy non-size attributes to element
  Object.entries(HTMLAttributes).forEach(([key, value]) => {
    if (value == null) return
    if (key === 'width' || key === 'height') return
    img.setAttribute(key, String(value))
  })

  // instantiate ResizableNodeView
  return new ResizableNodeView({
    element: img,
    node,
    getPos,
    onResize: (w, h) => {
      img.style.width = `${w}px`
      img.style.height = `${h}px`
    },
    onCommit: (w, h) => {
      const pos = getPos()
      if (pos === undefined) return
      // persist new size to the node
      editor.commands.updateAttributes('image', { width: w, height: h })
    },
    onUpdate: (updatedNode) => {
      if (updatedNode.type !== node.type) return false
      return true
    },
    options: {
      directions: ['bottom-right', 'bottom-left', 'top-right', 'top-left'],
      min: { width: 50, height: 50 },
      preserveAspectRatio: false, // hold Shift to lock aspect ratio
      className: {
        container: 'my-resize-container',
        wrapper: 'my-resize-wrapper',
        handle: 'my-resize-handle',
        resizing: 'is-resizing',
      },
    },
  })
}
```

---

## scrollIntoView command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/scroll-into-view

**Contents:**
- scrollIntoView command
- Use the scrollIntoView command
  - Was this page helpful?

scrollIntoView scrolls the view to the current selection or cursor position.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.scrollIntoView()
```

Example 2 (unknown):
```unknown
editor.commands.scrollIntoView()
```

---

## selectAll command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/select-all

**Contents:**
- selectAll command
- Use the selectAll command
  - Was this page helpful?

Selects the whole document at once.

**Examples:**

Example 1 (sql):
```sql
// Select the whole document
editor.commands.selectAll()
```

Example 2 (sql):
```sql
// Select the whole document
editor.commands.selectAll()
```

---

## Selection commands

**URL:** https://tiptap.dev/docs/editor/api/commands/selection

**Contents:**
- Selection commands
- Use Cases
- List of selection commands
  - Was this page helpful?

The Tiptap editor provides editor commands for managing selection and focus within your documents. Here’s an overview of the essential selection commands that help you manage cursor movement, selections, and focus behavior.

---

## selectNodeBackward command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/select-node-backward

**Contents:**
- selectNodeBackward command
- Use the selectNodeBackward command
  - Was this page helpful?

If the selection is empty and at the start of a textblock, selectNodeBackward will select the node before the current textblock if possible.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.selectNodeBackward()
```

Example 2 (unknown):
```unknown
editor.commands.selectNodeBackward()
```

---

## selectNodeForward command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/select-node-forward

**Contents:**
- selectNodeForward command
- Use the selectNodeForward command
  - Was this page helpful?

If the selection is empty and at the end of a textblock, selectNodeForward will select the node after the current textblock if possible.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.selectNodeForward()
```

Example 2 (unknown):
```unknown
editor.commands.selectNodeForward()
```

---

## selectParentNode command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/select-parent-node

**Contents:**
- selectParentNode command
- Use the selectParentNode command
  - Was this page helpful?

selectParentNode will try to get the parent node of the currently selected node and move the selection to that node.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.selectParentNode()
```

Example 2 (unknown):
```unknown
editor.commands.selectParentNode()
```

---

## selectTextblockEnd command

**URL:** https://tiptap.dev/docs/editor/api/commands/select-textblock-end

**Contents:**
- selectTextblockEnd command
- Use the selectTextblockEnd command
  - Was this page helpful?

The selectTextblockEnd will move the cursor to the end of the current textblock if the block is a valid textblock.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.selectTextblockEnd()
```

Example 2 (unknown):
```unknown
editor.commands.selectTextblockEnd()
```

---

## selectTextblockStart command

**URL:** https://tiptap.dev/docs/editor/api/commands/select-textblock-start

**Contents:**
- selectTextblockStart command
- Use the selectTextblockStart command
  - Was this page helpful?

The selectTextblockStart will move the cursor to the start of the current textblock if the block is a valid textblock.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.selectTextblockStart()
```

Example 2 (unknown):
```unknown
editor.commands.selectTextblockStart()
```

---

## setContent command

**URL:** https://tiptap.dev/docs/editor/api/commands/content/set-content

**Contents:**
- setContent command
- Parameters
  - content
  - options
- Examples
  - Was this page helpful?

The setContent command replaces the document with new content. You can pass JSON or HTML. It's basically the same as setting the content on initialization.

See also: insertContent, clearContent

The new content as string (JSON or HTML), Fragment, or ProseMirror Node. The editor will only render what's allowed according to the schema.

Optional configuration object with the following properties:

parseOptions?: Record<string, any> Options to configure the parsing. Read more about parseOptions in the ProseMirror documentation.

errorOnInvalidContent?: boolean Whether to throw an error if the content is invalid.

emitUpdate?: boolean (true) Whether to emit an update event. Defaults to true (Note: This changed from false in v2).

**Examples:**

Example 1 (json):
```json
// Plain text
editor.commands.setContent('Example Text')

// HTML
editor.commands.setContent('<p>Example Text</p>')

// JSON
editor.commands.setContent({
  type: 'doc',
  content: [
    {
      type: 'paragraph',
      content: [
        {
          type: 'text',
          text: 'Example Text',
        },
      ],
    },
  ],
})

// With options
editor.commands.setContent('<p>Example Text</p>', {
  emitUpdate: false,
  parseOptions: {
    preserveWhitespace: 'full',
  },
  errorOnInvalidContent: true,
})
```

Example 2 (json):
```json
// Plain text
editor.commands.setContent('Example Text')

// HTML
editor.commands.setContent('<p>Example Text</p>')

// JSON
editor.commands.setContent({
  type: 'doc',
  content: [
    {
      type: 'paragraph',
      content: [
        {
          type: 'text',
          text: 'Example Text',
        },
      ],
    },
  ],
})

// With options
editor.commands.setContent('<p>Example Text</p>', {
  emitUpdate: false,
  parseOptions: {
    preserveWhitespace: 'full',
  },
  errorOnInvalidContent: true,
})
```

---

## setMark command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/set-mark

**Contents:**
- setMark command
- Parameters
- Use the setMark command
  - Was this page helpful?

The setMark command will add a new mark at the current selection.

typeOrName: string | MarkType

The type of a mark to add. Can be a string or a MarkType.

attributes: Record<string, any>

The attributes that should be applied to the mark. This is optional.

**Examples:**

Example 1 (css):
```css
editor.commands.setMark('bold', { class: 'bold-tag' })
```

Example 2 (css):
```css
editor.commands.setMark('bold', { class: 'bold-tag' })
```

---

## setMeta command

**URL:** https://tiptap.dev/docs/editor/api/commands/set-meta

**Contents:**
- setMeta command
- Parameters
- Use the setMeta command
  - Was this page helpful?

Store a metadata property in the current transaction.

The name of your metadata. You can get its value at any time with getMeta.

Store any value within your metadata.

**Examples:**

Example 1 (sql):
```sql
// Prevent the update event from being triggered
editor.commands.setMeta('preventUpdate', true)

// Store any value in the current transaction.
// You can get this value at any time with tr.getMeta('foo').
editor.commands.setMeta('foo', 'bar')
```

Example 2 (sql):
```sql
// Prevent the update event from being triggered
editor.commands.setMeta('preventUpdate', true)

// Store any value in the current transaction.
// You can get this value at any time with tr.getMeta('foo').
editor.commands.setMeta('foo', 'bar')
```

---

## setNodeSelection command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/set-node-selection

**Contents:**
- setNodeSelection command
- Parameters
- Use the setNodeSelection command
  - Was this page helpful?

setNodeSelection creates a new NodeSelection at a given position. A node selection is a selection that points to a single node. See more

The position the NodeSelection will be created at.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.setNodeSelection(10)
```

Example 2 (unknown):
```unknown
editor.commands.setNodeSelection(10)
```

---

## setNode command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/set-node

**Contents:**
- setNode command
- Parameters
- Use the setNode command
  - Was this page helpful?

The setNode command will replace a given range with a given node. The range depends on the current selection. Important: Currently setNode only supports text block nodes.

typeOrName: string | NodeType

The type of the node that will replace the range. Can be a string or a NodeType.

attributes?: Record<string, any>

The attributes that should be applied to the node. This is optional.

**Examples:**

Example 1 (css):
```css
editor.commands.setNode('paragraph', { id: 'paragraph-01' })
```

Example 2 (css):
```css
editor.commands.setNode('paragraph', { id: 'paragraph-01' })
```

---

## setTextDirection command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/set-text-direction

**Contents:**
- setTextDirection command
- Parameters
  - direction
  - position
- Examples
  - Was this page helpful?

The setTextDirection command sets the text direction for nodes at the current selection or at a specified position. This is useful for controlling the direction of right-to-left (RTL) languages like Arabic and Hebrew, or for bidirectional text content.

See also: unsetTextDirection

The text direction to set. Can be 'ltr' (left-to-right), 'rtl' (right-to-left), or 'auto' (automatically detect based on content).

Optional. The position or range where the direction should be applied. If not provided, the command will use the current selection.

**Examples:**

Example 1 (css):
```css
// Set RTL direction on current selection
editor.commands.setTextDirection('rtl')

// Set LTR direction on current selection
editor.commands.setTextDirection('ltr')

// Set auto direction on current selection
editor.commands.setTextDirection('auto')

// Set direction on a specific position
editor.commands.setTextDirection('rtl', 5)

// Set direction on a specific range
editor.commands.setTextDirection('ltr', { from: 0, to: 10 })

// Chain with other commands
editor.chain().focus().setTextDirection('rtl').run()
```

Example 2 (css):
```css
// Set RTL direction on current selection
editor.commands.setTextDirection('rtl')

// Set LTR direction on current selection
editor.commands.setTextDirection('ltr')

// Set auto direction on current selection
editor.commands.setTextDirection('auto')

// Set direction on a specific position
editor.commands.setTextDirection('rtl', 5)

// Set direction on a specific range
editor.commands.setTextDirection('ltr', { from: 0, to: 10 })

// Chain with other commands
editor.chain().focus().setTextDirection('rtl').run()
```

---

## setTextSelection command

**URL:** https://tiptap.dev/docs/editor/api/commands/selection/set-text-selection

**Contents:**
- setTextSelection command
- Parameters
- Use the setTextSelection command
  - Was this page helpful?

If you think of selection in the context of an editor, you’ll probably think of a text selection. With setTextSelection you can control that text selection and set it to a specified range or position.

See also: focus, setNodeSelection, deleteSelection, selectAll

position: number | Range

Pass a number, or a Range, for example { from: 5, to: 10 }.

**Examples:**

Example 1 (css):
```css
// Set the cursor to the specified position
editor.commands.setTextSelection(10)

// Set the text selection to the specified range
editor.commands.setTextSelection({ from: 5, to: 10 })
```

Example 2 (css):
```css
// Set the cursor to the specified position
editor.commands.setTextSelection(10)

// Set the text selection to the specified range
editor.commands.setTextSelection({ from: 5, to: 10 })
```

---

## sinkListItem command

**URL:** https://tiptap.dev/docs/editor/api/commands/lists/sink-list-item

**Contents:**
- sinkListItem command
- Use the sinkListItem command
  - Was this page helpful?

The sinkListItem will try to sink the list item around the current selection down into a wrapping child list.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.sinkListItem()
```

Example 2 (unknown):
```unknown
editor.commands.sinkListItem()
```

---

## splitBlock command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/split-block

**Contents:**
- splitBlock command
- Parameters
- Use the splitBlock command
  - Was this page helpful?

splitBlock will split the current node into two nodes at the current NodeSelection. If the current selection is not splittable, the command will be ignored.

options: Record<string, any>

**Examples:**

Example 1 (css):
```css
// split the current node and keep marks
editor.commands.splitBlock()

// split the current node and don't keep marks
editor.commands.splitBlock({ keepMarks: false })
```

Example 2 (css):
```css
// split the current node and keep marks
editor.commands.splitBlock()

// split the current node and don't keep marks
editor.commands.splitBlock({ keepMarks: false })
```

---

## splitListItem command

**URL:** https://tiptap.dev/docs/editor/api/commands/lists/split-list-item

**Contents:**
- splitListItem command
- Parameters
- Use the splitListItem command
  - Was this page helpful?

splitListItem splits one list item into two separate list items. If this is a nested list, the wrapping list item should be split.

typeOrName: string | NodeType

The type of node that should be split into two separate list items.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.splitListItem('bulletList')
```

Example 2 (unknown):
```unknown
editor.commands.splitListItem('bulletList')
```

---

## State and change events

**URL:** https://tiptap.dev/docs/collaboration/provider/events

**Contents:**
- State and change events
- Use provider events
- Configure event listeners
  - Bind events dynamically
- Provider event examples
  - Display connection status
  - Sync document status
  - Handle authentication issues
  - Was this page helpful?

Events in Collaboration providers let you respond to various states and changes, such as successful connections or authentication updates. You can attach event listeners during provider initialization or add them later based on your application's needs.

To track events immediately, pass event listeners directly to the provider's constructor. This guarantees that listeners are active from the start.

To add or remove listeners after initialization, the provider supports dynamic binding and unbinding of event handlers.

Example: Binding event listeners during provider initialization

Example: Binding/unbinding event listeners after provider initialization

Use onConnect and onDisconnect to provide users with real-time connection status feedback, enhancing the user experience.

Use synced to alert users when the document is fully synced initially, ensuring they start working with the latest version.

Use authenticationFailed to catch authentication errors and prompt users to reauthenticate, ensuring secure access.

**Examples:**

Example 1 (javascript):
```javascript
const provider = new TiptapCollabProvider({
  appId: '', // Use for cloud setups, replace with baseUrl in case of on-prem
  name: 'example-document', // Document identifier
  token: '', // Your authentication JWT token
  document: ydoc,
  onOpen() {
    console.log('WebSocket connection opened.')
  },
  onConnect() {
    console.log('Connected to the server.')
  },
  // See below for more event listeners...
})
```

Example 2 (javascript):
```javascript
const provider = new TiptapCollabProvider({
  appId: '', // Use for cloud setups, replace with baseUrl in case of on-prem
  name: 'example-document', // Document identifier
  token: '', // Your authentication JWT token
  document: ydoc,
  onOpen() {
    console.log('WebSocket connection opened.')
  },
  onConnect() {
    console.log('Connected to the server.')
  },
  // See below for more event listeners...
})
```

Example 3 (javascript):
```javascript
const provider = new TiptapCollabProvider({
  // …
})

provider.on('synced', () => {
  console.log('Document synced.')
})
```

Example 4 (javascript):
```javascript
const provider = new TiptapCollabProvider({
  // …
})

provider.on('synced', () => {
  console.log('Document synced.')
})
```

---

## Static Renderer

**URL:** https://tiptap.dev/docs/editor/api/utilities/static-renderer

**Contents:**
- Static Renderer
- Why Static Render?
- Generating HTML strings from JSON
  - generateHTML API
- Generating Markdown from JSON
  - generateMarkdown API
- Generating React components from JSON
  - generateReactElement API
  - React NodeViews
- Shared Options

The Static Renderer helps render JSON content as HTML, markdown, or React components without an editor instance. All it needs is JSON content and a list of extensions.

The main use case for static rendering is to render a Tiptap/ProseMirror JSON document on the server-side, for example within a Next.js or Nuxt.js application. This way, you can render the content of your editor to HTML before sending it to the client, which can improve the performance of your application by not having to load the editor on the client or server.

Another use case is to render the content of your editor to another format like markdown, which can be useful if you want to send it to a markdown-based API. The static renderer is built in a way that the output can be anything you want, as long as you provide the correct mappings.

But what makes it static? The static renderer doesn't require a browser, DOM or even an editor instance to render the content. It's a pure JavaScript function that takes a document (as JSON or Prosemirror Node instance) and returns the target format back.

Given a JSON document, the renderToHTMLString function will return an HTML string representing the JSON content. The function takes three arguments: the JSON document, a list of extensions, and an options object.

Given a JSON document, the renderToMarkdown function will return a markdown string representing the JSON content. The function takes three arguments: the JSON document, a list of extensions, and an options object.

This package does not validate the markdown output, there are several markdown flavors and this package does not enforce any of them. It's up to you to ensure that the markdown output is valid.

Given a JSON document, the renderToReactElement function will return a React component representing the JSON content. The function takes three arguments: the JSON document, a list of extensions, and an options object.

The static renderer doesn't support node views automatically, so you need to provide a mapping for each node type that you want rendered as a node view. Here is an example of how you can render a node view as a React component:

But, what if you want to render the rich text content of the node view? You can do that by providing a NodeViewContent component as a child of the node view component:

The renderToHTMLString, renderToMarkdown, and renderToReactElement functions take an options object as an argument. This object can be used to customize the output of the renderer by providing custom node and mark mappings, or handling unhandled nodes and marks.

To cut down on bundle size in your application, the static renderer is split into three separate packages: @tiptap/static-renderer/pm/html, @tiptap/static-renderer/pm/markdown, and @tiptap/static-renderer/pm/react. This way, you only need to import the parts of the static renderer that you need. If you want the most flexibility, you can import the entire static renderer package with @tiptap/static-renderer.

Packages in the json namespace are also available for statically rendering without a runtime dependency on any ProseMirror packages. But, these packages cannot automatically map Prosemirror nodes and marks to the target format. You will need to provide custom mappings for every node and mark in these packages.

These packages have the same API as the pm namespace, but:

The static renderer uses default mappings for Prosemirror nodes and marks to the target format. These mappings can be overridden by providing custom mappings in the options object. This allows you to customize the output of the renderer to suit your needs.

To convert custom nodes and marks to the target format, you should provide a mapping function that takes a node or mark object as an argument and returns the appropriate target format element. If you encounter an unhandled node or mark, you can provide a function that will be called with the unhandled node or mark as an argument.

The static renderer packages in the json namespace map over the JSON content and call the appropriate mapping function for each node and mark. The renderJSONContentToString function returns a string representing the JSON content, while the renderJSONContentToReactElement function returns a React element representing the JSON content.

The static renderer packages in the pm namespace, extending the packages in the json namespace, utilize the renderHTML method of Tiptap extensions to generate default mappings of Prosemirror nodes/marks to the target format. These can be completely overridden by providing custom mappings in the options.

packages/static-renderer/

**Examples:**

Example 1 (json):
```json
import StarterKit from '@tiptap/starter-kit'
import { renderToHTMLString } from '@tiptap/static-renderer/pm/html'

renderToHTMLString({
  extensions: [StarterKit], // using your extensions
  content: {
    type: 'doc',
    content: [
      {
        type: 'paragraph',
        content: [
          {
            type: 'text',
            text: 'Hello World!',
          },
        ],
      },
    ],
  },
})
// returns: '<p>Hello World!</p>'
```

Example 2 (json):
```json
import StarterKit from '@tiptap/starter-kit'
import { renderToHTMLString } from '@tiptap/static-renderer/pm/html'

renderToHTMLString({
  extensions: [StarterKit], // using your extensions
  content: {
    type: 'doc',
    content: [
      {
        type: 'paragraph',
        content: [
          {
            type: 'text',
            text: 'Hello World!',
          },
        ],
      },
    ],
  },
})
// returns: '<p>Hello World!</p>'
```

Example 3 (typescript):
```typescript
function renderToHTMLString(options: {
  extensions: Extension[]
  content: ProsemirrorNode | JSONContent
  options?: TiptapHTMLStaticRendererOptions
}): string
```

Example 4 (typescript):
```typescript
function renderToHTMLString(options: {
  extensions: Extension[]
  content: ProsemirrorNode | JSONContent
  options?: TiptapHTMLStaticRendererOptions
}): string
```

---

## Suggestion utility

**URL:** https://tiptap.dev/docs/editor/api/utilities/suggestion

**Contents:**
- Suggestion utility
- Settings
  - char
  - pluginKey
  - allow
  - allowSpaces
  - allowedPrefixes
  - startOfLine
  - decorationTag
  - decorationClass

This utility helps with all kinds of suggestions in the editor. Have a look at the Mention or Emoji node to see it in action.

The character that triggers the autocomplete popup.

A ProseMirror PluginKey.

Default: SuggestionPluginKey

A function that returns a boolean to indicate if the suggestion should be active.

Default: (props: { editor: Editor; state: EditorState; range: Range, isActive?: boolean }) => true

Allows or disallows spaces in suggested items.

The prefix characters that are allowed to trigger a suggestion. Set to null to allow any prefix character.

Trigger the autocomplete popup at the start of a line only.

The HTML tag that should be rendered for the suggestion.

A CSS class that should be added to the suggestion.

Default: 'suggestion'

The content that should be rendered in the suggestion decoration.

A CSS class that should be added to the suggestion when it is empty.

Executed when a suggestion is selected.

Pass an array of filtered suggestions, can be async.

Default: ({ editor, query }) => []

A render function for the autocomplete popup.

Optional param to replace the built-in regex matching of editor content that triggers a suggestion. See the source for more detail.

Default: findSuggestionMatch(config: Trigger): SuggestionMatch

Sometimes you want your users to be able to exit an an open suggestion without selecting an item. To achieve this, users can either press Escape which will close the open suggestion. If you want to manually trigger the closing of the suggestion, you can use use exitSuggestion utility function to close existing suggestions on your view.

**Examples:**

Example 1 (python):
```python
import { exitSuggestion } from '@tiptap/suggestion'
import { PluginKey } from 'prosemirror-state' // optional, if you need to create a custom key

const MySuggestionPluginKey = new PluginKey('my-suggestions') // or use the default 'suggestion'

exitSuggestion(editor.view, MySuggestionPluginKey)

// Alternatively, use the default plugin key:
// exitSuggestion(editor.view, 'suggestion')
```

Example 2 (python):
```python
import { exitSuggestion } from '@tiptap/suggestion'
import { PluginKey } from 'prosemirror-state' // optional, if you need to create a custom key

const MySuggestionPluginKey = new PluginKey('my-suggestions') // or use the default 'suggestion'

exitSuggestion(editor.view, MySuggestionPluginKey)

// Alternatively, use the default plugin key:
// exitSuggestion(editor.view, 'suggestion')
```

---

## Tiptap for PHP utility

**URL:** https://tiptap.dev/docs/editor/api/utilities/tiptap-for-php

**Contents:**
- Tiptap for PHP utility
- Install
- Using the Tiptap PHP utility
- Documentation
  - Was this page helpful?

A PHP package to work with Tiptap content. You can transform Tiptap-compatible JSON to HTML, and the other way around, sanitize your content, or just modify it.

You can install the package via composer:

The PHP package mimics large parts of the JavaScript package. If you know your way around Tiptap, the PHP syntax will feel familiar to you. Here is an easy example:

There’s a lot more the PHP package can do. Check out the repository on GitHub.

**Examples:**

Example 1 (unknown):
```unknown
composer require ueberdosis/tiptap-php
```

Example 2 (unknown):
```unknown
composer require ueberdosis/tiptap-php
```

Example 3 (typescript):
```typescript
(new Tiptap\Editor)
    ->setContent('<p>Example Text</p>')
    ->getDocument();

// Returns:
// ['type' => 'doc', 'content' => …]
```

Example 4 (typescript):
```typescript
(new Tiptap\Editor)
    ->setContent('<p>Example Text</p>')
    ->getDocument();

// Returns:
// ['type' => 'doc', 'content' => …]
```

---

## Tiptap Utilities

**URL:** https://tiptap.dev/docs/editor/api/utilities

**Contents:**
- Tiptap Utilities
- All utilities
  - Was this page helpful?

Tiptap Utilities are complementing the Editor API, providing tools that improve and extend your interactions with the editor and content.

---

## toggleList command

**URL:** https://tiptap.dev/docs/editor/api/commands/lists/toggle-list

**Contents:**
- toggleList command
- Parameters
- Use the toggleList command
  - Was this page helpful?

toggleList will toggle between different types of lists.

listTypeOrName: string | NodeType

The type of node that should be used for the wrapping list

itemTypeOrName: string | NodeType

The type of node that should be used for the list items

If marks should be kept as list items or not

attributes?: Record<string, any>

The attributes that should be applied to the list. This is optional.

**Examples:**

Example 1 (unknown):
```unknown
// toggle a bullet list with list items
editor.commands.toggleList('bulletList', 'listItem')

// toggle a numbered list with list items
editor.commands.toggleList('orderedList', 'listItem')
```

Example 2 (unknown):
```unknown
// toggle a bullet list with list items
editor.commands.toggleList('bulletList', 'listItem')

// toggle a numbered list with list items
editor.commands.toggleList('orderedList', 'listItem')
```

---

## toggleMark command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/toggle-mark

**Contents:**
- toggleMark command
- Parameters
- Use the toggleMark command
  - Was this page helpful?

The toggleMark command toggles a specific mark on and off at the current selection.

typeOrName: string | MarkType

The type of mark that should be toggled.

attributes?: Record<string, any>

The attributes that should be applied to the mark. This is optional.

options?: Record<string, any>

**Examples:**

Example 1 (css):
```css
// toggles a bold mark
editor.commands.toggleMark('bold')

// toggles bold mark with a color attribute
editor.commands.toggleMark('bold', { color: 'red' })

// toggles a bold mark with a color attribute and removes the mark across the current selection
editor.commands.toggleMark('bold', { color: 'red' }, { extendEmptyMarkRange: true })
```

Example 2 (css):
```css
// toggles a bold mark
editor.commands.toggleMark('bold')

// toggles bold mark with a color attribute
editor.commands.toggleMark('bold', { color: 'red' })

// toggles a bold mark with a color attribute and removes the mark across the current selection
editor.commands.toggleMark('bold', { color: 'red' }, { extendEmptyMarkRange: true })
```

---

## toggleNode command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/toggle-node

**Contents:**
- toggleNode command
- Parameters
- Use the toggleNode command
  - Was this page helpful?

toggleNode will toggle a node with another node.

typeOrName: string | NodeType

The type of node that should be toggled.

toggleTypeOrName: string | NodeType

The type of node that should be used for the toggling.

attributes?: Record<string, any>

The attributes that should be applied to the node. This is optional.

**Examples:**

Example 1 (css):
```css
// toggle a paragraph with a heading node
editor.commands.toggleNode('paragraph', 'heading', { level: 1 })

// toggle a paragraph with a image node
editor.commands.toggleNode('paragraph', 'image', { src: 'https://example.com/image.png' })
```

Example 2 (css):
```css
// toggle a paragraph with a heading node
editor.commands.toggleNode('paragraph', 'heading', { level: 1 })

// toggle a paragraph with a image node
editor.commands.toggleNode('paragraph', 'image', { src: 'https://example.com/image.png' })
```

---

## toggleWrap command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/toggle-wrap

**Contents:**
- toggleWrap command
- Parameters
- Use the toggleWrap command
  - Was this page helpful?

toggleWrap wraps the current node with a new node or removes a wrapping node.

typeOrName: string | NodeType

The type of node that should be used for the wrapping node.

attributes?: Record<string, any>

The attributes that should be applied to the node. This is optional.

**Examples:**

Example 1 (css):
```css
// toggle wrap the current selection with a heading node
editor.commands.toggleWrap('heading', { level: 1 })
```

Example 2 (css):
```css
// toggle wrap the current selection with a heading node
editor.commands.toggleWrap('heading', { level: 1 })
```

---

## undoInputRule command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/undo-input-rule

**Contents:**
- undoInputRule command
- Use the undoInputRule command
  - Was this page helpful?

undoInputRule will undo the most recent input rule that was triggered.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.undoInputRule()
```

Example 2 (unknown):
```unknown
editor.commands.undoInputRule()
```

---

## unsetAllMarks command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/unset-all-marks

**Contents:**
- unsetAllMarks command
- Using the unsetAllMarks command
  - Was this page helpful?

unsetAllMarks will remove all marks from the current selection.

**Examples:**

Example 1 (unknown):
```unknown
editor.commands.unsetAllMarks()
```

Example 2 (unknown):
```unknown
editor.commands.unsetAllMarks()
```

---

## unsetMark command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/unset-mark

**Contents:**
- unsetMark command
- Parameters
- Use the unsetMark command
  - Was this page helpful?

unsetMark will remove the mark from the current selection. Can also remove all marks across the current selection.

typeOrName: string | MarkType

The type of mark that should be removed.

options?: Record<string, any>

**Examples:**

Example 1 (css):
```css
// removes a bold mark
editor.commands.unsetMark('bold')

// removes a bold mark across the current selection
editor.commands.unsetMark('bold', { extendEmptyMarkRange: true })
```

Example 2 (css):
```css
// removes a bold mark
editor.commands.unsetMark('bold')

// removes a bold mark across the current selection
editor.commands.unsetMark('bold', { extendEmptyMarkRange: true })
```

---

## unsetTextDirection command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/unset-text-direction

**Contents:**
- unsetTextDirection command
- Parameters
  - position
- Examples
  - Was this page helpful?

The unsetTextDirection command removes the text direction attribute from nodes at the current selection or at a specified position. This allows nodes to inherit the direction from the editor's global textDirection setting or to have no direction attribute at all.

See also: setTextDirection

Optional. The position or range where the direction should be removed. If not provided, the command will use the current selection.

**Examples:**

Example 1 (sql):
```sql
// Remove direction from current selection
editor.commands.unsetTextDirection()

// Remove direction from a specific position
editor.commands.unsetTextDirection(5)

// Remove direction from a specific range
editor.commands.unsetTextDirection({ from: 0, to: 10 })

// Chain with other commands
editor.chain().focus().unsetTextDirection().run()
```

Example 2 (sql):
```sql
// Remove direction from current selection
editor.commands.unsetTextDirection()

// Remove direction from a specific position
editor.commands.unsetTextDirection(5)

// Remove direction from a specific range
editor.commands.unsetTextDirection({ from: 0, to: 10 })

// Chain with other commands
editor.chain().focus().unsetTextDirection().run()
```

---

## updateAttributes command

**URL:** https://tiptap.dev/docs/editor/api/commands/nodes-and-marks/update-attributes

**Contents:**
- updateAttributes command
- Parameters
- Use the updateAttributes command
  - Was this page helpful?

The updateAttributes command sets attributes of a node or mark to new values. Not passed attributes won’t be touched.

See also: extendMarkRange

typeOrName: string | NodeType | MarkType

Pass the type you want to update, for example 'heading'.

attributes: Record<string, any>

This expects an object with the attributes that need to be updated. It doesn’t need to have all attributes.

**Examples:**

Example 1 (sql):
```sql
// Update node attributes
editor.commands.updateAttributes('heading', { level: 1 })

// Update mark attributes
editor.commands.updateAttributes('highlight', { color: 'pink' })
```

Example 2 (sql):
```sql
// Update node attributes
editor.commands.updateAttributes('heading', { level: 1 })

// Update mark attributes
editor.commands.updateAttributes('highlight', { color: 'pink' })
```

---

## wrapInList command

**URL:** https://tiptap.dev/docs/editor/api/commands/lists/wrap-in-list

**Contents:**
- wrapInList command
- Parameters
- Use the wrapInList command
  - Was this page helpful?

wrapInList will wrap a node in the current selection in a list.

typeOrName: string | NodeType

The type of node that should be wrapped in a list.

attributes?: Record<string, any>

The attributes that should be applied to the list. This is optional.

**Examples:**

Example 1 (unknown):
```unknown
// wrap a paragraph in a bullet list
editor.commands.wrapInList('paragraph')
```

Example 2 (unknown):
```unknown
// wrap a paragraph in a bullet list
editor.commands.wrapInList('paragraph')
```

---
