# Tiptap - Core Concepts

**Pages:** 98

---

## Add to an existing extension

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/extend-existing

**Contents:**
- Add to an existing extension
- Name
- Settings
- Storage
- Schema
- Attributes
  - Extend existing attributes
- Global attributes
- Render HTML
- Parse HTML

Every extension has an extend() method, which takes an object with everything you want to change or add to it.

Let’s say, you’d like to change the keyboard shortcut for the bullet list. You should start with looking at the source code of the extension, in that case the BulletList node. For the bespoken example to overwrite the keyboard shortcut, your code could look like this:

The same applies to every aspect of an existing extension, except to the name. Let’s look at all the things that you can change through the extend method. We focus on one aspect in every example, but you can combine all those examples and change multiple aspects in one extend() call too.

The extension name is used in a whole lot of places and changing it isn’t too easy. If you want to change the name of an existing extension, you can copy the whole extension and change the name in all occurrences.

The extension name is also part of the JSON. If you store your content as JSON, you need to change the name there too.

All settings can be configured through the extension anyway, but if you want to change the default settings, for example to provide a library on top of Tiptap for other developers, you can do it like this:

At some point you probably want to save some data within your extension instance. This data is mutable. You can access it within the extension under this.storage.

Outside the extension you have access via editor.storage. Make sure that each extension has a unique name.

Tiptap works with a strict schema, which configures how the content can be structured, nested, how it behaves and many more things. You can change all aspects of the schema for existing extensions. Let’s walk through a few common use cases.

The default Blockquote extension can wrap other nodes, like headings. If you want to allow nothing but paragraphs in your blockquotes, set the content attribute accordingly:

The schema even allows to make your nodes draggable, that’s what the draggable option is for. It defaults to false, but you can override that.

That’s just two tiny examples, but the underlying ProseMirror schema is really powerful.

You can use attributes to store additional information in the content. Let’s say you want to extend the default Paragraph node to have different colors:

That is already enough to tell Tiptap about the new attribute, and set 'pink' as the default value. All attributes will be rendered as a HTML attribute by default, and parsed from the content when initiated.

Let’s stick with the color example and assume you want to add an inline style to actually color the text. With the renderHTML function you can return HTML attributes which will be rendered in the output.

This examples adds a style HTML attribute based on the value of color:

You can also control how the attribute is parsed from the HTML. Maybe you want to store the color in an attribute called data-color (and not just color), here’s how you would do that:

You can completely disable the rendering of attributes with rendered: false.

If you want to add an attribute to an extension and keep existing attributes, you can access them through this.parent().

In some cases, it is undefined, so make sure to check for that case, or use optional chaining this.parent?.()

Attributes can be applied to multiple extensions at once. That’s useful for text alignment, line height, color, font family, and other styling related attributes.

Take a closer look at the full source code of the TextAlign extension to see a more complex example. But here is how it works in a nutshell:

With the renderHTML function you can control how an extension is rendered to HTML. We pass an attributes object to it, with all local attributes, global attributes, and configured CSS classes. Here is an example from the Bold extension:

The first value in the array should be the name of HTML tag. If the second element is an object, it’s interpreted as a set of attributes. Any elements after that are rendered as children.

The number zero (representing a hole) is used to indicate where the content should be inserted. Let’s look at the rendering of the CodeBlock extension with two nested tags:

If you want to add some specific attributes there, import the mergeAttributes helper from @tiptap/core:

The parseHTML() function tries to load the editor document from HTML. The function gets the HTML DOM element passed as a parameter, and is expected to return an object with attributes and their values. Here is a simplified example from the Bold mark:

This defines a rule to convert all <strong> tags to Bold marks. But you can get more advanced with this, here is the full example from the extension:

This checks for <strong> and <b> tags, and any HTML tag with an inline style setting the font-weight to bold.

As you can see, you can optionally pass a getAttrs callback, to add more complex checks, for example for specific HTML attributes. The callback gets passed the HTML DOM node, except when checking for the style attribute, then it’s the value.

You are wondering what’s that && null doing? ProseMirror expects null or undefined if the check is successful.

Pass priority to a rule to resolve conflicts with other extensions, for example if you build a custom extension which looks for paragraphs with a class attribute, but you already use the default paragraph extension.

The getAttrs function you’ve probably noticed in the example has two purposes:

You can return an object with the attribute as the key and the parsed value to set your mark or node attribute. We would recommend to use the parseHTML inside addAttributes(), though. That will keep your code cleaner.

Read more about getAttrs and all other ParseRule properties in the ProseMirror reference.

**Examples:**

Example 1 (javascript):
```javascript
// 1. Import the extension
import BulletList from '@tiptap/extension-bullet-list'

// 2. Overwrite the keyboard shortcuts
const CustomBulletList = BulletList.extend({
  addKeyboardShortcuts() {
    return {
      'Mod-l': () => this.editor.commands.toggleBulletList(),
    }
  },
})

// 3. Add the custom extension to your editor
new Editor({
  extensions: [
    CustomBulletList,
    // …
  ],
})
```

Example 2 (javascript):
```javascript
// 1. Import the extension
import BulletList from '@tiptap/extension-bullet-list'

// 2. Overwrite the keyboard shortcuts
const CustomBulletList = BulletList.extend({
  addKeyboardShortcuts() {
    return {
      'Mod-l': () => this.editor.commands.toggleBulletList(),
    }
  },
})

// 3. Add the custom extension to your editor
new Editor({
  extensions: [
    CustomBulletList,
    // …
  ],
})
```

Example 3 (python):
```python
import Heading from '@tiptap/extension-heading'

const CustomHeading = Heading.extend({
  addOptions() {
    return {
      ...this.parent?.(),
      levels: [1, 2, 3],
    }
  },
})
```

Example 4 (python):
```python
import Heading from '@tiptap/extension-heading'

const CustomHeading = Heading.extend({
  addOptions() {
    return {
      ...this.parent?.(),
      levels: [1, 2, 3],
    }
  },
})
```

---

## Awareness in Collaboration

**URL:** https://tiptap.dev/docs/collaboration/core-concepts/awareness

**Contents:**
- Awareness in Collaboration
- Necessary provider events
- Integrate awareness
  - Set the awareness field
  - Listen for changes
  - Track mouse movement
- Add carets and selections
  - Was this page helpful?

Awareness in Tiptap Collaboration, powered by Yjs, is helping you share real-time info on users' activities within a collaborative space. This can include details like user presence, cursor positions, and custom user states.

At its core, awareness utilizes its own Conflict-Free Replicated Data Type (CRDT) to ensure that this shared meta-information remains consistent and immediate across all users, without maintaining a historical record of these states.

You can read more about Awareness in the Yjs documentation on awareness.

Awareness updates trigger specific provider events to develop interactive features based on user actions and presence:

These events serve as hooks for integrating custom Awareness features.

With your collaborative environment set up, you're all set to integrate Awareness, which is natively supported by the Collaboration Provider.

To kick things off, update the Awareness state with any relevant information. As an example we'll use a user's name, cursor color, and mouse position as examples.

Let's assign a name, color, and mouse position to the user. This is just an example; feel free to use any data relevant to your application.

Set up an event listener to track changes in the Awareness states across all connected users:

You can now view these updates in your browser's console as you move on to the next step.

Next, we'll add an event listener to our app to track mouse movements and update the awareness' information accordingly.

Check your browser's console to see the stream of events as users move their mice.

With basic Awareness in place, consider adding the Collaboration Caret extension to your editor. This extension adds caret positions, text selections, and personalized details (such as names and colors) of all participating users to your editor.

**Examples:**

Example 1 (yaml):
```yaml
// Set the awareness field for the current user
provider.setAwarenessField('user', {
  // Share any information you like
  name: 'Kevin James',
  color: '#ffcc00',
})
```

Example 2 (yaml):
```yaml
// Set the awareness field for the current user
provider.setAwarenessField('user', {
  // Share any information you like
  name: 'Kevin James',
  color: '#ffcc00',
})
```

Example 3 (javascript):
```javascript
// Listen for updates to the states of all users
provider.on('awarenessChange', ({ states }) => {
  console.log(states)
})
```

Example 4 (javascript):
```javascript
// Listen for updates to the states of all users
provider.on('awarenessChange', ({ states }) => {
  console.log(states)
})
```

---

## Background Color extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/background-color

**Contents:**
- Background Color extension
- Install
- Settings
  - types
- Commands
  - setBackgroundColor()
  - unsetBackgroundColor()
- Source code
- Minimal Install
  - Was this page helpful?

This extension enables you to set the background color in the editor. It uses the TextStyle mark, which renders a <span> tag (and only that). The background color is applied as inline style then, for example <span style="background-color: #958DF1">.

This extension requires the TextStyle mark.

This extension is installed by default with the TextStyleKit extension, so you don’t need to install it separately.

A list of marks to which the color attribute should be applied to.

Default: ['textStyle']

Applies the given background color as inline style.

Removes any background color.

packages/extension-text-style/src/background-color/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-style
```

Example 2 (python):
```python
npm install @tiptap/extension-text-style
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, BackgroundColor } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, BackgroundColor],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, BackgroundColor } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, BackgroundColor],
})
```

---

## Blockquote extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/blockquote

**Contents:**
- Blockquote extension
- Install
- Settings
  - HTMLAttributes
- Commands
  - setBlockquote()
  - toggleBlockquote()
  - unsetBlockquote()
- Keyboard shortcuts
- Source code

The Blockquote extension enables you to use the <blockquote> HTML tag in the editor. This is great to … show quotes in the editor, you know?

Type > at the beginning of a new line and it will magically transform to a blockquote.

Custom HTML attributes that should be added to the rendered HTML tag.

Wrap content in a blockquote.

Wrap or unwrap a blockquote.

packages/extension-blockquote/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-blockquote
```

Example 2 (python):
```python
npm install @tiptap/extension-blockquote
```

Example 3 (css):
```css
Blockquote.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Blockquote.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Bold extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/bold

**Contents:**
- Bold extension
  - Restrictions
- Install
- Settings
  - HTMLAttributes
- Commands
  - setBold()
  - toggleBold()
  - unsetBold()
- Keyboard shortcuts

Use this extension to render text in bold. If you pass <strong>, <b> tags, or text with inline style attributes setting the font-weight CSS rule in the editor’s initial content, they all will be rendered accordingly.

Type **two asterisks** or __two underlines__ and it will magically transform to bold text while you type.

The extension will generate the corresponding <strong> HTML tags when reading contents of the Editor instance. All text marked bold, regardless of the method will be normalized to <strong> HTML tags.

Custom HTML attributes that should be added to the rendered HTML tag.

Toggle the bold mark.

Remove the bold mark.

packages/extension-bold/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-bold
```

Example 2 (python):
```python
npm install @tiptap/extension-bold
```

Example 3 (css):
```css
Bold.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Bold.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## BubbleMenu extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/bubble-menu

**Contents:**
- BubbleMenu extension
- Install
- Settings
  - element
  - updateDelay
  - resizeDelay
  - options
  - pluginKey
  - shouldShow
  - appendTo

This extension will make a contextual menu appear near a selection of text. Use it to let users apply marks to their text selection.

As always, the markup and styling is totally up to you.

The DOM element that contains your menu.

In the React version of the Bubble Menu, access the DOM element with the ref prop of the BubbleMenu component, by passing a ref into it.

The BubbleMenu debounces the update method to allow the bubble menu to not be updated on every selection update. This can be controlled in milliseconds. The BubbleMenuPlugin will come with a default delay of 250ms. This can be deactivated, by setting the delay to 0 which deactivates the debounce.

The BubbleMenu debounces the resize calculation for the bubble menu to allow the bubble menu to not be updated on every resize event. This can be controlled in milliseconds.

Under the hood, the BubbleMenu Floating UI. You can control the middleware and positioning of the floating menu with these options.

Default: { strategy: 'absolute', placement: 'right' }

The key for the underlying ProseMirror plugin. Make sure to use different keys if you add more than one instance.

Type: string | PluginKey

Default: 'bubbleMenu'

A callback to control whether the menu should be shown or not.

Type: (props) => boolean

The element to which the bubble menu should be appended to in the DOM. Can be a HTMLElement or a callback function that returns a HTMLElement.

Type: HTMLElement | (() => HTMLElement) | undefined

Default: undefined, the menu will be appended to the editor's parent element (editor.view.dom.parentElement).

A callback to provide the anchor coordinates used to position the menu. Should return a virtual element as expected by Floating UI.

Type: () => VirtualElement | null

Default: null, anchor is implied by the editor selection.

packages/extension-bubble-menu/

Check out the demo at the top of this page to see how to integrate the bubble menu extension with React or Vue.

Customize the logic for showing the menu with the shouldShow option. For components, shouldShow can be passed as a prop.

Use multiple menus by setting an unique pluginKey.

Alternatively you can pass a ProseMirror PluginKey.

If the bubble menu changes size after the initial render, its position will not be adjusted automatically. To fix this, you can force update the position of the bubble menu by emitting an 'updatePosition' event.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-bubble-menu
```

Example 2 (python):
```python
npm install @tiptap/extension-bubble-menu
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import BubbleMenu from '@tiptap/extension-bubble-menu'

new Editor({
  extensions: [
    BubbleMenu.configure({
      element: document.querySelector('.menu'),
    }),
  ],
})
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import BubbleMenu from '@tiptap/extension-bubble-menu'

new Editor({
  extensions: [
    BubbleMenu.configure({
      element: document.querySelector('.menu'),
    }),
  ],
})
```

---

## Build AI features on the server

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/server-ai-toolkit

**Contents:**
- Build AI features on the server
  - More details
  - Was this page helpful?

Build AI agents with document-editing superpowers on the server without a browser-based editor. The Server AI Toolkit provides flexible primitives and pre-built tools that enable your AI to read, edit, and manipulate Tiptap documents server-side.

For detailed information on installation, configuration, and comprehensive guides, please visit our Server AI Toolkit feature page.

---

## Build AI features with document editing

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/ai-toolkit

**Contents:**
- Build AI features with document editing
  - More details
  - Was this page helpful?

Build AI agents with document-editing superpowers or add custom AI functionality to your Tiptap editor. The AI Toolkit provides flexible primitives and pre-built tools that enable your AI to read, edit, and manipulate Tiptap documents.

For detailed information on installation, configuration, and comprehensive guides, please visit our AI Toolkit feature page.

---

## BulletList extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/bullet-list

**Contents:**
- BulletList extension
  - Modify backspace behavior
- Install
- Usage
- Settings
  - HTMLAttributes
  - itemTypeName
  - keepMarks
  - keepAttributes
- Commands

This extension enables you to use bullet lists in the editor. They are rendered as <ul> HTML tags. Type * , - or + at the beginning of a new line and it will magically transform to a bullet list.

If you want to modify the standard behavior of backspace and delete functions for lists, you should read about the ListKeymap extension.

This extension requires the ListItem node.

This extension is installed by default with the ListKit extension, so you don’t need to install it separately.

Custom HTML attributes that should be added to the rendered HTML tag.

Specify the list item name.

Decides whether to keep the marks from a previous line after toggling the list either using inputRule or using the button

Decides whether to keep the attributes from a previous line after toggling the list either using inputRule or using the button

Toggles a bullet list.

packages/extension-list/src/bullet-list/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-list
```

Example 2 (python):
```python
npm install @tiptap/extension-list
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { BulletList } from '@tiptap/extension-list'

new Editor({
  extensions: [BulletList],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { BulletList } from '@tiptap/extension-list'

new Editor({
  extensions: [BulletList],
})
```

---

## CharacterCount extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/character-count

**Contents:**
- CharacterCount extension
- Install
- Usage
- Settings
  - limit
  - mode
  - textCounter
  - wordCounter
- Storage
  - characters()

The CharacterCount extension limits the number of allowed characters to a specific length and is able to return the number of characters and words. That’s it, that’s all.

The maximum number of characters that should be allowed.

The mode by which the size is calculated.

The text counter function to use. Defaults to a simple character count.

Default: (text) => text.length

The word counter function to use. Defaults to a simple word count.

Default: (text) => text.split(' ').filter((word) => word !== '').length

Get the number of characters for the current document.

Get the number of words for the current document.

packages/extensions/src/character-count/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { CharacterCount } from '@tiptap/extensions'

new Editor({
  extensions: [CharacterCount],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { CharacterCount } from '@tiptap/extensions'

new Editor({
  extensions: [CharacterCount],
})
```

---

## CodeBlockLowlight extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/code-block-lowlight

**Contents:**
- CodeBlockLowlight extension
  - Syntax highlight dependency
- Install
- Settings
  - lowlight
  - HTMLAttributes
  - enableTabIndentation
  - tabSize
  - languageClassPrefix
  - defaultLanguage

With the CodeBlockLowlight extension you can add fenced code blocks to your documents. It’ll wrap the code in <pre> and <code> HTML tags.

This extension relies on the lowlight library to apply syntax highlight to the code block’s content.

Type ``` (three backticks and a space) or ∼∼∼ (three tildes and a space) and a code block is instantly added for you. You can even specify the language, try writing ```css . That should add a language-css class to the <code>-tag.

You should provide the lowlight module to this extension. Decoupling the lowlight package from the extension allows the client application to control which version of lowlight it uses and which programming language packages it needs to load.

Custom HTML attributes that should be added to the rendered HTML tag.

Enable tab key for indentation in code blocks.

The number of spaces to use for tab indentation.

Adds a prefix to language classes that are applied to code tags.

Define a default language instead of the automatic detection of lowlight.

Wrap content in a code block.

Toggle the code block.

packages/extension-code-block-lowlight/

**Examples:**

Example 1 (python):
```python
npm install lowlight @tiptap/extension-code-block-lowlight
```

Example 2 (python):
```python
npm install lowlight @tiptap/extension-code-block-lowlight
```

Example 3 (sql):
```sql
import { lowlight } from 'lowlight/lib/core'

CodeBlockLowlight.configure({
  lowlight,
})
```

Example 4 (sql):
```sql
import { lowlight } from 'lowlight/lib/core'

CodeBlockLowlight.configure({
  lowlight,
})
```

---

## CodeBlock extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/code-block

**Contents:**
- CodeBlock extension
  - No syntax highlighting
- Install
- Usage
- Settings
  - languageClassPrefix
  - exitOnTripleEnter
  - defaultLanguage
  - exitOnArrowDown
  - HTMLAttributes

With the CodeBlock extension you can add fenced code blocks to your documents. It’ll wrap the code in <pre> and <code> HTML tags.

Type ``` (three backticks and a space) or ∼∼∼ (three tildes and a space) and a code block is instantly added for you. You can even specify the language, try writing ```css . That should add a language-css class to the <code>-tag.

The CodeBlock extension doesn’t come with styling and has no syntax highlighting built-in. Try the CodeBlockLowlight extension if you’re looking for code blocks with syntax highlighting.

Adds a prefix to language classes that are applied to code tags.

Define whether the node should be exited on triple enter.

Define a default language instead of the automatic detection of lowlight.

Define whether the node should be exited on arrow down if there is no node after it.

Custom HTML attributes that should be added to the rendered HTML tag.

Enable tab key for indentation in code blocks.

The number of spaces to use for tab indentation.

Wrap content in a code block.

Toggle the code block.

packages/extension-code-block/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-code-block
```

Example 2 (python):
```python
npm install @tiptap/extension-code-block
```

Example 3 (python):
```python
import CodeBlock from '@tiptap/extension-code-block'

const editor = new Editor({
  extensions: [CodeBlock],
})
```

Example 4 (python):
```python
import CodeBlock from '@tiptap/extension-code-block'

const editor = new Editor({
  extensions: [CodeBlock],
})
```

---

## Code extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/code

**Contents:**
- Code extension
- Install
- Settings
  - HTMLAttributes
- Commands
  - setCode()
  - toggleCode()
  - unsetCode()
- Keyboard shortcuts
- Source code

The Code extensions enables you to use the <code> HTML tag in the editor. If you paste in text with <code> tags it will rendered accordingly.

Type something with `back-ticks around` and it will magically transform to inline code while you type.

Custom HTML attributes that should be added to the rendered HTML tag.

Mark text as inline code.

Toggle inline code mark.

Remove inline code mark.

packages/extension-code/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-code
```

Example 2 (python):
```python
npm install @tiptap/extension-code
```

Example 3 (css):
```css
Code.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Code.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## CollaborationCaret extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/collaboration-caret

**Contents:**
- CollaborationCaret extension
  - Public Demo
- Install
- Settings
  - provider
  - user
  - render
  - selectionRender
- Commands
  - updateUser()

This extension adds information about all connected users (like their name and a specified color), their current caret position and their text selection (if there’s one).

It requires a collaborative Editor, so make sure to check out the Tiptap Collaboration Docs for a fully hosted or on-premises collaboration server solution.

The content of this editor is shared with other users.

Open this page in multiple browser windows to test it.

This extension requires the Collaboration extension.

A Y.js network provider, for example a Tiptap Collaboration instance.

Attributes of the current user, assumes to have a name and a color, but can be used with any attribute. The values are synced with all other connected clients.

Default: { user: null, color: null }

A render function for the caret, look at the extension source code for an example.

A render function for the selection, look at the extension source code for an example.

Pass an object with updated attributes of the current user. It expects a name and a color, but you can add additional fields, too.

packages/extension-collaboration-caret/

Fasten your seatbelts! Make your rich text editor collaborative with Tiptap Collaboration.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-collaboration-caret
```

Example 2 (python):
```python
npm install @tiptap/extension-collaboration-caret
```

Example 3 (css):
```css
editor.commands.updateUser({
  name: 'John Doe',
  color: '#000000',
  avatar: 'https://unavatar.io/github/ueberdosis',
})
```

Example 4 (css):
```css
editor.commands.updateUser({
  name: 'John Doe',
  color: '#000000',
  avatar: 'https://unavatar.io/github/ueberdosis',
})
```

---

## Color extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/color

**Contents:**
- Color extension
- Install
- Settings
  - types
- Commands
  - setColor()
  - unsetColor()
- Source code
- Minimal Install
  - Was this page helpful?

This extension enables you to set the font color in the editor. It uses the TextStyle mark, which renders a <span> tag (and only that). The font color is applied as inline style then, for example <span style="color: #958DF1">.

This extension requires the TextStyle mark.

This extension is installed by default with the TextStyleKit extension, so you don’t need to install it separately.

A list of marks to which the color attribute should be applied to.

Default: ['textStyle']

Applies the given font color as inline style.

Removes any font color.

packages/extension-text-style/src/color/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-style
```

Example 2 (python):
```python
npm install @tiptap/extension-text-style
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, Color } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, Color],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, Color } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, Color],
})
```

---

## Convert Markdown with Tiptap

**URL:** https://tiptap.dev/docs/conversion/import-export/markdown/editor-extensions

**Contents:**
- Convert Markdown with Tiptap
- Editor Markdown Import
  - Configure the extension in your editor
  - Import your first document
  - Customize the import behavior
  - Options
  - Commands
  - import arguments
- Editor Markdown Export
  - Install the Export extension:

Tiptap’s Conversion tools support handling Markdown (.md) files in three ways:

The Conversion extensions are published in Tiptap’s private npm registry. Integrate the extensions by following the private registry guide.

Install the Import extension:

This uploads the chosen .md file to the Conversion API, converts it into Tiptap JSON, and replaces the current editor content.

**Examples:**

Example 1 (python):
```python
npm i @tiptap-pro/extension-import
```

Example 2 (python):
```python
npm i @tiptap-pro/extension-import
```

Example 3 (sql):
```sql
import { Import } from '@tiptap-pro/extension-import'

const editor = new Editor({
  // ...
  extensions: [
    // ...
    Import.configure({
      // Your Convert App ID from https://cloud.tiptap.dev/convert-settings
      appId: 'your-app-id',

      // JWT token you generated
      token: 'your-jwt',

      // If your Markdown includes images, you can provide a URL for image upload
      imageUploadCallbackUrl: 'https://your-image-upload-url.com',
    }),
  ],
})
```

Example 4 (sql):
```sql
import { Import } from '@tiptap-pro/extension-import'

const editor = new Editor({
  // ...
  extensions: [
    // ...
    Import.configure({
      // Your Convert App ID from https://cloud.tiptap.dev/convert-settings
      appId: 'your-app-id',

      // JWT token you generated
      token: 'your-jwt',

      // If your Markdown includes images, you can provide a URL for image upload
      imageUploadCallbackUrl: 'https://your-image-upload-url.com',
    }),
  ],
})
```

---

## Create a custom mark view

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/mark-views

**Contents:**
- Create a custom mark view
  - Render JavaScript/Vue/React
  - Was this page helpful?

While node views are used to render nodes, mark views are used to render marks. They are a little bit simpler, because they don’t have to deal with the complexity of a node and come with a simpler API.

Mark views are amazing to improve the in-editor experience, but can also be used in a read-only instance of Tiptap. They are unrelated to the HTML output by design, so you have full control about the in-editor experience and the output.

But what if you want to render your actual JavaScript/Vue/React code? Use the Static Renderer. This utility lets you render your content as HTML, Markdown, or React components, without an Editor instance.

---

## Create a custom node view

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/node-views

**Contents:**
- Create a custom node view
- Different types of node views
  - Editable text
  - Non-editable text
  - Mixed content
- Markup
  - What if you store JSON?
  - Render HTML
  - Parse HTML
  - Render JavaScript/Vue/React

Node views are the best thing since sliced bread, at least if you are a fan of customization (and bread). With node views you can add interactive nodes to your editor. That can literally be everything. If you can write it in JavaScript, you can use it in your editor.

Node views are amazing to improve the in-editor experience, but can also be used in a read-only instance of Tiptap. They are unrelated to the HTML output by design, so you have full control about the in-editor experience and the output.

Depending on what you would like to build, node views work a little bit different and can have their very specific capabilities, but also pitfalls. The main question is: How should your custom node look like?

Yes, node views can have editable text, just like a regular node. That’s simple. The cursor will exactly behave like you would expect it from a regular node. Existing commands work very well with those nodes.

That’s how the TaskItem node works.

Nodes can also have text, which is not editable. The cursor can’t jump into those, but you don’t want that anyway.

Tiptap adds a contenteditable="false" to those by default.

That’s how you could render mentions, which shouldn’t be editable. Users can add or delete them, but not delete single characters.

Statamic uses those for their Bard editor, which renders complex modules inside Tiptap, which can have their own text inputs.

You can even mix non-editable and editable text. That’s great to build complex things, and still use marks like bold and italic inside the editable content.

BUT, if there are other elements with non-editable text in your node view, the cursor can jump there. You can improve that with manually adding contenteditable="false" to the specific parts of your node view.

But what happens if you access the editor content? If you’re working with HTML, you’ll need to tell Tiptap how your node should be serialized.

The editor does not export the rendered JavaScript node, and for a lot of use cases you wouldn’t want that anyway.

Let’s say you have a node view which lets users add a video player and configure the appearance (autoplay, controls, …). You want the interface to do that in the editor, not in the output of the editor. The output of the editor should probably only have the video player.

I know, I know, it’s not that easy. Just keep in mind, that you‘re in full control of the rendering inside the editor and of the output.

That doesn’t apply to JSON. In JSON, everything is stored as an object. There is no need to configure the “translation” to and from JSON.

Okay, you’ve set up your node with an interactive node view and now you want to control the output. Even if your node view is pretty complex, the rendered HTML can be simple:

Make sure it’s something distinguishable, so it’s easier to restore the content from the HTML. If you just need something generic markup like a <div> consider to add a data-type="my-custom-node".

The same applies to restoring the content. You can configure what markup you expect, that can be something completely unrelated to the node view markup. It just needs to contain all the information you want to restore.

Attributes are automagically restored, if you registered them through addAttributes.

But what if you want to render your actual JavaScript/Vue/React code? Use the Static Renderer. This utility lets you render your content as HTML, Markdown, or React components, without an Editor instance.

**Examples:**

Example 1 (jsx):
```jsx
<div class="Prosemirror" contenteditable="true">
  <p>text</p>
  <node-view>text</node-view>
  <p>text</p>
</div>
```

Example 2 (jsx):
```jsx
<div class="Prosemirror" contenteditable="true">
  <p>text</p>
  <node-view>text</node-view>
  <p>text</p>
</div>
```

Example 3 (jsx):
```jsx
<div class="Prosemirror" contenteditable="true">
  <p>text</p>
  <node-view contenteditable="false">text</node-view>
  <p>text</p>
</div>
```

Example 4 (jsx):
```jsx
<div class="Prosemirror" contenteditable="true">
  <p>text</p>
  <node-view contenteditable="false">text</node-view>
  <p>text</p>
</div>
```

---

## Create a new extension

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/create-new

**Contents:**
- Create a new extension
  - Create an extension
  - Create a node
  - Create a mark
- Publish standalone extensions
- Share
  - Was this page helpful?

You can build your own extensions from scratch and you know what? It’s the same syntax as for extending existing extension described above.

Extensions add new capabilities to Tiptap and you’ll read the word extension here very often, even for nodes and marks. But there are literal extensions. Those can’t add to the schema (like marks and nodes do), but can add functionality or change the behaviour of the editor.

A good example to learn from is probably TextAlign.

You can also use a callback function to create an extension. This is useful if you want to encapsulate the logic of your extension, for example when you want to define event handlers or other custom logic.

Read more about the Extension API.

If you think of the document as a tree, then nodes are just a type of content in that tree. Good examples to learn from are Paragraph, Heading, or CodeBlock.

You can also use a callback function to create a node. This is useful if you want to encapsulate the logic of your extension, for example when you want to define event handlers or other custom logic.

Nodes don’t have to be blocks. They can also be rendered inline with the text, for example for @mentions.

Read more about the Node API.

One or multiple marks can be applied to nodes, for example to add inline formatting. Good examples to learn from are Bold, Italic and Highlight.

You can also use a callback function to create a mark. This is useful if you want to encapsulate the logic of your extension, for example when you want to define event handlers or other custom logic.

Read more about the Mark API.

If you want to create and publish your own extensions for Tiptap, you can use our CLI tool to bootstrap your project. Simply run npm init tiptap-extension and follow the instructions. The CLI will create a new folder with a pre-configured project for you including a build script running on Rollup.

If you want to test your extension locally, you can run npm link in the project folder and then npm link YOUR_EXTENSION in your project (for example a Vite app).

When everything is working fine, don’t forget to share it with the community or in our awesome-tiptap repository.

**Examples:**

Example 1 (sql):
```sql
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create({
  name: 'customExtension',

  // Your code goes here.
})
```

Example 2 (sql):
```sql
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create({
  name: 'customExtension',

  // Your code goes here.
})
```

Example 3 (javascript):
```javascript
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create(() => {
  // Define variables or functions to use inside your extension
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customExtension',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

Example 4 (javascript):
```javascript
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create(() => {
  // Define variables or functions to use inside your extension
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customExtension',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

---

## DetailsContent extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/details-content

**Contents:**
- DetailsContent extension
- Install
- Usage
- Settings
  - HTMLAttributes
  - Was this page helpful?

The Details extension enables you to use the <details> HTML tag in the editor. This is great to show and hide content.

Custom HTML attributes that should be added to the rendered HTML tag.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-details
```

Example 2 (python):
```python
npm install @tiptap/extension-details
```

Example 3 (sql):
```sql
import { Details, DetailsSummary, DetailsContent } from '@tiptap/extension-details'

const editor = new Editor({
  extensions: [Details, DetailsSummary, DetailsContent],
})
```

Example 4 (sql):
```sql
import { Details, DetailsSummary, DetailsContent } from '@tiptap/extension-details'

const editor = new Editor({
  extensions: [Details, DetailsSummary, DetailsContent],
})
```

---

## DetailsSummary extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/details-summary

**Contents:**
- DetailsSummary extension
- Install
- Usage
- Settings
  - HTMLAttributes
  - Was this page helpful?

The Details extension enables you to use the <details> HTML tag in the editor. This is great to show and hide content.

Custom HTML attributes that should be added to the rendered HTML tag.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-details
```

Example 2 (python):
```python
npm install @tiptap/extension-details
```

Example 3 (sql):
```sql
import { Details, DetailsSummary, DetailsContent } from '@tiptap/extension-details'

const editor = new Editor({
  extensions: [Details, DetailsSummary, DetailsContent],
})
```

Example 4 (sql):
```sql
import { Details, DetailsSummary, DetailsContent } from '@tiptap/extension-details'

const editor = new Editor({
  extensions: [Details, DetailsSummary, DetailsContent],
})
```

---

## Details extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/details

**Contents:**
- Details extension
- Install
- Settings
  - persist
  - openClassName
  - HTMLAttributes
- Commands
  - setDetails()
  - unsetDetails()
  - Was this page helpful?

The Details extension enables you to use the <details> HTML tag in the editor. This is great to show and hide content.

This extension requires the DetailsSummary and DetailsContent node.

Specify if the open status should be saved in the document. Defaults to false.

Specifies a CSS class that is set when toggling the content. Defaults to is-open.

Custom HTML attributes that should be added to the rendered HTML tag.

Wrap content in a details node.

Unwrap a details node.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-details
```

Example 2 (python):
```python
npm install @tiptap/extension-details
```

Example 3 (css):
```css
Details.configure({
  persist: true,
})
```

Example 4 (css):
```css
Details.configure({
  persist: true,
})
```

---

## Document extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/document

**Contents:**
- Document extension
  - Breaking Change
- Install
- Source code
  - Was this page helpful?

The Document extension is required, no matter what you build with Tiptap. It’s a so called “topNode”, a node that’s the home to all other nodes. Think of it like the <body> tag for your document.

The node is very tiny though. It defines a name of the node (doc), is configured to be a top node (topNode: true) and that it can contain multiple other nodes (block+). That’s all. But have a look yourself:

Tiptap v1 tried to hide that node from you, but it has always been there. You have to explicitly import it from now on (or use StarterKit).

packages/extension-document/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-document
```

Example 2 (python):
```python
npm install @tiptap/extension-document
```

---

## Drag Handle extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/drag-handle

**Contents:**
- Drag Handle extension
- Install
- Settings
  - render
  - computePositionConfig
  - getReferencedVirtualElement
  - locked
  - onNodeChange
- Commands
  - lockDragHandle()

Have you ever wanted to drag nodes around your editor? Well, we did too—so here’s an extension for that.

The DragHandle extension allows you to easily handle dragging nodes around in the editor. You can define custom render functions, placement, and more.

Renders an element that is positioned with the floating-ui/dom package. This is the element that will be displayed as the handle when dragging a node around.

Configuration for position computation of the drag handle using the floating-ui/dom package. You can pass any options that are available in the floating-ui documentation.

Default: { placement: 'left-start', strategy: 'absolute' }

A function that returns the virtual element for the drag handle. This is useful when the menu needs to be positioned relative to a specific DOM element.

Locks the draghandle in place and visibility. If the drag handle was visible, it will remain visible until unlocked. If it was hidden, it will remain hidden until unlocked.

Returns a node or null when a node is hovered over. This can be used to highlight the node that is currently hovered over.

Locks the draghandle in place and visibility. If the drag handle was visible, it will remain visible until unlocked. If it was hidden, it will remain hidden until unlocked.

This can be useful if you want to have a menu inside of the drag handle and want it to remain visible whether the drag handle is moused over or not.

Unlocks the draghandle. Resets to default visibility and behavior.

Toggle draghandle lock state. If the drag handle is locked, it will be unlocked and vice versa.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-drag-handle
```

Example 2 (python):
```python
npm install @tiptap/extension-drag-handle
```

Example 3 (typescript):
```typescript
DragHandle.configure({
  render: () => {
    const element = document.createElement('div')

    // Use as a hook for CSS to insert an icon
    element.classList.add('custom-drag-handle')

    return element
  },
})
```

Example 4 (typescript):
```typescript
DragHandle.configure({
  render: () => {
    const element = document.createElement('div')

    // Use as a hook for CSS to insert an icon
    element.classList.add('custom-drag-handle')

    return element
  },
})
```

---

## Drag Handle React extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/drag-handle-react

**Contents:**
- Drag Handle React extension
- Install
- Props
  - children
  - computePositionConfig
  - onNodeChange
  - getReferencedVirtualElement
  - locked
  - pluginKey
  - onElementDragStart

Have you ever wanted to drag nodes around your react-based editor? Well, we did too—so here’s an extension for that.

The DragHandleReact component allows you to easily handle dragging nodes around in the editor. You can define custom render functions, placement, and more. It essentially wraps the DragHandle extension in a React component that will automatically register/unregister the extension with the editor.

All props follow the same structure as the DragHandle extension.

The content that should be displayed inside of the drag handle.

Configuration for position computation of the drag handle using the floating-ui/dom package. You can pass any options that are available in the floating-ui documentation.

Default: { placement: 'left-start', strategy: 'absolute' }

Returns a node or null when a node is hovered over. This can be used to highlight the node that is currently hovered over.

A function that returns the virtual element for the drag handle. This is useful when the menu needs to be positioned relative to a specific DOM element.

Locks the draghandle in place and visibility. If the drag handle was visible, it will remain visible until unlocked. If it was hidden, it will remain hidden until unlocked.

The key that should be used to store the plugin in the editor. This is useful if you have multiple drag handles in the same editor.

A function that is called when the element starts to be dragged. This can be used to add custom logic when dragging starts.

A function that is called when the element stops being dragged. This can be used to add custom logic when dragging ends.

See the DragHandle extension for available editor commands.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-drag-handle-react @tiptap/extension-drag-handle @tiptap/extension-node-range @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-protocols
```

Example 2 (python):
```python
npm install @tiptap/extension-drag-handle-react @tiptap/extension-drag-handle @tiptap/extension-node-range @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-protocols
```

Example 3 (typescript):
```typescript
<DragHandle>
  <div>Drag Me!</div>
</DragHandle>
```

Example 4 (typescript):
```typescript
<DragHandle>
  <div>Drag Me!</div>
</DragHandle>
```

---

## Drag Handle vue extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/drag-handle-vue

**Contents:**
- Drag Handle vue extension
- Install
  - Vue 2 vs. Vue 3
- Props
  - children
  - computePositionConfig
  - onNodeChange
  - getReferencedVirtualElement
  - locked
  - pluginKey

Have you ever wanted to drag nodes around your vue-based editor? Well, we did too—so here’s an extension for that.

The DragHandleVue component allows you to easily handle dragging nodes around in the editor. You can define custom render functions, placement, and more. It essentially wraps the DragHandle extension in a vue component that will automatically register/unregister the extension with the editor.

There are two versions of the DragHandle extension available. Make sure to install the correct version for your Vue version. @tiptap/extension-drag-handle-vue-2 and @tiptap/extension-drag-handle-vue-3

All props follow the same structure as the DragHandle extension.

The content that should be displayed inside of the drag handle.

Configuration for position computation of the drag handle using the floating-ui/dom package. You can pass any options that are available in the floating-ui documentation.

Default: { placement: 'left-start', strategy: 'absolute' }

Returns a node or null when a node is hovered over. This can be used to highlight the node that is currently hovered over.

A function that returns the virtual element for the drag handle. This is useful when the menu needs to be positioned relative to a specific DOM element.

Locks the draghandle in place and visibility. If the drag handle was visible, it will remain visible until unlocked. If it was hidden, it will remain hidden until unlocked.

The key that should be used to store the plugin in the editor. This is useful if you have multiple drag handles in the same editor.

See the DragHandle extension for available editor commands.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-drag-handle-vue-3 @tiptap/extension-drag-handle @tiptap/extension-node-range @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-protocols
```

Example 2 (python):
```python
npm install @tiptap/extension-drag-handle-vue-3 @tiptap/extension-drag-handle @tiptap/extension-node-range @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-protocols
```

Example 3 (typescript):
```typescript
<drag-handle>
  <div>Drag Me!</div>
</drag-handle>
```

Example 4 (typescript):
```typescript
<drag-handle>
  <div>Drag Me!</div>
</drag-handle>
```

---

## Dropcursor extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/dropcursor

**Contents:**
- Dropcursor extension
- Install
- Usage
- Settings
  - color
  - width
  - class
- Source code
- Minimal Install
  - Was this page helpful?

This extension loads the ProseMirror Dropcursor plugin by Marijn Haverbeke, which shows a cursor at the drop position when something is dragged into the editor.

Note that Tiptap is headless, but the dropcursor needs CSS for its appearance. There are settings for the color and width, and you’re free to add a custom CSS class.

Color of the dropcursor.

Default: 'currentColor'

Width of the dropcursor.

One or multiple CSS classes that should be applied to the dropcursor.

packages/extensions/src/drop-cursor/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Dropcursor } from '@tiptap/extensions'

new Editor({
  extensions: [Dropcursor],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Dropcursor } from '@tiptap/extensions'

new Editor({
  extensions: [Dropcursor],
})
```

---

## Emoji extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/emoji

**Contents:**
- Emoji extension
- Install
- Dependencies
- Settings
  - HTMLAttributes
  - emojis
  - Skin tones
  - enableEmoticons
  - forceFallbackImages
    - Add custom emojis

The Emoji extension renders emojis as an inline node. All inserted (typed, pasted, etc.) emojis will be converted to this node. The benefit of this is that unsupported emojis can be rendered with a fallback image. As soon as you copy text out of the editor, they will be converted back to plain text.

To place the popups correctly, we’re using tippy.js in all our examples. You are free to bring your own library, but if you’re fine with it, just install what we use:

Custom HTML attributes that should be added to the rendered HTML tag.

Define a set of emojis. Tiptap provides two lists of emojis:

Skin tones are not yet supported ✌🏽

Specifies whether text should be converted to emoticons (e.g. <3 to ❤️). Defaults to false.

Specifies whether fallback images should always be rendered. Defaults to false.

It’s super easy to add your own custom emojis.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-emoji
```

Example 2 (python):
```python
npm install @tiptap/extension-emoji
```

Example 3 (unknown):
```unknown
npm install tippy.js
```

Example 4 (unknown):
```unknown
npm install tippy.js
```

---

## Extensions in Tiptap

**URL:** https://tiptap.dev/docs/editor/core-concepts/extensions

**Contents:**
- Extensions in Tiptap
- What are extensions?
- Create a new extension
  - Was this page helpful?

Extensions enhance Tiptap by adding new capabilities or modifying the editor's behavior. Whether it is adding new types of content, customizing the editor's appearance, or extending its functionality, extensions are the building blocks of Tiptap.

To add new types of content into your editor you can use nodes and marks which can render content in the editor.

The optional @tiptap/starter-kit includes the most commonly used extensions, simplifying setup. Read more about StarterKit.

Expand your editor's functionality with extensions created by the Tiptap community. Discover a variety of custom features and tools in the Awesome Tiptap Repository. For collaboration and support, engage with other developers in the Discussion Thread on community-built extensions.

Although Tiptap tries to hide most of the complexity of ProseMirror, it’s built on top of its APIs and we recommend you to read through the ProseMirror Guide for advanced usage. You’ll have a better understanding of how everything works under the hood and get more familiar with many terms and jargon used by Tiptap.

Existing nodes, marks and functionality can give you a good impression on how to approach your own extensions. To make it easier to switch between the documentation and the source code, we linked to the file on GitHub from every single extension documentation page.

We recommend to start with customizing existing extensions first, and create your own extensions with the gained knowledge later. That’s why all the examples below extend existing extensions, but all examples will work on newly created extensions as well.

You’re free to create your own extensions for Tiptap. Here is the boilerplate code that’s needed to create and register your own extension:

You can easily bootstrap a new extension via our CLI.

Learn more about custom extensions in our guide.

**Examples:**

Example 1 (sql):
```sql
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create({
  // Your code here
})

const editor = new Editor({
  extensions: [
    // Register your custom extension with the editor.
    CustomExtension,
    // … and don’t forget all other extensions.
    Document,
    Paragraph,
    Text,
    // …
  ],
})
```

Example 2 (sql):
```sql
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create({
  // Your code here
})

const editor = new Editor({
  extensions: [
    // Register your custom extension with the editor.
    CustomExtension,
    // … and don’t forget all other extensions.
    Document,
    Paragraph,
    Text,
    // …
  ],
})
```

Example 3 (unknown):
```unknown
npm init tiptap-extension
```

Example 4 (unknown):
```unknown
npm init tiptap-extension
```

---

## Extension API

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/create-new/extension

**Contents:**
- Extension API
- Creating an extension
- Extension options
  - name
  - priority
  - addOptions
  - addStorage
  - Notice
  - addCommands
  - Use the commands parameter inside of addCommands

The power of Tiptap lies in its flexibility. You can create your own extensions from scratch and build a unique editor experience tailored to your needs.

The base extension structure is the same for all extensions, whether you're creating a node, a mark, or a functionality change. And, everything in Tiptap is based on extensions.

Extensions add new capabilities to Tiptap. You'll read the word "extension" a lot in the docs, even for nodes and marks. But there are literal extensions, too. These can't add to the schema (like marks and nodes do), but they can add functionality or change the behavior of the editor.

A good example would be something that listens to the editor's events and does something with them. Like this:

You can also use a callback function to create an extension. This is useful if you want to encapsulate the logic of your extension, for example when you want to define event handlers or other custom logic.

This extension listens to the editor's update event and logs the editor's current JSON representation to the console.

It is installed to the editor like this:

This extensions array can contain any number of extensions. They will be installed in the order they are listed, or sorted by their priority property.

Now that we've seen the basic structure of an extension, let's dive into all of the extension options you can use to create your own extensions.

When creating an extension, you can define a set of options that can be configured by the user. These options can be used to customize the behavior of the extension, or to provide additional functionality.

The name of the extension. This is used to identify the extension in the editor's extension manager.

If the extension is a node or a mark, the name is used to identify the node or mark in the editor's schema and therefore persisted to the JSON representation of the editor's content. See store your content as JSON for more information.

The priority defines the order in which extensions are registered. The default priority is 100, that’s what most extension have. Extensions with a higher priority will be loaded earlier.

The order in which extensions are loaded influences two things:

The Link mark for example has a higher priority, which means it will be rendered as <a href="…"><strong>Example</strong></a> instead of <strong><a href="…">Example</a></strong>.

The addOptions method is used to define the extension's options. This method should return an object with the options that can be configured by the user.

This exposes configuration which can be set when installing the extension:

The addStorage method is used to define the extension's storage (essentially a simple state manager). This method should return an object with the storage that can be used by the extension.

This exposes storage which can be accessed within the extension:

The editor.storage is namespaced by the extension's name.

The addCommands method is used to define the extension's commands. This method should return an object with the commands that can be executed by the user.

To access other commands inside addCommands use the commands parameter that’s passed to it.

This exposes commands which can be executed by the user:

The addKeyboardShortcuts method is used to define keyboard shortcuts. This method should return an object with the keyboard shortcuts that can be used by the user.

This exposes keyboard shortcuts which can be used by the user.

With input rules you can define regular expressions to listen for user inputs. They are used for markdown shortcuts, or for example to convert text like (c) to a © (and many more) with the Typography extension. Use the markInputRule helper function for marks, and the nodeInputRule for nodes.

By default text between two tildes on both sides is transformed to striked text. If you want to think one tilde on each side is enough, you can overwrite the input rule like this:

Now, when you type ~striked text~, it will be transformed to striked text.

Want to learn more about input rules? Check out the Input Rules documentation.

Paste rules work like input rules (see above) do. But instead of listening to what the user types, they are applied to pasted content.

There is one tiny difference in the regular expression. Input rules typically end with a $ dollar sign (which means “asserts position at the end of a line”), paste rules typically look through all the content and don’t have said $ dollar sign.

Taking the example from above and applying it to the paste rule would look like the following example.

Want to learn more about paste rules? Check out the Paste Rules documentation.

You can even move your event listeners to a separate extension. Here is an example with listeners for all events:

You can add ProseMirror plugins to your extension. This is useful if you want to extend the editor with ProseMirror plugins.

You can wrap existing ProseMirror plugins in Tiptap extensions like shown in the example below.

You can also create custom ProseMirror plugins. Here is an example of a custom ProseMirror plugin that logs a message to the console.

To learn more about ProseMirror plugins, check out the ProseMirror documentation.

You can add more extensions to your extension. This is useful if you want to create a bundle of extensions that belong together.

You can extend the editor's NodeConfig with the extendNodeSchema method. This is useful if you want to add additional attributes to the node schema.

You can extend the editor's MarkConfig with the extendMarkSchema method. This is useful if you want to add additional attributes to the mark schema.

Those extensions aren’t classes, but you still have a few important things available in this everywhere in the extension.

**Examples:**

Example 1 (javascript):
```javascript
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create({
  name: 'customExtension',

  onUpdate() {
    console.log(this.editor.getJSON())
  },
})
```

Example 2 (javascript):
```javascript
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create({
  name: 'customExtension',

  onUpdate() {
    console.log(this.editor.getJSON())
  },
})
```

Example 3 (javascript):
```javascript
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create(() => {
  // Define variables or functions to use inside your extension
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customExtension',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

Example 4 (javascript):
```javascript
import { Extension } from '@tiptap/core'

const CustomExtension = Extension.create(() => {
  // Define variables or functions to use inside your extension
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customExtension',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

---

## FileHandler extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/filehandler

**Contents:**
- FileHandler extension
  - No Server Upload Functionality
- Install
- Settings
  - onPaste
  - onDrop
  - allowedMimeTypes
  - Was this page helpful?

Have you ever wanted to drag and drop or paste files into your editor? Well, we did too—so here’s an extension for that.

The FileHandler extension allows you to easily handle file drops and pastes in the editor. You can define custom handlers for both events & manage allowed file types.

By default, the extension does not display the uploaded file when it is pasted or dropped. Instead, it triggers an event that you can respond to by inserting a new Node into the editor. For example, to display the uploaded image file, use the image extension.

This extension is only responsible for handling the event of dropping or pasting a file into the editor. It does not implement server file uploads.

The callback function that will be called when a file is pasted into the editor. You will have access to the editor instance & the files pasted.

The callback function that will be called when a file is dropped into the editor. You will have access to the editor instance, the files dropped and the position the file was dropped at.

This option controls which file types are allowed to be dropped or pasted into the editor. You can define a list of mime types or a list of file extensions. If no mime types or file extensions are defined, all files will be allowed.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-file-handler
```

Example 2 (python):
```python
npm install @tiptap/extension-file-handler
```

Example 3 (typescript):
```typescript
FileHandler.configure({
  onPaste: (editor, files, htmlContent) => {
    // do something with the files
    // and insert the file into the editor
    // in some cases (for example copy / pasted gifs from other apps) you should probably not use the file directly
    // as the file parser will only have a single gif frame as png
    // in this case, you can extract the url from the htmlContent and use it instead, let other inputRules handle insertion
    // or do anything else with the htmlContent pasted into here
  },
})
```

Example 4 (typescript):
```typescript
FileHandler.configure({
  onPaste: (editor, files, htmlContent) => {
    // do something with the files
    // and insert the file into the editor
    // in some cases (for example copy / pasted gifs from other apps) you should probably not use the file directly
    // as the file parser will only have a single gif frame as png
    // in this case, you can extract the url from the htmlContent and use it instead, let other inputRules handle insertion
    // or do anything else with the htmlContent pasted into here
  },
})
```

---

## FloatingMenu extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/floatingmenu

**Contents:**
- FloatingMenu extension
- Install the extension
- Settings
  - element
  - appendTo
  - options
  - pluginKey
  - shouldShow
- Source code
- Use in Vanilla JavaScript

Use the Floating Menu extension in Tiptap to make a menu appear on an empty line.

Install the Floating Menu extension and the Floating UI library.

The DOM element that contains your menu.

The element to which the bubble menu should be appended to in the DOM. Can be a HTMLElement or a callback function that returns a HTMLElement.

Type: HTMLElement | (() => HTMLElement) | undefined

Default: undefined, the menu will be appended to document.body.

Under the hood, the FloatingMenu uses Floating UI. You can control the middleware and positioning of the floating menu with these options.

Default: { strategy: 'absolute', placement: 'right' }

The key for the underlying ProseMirror plugin. Make sure to use different keys if you add more than one instance.

Type: string | PluginKey

Default: 'floatingMenu'

A callback to control whether the menu should be shown or not.

Type: (props) => boolean

packages/extension-floating-menu/

Check out the demo at the top of this page to see how to integrate the floating menu extension with React or Vue.

Customize the logic for showing the menu with the shouldShow option. For components, shouldShow can be passed as a prop.

Use multiple menus by setting an unique pluginKey.

Alternatively you can pass a ProseMirror PluginKey.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-floating-menu @floating-ui/dom@^1.6.0
```

Example 2 (python):
```python
npm install @tiptap/extension-floating-menu @floating-ui/dom@^1.6.0
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import FloatingMenu from '@tiptap/extension-floating-menu'

new Editor({
  extensions: [
    FloatingMenu.configure({
      element: document.querySelector('.menu'),
    }),
  ],
})
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import FloatingMenu from '@tiptap/extension-floating-menu'

new Editor({
  extensions: [
    FloatingMenu.configure({
      element: document.querySelector('.menu'),
    }),
  ],
})
```

---

## Focus extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/focus

**Contents:**
- Focus extension
- Install
- Settings
  - className
  - mode
- Source code
  - Was this page helpful?

The Focus extension adds a CSS class to focused nodes. By default it adds .has-focus, but you can change that.

Note that it’s only a class, the styling is totally up to you. The usage example below has some CSS for that class.

And import it in your editor:

The class that is applied to the focused element.

Apply the class to 'all', the 'shallowest' or the 'deepest' node.

packages/extensions/focus/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Focus } from '@tiptap/extensions'

new Editor({
  extensions: [Focus],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Focus } from '@tiptap/extensions'

new Editor({
  extensions: [Focus],
})
```

---

## FontFamily extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/fontfamily

**Contents:**
- FontFamily extension
  - Heads-up!
- Install
- Settings
  - types
- Commands
  - setFontFamily()
  - unsetFontFamily()
- Source code
- Minimal Install

This extension enables you to set the font family in the editor. It uses the TextStyle mark, which renders a <span> tag. The font family is applied as inline style, for example <span style="font-family: Arial">.

Be aware that editor.isActive('textStyle', { fontFamily: 'Font Family' }) will return the font family as set by the browser's CSS rules and not as you would have expected when setting the font family.

This extension requires the TextStyle mark.

This extension is installed by default with the TextStyleKit extension, so you don’t need to install it separately.

A list of marks to which the font family attribute should be applied to.

Default: ['textStyle']

Applies the given font family as inline style.

Removes any font family.

packages/extension-text-style/src/font-family/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-style
```

Example 2 (python):
```python
npm install @tiptap/extension-text-style
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, FontFamily } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, FontFamily],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, FontFamily } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, FontFamily],
})
```

---

## FontSize extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/fontsize

**Contents:**
- FontSize extension
- Install
- Settings
  - types
- Commands
  - setFontSize()
  - unsetFontSize()
- Source code
- Minimal Install
  - Was this page helpful?

This extension enables you to set the font size in the editor. It uses the TextStyle mark, which renders a <span> tag. The font size is applied as inline style, for example <span style="font-size: 14px">.

This extension requires the TextStyle mark.

This extension is installed by default with the TextStyleKit extension, so you don’t need to install it separately.

A list of marks to which the font family attribute should be applied to.

Default: ['textStyle']

Applies the given font family as inline style.

Removes any font family.

packages/extension-text-style/src/font-size/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-style
```

Example 2 (python):
```python
npm install @tiptap/extension-text-style
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, FontSize } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, FontSize],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, FontSize } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, FontSize],
})
```

---

## Functionality extensions

**URL:** https://tiptap.dev/docs/editor/extensions/functionality

**Contents:**
- Functionality extensions
  - Was this page helpful?

Extensions do not always render content, but can also provide additional functionality to the editor. This includes tools for collaboration, text editing, and more.

---

## Gapcursor extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/gapcursor

**Contents:**
- Gapcursor extension
- Install
- Usage
- Source code
- Minimal Install
  - Was this page helpful?

This extension loads the ProseMirror Gapcursor plugin by Marijn Haverbeke, which adds a gap for the cursor in places that don’t allow regular selection. For example, after a table at the end of a document.

Note that Tiptap is headless, but the gapcursor needs CSS for its appearance. The default CSS is loaded through the Editor class.

packages/extensions/src/gap-cursor/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Gapcursor } from '@tiptap/extensions'

new Editor({
  extensions: [Gapcursor],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Gapcursor } from '@tiptap/extensions'

new Editor({
  extensions: [Gapcursor],
})
```

---

## HardBreak extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/hard-break

**Contents:**
- HardBreak extension
- Install
- Settings
  - HTMLAttributes
  - keepMarks
- Commands
  - setHardBreak()
- Keyboard shortcuts
- Source code
  - Was this page helpful?

The HardBreak extensions adds support for the <br> HTML tag, which forces a line break.

Custom HTML attributes that should be added to the rendered HTML tag.

Decides whether to keep marks after a line break. Based on the keepOnSplit option for marks.

packages/extension-hard-break/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-hard-break
```

Example 2 (python):
```python
npm install @tiptap/extension-hard-break
```

Example 3 (css):
```css
HardBreak.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
HardBreak.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Heading extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/heading

**Contents:**
- Heading extension
- Install
- Settings
  - HTMLAttributes
  - levels
- Commands
  - setHeading()
  - toggleHeading()
- Keyboard shortcuts
- Source code

The Heading extension adds support for headings of different levels. Headings are rendered with <h1>, <h2>, <h3>, <h4>, <h5> or <h6> HTML tags. By default all six heading levels (or styles) are enabled, but you can pass an array to only allow a few levels. Check the usage example to see how this is done.

Type # at the beginning of a new line and it will magically transform to a heading, same for ## , ### , #### , ##### and ###### .

Custom HTML attributes that should be added to the rendered HTML tag.

Specifies which heading levels are supported.

Default: [1, 2, 3, 4, 5, 6]

Creates a heading node with the specified level.

Toggles a heading node with the specified level.

packages/extension-heading/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-heading
```

Example 2 (python):
```python
npm install @tiptap/extension-heading
```

Example 3 (css):
```css
Heading.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Heading.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Highlight extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/highlight

**Contents:**
- Highlight extension
- Install
- Settings
  - HTMLAttributes
  - multicolor
- Commands
  - setHighlight()
  - toggleHighlight()
  - unsetHighlight()
- Keyboard shortcuts

Use this extension to render highlighted text with <mark>. You can use only default <mark> HTML tag, which has a yellow background color by default, or apply different colors.

Type ==two equal signs== and it will magically transform to highlighted text while you type.

Custom HTML attributes that should be added to the rendered HTML tag.

Add support for multiple colors.

Mark text as highlighted.

Toggle a text highlight.

Removes the highlight.

packages/extension-highlight/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-highlight
```

Example 2 (python):
```python
npm install @tiptap/extension-highlight
```

Example 3 (css):
```css
Highlight.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Highlight.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## HorizontalRule extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/horizontal-rule

**Contents:**
- HorizontalRule extension
- Install
- Settings
  - HTMLAttributes
- Commands
  - setHorizontalRule()
- Source code
  - Was this page helpful?

Use this extension to render a <hr> HTML tag. If you pass <hr> in the editor’s initial content, it’ll be rendered accordingly.

Type three dashes (---) or three underscores and a space (___ ) at the beginning of a new line and it will magically transform to a horizontal rule.

Custom HTML attributes that should be added to the rendered HTML tag.

Create a horizontal rule.

packages/extension-horizontal-rule/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-horizontal-rule
```

Example 2 (python):
```python
npm install @tiptap/extension-horizontal-rule
```

Example 3 (css):
```css
HorizontalRule.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
HorizontalRule.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## How to develop a custom extension

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions

**Contents:**
- How to develop a custom extension
  - Customize and create extensions
  - Customize and add to an existing extension
  - Create a new extensions from scratch
  - Create custom and interactive nodes
  - Learn from custom node view examples
  - Was this page helpful?

One of the strengths of Tiptap is its extendability. You don’t depend on the provided extensions, it is intended to extend the editor to your liking.

With custom extensions you can add new content types and new functionalities, on top of what already exists or from scratch. Let’s start with a few common examples of how you can extend existing nodes, marks and extensions.

You’ll learn how you start from scratch in the Create new page, but you’ll need the same knowledge for extending existing and creating new extensions.

---

## Image extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/image

**Contents:**
- Image extension
  - No Server Functionality
- Install
- Settings
  - inline
  - resize
  - allowBase64
  - HTMLAttributes
- Commands
  - setImage()

Use this extension to render <img> HTML tags. By default, those images are blocks. If you want to render images in line with text set the inline option to true.

This extension is only responsible for displaying images. It doesn’t upload images to your server, for that you can integrate the FileHandler extension

Renders the image node inline, for example in a paragraph tag: <p><img src="spacer.gif"></p>. By default images are on the same level as paragraphs.

It totally depends on what kind of editing experience you’d like to have, but can be useful if you (for example) migrate from Quill to Tiptap.

Options for resizable images. If defined the node will be wrapped in a resizable node view making it possible to resize the image via resize handles.

Allow images to be parsed as base64 strings <img src="data:image/jpg;base64...">.

Custom HTML attributes that should be added to the rendered HTML tag.

Makes the current node an image.

packages/extension-image/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-image
```

Example 2 (python):
```python
npm install @tiptap/extension-image
```

Example 3 (css):
```css
Image.configure({
  inline: true,
})
```

Example 4 (css):
```css
Image.configure({
  inline: true,
})
```

---

## Import and export custom nodes with .docx

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/custom-node-conversion

**Contents:**
- Import and export custom nodes with .docx
  - Custom node conventions
- Export custom nodes to .docx
- Import custom nodes from .docx
  - DOCX, "prosemirrorNodes" and "prosemirrorMarks"
  - Was this page helpful?

One of the biggest advantages of the @tiptap-pro/extension-export-docx and @tiptap-pro/extension-import-docx extensions is the ability to define how custom nodes in your Tiptap schema should be rendered in DOCX.

This allows you to preserve application-specific content in the exported Word file.

Custom node converters must adhere to the underlying DOCX generation library’s requirements. In practice, a custom converter function for DOCX should return one of the allowed DOCX elements for that node: a Paragraph class (or an array of Paragraph classes), a Table class, or null if the node should be skipped in the output.

When calling editor.exportDocx(), you can pass an array of custom node definitions in the ExportDocxOptions argument. Each definition specifies the node type and a render function.

For the sake of the example, suppose your editor has a custom node type hintbox (a callout-styled box). You can define how it should appear in DOCX.

Here's how the Hintbox extension's custom node might look like:

And we will define how the Hintbox custom node should be rendered in the DOCX:

Then, at a later point in your application, you can export the editor content to a DOCX file:

You can construct any supported DOCX elements in the render function using the Docx library classes (Paragraph, TextRun, Table, etc.) that are provided via the Docx import from the @tiptap-pro/extension-export-docx package.

When importing a DOCX file, you can also define how custom nodes should be converted back to Tiptap nodes. This is done by passing an array of custom node definitions to the import command.

The latest version of the @tiptap-pro/extension-import-docx has available the prosemirrorNodes configuration option. This option allows you to map custom nodes from the DOCX to your Tiptap schema. In the example above, we are mapping the Hintbox custom node from the DOCX to the hintbox custom node in our Tiptap schema. By doing so, whenever the DOCX contains a Hintbox custom node, it will be converted to a hintbox node in Tiptap when imported.

Please note that the prosemirrorNodes and prosemirrorMarks options will only work if you're importing a .docx file. If you're importing another type of file, eg: an .odt file, the /import endpoint will be used instead of the /import-docx endpoint, and the prosemirrorNodes and prosemirrorMarks options will not be available, and therefore you'd need to rely on the custom node and mark mapping API for those endpoints.

**Examples:**

Example 1 (typescript):
```typescript
import { mergeAttributes, Node } from '@tiptap/core'

export interface ParagraphOptions {
  /**
   * The HTML attributes for a paragraph node.
   * @default {}
   * @example { class: 'foo' }
   */
  HTMLAttributes: Record<string, any>
}

declare module '@tiptap/core' {
  interface Commands<ReturnType> {
    hintbox: {
      /**
       * Set a hintbox
       * @example editor.commands.setHintbox()
       */
      setHintbox: () => ReturnType
      /**
       * Toggle a hintbox
       * @example editor.commands.toggleHintbox()
       */
      toggleHintbox: () => ReturnType
    }
  }
}

/**
 * This extension allows you to create hintboxes.
 * @see https://www.tiptap.dev/api/nodes/paragraph
 */
export const Hintbox = Node.create<ParagraphOptions>({
  name: 'hintbox',

  priority: 1000,

  addOptions() {
    return {
      HTMLAttributes: {
        style: 'padding: 20px; border: 1px solid #b8d8ff; border-radius: 5px; background-color: #e6f3ff;',
      },
    }
  },

  group: 'block',

  content: 'inline*',

  parseHTML() {
    return [{ tag: 'p' }]
  },

  renderHTML({ HTMLAttributes }) {
    return ['p', mergeAttributes(this.options.HTMLAttributes, HTMLAttributes), 0]
  },

  addCommands() {
    return {
      setHintbox:
        () =>
        ({ commands }) => {
          return commands.setNode(this.name)
        },
      toggleHintbox:
        () =>
        ({ commands }) => {
          return commands.toggleNode(this.name, 'paragraph')
        },
    }
  },

  addKeyboardShortcuts() {
    return {
      'Mod-Alt-h': () => this.editor.commands.toggleHintbox(),
    }
  },
})
```

Example 2 (typescript):
```typescript
import { mergeAttributes, Node } from '@tiptap/core'

export interface ParagraphOptions {
  /**
   * The HTML attributes for a paragraph node.
   * @default {}
   * @example { class: 'foo' }
   */
  HTMLAttributes: Record<string, any>
}

declare module '@tiptap/core' {
  interface Commands<ReturnType> {
    hintbox: {
      /**
       * Set a hintbox
       * @example editor.commands.setHintbox()
       */
      setHintbox: () => ReturnType
      /**
       * Toggle a hintbox
       * @example editor.commands.toggleHintbox()
       */
      toggleHintbox: () => ReturnType
    }
  }
}

/**
 * This extension allows you to create hintboxes.
 * @see https://www.tiptap.dev/api/nodes/paragraph
 */
export const Hintbox = Node.create<ParagraphOptions>({
  name: 'hintbox',

  priority: 1000,

  addOptions() {
    return {
      HTMLAttributes: {
        style: 'padding: 20px; border: 1px solid #b8d8ff; border-radius: 5px; background-color: #e6f3ff;',
      },
    }
  },

  group: 'block',

  content: 'inline*',

  parseHTML() {
    return [{ tag: 'p' }]
  },

  renderHTML({ HTMLAttributes }) {
    return ['p', mergeAttributes(this.options.HTMLAttributes, HTMLAttributes), 0]
  },

  addCommands() {
    return {
      setHintbox:
        () =>
        ({ commands }) => {
          return commands.setNode(this.name)
        },
      toggleHintbox:
        () =>
        ({ commands }) => {
          return commands.toggleNode(this.name, 'paragraph')
        },
    }
  },

  addKeyboardShortcuts() {
    return {
      'Mod-Alt-h': () => this.editor.commands.toggleHintbox(),
    }
  },
})
```

Example 3 (typescript):
```typescript
// Import the ExportDocx extension
import {
  convertTextNode,
  Docx,
  ExportDocx,
  lineHeightToDocx,
  pixelsToHalfPoints,
  pointsToTwips,
} from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // Other extensions ...
    ExportDocx.configure({
      onCompleteExport: result => {
        setIsLoading(false)
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
      exportType: 'blob',
      customNodes: [
        {
          type: 'hintbox',
          render: node => {
            // Here we define how our custom Hintbox node should be rendered in the DOCX.
            // Per the documentation, we should return a Docx node
            // that's either a Paragraph, an array of Paragraphs, or a Table.
            return new Docx.Paragraph({
              children: node.content.map(content => convertTextNode(content)),
              style: 'Hintbox', // Here we apply our custom style to the Paragraph node.
            })
            },
        },
      ], // Custom nodes
      styleOverrides: {
        paragraphStyles: [
          // Here we define our custom styles for our custom Hintbox node.
          {
            id: 'Hintbox',
            name: 'Hintbox',
            basedOn: 'Normal',
            next: 'Normal',
            quickFormat: false,
            run: {
              font: 'Aptos Light',
              size: pixelsToHalfPoints(16),
            },
            paragraph: {
              spacing: {
                before: pointsToTwips(12),
                after: pointsToTwips(12),
                line: lineHeightToDocx(1),
              },
              border: {
                // DOCX colors are in Hexadecimal without the leading #
                top: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
                bottom: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
                right: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
                left: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
              },
              shading: {
                type: Docx.ShadingType.SOLID,
                color: 'e6f3ff',
              },
            },
          },
        ],
      }, // Style overrides
    }),
    // Other extensions ...
  ],
  // Other editor settings ...
})
```

Example 4 (typescript):
```typescript
// Import the ExportDocx extension
import {
  convertTextNode,
  Docx,
  ExportDocx,
  lineHeightToDocx,
  pixelsToHalfPoints,
  pointsToTwips,
} from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // Other extensions ...
    ExportDocx.configure({
      onCompleteExport: result => {
        setIsLoading(false)
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
      exportType: 'blob',
      customNodes: [
        {
          type: 'hintbox',
          render: node => {
            // Here we define how our custom Hintbox node should be rendered in the DOCX.
            // Per the documentation, we should return a Docx node
            // that's either a Paragraph, an array of Paragraphs, or a Table.
            return new Docx.Paragraph({
              children: node.content.map(content => convertTextNode(content)),
              style: 'Hintbox', // Here we apply our custom style to the Paragraph node.
            })
            },
        },
      ], // Custom nodes
      styleOverrides: {
        paragraphStyles: [
          // Here we define our custom styles for our custom Hintbox node.
          {
            id: 'Hintbox',
            name: 'Hintbox',
            basedOn: 'Normal',
            next: 'Normal',
            quickFormat: false,
            run: {
              font: 'Aptos Light',
              size: pixelsToHalfPoints(16),
            },
            paragraph: {
              spacing: {
                before: pointsToTwips(12),
                after: pointsToTwips(12),
                line: lineHeightToDocx(1),
              },
              border: {
                // DOCX colors are in Hexadecimal without the leading #
                top: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
                bottom: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
                right: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
                left: { style: Docx.BorderStyle.SINGLE, size: 1, color: 'b8d8ff', space: 5 },
              },
              shading: {
                type: Docx.ShadingType.SOLID,
                color: 'e6f3ff',
              },
            },
          },
        ],
      }, // Style overrides
    }),
    // Other extensions ...
  ],
  // Other editor settings ...
})
```

---

## Integration Markdown in Custom Extensions

**URL:** https://tiptap.dev/docs/editor/markdown/guides/integrate-markdown-in-your-extension

**Contents:**
- Integration Markdown in Custom Extensions
- Basic Extension Integration
- The Markdown Configuration Explained
  - Using markdownTokenName
- Node Extensions
- Mark Extensions
- Testing Your Extension
  - Unit Test Parse Handler
  - Unit Test Render Handler
  - Integration Test

This guide shows you how to add Markdown support to your Tiptap extensions.

Tip: For standard patterns like Pandoc blocks (:::name) or shortcodes ([name]), check out the Utility Functions to generate Markdown specs with minimal code.

To add Markdown support to an extension, define a Markdown configuration in your extension configuration:

The extension spec allows for the following options:

Creating Markdown support for nodes is straightforward. Here are some common patterns.

Depending on the type of content you expect, you may need to use different helper functions in your parseMarkdown and renderMarkdown methods.

Marks work differently because they wrap inline content. To add Markdown support to your mark extensions, use the applyMark and renderChildren helper functions.

If you want to apply attributes to your marks, use the applyMark helper with an attributes object.

**Examples:**

Example 1 (javascript):
```javascript
import { Node } from '@tiptap/core'

const MyNode = Node.create({
  name: 'myNode',

  // ... other configuration (parseHTML, renderHTML, etc.)

  parseMarkdown: (token, helpers) => {
    /* ... */
  },

  renderMarkdown: (node, helpers) => {
    /* ... */
  },
})
```

Example 2 (javascript):
```javascript
import { Node } from '@tiptap/core'

const MyNode = Node.create({
  name: 'myNode',

  // ... other configuration (parseHTML, renderHTML, etc.)

  parseMarkdown: (token, helpers) => {
    /* ... */
  },

  renderMarkdown: (node, helpers) => {
    /* ... */
  },
})
```

Example 3 (yaml):
```yaml
markdownTokenName: 'strong' // for a Bold mark extension where the Markdown token is called "strong"
```

Example 4 (yaml):
```yaml
markdownTokenName: 'strong' // for a Bold mark extension where the Markdown token is called "strong"
```

---

## Introduction into Markdown with Tiptap

**URL:** https://tiptap.dev/docs/editor/markdown

**Contents:**
- Introduction into Markdown with Tiptap
- Core Capabilities
- How It Works
  - Architecture
- Limitations
- Why MarkedJS?
  - Was this page helpful?

Important: The markdown extension is a early release and can be subject to change or may have edge cases that may not be supported yet. If you are encountering a bug or have a feature request, please open an issue on GitHub.

The Markdown extension provides bidirectional Markdown support for your Tiptap editor—parse Markdown strings into Tiptap's JSON format and serialize editor content back to Markdown.

The Markdown extension acts as a bridge between Markdown text and Tiptap's JSON document structure.

It extends the base editor functionality by overwriting existing methods & properties with markdown-ready implementations, allowing for seamless integration between Markdown and Tiptap's rich text editor.

The current implementation of the Markdown extension has some limitations:

This extension integrates MarkedJS as its parser:

The Lexer API breaks Markdown into tokens that map naturally to Tiptap's node structure, making the integration clean and maintainable. The extension works identically in browser and server environments.

**Examples:**

Example 1 (css):
```css
// Set initial content
const editor = new Editor({
  extensions: [StarterKit, Markdown],
  content: '# Hello World\n\nThis is **Markdown**!',
  contentType: 'markdown',
})

// Insert content
editor.commands.insertContent('# Hello World\n\nThis is **Markdown**!')
```

Example 2 (css):
```css
// Set initial content
const editor = new Editor({
  extensions: [StarterKit, Markdown],
  content: '# Hello World\n\nThis is **Markdown**!',
  contentType: 'markdown',
})

// Insert content
editor.commands.insertContent('# Hello World\n\nThis is **Markdown**!')
```

Example 3 (unknown):
```unknown
Markdown String
      ↓
   MarkedJS Lexer (Tokenization)
      ↓
   Markdown Tokens
      ↓
   Extension Parse Handlers
      ↓
   Tiptap JSON
```

Example 4 (unknown):
```unknown
Markdown String
      ↓
   MarkedJS Lexer (Tokenization)
      ↓
   Markdown Tokens
      ↓
   Extension Parse Handlers
      ↓
   Tiptap JSON
```

---

## InvisibleCharacters extensions

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/invisiblecharacters

**Contents:**
- InvisibleCharacters extensions
- Install
- Settings
  - visible
  - builders
  - injectCSS
  - injectNonce
- Storage
  - visibility()
- Commands

This extension adds decorators to show non-printable characters and help you increase accessibility.

Define default visibility.

An array of invisible characters – by default it contains: spaces, hard breaks and paragraphs.

Default: [new SpaceCharacter(), new HardBreakNode(), new ParagraphNode()]

By default, the extension injects some CSS. With injectCSS you can disable that.

When you use a Content-Security-Policy with nonce, you can specify a nonce to be added to dynamically created elements. Here is an example:

Find out whether the visibility of invisible characters is active or not.

Show invisible characters. You can also pass false to use the same command to hide them.

Hide invisible characters.

Toggle visibility of invisible characters.

To create a custom invisible characters, you can extend the classes provided by the package.

To select the decoration within CSS, we can use the following selector:

To select the decoration within CSS, we can use the following selector:

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-invisible-characters
```

Example 2 (python):
```python
npm install @tiptap/extension-invisible-characters
```

Example 3 (css):
```css
InvisibleCharacters.configure({
  visible: false,
})
```

Example 4 (css):
```css
InvisibleCharacters.configure({
  visible: false,
})
```

---

## Italic extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/italic

**Contents:**
- Italic extension
  - Restrictions
- Install
- Settings
  - HTMLAttributes
- Commands
  - setItalic()
  - toggleItalic()
  - unsetItalic()
- Keyboard shortcuts

Use this extension to render text in italic. If you pass <em>, <i> tags, or text with inline style attributes setting font-style: italic in the editor’s initial content, they all will be rendered accordingly.

Type *one asterisk* or _one underline_ and it will magically transform to italic text while you type.

The extension will generate the corresponding <em> HTML tags when reading contents of the Editor instance. All text marked italic, regardless of the method will be normalized to <em> HTML tags.

Custom HTML attributes that should be added to the rendered HTML tag.

Mark the text italic.

Toggle the italic mark.

Remove the italic mark.

packages/extension-italic/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-italic
```

Example 2 (python):
```python
npm install @tiptap/extension-italic
```

Example 3 (css):
```css
Italic.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Italic.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Keyboard shortcuts in Tiptap

**URL:** https://tiptap.dev/docs/editor/core-concepts/keyboard-shortcuts

**Contents:**
- Keyboard shortcuts in Tiptap
- Predefined keyboard shortcuts
  - Essentials
  - Text Formatting
  - Paragraph Formatting
  - Text Selection
- Overwrite keyboard shortcuts
  - Was this page helpful?

Tiptap comes with sensible keyboard shortcut defaults. Depending on what you want to use it for, you’ll probably want to change those keyboard shortcuts to your liking. Let’s have a look at what we defined for you, and show you how to change it then!

Most of the core extensions register their own keyboard shortcuts. Depending on what set of extension you use, not all of the below listed keyboard shortcuts work for your editor.

Keyboard shortcuts may be strings like 'Shift-Control-Enter'. Keys are based on the strings that can appear in event.key, concatenated with a -. There is a little tool called keycode.info, which shows the event.key interactively.

Use lowercase letters to refer to letter keys (or uppercase letters if you want shift to be held). You may use Space as an alias for the .

Modifiers can be given in any order. Shift, Alt, Control and Cmd are recognized. For characters that are created by holding shift, the Shift prefix is implied, and should not be added explicitly.

You can use Mod as a shorthand for Cmd on Mac and Control on other platforms.

Here is an example how you can overwrite the keyboard shortcuts for an existing extension:

**Examples:**

Example 1 (javascript):
```javascript
// 1. Import the extension
import BulletList from '@tiptap/extension-bullet-list'

// 2. Overwrite the keyboard shortcuts
const CustomBulletList = BulletList.extend({
  addKeyboardShortcuts() {
    return {
      // ↓ your new keyboard shortcut
      'Mod-l': () => this.editor.commands.toggleBulletList(),
    }
  },
})

// 3. Add the custom extension to your editor
new Editor({
  extensions: [
    CustomBulletList(),
    // …
  ],
})
```

Example 2 (javascript):
```javascript
// 1. Import the extension
import BulletList from '@tiptap/extension-bullet-list'

// 2. Overwrite the keyboard shortcuts
const CustomBulletList = BulletList.extend({
  addKeyboardShortcuts() {
    return {
      // ↓ your new keyboard shortcut
      'Mod-l': () => this.editor.commands.toggleBulletList(),
    }
  },
})

// 3. Add the custom extension to your editor
new Editor({
  extensions: [
    CustomBulletList(),
    // …
  ],
})
```

---

## Line Height extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/line-height

**Contents:**
- Line Height extension
- Install
- Settings
  - types
- Commands
  - setLineHeight()
  - unsetLineHeight()
- Source code
- Minimal Install
  - Was this page helpful?

This extension enables you to set the line height in the editor. It uses the TextStyle mark, which renders a <span> tag (and only that). The line height is applied as inline style then, for example <span style="line-height: 1.5">.

This extension requires the TextStyle mark.

This extension is installed by default with the TextStyleKit extension, so you don’t need to install it separately.

A list of marks to which the lineHeight attribute should be applied to.

Default: ['textStyle']

Applies the given line height as inline style.

Removes any line height.

packages/extension-text-style/src/line-height/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-style
```

Example 2 (python):
```python
npm install @tiptap/extension-text-style
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, LineHeight } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, LineHeight],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyle, LineHeight } from '@tiptap/extension-text-style'

new Editor({
  extensions: [TextStyle, LineHeight],
})
```

---

## Link extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/link

**Contents:**
- Link extension
- Install
- Settings
  - protocols
  - autolink
  - openOnClick
  - enableClickSelection
  - linkOnPaste
  - defaultProtocol
  - HTMLAttributes

The Link extension adds support for <a> tags to the editor. The extension is headless too, there is no actual UI to add, modify or delete links. The usage example below uses the native JavaScript prompt to show you how that could work.

In a real world application, you would probably add a more sophisticated user interface.

Pasted URLs will be transformed to links automatically.

Additional custom protocols you would like to be recognized as links.

By default, linkify adds // to the end of a protocol however this behavior can be changed by passing optionalSlashes option

If enabled, it adds links as you type.

If enabled, links will be opened on click.

If enabled, clicking on a link will select the link.

Adds a link to the current selection if the pasted content only contains an url.

The default protocol used by linkOnPaste and autolink when no protocol is defined.

By default, the href generated for example.com is http://example.com and this option allows that protocol to be customized.

Custom HTML attributes that should be added to the rendered HTML tag.

You can add rel: null to HTMLAttributes to remove the default rel="noopener noreferrer nofollow". You can also override the default by using rel: "your-value".

This can also be used to change the target from the default value of _blank.

A function that allows customization of link validation, modifying the default verification logic. This function accepts the URL and a context object with additional properties.

Returns: boolean - true if the URL is valid, false otherwise.

This function enables you to enforce rules on allowed protocols or domains when autolinking URLs.

This function has been deprecated in favor of the more descriptive shouldAutoLink function. If provided, the validate function will replace the shouldAutoLink function.

Defines whether a valid link should be automatically linked within the editor content.

Returns: boolean - true if the link should be auto-linked, false if it should not.

Use this function to control autolinking behavior based on the URL.

Links the selected text.

Adds or removes a link from the selected text.

This extension doesn’t bind a specific keyboard shortcut. You would probably open your custom UI on Mod-k though.

Did you know that you can use getAttributes to find out which attributes, for example which href, is currently set? Don’t confuse it with a command (which changes the state), it’s just a method. Here is how that could look like:

packages/extension-link/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-link
```

Example 2 (python):
```python
npm install @tiptap/extension-link
```

Example 3 (css):
```css
Link.configure({
  protocols: ['ftp', 'mailto'],
})
```

Example 4 (css):
```css
Link.configure({
  protocols: ['ftp', 'mailto'],
})
```

---

## ListItem extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/list-item

**Contents:**
- ListItem extension
  - Modify backspace behavior
- Install
- Usage
- Settings
  - HTMLAttributes
- Keyboard shortcuts
- Source code
- Minimal Install
  - Was this page helpful?

The ListItem extension adds support for the <li> HTML tag. It’s used for bullet lists and ordered lists and can’t really be used without them.

If you want to modify the standard behavior of backspace and delete functions for lists, you should read about the ListKeymap extension.

This extension requires the BulletList or OrderedList node.

This extension is installed by default with the ListKit extension, so you don’t need to install it separately.

Custom HTML attributes that should be added to the rendered HTML tag.

packages/extension-list/src/item/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-list
```

Example 2 (python):
```python
npm install @tiptap/extension-list
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { ListItem } from '@tiptap/extension-list'

new Editor({
  extensions: [ListItem],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { ListItem } from '@tiptap/extension-list'

new Editor({
  extensions: [ListItem],
})
```

---

## List Keymap extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/listkeymap

**Contents:**
- List Keymap extension
- Install
- Settings
  - listTypes
- Source code
- Minimal Install
  - Was this page helpful?

The List Keymap extension modifies the default ProseMirror and Tiptap behavior. Without this extension, pressing backspace at the start of a list item keeps the list item content on the same line. With the List Keymap, the content is lifted into the list item above.

And import it in your editor:

This extension is installed by default with the ListKit extension, so you don’t need to install it separately.

An array of list items and their parent wrapper node types.

packages/extension-list/src/keymap/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-list
```

Example 2 (python):
```python
npm install @tiptap/extension-list
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { ListKeymap } from '@tiptap/extension-list'

new Editor({
  extensions: [ListKeymap],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { ListKeymap } from '@tiptap/extension-list'

new Editor({
  extensions: [ListKeymap],
})
```

---

## Mark API

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/create-new/mark

**Contents:**
- Mark API
- Creating a mark
- Mark options
  - parseHTML
  - renderHTML
  - addAttributes
  - keepOnSplit
  - inclusive
  - excludes
  - exitable

The power of Tiptap lies in its flexibility. You can create your own extensions from scratch and build a unique editor experience tailored to your needs.

Marks are used to add inline formatting to text in Tiptap. Common examples include bold, italic, and underline formatting.

They extend all of the options and methods from the Extension API and add a few more specific to marks.

Let's add a simple mark extension to see how it works.

You can also use a callback function to create a mark. This is useful if you want to encapsulate the logic of your extension, for example when you want to define event handlers or other custom logic.

This code creates a new mark extension named HighlightMark. It adds an addOptions method to define the mark's options, which are configurable by the user. It also adds parseHTML and renderHTML methods to define how the mark is parsed and rendered as HTML.

It is installed to the editor just like any other extension by adding it to the extensions array.

Now let's take a closer look at the options and methods available for marks.

When creating a mark extension, you can define options that are configurable by the user. These options can be used to customize the behavior or appearance of the mark.

The parseHTML method is used to define how the mark is parsed from HTML. It should return an array of objects representing the mark's attributes.

Maps to the parseDOM attribute in the ProseMirror schema.

This will be used during paste events to parse the HTML content into a mark.

The renderHTML method is used to define how the mark is rendered as HTML. It should return an array representing the mark's HTML representation.

Maps to the toDOM attribute in the ProseMirror schema.

This will be used during copy events to render the mark as HTML. For more details, see the extend existing extensions guide.

The addAttributes method is used to define custom attributes for the mark. It should return an object with the attribute names and their default values.

Maps to the attrs attribute in the ProseMirror schema.

For more details, see the extend existing extensions guide.

By default, when a node is split, marks are removed from the content. You can set keepOnSplit to true to keep the mark on the new node.

By default, marks are not inclusive, meaning they cannot be applied to the start or end of a node. You can set inclusive to true to allow the mark to be applied to the start or end of a node.

Maps to the inclusive attribute in the ProseMirror schema.

By default, marks do not exclude other marks. You can define a list of marks that should be excluded when this mark is applied.

Maps to the excludes attribute in the ProseMirror schema.

By default, marks are not exitable, meaning they cannot be removed by pressing backspace at the start of the mark. You can set exitable to true to allow the mark to be removed in this way.

Maps to the exitable attribute in the ProseMirror schema.

By default, marks are not grouped, meaning they are applied individually. You can define a group for the mark to ensure that only one mark from the group can be applied at a time.

Maps to the group attribute in the ProseMirror schema.

By default, marks do not span multiple nodes. You can set spanning to true to allow the mark to span multiple nodes.

Maps to the spanning attribute in the ProseMirror schema.

By default, marks are not treated as code marks. You can set code to true to treat the mark as a code mark.

Maps to the code attribute in the ProseMirror schema.

**Examples:**

Example 1 (json):
```json
import { Mark } from '@tiptap/core'

const HighlightMark = Mark.create({
  name: 'highlight',

  addOptions() {
    return {
      HTMLAttributes: {},
    }
  },

  parseHTML() {
    return [
      {
        tag: 'mark',
      },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['mark', HTMLAttributes, 0]
  },
})
```

Example 2 (json):
```json
import { Mark } from '@tiptap/core'

const HighlightMark = Mark.create({
  name: 'highlight',

  addOptions() {
    return {
      HTMLAttributes: {},
    }
  },

  parseHTML() {
    return [
      {
        tag: 'mark',
      },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['mark', HTMLAttributes, 0]
  },
})
```

Example 3 (javascript):
```javascript
import { Mark } from '@tiptap/core'

const CustomMark = Mark.create(() => {
  // Define variables or functions to use inside your extension
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customMark',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

Example 4 (javascript):
```javascript
import { Mark } from '@tiptap/core'

const CustomMark = Mark.create(() => {
  // Define variables or functions to use inside your extension
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customMark',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

---

## Mark extensions

**URL:** https://tiptap.dev/docs/editor/extensions/marks

**Contents:**
- Mark extensions
  - Was this page helpful?

Learn about mark extensions like Bold, Code, Link, and more to improve your users’ text editor experience in Tiptap.

---

## Mark views with JavaScript

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/mark-views/javascript

**Contents:**
- Mark views with JavaScript
- Render a mark view with JavaScript
- Access mark attributes
- Adding a content editable
- Updating mark view attributes
  - Was this page helpful?

Using frameworks like Vue or React can feel too complex, if you’re used to work without those two. Good news: You can use Vanilla JavaScript in your mark views. There is just a little bit you need to know, but let’s go through this one by one.

Here is what you need to do to render a mark view inside your editor:

This is how your mark extension could look like:

Got it? Let’s see it in action. Feel free to copy the below example to get started.

That mark view even interacts with the editor. Time to see how that is wired up.

The editor passes a few helpful things to your render function. One of them is the mark prop. This one enables you to access mark attributes in your mark view. Let’s say you have added an attribute named color to your mark extension. You could access the attribute like this:

A mark is wrapping a part of the text in the editor. If you want to make the content of the mark editable, you can add the contenteditable attribute to the contentDOM element.

Got it? You’re free to do anything you like, as long as you return a container for the mark view and another one for the content.

If you want to update the attributes of your mark view, you can call the updateAttributes method on the MarkView instance.

**Examples:**

Example 1 (javascript):
```javascript
import { Mark } from '@tiptap/core'

export default Mark.create({
  // Other options...
  addMarkView() {
    return ({ mark, HTMLAttributes }) => {
      const dom = document.createElement('b')
      const contentDOM = document.createElement('span')

      dom.appendChild(contentDOM)

      return {
        dom,
        contentDOM,
      }
    }
  },
})
```

Example 2 (javascript):
```javascript
import { Mark } from '@tiptap/core'

export default Mark.create({
  // Other options...
  addMarkView() {
    return ({ mark, HTMLAttributes }) => {
      const dom = document.createElement('b')
      const contentDOM = document.createElement('span')

      dom.appendChild(contentDOM)

      return {
        dom,
        contentDOM,
      }
    }
  },
})
```

Example 3 (javascript):
```javascript
addMarkView() {
  return ({ mark }) => {
    console.log(mark.attrs.color)
  }
}
```

Example 4 (javascript):
```javascript
addMarkView() {
  return ({ mark }) => {
    console.log(mark.attrs.color)
  }
}
```

---

## Mark views with React

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/mark-views/react

**Contents:**
- Mark views with React
- Render a React component
- Updating the mark view attributes
  - Was this page helpful?

Using Vanilla JavaScript can feel complex if you are used to work in React. Good news: You can use regular React components in your mark views, too. There is just a little bit you need to know, but let’s go through this one by one.

Here is what you need to do to render React components inside your editor:

This is how your node extension could look like:

And here is an example of a React component:

Got it? Let’s see it in action. Feel free to copy the below example to get started.

Updating your mark view's attributes is very straightforward. You can use the updateAttributes method provided by the MarkViewRendererProps to update the attributes of your mark view.

**Examples:**

Example 1 (python):
```python
import { Mark } from '@tiptap/core'
import { ReactMarkViewRenderer } from '@tiptap/react'

import Component from './Component.jsx'

export default Mark.create({
  // options…

  addMarkView() {
    return ReactMarkViewRenderer(Component)
  },
})
```

Example 2 (python):
```python
import { Mark } from '@tiptap/core'
import { ReactMarkViewRenderer } from '@tiptap/react'

import Component from './Component.jsx'

export default Mark.create({
  // options…

  addMarkView() {
    return ReactMarkViewRenderer(Component)
  },
})
```

Example 3 (jsx):
```jsx
import { MarkViewContent, MarkViewRendererProps } from '@tiptap/react'
import React from 'react'

export default (props: MarkViewRendererProps) => {
  const [count, setCount] = React.useState(0)

  return (
    <span className="content" data-test-id="mark-view">
      <MarkViewContent />
      <label contentEditable={false}>
        React component:
        <button
          onClick={() => {
            setCount(count + 1)
          }}
        >
          This button has been clicked {count} times.
        </button>
      </label>
    </span>
  )
}
```

Example 4 (jsx):
```jsx
import { MarkViewContent, MarkViewRendererProps } from '@tiptap/react'
import React from 'react'

export default (props: MarkViewRendererProps) => {
  const [count, setCount] = React.useState(0)

  return (
    <span className="content" data-test-id="mark-view">
      <MarkViewContent />
      <label contentEditable={false}>
        React component:
        <button
          onClick={() => {
            setCount(count + 1)
          }}
        >
          This button has been clicked {count} times.
        </button>
      </label>
    </span>
  )
}
```

---

## Mark views with Vue

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/mark-views/vue

**Contents:**
- Mark views with Vue
- Render a Vue component
- Updating the mark view attributes
  - Was this page helpful?

Using Vanilla JavaScript can feel complex if you are used to work in Vue. Good news: You can use regular Vue components in your mark views, too. There is just a little bit you need to know, but let’s go through this one by one.

Here is what you need to do to render Vue components inside your editor:

This is how your node extension could look like:

And here is an example of a Vue component:

Got it? Let’s see it in action. Feel free to copy the below example to get started.

Updating your mark view's attributes is very straightforward. You can use the updateAttributes method provided by the component's props.

**Examples:**

Example 1 (python):
```python
import { Mark } from '@tiptap/core'
import { VueMarkViewRenderer } from '@tiptap/vue-3'

import Component from './Component.jsx'

export default Mark.create({
  // options…

  addMarkView() {
    return VueMarkViewRenderer(Component)
  },
})
```

Example 2 (python):
```python
import { Mark } from '@tiptap/core'
import { VueMarkViewRenderer } from '@tiptap/vue-3'

import Component from './Component.jsx'

export default Mark.create({
  // options…

  addMarkView() {
    return VueMarkViewRenderer(Component)
  },
})
```

Example 3 (vue):
```vue
<template>
  <span className="content" data-test-id="mark-view">
    <mark-view-content />
    <label contenteditable="false"
      >Vue Component::
      <button @click="increase" class="primary">
        This button has been clicked {{ count }} times.
      </button>
    </label>
  </span>
</template>

<script>
import { MarkViewContent, markViewProps } from '@tiptap/vue-3'
export default {
  components: {
    MarkViewContent,
  },
  data() {
    return {
      count: 0,
    }
  },
  props: markViewProps,
  methods: {
    increase() {
      this.count += 1
    },
  },
}
</script>
```

Example 4 (vue):
```vue
<template>
  <span className="content" data-test-id="mark-view">
    <mark-view-content />
    <label contenteditable="false"
      >Vue Component::
      <button @click="increase" class="primary">
        This button has been clicked {{ count }} times.
      </button>
    </label>
  </span>
</template>

<script>
import { MarkViewContent, markViewProps } from '@tiptap/vue-3'
export default {
  components: {
    MarkViewContent,
  },
  data() {
    return {
      count: 0,
    }
  },
  props: markViewProps,
  methods: {
    increase() {
      this.count += 1
    },
  },
}
</script>
```

---

## Mathematics extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/mathematics

**Contents:**
- Mathematics extension
- Install
- Usage
- Additional Setup
  - Styling the KaTeX rendering
  - Configuring the extension and updating math nodes
  - Migrating existing math decorations to math nodes
  - Only importing one type of math node
- Settings
  - inlineOptions

This extension allows you to insert math formulas into your editor. It uses KaTeX to render math formulas written in LaTeX.

You are free to style the rendering element and the editor input.

Import of KaTeX styling (needed).

The following classes allow you to select and style the math nodes. For an example, see demonstration code at the end of this page.

The extension comes with a few options to configure the behavior of the math nodes.

Since the Math extension used to be a decoration extension, you might have existing math decorations in your editor. To migrate them to the new math nodes, you can use the migrateMathStrings utility function provided by the extension.

If you only want to use one type of math node (either inline or block), you can import the respective extension directly:

This option allows you to configure the inline math node. You can pass any options that are supported by the inline math node, such as onClick to handle click events on the node.

This option allows you to configure the block math node. You can pass any options that are supported by the block math node, such as onClick to handle click events on the node.

This option allows you to configure the KaTeX renderer. You can see all options here.

This option allows you to handle click events on the inline math node. You can use it to open a dialog to edit the math node or just a prompt to edit the LaTeX code for a quick prototype.

This option allows you to configure the KaTeX renderer for the inline math node. You can see all options here.

This option allows you to handle click events on the block math node. You can use it to open a dialog to edit the math node or just a prompt to edit the LaTeX code for a quick prototype.

This option allows you to configure the KaTeX renderer for the block math node. You can see all options here.

Inserts a new inline math node with the given LaTeX code at the specified position. If pos is not provided, it will insert the node at the current selection.

Deletes the inline math node at the specified position. If pos is not provided, it will delete the node at the current selection.

Updates the inline math node at the specified position with the given LaTeX code. If pos is not provided, it will update the node at the current selection.

Inserts a new block math node with the given LaTeX code at the specified position. If pos is not provided, it will insert the node at the current selection.

Deletes the block math node at the specified position. If pos is not provided, it will delete the node at the current selection.

Updates the block math node at the specified position with the given LaTeX code. If pos is not provided, it will update the node at the current selection.

The default regular expression used to find and migrate math strings in the editor. It matches LaTeX expressions wrapped in $ symbols.

Creates a ProseMirror transaction that migrates math strings in the editor document to math nodes. It uses the provided regular expression to find math strings.

Creates and runs a migration transaction to convert math strings in the editor document to math nodes. It uses the provided regular expression to find math strings.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-mathematics katex
```

Example 2 (python):
```python
npm install @tiptap/extension-mathematics katex
```

Example 3 (csharp):
```csharp
import { Mathematics } from '@tiptap/extension-mathematics'

const editor = new Editor({
  extensions: [
    Mathematics.configure({
      inlineOptions: {
        // optional options for the inline math node
      },
      blockOptions: {
        // optional options for the block math node
      },
      katexOptions: {
        // optional options for the KaTeX renderer
      },
    }),
  ],
})
```

Example 4 (csharp):
```csharp
import { Mathematics } from '@tiptap/extension-mathematics'

const editor = new Editor({
  extensions: [
    Mathematics.configure({
      inlineOptions: {
        // optional options for the inline math node
      },
      blockOptions: {
        // optional options for the block math node
      },
      katexOptions: {
        // optional options for the KaTeX renderer
      },
    }),
  ],
})
```

---

## Mention extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/mention

**Contents:**
- Mention extension
- Install
- Dependencies
- Settings
  - HTMLAttributes
  - renderText
  - renderHTML
  - deleteTriggerWithBackspace
  - suggestion
  - suggestions

Honestly, the mention node is amazing. It adds support for @mentions, for example to ping users, and provides full control over the rendering.

Literally everything can be customized. You can pass a custom component for the rendering. All examples use .filter() to search through items, but feel free to send async queries to an API or add a more advanced library like fuse.js to your project.

To place the popups correctly, we’re using tippy.js in all our examples. You are free to bring your own library, but if you’re fine with it, just install what we use:

Since 2.0.0-beta.193 we marked the @tiptap/suggestion as a peer dependency. That means, you will need to install it manually.

Custom HTML attributes that should be added to the rendered HTML tag.

Define how a mention text should be rendered.

Define how a mention html element should be rendered, this is useful if you want to render an element other than span (e.g a)

Toggle whether the suggestion character(s) should also be deleted on deletion of a mention node. Default is false.

Options for the Suggestion utility. Used to define what character triggers the suggestion popup, among other parameters. Read more.

Allows you to define multiple types of mentions within the same editor. For example, define a mention for people with the @ trigger and one for movies with the # trigger. Read more about the Suggestion utility.

Below is an example demo:

packages/extension-mention/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-mention
```

Example 2 (python):
```python
npm install @tiptap/extension-mention
```

Example 3 (unknown):
```unknown
npm install tippy.js
```

Example 4 (unknown):
```unknown
npm install tippy.js
```

---

## Nodes and Marks

**URL:** https://tiptap.dev/docs/editor/core-concepts/nodes-and-marks

**Contents:**
- Nodes and Marks
- Differences
  - Was this page helpful?

If you think of the document as a tree, then nodes are just a type of content in that tree. Examples of nodes are paragraphs, headings, or code blocks. But nodes don’t have to be blocks. They can also be rendered inline with the text, for example for @mentions. Think of them as unique pieces of content that can be styled and manipulated in different ways.

Marks can be applied to specific parts of a node. That’s the case for bold, italic or striked text. Links are marks, too. Think of them as a way to style or annotate text.

Nodes and marks are similar in some ways, but they have different use cases. Nodes are the building blocks of your document. They define the structure and hierarchy of your content. Marks, on the other hand, are used to style or annotate text. They can be applied to any part of a node, but they don’t change the structure of the document.

---

## Nodes extensions

**URL:** https://tiptap.dev/docs/editor/extensions/nodes

**Contents:**
- Nodes extensions
  - Was this page helpful?

If you think of the document as a tree, then nodes are just a type of content in that tree. Examples of nodes are paragraphs, headings, or code blocks. But nodes don’t have to be blocks. They can also be rendered inline with the text, for example for @mentions.

---

## Node API

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/create-new/node

**Contents:**
- Node API
- Creating a node
- Node options
  - parseHTML
  - renderHTML
  - addAttributes
  - topNode
  - content
  - marks
  - group

The power of Tiptap lies in its flexibility. You can create your own extensions from scratch and build a unique editor experience tailored to your needs.

Nodes are the building blocks of your editor. They can be blocks or inline nodes. Good examples to learn from are Paragraph, Heading, or CodeBlock.

They extend all of the options and methods from the Extension API and add a few more specific to nodes.

Let's add a simple node extension to see how it works.

You can also use a callback function to create a node. This is useful if you want to encapsulate the logic of your extension, for example when you want to define event handlers or other custom logic.

This code creates a new node extension named CustomNode. It adds an addOptions method to define the node's options, which are configurable by the user. It also adds parseHTML and renderHTML methods to define how the node is parsed and rendered as HTML.

It is installed to the editor just like any other extension by adding it to the extensions array.

Now let's take a closer look at the options and methods available for nodes.

When creating a node, you can define options that are configurable by the user. These options can be used to customize the behavior or appearance of the node.

The parseHTML method is used to define how the mark is parsed from HTML. It should return an array of objects representing the mark's attributes.

Maps to the parseDOM attribute in the ProseMirror schema.

This will be used during paste events to parse the HTML content into a mark.

The renderHTML method is used to define how the mark is rendered as HTML. It should return an array representing the mark's HTML representation.

Maps to the toDOM attribute in the ProseMirror schema.

This will be used during copy events to render the mark as HTML. For more details, see the extend existing extensions guide.

The addAttributes method is used to define custom attributes for the mark. It should return an object with the attribute names and their default values.

Maps to the attrs attribute in the ProseMirror schema.

For more details, see the extend existing extensions guide.

Defines if this node should be a top-level node (doc).

Maps to the topNode attribute in the ProseMirror schema.

The content expression for this node, as described in the schema guide. When not given, the node does not allow any content.

You can read more about it on the Prosemirror documentation here.

The marks that are allowed inside of this node. May be a space-separated string referring to mark names or groups, "_" to explicitly allow all marks, or "" to disallow marks. When not given, nodes with inline content default to allowing all marks, other nodes default to not allowing marks.

Maps to the marks attribute in the ProseMirror schema.

The group or space-separated groups to which this node belongs, which can be referred to in the content expressions for the schema.

By default, Tiptap uses the groups 'block' and 'inline' for nodes. You can also use custom groups if you want to group specific nodes together and handle them in your schema.

Maps to the group attribute in the ProseMirror schema.

Should be set to true for inline nodes. (Implied for text nodes).

Maps to the inline attribute in the ProseMirror schema.

Can be set to true to indicate that, though this isn't a leaf node, it doesn't have directly editable content and should be treated as a single unit in the view.

Maps to the atom attribute in the ProseMirror schema.

Controls whether nodes of this type can be selected as a node selection. Defaults to true for non-text nodes.

Maps to the selectable attribute in the ProseMirror schema.

Determines whether nodes of this type can be dragged without being selected. Defaults to false.

Maps to the draggable attribute in the ProseMirror schema.

Can be used to indicate that this node contains code, which causes some commands to behave differently.

Maps to the code attribute in the ProseMirror schema.

Controls the way whitespace in this a node is parsed. The default is "normal", which causes the DOM parser to collapse whitespace in normal mode, and normalize it (replacing newlines and such with spaces) otherwise. "pre" causes the parser to preserve spaces inside the node. When this option isn't given, but code is true, whitespace will default to "pre".

Maps to the whitespace attribute in the ProseMirror schema.

Allows a single node to be set as a linebreak equivalent (e.g. hardBreak). When converting between block types that have whitespace set to "pre" and don't support the linebreak node (e.g. codeBlock) and other block types that do support the linebreak node (e.g. paragraphs) - this node will be used as the linebreak instead of stripping the newline.

Maps to the linebreakReplacement attribute in the ProseMirror schema.

When enabled, enables both definingAsContext and definingForContent.

Maps to the defining attribute in the ProseMirror schema.

When enabled (default is false), the sides of nodes of this type count as boundaries that regular editing operations, like backspacing or lifting, won't cross. An example of a node that should probably have this enabled is a table cell.

Maps to the isolating attribute in the ProseMirror schema.

For advanced use cases, where you need to execute JavaScript inside your nodes, for example to render a sophisticated interface around an image, you need to learn about node views.

They are really powerful, but also complex. In a nutshell, you need to return a parent DOM element, and a DOM element where the content should be rendered in. Look at the following, simplified example:

There is a whole lot to learn about node views, so head over to the dedicated section in our guide about node views for more information. If you are looking for a real-world example, look at the source code of the TaskItem node. This is using a node view to render the checkboxes.

**Examples:**

Example 1 (json):
```json
import { Node } from '@tiptap/core'

const CustomNode = Node.create({
  name: 'customNode',

  addOptions() {
    return {
      HTMLAttributes: {},
    }
  },

  parseHTML() {
    return [
      {
        tag: 'div',
      },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['div', HTMLAttributes, 0]
  },
})
```

Example 2 (json):
```json
import { Node } from '@tiptap/core'

const CustomNode = Node.create({
  name: 'customNode',

  addOptions() {
    return {
      HTMLAttributes: {},
    }
  },

  parseHTML() {
    return [
      {
        tag: 'div',
      },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['div', HTMLAttributes, 0]
  },
})
```

Example 3 (javascript):
```javascript
import { Node } from '@tiptap/core'

const CustomNode = Node.create(() => {
  // here you could define variables or function that you can use on your schema definition
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customNode',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

Example 4 (javascript):
```javascript
import { Node } from '@tiptap/core'

const CustomNode = Node.create(() => {
  // here you could define variables or function that you can use on your schema definition
  const customVariable = 'foo'

  function onCreate() {}
  function onUpdate() {}

  return {
    name: 'customNode',
    onCreate,
    onUpdate,

    // Your code goes here.
  }
})
```

---

## Node views with JavaScript

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/node-views/javascript

**Contents:**
- Node views with JavaScript
- Render a node view with JavaScript
- Access node attributes
- Update node attributes
- Adding a content editable
  - Was this page helpful?

Using frameworks like Vue or React can feel too complex, if you’re used to work without those two. Good news: You can use Vanilla JavaScript in your node views. There is just a little bit you need to know, but let’s go through this one by one.

Here is what you need to do to render a node view inside your editor:

This is how your node extension could look like:

Got it? Let’s see it in action. Feel free to copy the below example to get started.

That node view even interacts with the editor. Time to see how that is wired up.

The editor passes a few helpful things to your render function. One of them is the node prop. This one enables you to access node attributes in your node view. Let’s say you have added an attribute named count to your node extension. You could access the attribute like this:

You can even update node attributes from your node view, with the help of the getPos prop passed to your render function. Dispatch a new transaction with an object of the updated attributes:

Does seem a little bit too complex? Consider using React or Vue, if you have one of those in your project anyway. It get’s a little bit easier with those two.

To add editable content to your node view, you need to pass a contentDOM, a container element for the content. Here is a simplified version of a node view with non-editable and editable text content:

Got it? You’re free to do anything you like, as long as you return a container for the node view and another one for the content. Here is the above example in action:

Keep in mind that this content is rendered by Tiptap. That means you need to tell what kind of content is allowed, for example with content: 'inline*' in your node extension (that’s what we use in the above example).

**Examples:**

Example 1 (javascript):
```javascript
import { Node } from '@tiptap/core'

export default Node.create({
  // configuration …

  addNodeView() {
    return ({ editor, node, getPos, HTMLAttributes, decorations, extension }) => {
      const dom = document.createElement('div')

      dom.innerHTML = 'Hello, I’m a node view!'

      return {
        dom,
      }
    }
  },
})
```

Example 2 (javascript):
```javascript
import { Node } from '@tiptap/core'

export default Node.create({
  // configuration …

  addNodeView() {
    return ({ editor, node, getPos, HTMLAttributes, decorations, extension }) => {
      const dom = document.createElement('div')

      dom.innerHTML = 'Hello, I’m a node view!'

      return {
        dom,
      }
    }
  },
})
```

Example 3 (javascript):
```javascript
addNodeView() {
  return ({ node }) => {
    console.log(node.attrs.count)

    // …
  }
}
```

Example 4 (javascript):
```javascript
addNodeView() {
  return ({ node }) => {
    console.log(node.attrs.count)

    // …
  }
}
```

---

## Node views with React

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/node-views/react

**Contents:**
- Node views with React
- Render a React component
- Access node attributes
- Update node attributes
- Adding a content editable
- Changing the default content tag for a node view
- Changing the wrapping DOM element
- All available props
- Dragging
  - Was this page helpful?

Using Vanilla JavaScript can feel complex if you are used to work in React. Good news: You can use regular React components in your node views, too. There is just a little bit you need to know, but let’s go through this one by one.

Here is what you need to do to render React components inside your editor:

This is how your node extension could look like:

There is a little bit of magic required to make this work. But don’t worry, we provide a wrapper component you can use to get started easily. Don’t forget to add it to your custom React component, like shown below:

Got it? Let’s see it in action. Feel free to copy the below example to get started.

That component doesn’t interact with the editor, though. Time to wire it up.

The ReactNodeViewRenderer which you use in your node extension, passes a few very helpful props to your custom React component. One of them is the node prop. Let’s say you have added an attribute named count to your node extension (like we did in the above example) you could access it like this:

You can even update node attributes from your node, with the help of the updateAttributes prop passed to your component. Pass an object with updated attributes to the updateAttributes prop:

And yes, all of that is reactive, too. A pretty seamless communication, isn’t it?

There is another component called NodeViewContent which helps you adding editable content to your node view. Here is an example:

You don’t need to add those className attributes, feel free to remove them or pass other class names. Try it out in the following example:

Keep in mind that this content is rendered by Tiptap. That means you need to tell what kind of content is allowed, for example with content: 'inline*' in your node extension (that’s what we use in the above example).

The NodeViewWrapper and NodeViewContent components render a <div> HTML tag (<span> for inline nodes), but you can change that. For example <NodeViewContent as="p"> should render a paragraph. One limitation though: That tag must not change during runtime.

By default a node view rendered by ReactNodeViewRenderer will always have a wrapping div inside. If you want to change the type of this node, you can the contentDOMElementTag to the ReactNodeViewRenderer options:

To change the wrapping DOM elements tag, you can use the as option on the ReactNodeViewRenderer function to change the default tag name.

Here is the full list of what props you can expect:

To make your node views draggable, set draggable: true in the extension and add data-drag-handle to the DOM element that should function as the drag handle.

**Examples:**

Example 1 (python):
```python
import { Node } from '@tiptap/core'
import { ReactNodeViewRenderer } from '@tiptap/react'
import Component from './Component.jsx'

export default Node.create({
  // configuration …

  addNodeView() {
    return ReactNodeViewRenderer(Component)
  },
})
```

Example 2 (python):
```python
import { Node } from '@tiptap/core'
import { ReactNodeViewRenderer } from '@tiptap/react'
import Component from './Component.jsx'

export default Node.create({
  // configuration …

  addNodeView() {
    return ReactNodeViewRenderer(Component)
  },
})
```

Example 3 (jsx):
```jsx
<NodeViewWrapper className="react-component">React Component</NodeViewWrapper>
```

Example 4 (jsx):
```jsx
<NodeViewWrapper className="react-component">React Component</NodeViewWrapper>
```

---

## Node views with Vue

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/node-views/vue

**Contents:**
- Node views with Vue
- Render a Vue component
- Access node attributes
- Update node attributes
- Adding a content editable
- All available props
  - editor
  - node
  - decorations
  - selected

Using Vanilla JavaScript can feel complex if you are used to work in Vue. Good news: You can use regular Vue components in your node views, too. There is just a little bit you need to know, but let’s go through this one by one.

Here is what you need to do to render Vue components inside your editor:

This is how your node extension could look like:

There is a little bit of magic required to make this work. But don’t worry, we provide a wrapper component you can use to get started easily. Don’t forget to add it to your custom Vue component, like shown below:

Got it? Let’s see it in action. Feel free to copy the below example to get started.

That component doesn’t interact with the editor, though. Time to wire it up.

The VueNodeViewRenderer which you use in your node extension, passes a few very helpful props to your custom Vue component. One of them is the node prop. Add this snippet to your Vue component to directly access the node:

That enables you to access node attributes in your Vue component. Let’s say you have added an attribute named count to your node extension (like we did in the above example) you could access it like this:

You can even update node attributes from your node, with the help of the updateAttributes prop passed to your component. Just add this snippet to your component:

Pass an object with updated attributes to the function:

And yes, all of that is reactive, too. A pretty seamless communication, isn’t it?

There is another component called NodeViewContent which helps you adding editable content to your node view. Here is an example:

You don’t need to add those class attributes, feel free to remove them or pass other class names. Try it out in the following example:

Keep in mind that this content is rendered by Tiptap. That means you need to tell what kind of content is allowed, for example with content: 'inline*' in your node extension (that’s what we use in the above example).

The NodeViewWrapper and NodeViewContent components render a <div> HTML tag (<span> for inline nodes), but you can change that. For example <node-view-content as="p"> should render a paragraph. One limitation though: That tag must not change during runtime.

For advanced use cases, we pass a few more props to the component.

Access the current node.

An array of decorations.

true when there is a NodeSelection at the current node view.

Access to the node extension, for example to get options.

Get the document position of the current node.

Update attributes of the current node.

Delete the current node.

Here is the full list of what props you can expect:

If you just want to have all (and TypeScript support) you can import all props:

To make your node views draggable, set draggable: true in the extension and add data-drag-handle to the DOM element that should function as the drag handle.

**Examples:**

Example 1 (python):
```python
import { Node } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-2'
import Component from './Component.vue'

export default Node.create({
  // configuration …

  addNodeView() {
    return VueNodeViewRenderer(Component)
  },
})
```

Example 2 (python):
```python
import { Node } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-2'
import Component from './Component.vue'

export default Node.create({
  // configuration …

  addNodeView() {
    return VueNodeViewRenderer(Component)
  },
})
```

Example 3 (vue):
```vue
<template>
  <node-view-wrapper> Vue Component </node-view-wrapper>
</template>
```

Example 4 (vue):
```vue
<template>
  <node-view-wrapper> Vue Component </node-view-wrapper>
</template>
```

---

## Node view examples

**URL:** https://tiptap.dev/docs/editor/extensions/custom-extensions/node-views/examples

**Contents:**
- Node view examples
- Drag handles
- Drawing in the editor
  - Was this page helpful?

Node views enable you to fully customize your nodes. We are collecting a few different examples here. Feel free to copy them and start building on them.

Keep in mind that those are just examples to get you started, not officially supported extensions. We don’t have tests for them, and don’t plan to maintain them with the same attention as we do with official extensions.

Drag handles aren’t that easy to add. We are still on the lookout what’s the best way to add them. Official support will come at some point, but there’s no timeline yet.

The drawing example shows a SVG that enables you to draw inside the editor.

It’s not working very well with the Collaboration extension. It’s sending all data on every change, which can get pretty huge with Y.js. If you plan to use those two in combination, you need to improve it or your WebSocket backend will melt.

---

## OrderedList extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/ordered-list

**Contents:**
- OrderedList extension
  - Modify backspace behavior
- Install
- Usage
- Settings
  - HTMLAttributes
  - itemTypeName
  - keepMarks
  - keepAttributes
- Commands

This extension enables you to use ordered lists in the editor. They are rendered as <ol> HTML tags.

Type 1. (or any other number followed by a dot) at the beginning of a new line and it will magically transform to a ordered list.

If you want to modify the standard behavior of backspace and delete functions for lists, you should read about the ListKeymap extension.

This extension requires the ListItem node.

This extension is installed by default with the ListKit extension, so you don’t need to install it separately.

Custom HTML attributes that should be added to the rendered HTML tag.

Specify the list item name.

Decides whether to keep the marks from a previous line after toggling the list either using inputRule or using the button

Decides whether to keep the attributes from a previous line after toggling the list either using inputRule or using the button

Toggle an ordered list.

packages/extension-list/src/ordered-list/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-list
```

Example 2 (python):
```python
npm install @tiptap/extension-list
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { OrderedList } from '@tiptap/extension-list'

new Editor({
  extensions: [OrderedList],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { OrderedList } from '@tiptap/extension-list'

new Editor({
  extensions: [OrderedList],
})
```

---

## Paragraph extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/paragraph

**Contents:**
- Paragraph extension
  - Breaking Change
- Install
- Settings
  - HTMLAttributes
- Commands
  - setParagraph()
- Keyboard shortcuts
- Source code
  - Was this page helpful?

Yes, the schema is very strict. Without this extension you won’t even be able to use paragraphs in the editor.

Tiptap v1 tried to hide that node from you, but it has always been there. You have to explicitly import it from now on (or use StarterKit).

Custom HTML attributes that should be added to the rendered HTML tag.

Transforms all selected nodes to paragraphs.

packages/extension-paragraph/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-paragraph
```

Example 2 (python):
```python
npm install @tiptap/extension-paragraph
```

Example 3 (css):
```css
Paragraph.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Paragraph.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Persistence

**URL:** https://tiptap.dev/docs/editor/core-concepts/persistence

**Contents:**
- Persistence
- Persisting the state to LocalStorage
- Persisting the state to a database
- Restoring the editor state in React
  - Was this page helpful?

After you have set up your editor, configured it and added some content you might wonder how to persist the editor. Since Tiptap is able to return HTML or JSON you can easily save the content to a database, LocalStorage or any other storage solution like sqlite or IndexedDB.

While saving HTML is possible and may be the easiest way to get renderable content, we recommend using JSON to persist the editor state as it is more flexible, easier to parse and allows for external edits if needed without running an additional HTML parser over it.

You can use the localStorage API to persist the editor state in the browser. Here's a simple example of how to save and restore the editor content using LocalStorage:

You can also get data from the localStorage when initializing the editor:

To persist the editor state to a database, you can use the same approach as with LocalStorage, but instead of using localStorage, you would send the JSON data to your backend API.

In this example we'll use the Fetch API to send the editor content to a hypothetical endpoint:

To restore the editor content from the database, you would fetch the content from your API and set it in the editor:

If you are using React, you can use the useEffect hook to restore the editor state when the component mounts. Here's an example of how to do this for the LocalStorage case:

**Examples:**

Example 1 (sql):
```sql
// Save the editor content to LocalStorage
localStorage.setItem('editorContent', JSON.stringify(editor.getJSON()))

// Restore the editor content from LocalStorage
const savedContent = localStorage.getItem('editorContent')
if (savedContent) {
  editor.setContent(JSON.parse(savedContent))
}
```

Example 2 (sql):
```sql
// Save the editor content to LocalStorage
localStorage.setItem('editorContent', JSON.stringify(editor.getJSON()))

// Restore the editor content from LocalStorage
const savedContent = localStorage.getItem('editorContent')
if (savedContent) {
  editor.setContent(JSON.parse(savedContent))
}
```

Example 3 (css):
```css
const savedContent = localStorage.getItem('editorContent')
const editor = new Editor({
  content: savedContent ? JSON.parse(savedContent) : '',
  extensions: [
    // your extensions here
  ],
})
```

Example 4 (css):
```css
const savedContent = localStorage.getItem('editorContent')
const editor = new Editor({
  content: savedContent ? JSON.parse(savedContent) : '',
  extensions: [
    // your extensions here
  ],
})
```

---

## Placeholder extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/placeholder

**Contents:**
- Placeholder extension
- Install
- Usage
  - Additional Setup
- Settings
  - emptyEditorClass
  - emptyNodeClass
  - placeholder
  - showOnlyWhenEditable
  - showOnlyCurrent

This extension provides placeholder support. Give your users an idea what they should write with a tiny hint. There is a handful of things to customize, if you feel like it.

Placeholders are displayed with the help of CSS.

Display a Placeholder only for the first line in an empty editor.

Display Placeholders on every new line.

The added CSS class if the editor is empty.

Default: 'is-editor-empty'

The added CSS class if the node is empty.

The placeholder text added as data-placeholder attribute.

Default: 'Write something …'

You can even use a function to add placeholder depending on the node:

Show decorations only when editor is editable.

Show decorations only in currently selected node.

Show decorations also for nested nodes.

packages/extensions/src/placeholder

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Placeholder } from '@tiptap/extensions'

new Editor({
  extensions: [
    Placeholder.configure({
      placeholder: 'Write something …',
    }),
  ],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Placeholder } from '@tiptap/extensions'

new Editor({
  extensions: [
    Placeholder.configure({
      placeholder: 'Write something …',
    }),
  ],
})
```

---

## Professional page-based layout

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/pages

**Contents:**
- Professional page-based layout
  - More details
  - Was this page helpful?

Transform your editing experience from a traditional single-block editor into a sophisticated page-based interface. Tiptap Pages provides proper margins, page breaks, and layout controls for creating documents that maintain a professional, page-based structure.

For detailed information on installation, configuration, and integration, please visit our Pages feature page.

---

## ProseMirror

**URL:** https://tiptap.dev/docs/editor/core-concepts/prosemirror

**Contents:**
- ProseMirror
- Install
- Integrate packages
  - Was this page helpful?

Tiptap is built on top of ProseMirror, which has a pretty powerful API. To access it, we provide the package @tiptap/pm. This package provides all important ProseMirror packages like prosemirror-state, prosemirror-view or prosemirror-model.

Using the package for custom development makes sure that you always have the same version of ProseMirror which is used by Tiptap as well. This way, we can make sure that Tiptap and all extensions are compatible with each other and prevent version clashes.

Another plus is that you don't need to install all ProseMirror packages manually, especially if you are not using npm or any other package manager that supports automatic peer dependency resolution.

After that you can access all internal ProseMirror packages like this:

The following packages are available:

You can find out more about those libraries in the ProseMirror documentation.

**Examples:**

Example 1 (python):
```python
npm i @tiptap/pm
```

Example 2 (python):
```python
npm i @tiptap/pm
```

Example 3 (go):
```go
// this example loads the EditorState class from the ProseMirror state package
import { EditorState } from '@tiptap/pm/state'
```

Example 4 (go):
```go
// this example loads the EditorState class from the ProseMirror state package
import { EditorState } from '@tiptap/pm/state'
```

---

## Selection extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/selection

**Contents:**
- Selection extension
- Install
- Settings
  - className
- Source code
- Minimal Install
  - Was this page helpful?

The Selection extension adds a CSS class to the current selection when the editor is blurred. By default it adds .selection, but you can change that.

Note that it’s only a class, the styling is totally up to you. The usage example below has some CSS for that class.

And import it in your editor:

The class that is applied to the current selection.

packages/extensions/selection/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Selection } from '@tiptap/extensions'

new Editor({
  extensions: [Selection],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { Selection } from '@tiptap/extensions'

new Editor({
  extensions: [Selection],
})
```

---

## StarterKit extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/starterkit

**Contents:**
- StarterKit extension
- Install
- Included extensions
  - Nodes
  - Marks
  - Extensions
- Source code
- Using the StarterKit extension
  - Was this page helpful?

The StarterKit is a collection of the most popular Tiptap extensions. If you’re just getting started, this extension is for you.

packages/starter-kit/

Pass StarterKit to the editor to load all included extension at once.

You can configure the included extensions, or even disable a few of them, like shown below.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/starter-kit
```

Example 2 (python):
```python
npm install @tiptap/starter-kit
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  content: '<p>Example Text</p>',
  extensions: [StarterKit],
})
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'

const editor = new Editor({
  content: '<p>Example Text</p>',
  extensions: [StarterKit],
})
```

---

## Strike extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/strike

**Contents:**
- Strike extension
  - Restrictions
- Install
- Settings
  - HTMLAttributes
- Commands
  - setStrike()
  - toggleStrike()
  - unsetStrike()
- Keyboard shortcuts

Use this extension to render striked text. If you pass <s>, <del>, <strike> tags, or text with inline style attributes setting text-decoration: line-through in the editor’s initial content, they all will be rendered accordingly.

Type ~~ text between tildes ~~ and it will be magically striked through while you type.

The extension will generate the corresponding <s> HTML tags when reading contents of the Editor instance. All text striked through, regardless of the method will be normalized to <s> HTML tags.

Custom HTML attributes that should be added to the rendered HTML tag.

Mark text as striked.

packages/extension-strike/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-strike
```

Example 2 (python):
```python
npm install @tiptap/extension-strike
```

Example 3 (css):
```css
Strike.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Strike.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Subscript extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/subscript

**Contents:**
- Subscript extension
  - Restrictions
- Install
- Settings
  - HTMLAttributes
- Commands
  - setSubscript()
  - toggleSubscript()
  - unsetSubscript()
- Keyboard shortcuts

Use this extension to render text in subscript. If you pass <sub> or text with vertical-align: sub as inline style in the editor’s initial content, both will be rendered accordingly.

The extension will generate the corresponding <sub> HTML tags when reading contents of the Editor instance. All text in subscript, regardless of the method will be normalized to <sub> HTML tags.

Custom HTML attributes that should be added to the rendered HTML tag.

Mark text as subscript.

Toggle subscript mark.

Remove subscript mark.

packages/extension-subscript/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-subscript
```

Example 2 (python):
```python
npm install @tiptap/extension-subscript
```

Example 3 (css):
```css
Subscript.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Subscript.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Superscript extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/superscript

**Contents:**
- Superscript extension
  - Restrictions
- Install
- Settings
  - HTMLAttributes
- Commands
  - setSuperscript()
  - toggleSuperscript()
  - unsetSuperscript()
- Keyboard shortcuts

Use this extension to render text in superscript. If you pass <sup> or text with vertical-align: super as inline style in the editor’s initial content, both will be rendered accordingly.

The extension will generate the corresponding <sup> HTML tags when reading contents of the Editor instance. All text in superscript, regardless of the method will be normalized to <sup> HTML tags.

Custom HTML attributes that should be added to the rendered HTML tag.

Mark text as superscript.

Toggle superscript mark.

Remove superscript mark.

packages/extension-superscript/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-superscript
```

Example 2 (python):
```python
npm install @tiptap/extension-superscript
```

Example 3 (css):
```css
Superscript.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Superscript.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## TableCell extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/table-cell

**Contents:**
- TableCell extension
- Install
- Source code
- Minimal Install
  - Was this page helpful?

Don’t try to use tables without table cells. It won’t be fun.

This extension requires the Table extension to be installed.

This extension is installed by default with the TableKit extension, so you don’t need to install it separately.

packages/extension-table-cell/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-table
```

Example 2 (python):
```python
npm install @tiptap/extension-table
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

---

## TableHeader extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/table-header

**Contents:**
- TableHeader extension
- Install
- Usage
- Source code
- Minimal Install
  - Was this page helpful?

This extension complements the Table extension and adds… you guessed it… table headers to them.

This extension requires the Table extension to be installed.

This extension is installed by default with the TableKit extension, so you don’t need to install it separately.

Table headers are optional. But come on, you want them, don’t you? If you don’t want them, update the content attribute of the TableRow extension, like this:

This is the default, which allows table headers:

packages/extension-table-header/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-table
```

Example 2 (python):
```python
npm install @tiptap/extension-table
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

---

## TableRow extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/table-row

**Contents:**
- TableRow extension
- Install
- Source code
- Minimal Install
  - Was this page helpful?

What’s a table without rows? Add this extension to make your tables usable.

This extension requires the Table extension to be installed.

This extension is installed by default with the TableKit extension, so you don’t need to install it separately.

packages/extension-table/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-table
```

Example 2 (python):
```python
npm install @tiptap/extension-table
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

---

## Table extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/table

**Contents:**
- Table extension
- Install
- Settings
  - HTMLAttributes
  - resizable
  - renderWrapper
  - handleWidth
  - cellMinWidth
  - View
  - lastColumnResizable

Nothing is as much fun as a good old HTML table. The Table extension enables you to add this holy grail of WYSIWYG editing to your editor.

Don’t forget to add a spacer.gif. (Just joking. If you don’t know what that is, don’t listen.)

This extension is installed by default with the TableKit extension, so you don’t need to install it separately.

Custom HTML attributes that should be added to the rendered HTML tag.

Controls whether a wrapper <div> is rendered around the table when the table is not resizable or the editor is not editable, maintaining layout consistency with the node view used for resizable tables.

Creates a new three-by-three table by default, including a header row. You can specify custom rows, columns, and header preferences:

Adds a column before the current column.

Adds a column after the current column.

Deletes the current column.

Adds a row above the current row.

Adds a row below the current row.

Deletes the current row.

Deletes the whole table.

Merge all selected cells to a single cell.

Splits the current cell.

Makes the current column a header column.

Makes the current row a header row.

Makes the current cell a header cell.

If multiple cells are selected, they are merged. If a single cell is selected, the cell is splitted into two cells.

Sets the given attribute for the current cell. Can be whatever you define on the TableCell extension, for example a background color. Just make sure to register your custom attribute first.

Go to the previous cell.

Inspects all tables in the document and fixes them, if necessary.

packages/extension-table/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-table
```

Example 2 (python):
```python
npm install @tiptap/extension-table
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

new Editor({
  extensions: [TableKit],
})
```

---

## Table of Contents extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/table-of-contents

**Contents:**
- Table of Contents extension
- Install
- Settings
  - anchorTypes
  - getIndex
  - getLevel
  - getId
  - scrollParent
  - onUpdate
- Storage

The TableOfContents extension lets you get a list of anchors from your document and passes on important information about each anchor (for example the depth, the content and a unique ID for each heading but also the active state and scroll states for each anchor). This can be used to render the table of content on your own.

Once done, you can install the extension from our private registry:

The types of the nodes you want to use for your Table of Content. By default this is ["heading"] but in case you create your own custom Heading extension OR extend the existing one and use a different name, you can pass that name here.

This option can be used to customize how the item indexes are calculated. By default this is using an internal function but it can be overwritten to do some custom logic.

We expose two ready to use functions - one to generate linear indexes which continue to count from 1 to n and one to generate hierarchical indexes that will count from 1 to n for each level.

This option can be used to customize how item levels are generated. By default the normal level generation is used that checks for heading element level attributes. If you want to customize this because for example you want to include custom anchors in your heading generation, you can use this to do so.

A builder function that returns a unique ID for each heading. Inside the argument you get access to the headings text content (for example you want to generate IDs based on the text content of the heading).

By default this is a function that uses the uuid package to generate a unique ID.

Default: () => uuid()

The scroll parent you want to attach to. This is used to determine which heading currently is active or was already scrolled over. By default this is a callback function that returns the window but you can pass a callback that returns any HTML element here.

Default: () => window

The most important option that you must set to use this extension. This is a callback function that gets called whenever the Table of Content updates. You get access to an array of heading data (see below) which you can use to render your own Table of Content.

To render the table of content you can render it by any means you want. You can use a framework like Vue, React or Svelte or you can use a simple templating engine like Handlebars or Pug. You can also use a simple document.createElement to render the table of content.

You can pass a second argument to get the information whether this is the initial creation step for the ToC data.

The heading content of the current document

An array of HTML nodes

The scrollHandler used by the scroll function. Should not be changed or edited but could be used to manually bind this function somewhere else

The current scrollPosition inside the scrollParent.

The array returned by the storage or the onUpdate function includes objects structured like this:

This should give you enough flexibility to render your own table of content.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-table-of-contents
```

Example 2 (python):
```python
npm install @tiptap/extension-table-of-contents
```

Example 3 (css):
```css
TableOfContents.configure({
  anchorTypes: ['heading', 'customAnchorType'],
})
```

Example 4 (css):
```css
TableOfContents.configure({
  anchorTypes: ['heading', 'customAnchorType'],
})
```

---

## TaskItem extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/task-item

**Contents:**
- TaskItem extension
- Install
- Usage
- Settings
  - HTMLAttributes
  - nested
  - onReadOnlyChecked
  - taskListTypeName
  - a11y
- Keyboard shortcuts

This extension renders a task item list element, which is a <li> tag with a data-type attribute set to taskItem. It also renders a checkbox inside the list element, which updates a checked attribute.

This extension doesn’t require any JavaScript framework, it’s based on Vanilla JavaScript.

This extension requires the TaskList node.

This extension is installed by default with the ListKit extension, so you don’t need to install it separately.

Custom HTML attributes that should be added to the rendered HTML tag.

Whether the task items are allowed to be nested within each other.

A handler for when the task item is checked or unchecked while the editor is set to readOnly. If this is not supplied, the task items are immutable while the editor is readOnly. If this function returns false, the check state will be preserved (readOnly).

The type name of the task list that this task item belongs to. This is used to determine the parent task list type.

a11y specific settings for the task item. It includes the following options:

packages/extension-list/src/task-item/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-list
```

Example 2 (python):
```python
npm install @tiptap/extension-list
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TaskItem } from '@tiptap/extension-list'

new Editor({
  extensions: [TaskItem],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TaskItem } from '@tiptap/extension-list'

new Editor({
  extensions: [TaskItem],
})
```

---

## TaskList extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/task-list

**Contents:**
- TaskList extension
- Install
- Usage
- Settings
  - HTMLAttributes
  - itemTypeName
- Commands
- toggleTaskList()
- Keyboard shortcuts
- Source code

This extension enables you to use task lists in the editor. They are rendered as <ul data-type="taskList">. This implementation doesn’t require any framework, it’s using Vanilla JavaScript only.

Type [ ] or [x] at the beginning of a new line and it will magically transform to a task list.

This extension requires the TaskItem extension.

This extension is installed by default with the ListKit extension, so you don’t need to install it separately.

Custom HTML attributes that should be added to the rendered HTML tag.

Specify the list item name.

packages/extension-list/src/task-list/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-list
```

Example 2 (python):
```python
npm install @tiptap/extension-list
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TaskList } from '@tiptap/extension-list'

new Editor({
  extensions: [TaskList],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TaskList } from '@tiptap/extension-list'

new Editor({
  extensions: [TaskList],
})
```

---

## TextAlign extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/textalign

**Contents:**
- TextAlign extension
  - Firefox bug
- Install
- Settings
  - types
  - alignments
  - defaultAlignment
- Commands
  - setTextAlign()
  - unsetTextAlign()

This extension adds a text align attribute to a specified list of nodes. The attribute is used to align the text.

text-align: justify doesn’t work together with white-space: pre-wrap in Firefox, that’s a known issue.

A list of nodes where the text align attribute should be applied to. Usually something like ['heading', 'paragraph'].

A list of available options for the text align attribute.

Default: ['left', 'center', 'right', 'justify']

The default text align.

Set the text align to the specified value.

Remove the text align value.

Toggles the text align value. If the current value is the same as the specified value, it will be removed.

packages/extension-text-align/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-align
```

Example 2 (python):
```python
npm install @tiptap/extension-text-align
```

Example 3 (css):
```css
TextAlign.configure({
  types: ['heading', 'paragraph'],
})
```

Example 4 (css):
```css
TextAlign.configure({
  types: ['heading', 'paragraph'],
})
```

---

## TextStyle extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/text-style

**Contents:**
- TextStyle extension
- Install
- Commands
  - removeEmptyTextStyle()
- Source code
  - Was this page helpful?

This mark renders a <span> HTML tag and enables you to add a list of styling related attributes, for example font-family, font-size, or color. The extension doesn’t add any styling attribute by default, but other extensions use it as the foundation, for example FontFamily or Color.

Remove <span> tags without an inline style.

packages/extension-text-style/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text-style
```

Example 2 (python):
```python
npm install @tiptap/extension-text-style
```

Example 3 (unknown):
```unknown
editor.command.removeEmptyTextStyle()
```

Example 4 (unknown):
```unknown
editor.command.removeEmptyTextStyle()
```

---

## Text extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/text

**Contents:**
- Text extension
  - Breaking Change
- Install
- Source code
  - Was this page helpful?

The Text extension is required, at least if you want to work with text of any kind and that’s very likely. This extension is a little bit different, it doesn’t even render HTML. It’s plain text, that’s all.

Tiptap v1 tried to hide that node from you, but it has always been there. You have to explicitly import it from now on (or use StarterKit).

packages/extension-text/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-text
```

Example 2 (python):
```python
npm install @tiptap/extension-text
```

---

## Tiptap Concepts

**URL:** https://tiptap.dev/docs/editor/core-concepts/introduction

**Contents:**
- Tiptap Concepts
- Structure
- State
- Content
- Extensions
- Vocabulary
  - Was this page helpful?

Explore the foundational elements of Tiptap's API, designed for intricate rich text editing based on ProseMirror's architecture.

ProseMirror works with a strict Schema, which defines the allowed structure of a document. A document is a tree of headings, paragraphs and other elements, called nodes. Marks can be attached to a node, e. g. to emphasize part of it. Commands change that document programmatically.

The document is stored in a state. Changes are applied as transactions to the state. The state has details about the current content, cursor position and selection. You can hook into events, for example to alter transactions before they get applied.

The document is stored internally as a ProseMirror node, and can be retrieved as a Tiptap JSON object calling editor.getJSON().

Tiptap JSON is the recommended format for storing the document and working with it. Below is an example Tiptap JSON document:

A Tiptap JSON document is a tree of nodes. Some nodes can have children, but only text nodes (those with type: 'text') can contain text. Text nodes and other inline nodes can have marks applied to them. Some nodes and marks can have attributes.

Extensions add nodes, marks and/or functionalities to the editor. A lot of those extensions bound their commands to common keyboard shortcuts.

ProseMirror has its own vocabulary and you’ll stumble upon all those words now and then. Here is a short overview of the most common words we use in the documentation.

**Examples:**

Example 1 (json):
```json
{
  "type": "doc",
  "content": [
    {
      "type": "paragraph",
      "attrs": {
        "textAlign": "center"
      },
      "content": [
        { "type": "text", "text": "Hello, " },
        {
          "type": "text",
          "text": "world",
          "marks": [{ "type": "bold" }, { "type": "italic" }]
        },
        { "type": "text", "text": "!" }
      ]
    }
  ]
}
```

Example 2 (json):
```json
{
  "type": "doc",
  "content": [
    {
      "type": "paragraph",
      "attrs": {
        "textAlign": "center"
      },
      "content": [
        { "type": "text", "text": "Hello, " },
        {
          "type": "text",
          "text": "world",
          "marks": [{ "type": "bold" }, { "type": "italic" }]
        },
        { "type": "text", "text": "!" }
      ]
    }
  ]
}
```

---

## Tiptap Schemas

**URL:** https://tiptap.dev/docs/editor/core-concepts/schema

**Contents:**
- Tiptap Schemas
- How a schema looks like
- Nodes and marks
  - Differences
  - The node schema
    - Content
    - Marks
    - Group
    - Inline
    - Atom

Unlike many other editors, Tiptap is based on a schema that defines how your content is structured. That enables you to define the kind of nodes that may occur in the document, its attributes and the way they can be nested.

This schema is very strict. You can’t use any HTML element or attribute that is not defined in your schema.

Let me give you one example: If you paste something like This is <strong>important</strong> into Tiptap, but don’t have any extension that handles strong tags, you’ll only see This is important – without the strong tags.

If you want to know when this happens, you can listen to the contentError event after enabling the enableContentCheck option.

When you’ll work with the provided extensions only, you don’t have to care that much about the schema. If you’re building your own extensions, it’s probably helpful to understand how the schema works. Let’s look at the most simple schema for a typical ProseMirror editor:

We register three nodes here. doc, paragraph and text. doc is the root node which allows one or more block nodes as children (content: 'block+'). Since paragraph is in the group of block nodes (group: 'block') our document can only contain paragraphs. Our paragraphs allow zero or more inline nodes as children (content: 'inline*') so there can only be text in it. parseDOM defines how a node can be parsed from pasted HTML. toDOM defines how it will be rendered in the DOM.

In Tiptap every node, mark and extension is living in its own file. This allows us to split the logic. Under the hood the whole schema will be merged together:

Nodes are like blocks of content, for example paragraphs, headings, code blocks, blockquotes and many more.

Marks can be applied to specific parts of a node. That’s the case for bold, italic or striked text. Links are marks, too.

The content attribute defines exactly what kind of content the node can have. ProseMirror is really strict with that. That means, content which doesn’t fit the schema is thrown away. It expects a name or group as a string. Here are a few examples:

You can define which marks are allowed inside of a node with the marks setting of the schema. Add a one or more names or groups of marks, allow all or disallow all marks like this:

Add this node to a group of extensions, which can be referred to in the content attribute of the schema.

Nodes can be rendered inline, too. When setting inline: true nodes are rendered in line with the text. That’s the case for mentions. The result is more like a mark, but with the functionality of a node. One difference is the resulting JSON document. Multiple marks are applied at once, inline nodes would result in a nested structure.

For some cases where you want features that aren’t available in marks, for example a node view, try if an inline node would work:

Inline nodes can be tricky to select, especially at line edges. A quick fix: add a zero-width space right after the element using CSS:

Nodes with atom: true aren’t directly editable and should be treated as a single unit. It’s not so likely to use that in a editor context, but this is how it would look like:

One example is the Mention extension, which somehow looks like text, but behaves more like a single unit. As this doesn’t have editable text content, it’s empty when you copy such node. Good news though, you can control that. Here is the example from the Mention extension:

Besides the already visible text selection, there is an invisible node selection. If you want to make your nodes selectable, you can configure it like this:

All nodes can be configured to be draggable (by default they aren’t) with this setting:

Users expect code to behave very differently. For all kind of nodes containing code, you can set code: true to take this into account.

Controls the way whitespace in this node is parsed.

Nodes get dropped when their entire content is replaced (for example, when pasting new content) by default. If a node should be kept for such replace operations, configure them as defining.

Typically, that applies to Blockquote, CodeBlock, Heading, and ListItem.

For nodes that should fence the cursor for regular editing operations like backspacing, for example a TableCell, set isolating: true.

The Gapcursor extension registers a new schema attribute to control if gap cursors are allowed everywhere in that node.

The Table extension registers a new schema attribute to configure which role an Node has. Allowed values are table, row, cell, and header_cell.

If you don’t want the mark to be active when the cursor is at its end, set inclusive to false. For example, that’s how it’s configured for Link marks:

By default all marks can be applied at the same time. With the excludes attribute you can define which marks must not coexist with the mark. For example, the inline code mark excludes any other mark (bold, italic, and all others).

By default a mark will "trap" the cursor, meaning the cursor can't get out of the mark except by moving the cursor left to right into text without a mark. If this is set to true, the mark will be exitable when the mark is at the end of a node. This is handy for example using code marks.

Add this mark to a group of extensions, which can be referred to in the content attribute of the schema.

Users expect code to behave very differently. For all kind of marks containing code, you can set code: true to take this into account.

By default marks can span multiple nodes when rendered as HTML. Set spanning: false to indicate that a mark must not span multiple nodes.

There are a few use cases where you need to work with the underlying schema. You’ll need that if you’re using the Tiptap collaborative text editing features or if you want to manually render your content as HTML.

If you need this on the client side and need an editor instance anyway, it’s available through the editor:

If you just want to have the schema without initializing an actual editor, you can use the getSchema helper function. It needs an array of available extensions and conveniently generates a ProseMirror schema for you:

To track and respond to content errors, Tiptap supports checking that the content provided matches the schema derived from the registered extensions. To use this, set the enableContentCheck option to true, which activates checking the content and emitting contentError events. These events can be listened to with the onContentError callback. By default, this flag is set to false to maintain compatibility with previous versions.

The content checking that Tiptap runs is 100% accurate on JSON content types. But, if you provide your content as HTML, we have done our best to try to alert on missing nodes but marks can be missed in certain situations, therefore, falling back to the default behavior of stripping that unrecognized content by default.

The contentError event is emitted when the initial content provided during editor setup is incompatible with the schema.

As part of the error context, you are provided with a disableCollaboration function. Invoking this function reinitializes the editor without the collaboration extension, ensuring that any removed content is not synchronized with other users.

This event can be handled either directly as an option through onContentError like:

Or, by attaching a listener to the contentError event on the editor instance.

For more implementation examples, refer to the events section.

If you want to listen to the contentError event without enabling content checking, set emitContentError to true when initializing your Tiptap editor:

This setting allows you to have invalid content in your editor, but still be notified when the content is invalid.

How you handle schema errors will be specific to your application and requirements but, here are our suggestions:

Depending on your use case, the default behavior of stripping unknown content keeps your content in a known valid state for future editing.

Depending on your use case, you may want to set the enableContentCheck flag and listen to contentError events. When this event is received, you may want to respond similarly to this example:

**Examples:**

Example 1 (json):
```json
// the underlying ProseMirror schema
{
  nodes: {
    doc: {
      content: 'block+',
    },
    paragraph: {
      content: 'inline*',
      group: 'block',
      parseDOM: [{ tag: 'p' }],
      toDOM: () => ['p', 0],
    },
    text: {
      group: 'inline',
    },
  },
}
```

Example 2 (json):
```json
// the underlying ProseMirror schema
{
  nodes: {
    doc: {
      content: 'block+',
    },
    paragraph: {
      content: 'inline*',
      group: 'block',
      parseDOM: [{ tag: 'p' }],
      toDOM: () => ['p', 0],
    },
    text: {
      group: 'inline',
    },
  },
}
```

Example 3 (sql):
```sql
// the Tiptap schema API
import { Node } from '@tiptap/core'

const Document = Node.create({
  name: 'doc',
  topNode: true,
  content: 'block+',
})

const Paragraph = Node.create({
  name: 'paragraph',
  group: 'block',
  content: 'inline*',
  parseHTML() {
    return [{ tag: 'p' }]
  },
  renderHTML({ HTMLAttributes }) {
    return ['p', HTMLAttributes, 0]
  },
})

const Text = Node.create({
  name: 'text',
  group: 'inline',
})
```

Example 4 (sql):
```sql
// the Tiptap schema API
import { Node } from '@tiptap/core'

const Document = Node.create({
  name: 'doc',
  topNode: true,
  content: 'block+',
})

const Paragraph = Node.create({
  name: 'paragraph',
  group: 'block',
  content: 'inline*',
  parseHTML() {
    return [{ tag: 'p' }]
  },
  renderHTML({ HTMLAttributes }) {
    return ['p', HTMLAttributes, 0]
  },
})

const Text = Node.create({
  name: 'text',
  group: 'inline',
})
```

---

## Trailing Node extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/trailing-node

**Contents:**
- Trailing Node extension
- Install
- Usage
- Settings
  - node
  - notAfter
- Source code
- Minimal Install
  - Was this page helpful?

This extension adds a node after the last block node in the editor. This can be useful for adding a trailing node like a placeholder or a button.

The node type that should be inserted at the end of the document. When you leave it unset, Trailing Node asks the ProseMirror schema for whatever fallback node it defines (usually the default block node), but you can override that behavior with the node option. This is particularly relevant when you define a custom document structure - see the Forced content structure example for how custom documents may affect the assumed fallback node.

The node types after which the trailing node should not be inserted.

Default: ['paragraph']

packages/extensions/src/trailing-node/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TrailingNode } from '@tiptap/extensions'

new Editor({
  extensions: [TrailingNode],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TrailingNode } from '@tiptap/extensions'

new Editor({
  extensions: [TrailingNode],
})
```

---

## Twitch extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/twitch

**Contents:**
- Twitch extension
- Supported Content Types
- Install
- Settings
  - inline
  - width
  - height
  - allowFullscreen
  - autoplay
  - muted

This extension adds support for embedding Twitch videos, clips, and live channels into your editor.

Controls if the node should be handled inline or as a block.

Controls the default width of added videos.

Controls the default height of added videos.

Allows the iframe to be played in fullscreen.

Allows the iframe to start playing after the player is loaded.

Specifies whether the initial state of the video is muted. This is useful when autoplay is enabled, as most browsers require videos to be muted for autoplay to work.

The time in the video where playback starts. Only applicable for video embeds, not for clips or channels. Format: 1h2m3s (hours, minutes, seconds).

Specifies the domain that is embedding the Twitch player. This is required for the Twitch embed to function properly. You should set this to your domain.

Controls if the paste handler for Twitch video URLs should be added. When enabled, pasting a Twitch video URL will automatically create an embed.

Pass custom HTML attributes to the iframe element.

Inserts a Twitch video iframe embed at the current position.

You can override the extension options for individual embeds by specifying attributes when inserting the video. This allows you to have different settings for different embeds, while using the extension options as defaults:

The extension supports the following Twitch URL formats:

When addPasteHandler is enabled (default), you can simply paste a Twitch video URL into the editor, and it will automatically be converted to an embed.

packages/extension-twitch/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-twitch
```

Example 2 (python):
```python
npm install @tiptap/extension-twitch
```

Example 3 (css):
```css
Twitch.configure({
  inline: false,
})
```

Example 4 (css):
```css
Twitch.configure({
  inline: false,
})
```

---

## Typography extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/typography

**Contents:**
- Typography extension
- Install
- Rules
- Keyboard shortcuts
- Source code
  - Disabling rules
  - Overriding rules
  - Was this page helpful?

This extension tries to help with common text patterns with the correct typographic character. Under the hood all rules are input rules.

packages/extension-typography/

You can configure the included rules, or even disable a few of them, like shown below.

You can override the output of a rule by passing a string to the option you want to override.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-typography
```

Example 2 (python):
```python
npm install @tiptap/extension-typography
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import Typography from '@tiptap/extension-typography'

const editor = new Editor({
  extensions: [
    // Disable some included rules
    Typography.configure({
      oneHalf: false,
      oneQuarter: false,
      threeQuarters: false,
    }),
  ],
})
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import Typography from '@tiptap/extension-typography'

const editor = new Editor({
  extensions: [
    // Disable some included rules
    Typography.configure({
      oneHalf: false,
      oneQuarter: false,
      threeQuarters: false,
    }),
  ],
})
```

---

## Underline extension

**URL:** https://tiptap.dev/docs/editor/extensions/marks/underline

**Contents:**
- Underline extension
  - Restrictions
- Install
- Settings
  - HTMLAttributes
- Commands
  - setUnderline()
  - toggleUnderline()
  - unsetUnderline()
- Keyboard shortcuts

Use this extension to render text underlined. If you pass <u> tags, or text with inline style attributes setting text-decoration: underline in the editor’s initial content, they all will be rendered accordingly.

Be aware that underlined text in the internet usually indicates that it’s a clickable link. Don’t confuse your users with underlined text.

The extension will generate the corresponding <u> HTML tags when reading contents of the Editor instance. All text marked underlined, regardless of the method will be normalized to <u> HTML tags.

Custom HTML attributes that should be added to the rendered HTML tag.

Marks a text as underlined.

Toggles an underline mark.

Removes an underline mark.

packages/extension-underline/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-underline
```

Example 2 (python):
```python
npm install @tiptap/extension-underline
```

Example 3 (css):
```css
Underline.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

Example 4 (css):
```css
Underline.configure({
  HTMLAttributes: {
    class: 'my-custom-class',
  },
})
```

---

## Undo/Redo extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/undo-redo

**Contents:**
- Undo/Redo extension
- Install
- Usage
- Settings
  - depth
  - newGroupDelay
- Commands
  - undo()
  - redo()
- Keyboard shortcuts

This extension provides undo and redo support. All changes to the document will be tracked and can be removed with undo. Undone changes can be applied with redo again.

You should only integrate this extension if you don't plan to make your editor collaborative. The Collaboration extension has its own undo/redo support because people generally don't want to revert changes made by others.

The amount of history events that are collected before the oldest events are discarded.

The delay between changes after which a new group should be started (in milliseconds). When changes aren’t adjacent, a new group is always started.

Undo the last change.

Redo the last change.

packages/extensions/src/undo-redo/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extensions
```

Example 2 (python):
```python
npm install @tiptap/extensions
```

Example 3 (sql):
```sql
import { Editor } from '@tiptap/core'
import { UndoRedo } from '@tiptap/extensions'

new Editor({
  extensions: [UndoRedo],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { UndoRedo } from '@tiptap/extensions'

new Editor({
  extensions: [UndoRedo],
})
```

---

## UniqueID extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/uniqueid

**Contents:**
- UniqueID extension
- Install
- Settings
  - attributeName
  - types
  - generateID
  - filterTransaction
  - updateDocument
- Server side Unique ID utility
  - Parameters

The UniqueID extension adds unique IDs to all nodes. The extension keeps track of your nodes, even if you split them, merge them, undo/redo changes, crop content, paste content … It just works. Also, you can configure which node types get an unique ID, and which not, and you can customize how those IDs are generated.

Name of the attribute that is attached to the HTML tag (will be prefixed with data-).

All types that should get a unique ID, for example ['heading', 'paragraph']

Function that generates and returns a unique ID. It receives a context object (for example { node, pos }), so you can customize ID generation based on the node's type or its position.

Default: ({ node, pos }) => uuidv4()

Ignore some mutations, for example applied from other users through the collaboration plugin.

Whether to update the document by adding unique IDs to the nodes. Set this property to false if the document is in readonly mode, is immutable, or you don't want it to be modified.

The generateUniqueIds function allows you to add unique IDs to a Tiptap document on the server side, without needing to create an Editor instance. This is useful for processing documents server-side or when you need to add IDs to existing content.

Returns a new Tiptap document (a JSONContent object) with unique IDs added to the nodes.

The function automatically picks up the configuration from the UniqueID extension, including options like types, attributeName, and generateID.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-unique-id
```

Example 2 (python):
```python
npm install @tiptap/extension-unique-id
```

Example 3 (css):
```css
UniqueID.configure({
  attributeName: 'uid',
})
```

Example 4 (css):
```css
UniqueID.configure({
  attributeName: 'uid',
})
```

---

## Webhooks in Collaboration

**URL:** https://tiptap.dev/docs/collaboration/core-concepts/webhooks

**Contents:**
- Webhooks in Collaboration
- Configure Webhooks
  - Add Comments support to your webhook
- Example payload
- Retries
- Enable the Comments webhook
- Loader Webhook
- Awareness Webhooks
  - Was this page helpful?

You can define a URL and we will call it every time a document has changed. This is useful for getting the JSON representation of the Yjs document in your own application.

We call your webhook URL when the document is saved to our database. This operation is debounced by 2-10 seconds. So your application won't be flooded by us. Right now we're only exporting the fragment default of the Yjs document.

To configure webhooks for document and comments notifications:

After adding your URL, the webhook is immediately live. You'll start receiving notifications for the specified events without any delay.

If you want to add webhook support for the comments feature and your Document server was created before March 2024, please upgrade your webhook as described below.

All requests to your webhook URL will contain a header called X-Hocuspocus-Signature-256 that signs the entire message with your secret. You can find it in the settings of your Tiptap Collab app.

Webhooks are not retried by default, but you can enable retries by setting webhook_retries to 1 (see Configure Runtime). The retry schedule is as follows:

All retries include a header X-Hocuspocus-Retry with the current retry count. The time property in the payload is the timestamp of the initial attempt.

The webhook that supports comments is automatically enabled for all users that have created their account after March, 2024.

If your account was created before March, 2024 and you're using an older version of the webhook system, you'll need to manually enable the new comments webhooks. Here's how:

This upgrade is necessary to accommodate the introduction of multiple new events being routed to the same webhook endpoint, distinguished by a new type and trigger field.

If you do not wish to use the comments webhook, no upgrade is necessary.

In order to initialize documents, you can use the webhook_loader_url setting (see configure runtime). This URL will be called if a new document is requested. The webhook will contain a header Authorization with your secret, and document-name with the name of the requested document.

If you return a yjs update (Y.encodeStateAsUpdate on your side), it will be applied to the document. You can also return Tiptap JSON, if you send Content-Type: application/json (from January 2026 / v3.67.0). If you return anything else, the document will be initialized with an empty document. Note that the loader webhook is called only once when the document is created.

The request looks like this:

If you want to get notified whenever a user connects to or disconnects from a document, you can enable awareness webhooks here. If you need the user parameter, please make sure to pass it to the TiptapCollabProvider, as mentioned here.

The events look like this:

**Examples:**

Example 1 (json):
```json
{
  "appName": '', // name of your app
  "name": '', // name of the document (URI encoded if necessary)
  "time": // current time as ISOString (new Date()).toISOString())
  "tiptapJson": {}, // JSON output from Tiptap (see https://tiptap.dev/guide/output#option-1-json): TiptapTransformer.fromYdoc()
  "ydocState"?: {}, // optionally contains the entire yDoc as base64. This can be enabled in the runtime configuration (see https://tiptap.dev/docs/collaboration/operations/configure `webhook_include_ydoc_state`)
  "clientsCount": 100,// number of currently connected clients
  "type": '', // the payload type (if the document was changed, this is DOCUMENT) ; only available if you are on webhooks v2
  "trigger": '', // what triggered the event (usually "document.saved") ; only available if you are on webhooks v2
  "users": [] // list of users who changed the content since the last webhook ("sub" field from the JWT)
}
```

Example 2 (json):
```json
{
  "appName": '', // name of your app
  "name": '', // name of the document (URI encoded if necessary)
  "time": // current time as ISOString (new Date()).toISOString())
  "tiptapJson": {}, // JSON output from Tiptap (see https://tiptap.dev/guide/output#option-1-json): TiptapTransformer.fromYdoc()
  "ydocState"?: {}, // optionally contains the entire yDoc as base64. This can be enabled in the runtime configuration (see https://tiptap.dev/docs/collaboration/operations/configure `webhook_include_ydoc_state`)
  "clientsCount": 100,// number of currently connected clients
  "type": '', // the payload type (if the document was changed, this is DOCUMENT) ; only available if you are on webhooks v2
  "trigger": '', // what triggered the event (usually "document.saved") ; only available if you are on webhooks v2
  "users": [] // list of users who changed the content since the last webhook ("sub" field from the JWT)
}
```

Example 3 (yaml):
```yaml
GET {{webhook_loader_url}}

Authorization: {{jwt secret}}
document-name: {{requested document name}}
```

Example 4 (yaml):
```yaml
GET {{webhook_loader_url}}

Authorization: {{jwt secret}}
document-name: {{requested document name}}
```

---

## Youtube extension

**URL:** https://tiptap.dev/docs/editor/extensions/nodes/youtube

**Contents:**
- Youtube extension
- Install
- Settings
  - inline
  - width
  - height
  - controls
  - nocookie
  - allowFullscreen
  - autoplay

This extension adds a new YouTube embed node to the editor.

Controls if the node should be handled inline or as a block.

Controls the default width of added videos

Controls the default height of added videos

Enables or disables YouTube video controls

Enables the nocookie mode for YouTube embeds

Allows the iframe to be played in fullscreen

Allows the iframe to start playing after the player is loaded

Specifies the default language that the player will use to display closed captions. Set the parameter’s value to an ISO 639-1 two-letter language code. For example, setting it to es will cause the captions to be in spanish

Setting this parameter’s value to true causes closed captions to be shown by default, even if the user has turned captions off

Disables the keyboards controls for the iframe player

Enables the player to be controlled via IFrame Player API calls

This parameter provides an extra security measure for the IFrame API and is only supported for IFrame embeds. If you are using the IFrame API, which means you are setting the enableIFrameApi parameter value to true, you should always specify your domain as the origin parameter value.

This parameter specifies the time, measured in seconds from the start of the video, when the player should stop playing the video. For example, setting it to 15 will make the video stop at the 15 seconds mark

Sets the player’s interface language. The parameter value is an ISO 639-1 two-letter language code. For example, setting it to fr will cause the interface to be in french

Setting this to 1 causes video annotations to be shown by default, whereas setting to 3 causes video annotations to not be shown by default

This parameter has limited support in IFrame embeds. To loop a single video, set the loop parameter value to true and set the playlist parameter value to the same video ID already specified in the Player API URL.

This parameter specifies a comma-separated list of video IDs to play.

Disables the Youtube logo on the control bar of the player. Note that a small YouTube text label will still display in the upper-right corner of a paused video when the user's mouse pointer hovers over the player

This parameter specifies the color that will be used in the player's video progress bar. Note that setting the color parameter to white will disable the modestBranding parameter

Inserts a YouTube iframe embed at the current position

packages/extension-youtube/

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-youtube
```

Example 2 (python):
```python
npm install @tiptap/extension-youtube
```

Example 3 (css):
```css
Youtube.configure({
  inline: false,
})
```

Example 4 (css):
```css
Youtube.configure({
  inline: false,
})
```

---
