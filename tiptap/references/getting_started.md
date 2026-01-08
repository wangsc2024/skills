# Tiptap - Getting Started

**Pages:** 35

---

## Alpine

**URL:** https://tiptap.dev/docs/editor/getting-started/install/alpine

**Contents:**
- Alpine
  - Requirements
- Create a project (optional)
  - Install the dependencies
- Integrate Tiptap
  - Add it to your app
- Next steps
  - Was this page helpful?

The following guide describes how to integrate Tiptap with version 3 of Alpine.js. For the sake of this guide, we'll use Vite to quickly set up a project, but you can use whatever you're used to. Vite is just really fast and we love it!

If you already have an Alpine.js project, that's fine too. Just skip this step.

For the purpose of this guide, start with a fresh Vite project called my-tiptap-project. Vite sets up everything we need, just select the Vanilla JavaScript template.

Okay, enough of the boring boilerplate work. Let's finally install Tiptap! For the following example, you'll need alpinejs, the @tiptap/core package, the @tiptap/pm package, and the @tiptap/starter-kit, which includes the most common extensions to get started quickly.

If you followed step 1, you can now start your project with npm run dev, and open http://localhost:5173 in your favorite browser. This might be different if you're working with an existing project.

To actually start using Tiptap, you'll need to write a little bit of JavaScript. Let's put the following example code in a file called main.js.

This is the fastest way to get Tiptap up and running with Alpine.js. It will give you a very basic version of Tiptap. No worries, you will be able to add more functionality soon.

Now, let's replace the contents of index.html with the following example code to use the editor in our app.

Tiptap should now be visible in your browser. Time to give yourself a pat on the back! :)

**Examples:**

Example 1 (python):
```python
npm init vite@latest my-tiptap-project -- --template vanilla
cd my-tiptap-project
npm install
npm run dev
```

Example 2 (python):
```python
npm init vite@latest my-tiptap-project -- --template vanilla
cd my-tiptap-project
npm install
npm run dev
```

Example 3 (python):
```python
npm install alpinejs @tiptap/core @tiptap/pm @tiptap/starter-kit
```

Example 4 (python):
```python
npm install alpinejs @tiptap/core @tiptap/pm @tiptap/starter-kit
```

---

## Authenticate and authorize in Collaboration

**URL:** https://tiptap.dev/docs/collaboration/getting-started/authenticate

**Contents:**
- Authenticate and authorize in Collaboration
  - Need help with JWT?
- Set up authorization
  - Caution
  - Allow full access to every document
  - Limit access to specific documents
  - Block access to all documents
- Set Read-Only access
- Allow commenting while on read-only access
- Authorize with Wildcards

After setting up a collaborative editor in the installation guide, it's crucial to address authentication for longer-term use. The temporary JWT provided in your Tiptap account is only suitable for brief testing sessions.

If you need assistance with setting up server-side JWT authentication, you can find guidance at the bottom of the page.

Setting up the right access controls is important for keeping your documents secure and workflows smooth in Tiptap Collaboration.

This part of the guide walks you through how to use JSON Web Tokens (JWTs) to fine-tune who gets to see and edit what. Whether you need to give someone full access, restrict them to certain documents, or block access entirely, we've got you covered with minimalistic examples.

If you exclude the allowedDocumentNames property from your JWT setup, users can access all documents in your system!

Omitting the allowedDocumentNames property from the JWT payload grants the user access to all documents. This is useful for users who need unrestricted access.

To restrict a user's access to specific documents, include those document names in the allowedDocumentNames array within the JWT payload. This ensures the user can only access the listed documents.

To prohibit a user from accessing any documents, provide an empty array for allowedDocumentNames in the JWT payload. This effectively blocks access to all documents, except if granted using readonlyDocumentNames.

The readonlyDocumentNames property in your JWT setup plays a crucial role when you need to allow users to view documents without the ability to edit them. This feature is particularly useful in scenarios where you want to share information with team members for review or reference purposes but need to maintain the integrity of the original document.

By specifying document names in the readonlyDocumentNames array, you grant users read-only access to those documents. Users can open and read the documents, but any attempts to modify the content will be restricted. This ensures that sensitive or critical information remains unchanged while still being accessible for necessary personnel.

In this example, we grant read-only access to two documents, annual-report-2024 and policy-document-v3. Users with this JWT can view these documents but cannot make any edits.

Incorporating the readonlyDocumentNames property into your JWT strategy improves document security by ensuring that only authorized edits are made, preserving the integrity of your critical documents.

If you want to forbid editing the document but still allow comments, you can add commentDocumentNames to the JWT.

Wildcards in JWTs offer a dynamic way to manage document access, allowing for broader permissions within specific criteria without listing each document individually. This method is particularly useful in scenarios where documents are grouped by certain attributes, such as projects or teams.

For teams working on multiple projects, it's essential to ensure that members have access only to the documents relevant to their current projects. By using project identifiers with wildcards, you can streamline access management.

In this example, users will have access to all documents under 'project-alpha' and 'project-beta', making it easier to manage permissions as new documents are added to these projects.

You may want to limit the lifetime of your JWTs by setting an expiration time. JWTs are validated every few seconds, so an expired token will be rejected shortly after expiration and won't be able to reconnect.

Needless to say, with expiring tokens it's essential to handle the expiration case - both mid-session, but also on initial connect or reconnect. Your users may leave tabs open for weeks, and you don't want them to lose data because of an expired token (see also unsynced Changes).

If the JWT is not valid when creating the connection (or reconnecting), the onAuthenticationFailed callback is triggered (reason permission-denied) If the JWT expired during an established connection, the onClose callback is triggered with a reason JWT verification failed.

In both cases, you need to re-create the provider and supply a new JWT.

Tiptap Collab provides an API to revoke JWTs in case you need to, so we don't recommend making your tokens too short-lived. (see Revoke JWT API)

JWT, or JSON Web Token, is a compact, URL-safe means of representing claims to be transferred between two parties. The information in a JWT is digitally signed using a cryptographic algorithm to ensure that the claims cannot be altered after the token is issued. This digital signature makes the JWT a reliable vehicle for secure information exchange in web applications, providing a method to authenticate and exchange information.

For testing purposes, you might not want to set up a complete backend system to generate JWTs. In such cases, using online tools like http://jwtbuilder.jamiekurtz.com/ can be a quick workaround. These tools allow you to create a JWT by inputting the necessary payload and signing it with a secret key.

When using these tools, ensure that the "Key" field is replaced with the secret key from your Collaboration settings page. You don’t need to change any other information.

Remember, this approach is only recommended for testing due to security risks associated with exposing your secret key.

For production-level applications, generating JWTs on the server side is a necessity to maintain security. Exposing your secret key in client-side code would compromise the security of your application. Here’s an example using NodeJS for creating JWTs server-side:

This JWT should be incorporated into API requests within the token field of your authentication provider, safeguarding user sessions and data access.

To fully integrate JWT into your application, consider setting up a dedicated server or API endpoint, such as GET /getCollabToken. This endpoint would dynamically generate JWTs based on a secret stored securely on the server and user-specific information, like document access permissions.

This setup not only increases security but also provides a scalable solution for managing user sessions and permissions in your collaborative application.

Ensure the secret key is stored as an environment variable on the server, or define it directly in the server code. Avoid sending it from the client side.

A full server / API example is available here.

**Examples:**

Example 1 (sql):
```sql
import jsonwebtoken from 'jsonwebtoken'

const data = { sub: 'your_local_user_identifier' }

const jwt = jsonwebtoken.sign(data, 'your_secret')
```

Example 2 (sql):
```sql
import jsonwebtoken from 'jsonwebtoken'

const data = { sub: 'your_local_user_identifier' }

const jwt = jsonwebtoken.sign(data, 'your_secret')
```

Example 3 (sql):
```sql
import jsonwebtoken from 'jsonwebtoken'

const data = {
  sub: 'your_local_user_identifier',
  allowedDocumentNames: ['user-specific-document-1', 'user-specific-document-2'],
}

const jwt = jsonwebtoken.sign(data, 'your_secret')
```

Example 4 (sql):
```sql
import jsonwebtoken from 'jsonwebtoken'

const data = {
  sub: 'your_local_user_identifier',
  allowedDocumentNames: ['user-specific-document-1', 'user-specific-document-2'],
}

const jwt = jsonwebtoken.sign(data, 'your_secret')
```

---

## Authenticate your conversion service

**URL:** https://tiptap.dev/docs/conversion/getting-started/install

**Contents:**
- Authenticate your conversion service
- Set up authorization
  - Export DOCX Extension
- Explore file-type integrations
  - Was this page helpful?

Tiptap Conversion lets you import and export Tiptap JSON to and from DOCX, ODT, and Markdown. You can integrate import/export directly in your Tiptap editor with dedicated extensions, or use the Conversion REST API on your server.

Most conversion operations require authentication. Generate a JWT (JSON Web Token) using the secret key from your Tiptap Cloud account. Include this JWT in requests or extension configs.

The extension-export-docx package does not require authentication! Feel free to skip these steps if you only need that extension.

Depending on which format you want to work with, each file type has its own guide to installing and configuring the relevant import and export extensions:

These guides explain exactly how to integrate the respective extension and REST API endpoints into your application and configure any necessary options.

---

## Basic Usage

**URL:** https://tiptap.dev/docs/editor/markdown/getting-started/basic-usage

**Contents:**
- Basic Usage
- Getting Markdown from the Editor
- Setting Content from Markdown
- Using the MarkdownManager Directly
- GitHub Flavored Markdown (GFM)
- Inline Formatting
- Working with Block Elements
  - Headings
  - Lists
  - Code Blocks

This guide covers the core operations for working with Markdown: parsing Markdown into your editor and serializing editor content back to Markdown.

Use getMarkdown() to serialize your editor content to Markdown:

All content commands support the contentType option:

For more control, access the MarkdownManager via editor.markdown:

This is useful when working with JSON content outside the editor context.

Enable GFM for features like tables and task lists:

Standard Markdown formatting works automatically:

Block elements like headings, lists, and code blocks work as expected:

The Markdown extension can parse HTML embedded in Markdown using Tiptap's existing parseHTML methods:

The HTML is parsed according to your extensions' parseHTML rules, allowing you to support custom HTML nodes.

Always use contentType and set it to markdown when setting Markdown content (otherwise it's treated as HTML):

Include all needed extensions or content may be lost:

Test round-trip conversion to ensure your custom Markdown content survives parse → serialize:

The MarkdownManager class is the core engine that handles parsing and serialization. It:

The Markdown extension is the main extension that you add to your editor. It provides:

Each Tiptap extension can provide Markdown support by configuring the extension:

The handlers translate between Markdown tokens and Tiptap nodes in both directions and are automatically registered by the MarkdownManager, creating Tokenizers out of them and registering those to the Lexer.

**Examples:**

Example 1 (javascript):
```javascript
const markdown = editor.getMarkdown()
console.log(markdown)
// # Hello
//
// This is a **test**.
```

Example 2 (javascript):
```javascript
const markdown = editor.getMarkdown()
console.log(markdown)
// # Hello
//
// This is a **test**.
```

Example 3 (css):
```css
// 1. Initial content
const editor = new Editor({
  extensions: [StarterKit, Markdown],
  content: '# Hello World\n\nThis is **markdown**!',
  contentType: 'markdown',
})

// 2. Replace all content
editor.commands.setContent('# New Content', { contentType: 'markdown' })

// 3. Insert at cursor
editor.commands.insertContent('**Bold** text', { contentType: 'markdown' })

// 4. Insert at specific position
editor.commands.insertContentAt(10, '## Heading', { contentType: 'markdown' })

// 5. Replace a range
editor.commands.insertContentAt({ from: 10, to: 20 }, '**Replace**', { contentType: 'markdown' })
```

Example 4 (css):
```css
// 1. Initial content
const editor = new Editor({
  extensions: [StarterKit, Markdown],
  content: '# Hello World\n\nThis is **markdown**!',
  contentType: 'markdown',
})

// 2. Replace all content
editor.commands.setContent('# New Content', { contentType: 'markdown' })

// 3. Insert at cursor
editor.commands.insertContent('**Bold** text', { contentType: 'markdown' })

// 4. Insert at specific position
editor.commands.insertContentAt(10, '## Heading', { contentType: 'markdown' })

// 5. Replace a range
editor.commands.insertContentAt({ from: 10, to: 20 }, '**Replace**', { contentType: 'markdown' })
```

---

## CDN

**URL:** https://tiptap.dev/docs/editor/getting-started/install/cdn

**Contents:**
- CDN
- Next steps
  - Was this page helpful?

For testing purposes or demos, use our esm.sh CDN builds. Here are a few lines of code you need to get started.

Tiptap should now be visible in your browser. Time to give yourself a pat on the back! :)

**Examples:**

Example 1 (html):
```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
  </head>
  <body>
    <div class="element"></div>
    <script type="module">
      import { Editor } from 'https://esm.sh/@tiptap/core'
      import StarterKit from 'https://esm.sh/@tiptap/starter-kit'
      const editor = new Editor({
        element: document.querySelector('.element'),
        extensions: [StarterKit],
        content: '<p>Hello World!</p>',
      })
    </script>
  </body>
</html>
```

Example 2 (html):
```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
  </head>
  <body>
    <div class="element"></div>
    <script type="module">
      import { Editor } from 'https://esm.sh/@tiptap/core'
      import StarterKit from 'https://esm.sh/@tiptap/starter-kit'
      const editor = new Editor({
        element: document.querySelector('.element'),
        extensions: [StarterKit],
        content: '<p>Hello World!</p>',
      })
    </script>
  </body>
</html>
```

---

## Collaboration extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/collaboration

**Contents:**
- Collaboration extension
- Install
  - More details
- Settings
  - document
  - field
  - fragment
- Commands
  - undo()
  - redo()

This small guide quickly shows how to integrate basic collaboration functionality into your editor. For a proper collaboration integration, review the documentation of Tiptap Collaboration, which is a cloud and on-premises collaboration server solution.

For more detailed information on how to integrate, install, and configure the Collaboration extension with the Tiptap Collaboration product, please visit our feature page.

An initialized Y.js document.

Name of a Y.js fragment, can be changed to sync multiple fields with one Y.js document.

A raw Y.js fragment, can be used instead of document and field.

The Collaboration extension comes with its own history extension. Make sure to disable the UndoRedo extension, if you’re working with the StarterKit.

Undo the last change.

Redo the last change.

packages/extension-collaboration/

Your editor is now collaborative! Invite your friends and start typing together 🙌🏻 If you want to continue building out your collaborative editing features, make sure to check out the Tiptap Collaboration Docs for a fully hosted on on-premises collaboration server solution.

Fasten your seatbelts! Make your rich text editor collaborative with Tiptap Collaboration.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-websocket
```

Example 2 (python):
```python
npm install @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-websocket
```

Example 3 (css):
```css
Collaboration.configure({
  document: new Y.Doc(),
})
```

Example 4 (css):
```css
Collaboration.configure({
  document: new Y.Doc(),
})
```

---

## Configure the Editor

**URL:** https://tiptap.dev/docs/editor/getting-started/configure

**Contents:**
- Configure the Editor
- Add your configuration
- Nodes, marks, and extensions
  - Configure extensions
  - A bundle with the most common extensions
  - Disable specific StarterKit extensions
  - Additional extensions
- Next steps
  - Was this page helpful?

To configure Tiptap, specify three key elements:

While this setup works for most cases, you can configure additional options.

To configure the editor, pass an object with settings to the Editor class, as shown below:

Most editing features are packaged as nodes, marks, or functionality. Import what you need and pass them as an array to the editor.

Here's the minimal setup with only three extensions:

Many extensions can be configured with the .configure() method. You can pass an object with specific settings.

For example, to limit the heading levels to 1, 2, and 3, configure the Heading extension as shown below:

Refer to the extension's documentation for available settings.

We have bundled a few of the most common extensions into the StarterKit. Here's how to use it:

You can configure all extensions included in the StarterKit by passing an object. To target specific extensions, prefix their configuration with the name of the extension. For example, to limit heading levels to 1, 2, and 3:

To exclude certain extensions StarterKit, you can set them to false in the configuration. For example, to disable the Undo/Redo History extension:

When using Tiptap's Collaboration, which comes with its own history extension, you must disable the Undo/Redo History extension included in the StarterKit to avoid conflicts.

The StarterKit doesn't include all available extensions. To add more features to your editor, list them in the extensions array. For example, to add the Strike extension:

**Examples:**

Example 1 (typescript):
```typescript
import { Editor } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Text from '@tiptap/extension-text'

new Editor({
  // bind Tiptap to the `.element`
  element: document.querySelector('.element'),
  // register extensions
  extensions: [Document, Paragraph, Text],
  // set the initial content
  content: '<p>Example Text</p>',
  // place the cursor in the editor after initialization
  autofocus: true,
  // make the text editable (default is true)
  editable: true,
  // prevent loading the default ProseMirror CSS that comes with Tiptap
  // should be kept as `true` for most cases as it includes styles
  // important for Tiptap to work correctly
  injectCSS: false,
})
```

Example 2 (typescript):
```typescript
import { Editor } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Text from '@tiptap/extension-text'

new Editor({
  // bind Tiptap to the `.element`
  element: document.querySelector('.element'),
  // register extensions
  extensions: [Document, Paragraph, Text],
  // set the initial content
  content: '<p>Example Text</p>',
  // place the cursor in the editor after initialization
  autofocus: true,
  // make the text editable (default is true)
  editable: true,
  // prevent loading the default ProseMirror CSS that comes with Tiptap
  // should be kept as `true` for most cases as it includes styles
  // important for Tiptap to work correctly
  injectCSS: false,
})
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Text from '@tiptap/extension-text'

new Editor({
  element: document.querySelector('.element'),
  extensions: [Document, Paragraph, Text],
})
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Text from '@tiptap/extension-text'

new Editor({
  element: document.querySelector('.element'),
  extensions: [Document, Paragraph, Text],
})
```

---

## Convert ODT with Tiptap

**URL:** https://tiptap.dev/docs/conversion/import-export/odt/editor-extensions

**Contents:**
- Convert ODT with Tiptap
- Editor ODT Import
  - Configure the extension
  - Import your first document
  - Customize the import behavior
  - Options
  - Commands
  - import
    - Arguments
- Editor ODT Export

OpenDocument Text .odt is a widely used format in LibreOffice and OpenOffice. Tiptap’s Conversion tools provide three ways to work with ODT files:

The Conversion extensions are published in Tiptap’s private npm registry. Integrate the extensions by following the private registry guide.

To import .odt documents into your editor install the Import extension

Add and configure the extension in your editor setup

To use the convert extension, you need to install the @tiptap-pro/extension-export package:

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
// Start with importing the extension
import { Import } from '@tiptap-pro/extension-import'

const editor = new Editor({
  // ...
  extensions: [
    // ...
    Import.configure({
      // The Convert App-ID from the Convert settings page: https://cloud.tiptap.dev/convert-settings
      appId: 'your-app-id',

      // The JWT token you generated in the previous step
      token: 'your-jwt',

      // The URL to upload images to, if not provided, images will be stripped from the document
      imageUploadCallbackUrl: 'https://your-image-upload-url.com',
    }),
  ],
})
```

Example 4 (sql):
```sql
// Start with importing the extension
import { Import } from '@tiptap-pro/extension-import'

const editor = new Editor({
  // ...
  extensions: [
    // ...
    Import.configure({
      // The Convert App-ID from the Convert settings page: https://cloud.tiptap.dev/convert-settings
      appId: 'your-app-id',

      // The JWT token you generated in the previous step
      token: 'your-jwt',

      // The URL to upload images to, if not provided, images will be stripped from the document
      imageUploadCallbackUrl: 'https://your-image-upload-url.com',
    }),
  ],
})
```

---

## Don’t bend it, extend it.

**URL:** https://tiptap.dev/docs/editor/extensions/overview

**Contents:**
- Don’t bend it, extend it.
  - Was this page helpful?

Our editor does what you would expect from a modern editor, and probably way more than that. With tons of extensions there’s a lot to explore for you.

---

## Export

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/export

**Contents:**
- Export
  - More details
  - Was this page helpful?

Export Tiptap's editor content to various formats like docx, odt, and markdown.

For more detailed information on how to integrate, install, and configure the Conversion feature, please visit our feature page.

---

## How to develop a custom menu

**URL:** https://tiptap.dev/docs/editor/getting-started/style-editor/custom-menus

**Contents:**
- How to develop a custom menu
- Menus
  - Fixed menu
  - Bubble menu
  - Floating menu
  - Slash commands (work in progress)
- Buttons
  - Commands
  - Keep the focus
  - The active state

Tiptap comes very raw, but that's a good thing. You have full control over the editor's appearance.

When we say full control, we mean it. You can (and have to) build a menu on your own, although Tiptap will help you wire everything up.

The editor provides a fluent API to trigger commands and add active states. You can use any markup you like. To simplify menu positioning, Tiptap provides a few utilities and components. Let's go through the most typical use cases one by one.

A fixed menu is one that permanently sits in one location. For example, it's popular to place a fixed menu above the editor. Tiptap doesn't come with a fixed menu, but you can build one by creating a <div> element and filling it with <button> elements. See below to learn how those buttons can trigger commands in the editor, for example bolding or italicizing text.

A bubble menu is one that appears when selecting text. The markup and styling are entirely up to you.

A floating menu appears in the editor when you place your cursor on an empty line. Again, the markup and styling are entirely up to you.

Although there isn't an official extension yet, there is an experiment that allows you to use "slash commands." With slash commands, typing / at the beginning of a new line will reveal a popup menu.

Okay, you've got your menu. But how do you wire things up?

You've got the editor running already and want to add your first button. You need a <button> HTML tag with a click handler. Depending on your setup, that can look like the following example:

Oh, that's a long command, right? Actually, it's a chain of commands. Let's go through this one by one:

In other words: This will be a typical Bold button for your text editor.

The available commands depend on the extensions registered with the editor. Most extensions come with a set…(), unset…(), and toggle…() command. Read the extension documentation to see what's actually available or just surf through your code editor's autocomplete.

You have already seen the focus() command in the above example. When you click on the button, the browser focuses that DOM element and the editor loses focus. It's likely you want to add focus() to all your menu buttons, so the writing flow of your users isn't interrupted.

The editor provides an isActive() method to check if something is applied to the selected text already. In Vue.js you can toggle a CSS class with help of that function:

Important: If you are using React, values from the editor state are not automatically reactive to avoid performance issues, that means the isActive() value will always return the value of the initial editor state. To subscribe to the editor state you can use the useEditorState hook as described here.

This toggles the .is-active class for nodes and marks. You can even check for specific attributes. Here is an example with the Highlight mark that ignores different attributes:

And an example that compares the given attribute(s):

There is even support for regular expressions:

You can even check nodes and marks, but check for the attributes only. Here is an example with the TextAlign extension:

If your selection spans multiple nodes or marks, or only part of the selection has a mark, isActive() will return false and indicate nothing is active. This behavior is intentional, as it allows users to apply a new node or mark to the selection right away.

When designing a great user experience you should consider a few things.

Most editor menus use icons for their buttons. In some of our demos, we use the open source icon set Remix Icon. However, you can use any icon set you prefer. Here are a few suggestions:

**Examples:**

Example 1 (jsx):
```jsx
<button onclick="editor.chain().focus().toggleBold().run()">Bold</button>
```

Example 2 (jsx):
```jsx
<button onclick="editor.chain().focus().toggleBold().run()">Bold</button>
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

## Import

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/import

**Contents:**
- Import
  - More details
  - Was this page helpful?

Import documents from various formats like docx, odt, and markdown and convert them to Tiptap's JSON format.

For more detailed information on how to integrate, install, and configure the Conversion feature, please visit our feature page.

---

## Import and export documents with Tiptap

**URL:** https://tiptap.dev/docs/conversion/getting-started/overview

**Contents:**
- Import and export documents with Tiptap
  - Conversion service
  - Was this page helpful?

The Tiptap Conversion Service makes it easy to import and export content between Tiptap and document file formats. It offers:

We are continuously improving the conversion service. Support for page breaks, headers/footers, references like footnotes, and other complex features are in active development.

---

## Install and Setup the Markdown Package

**URL:** https://tiptap.dev/docs/editor/markdown/getting-started/installation

**Contents:**
- Install and Setup the Markdown Package
- Installation
- Basic Setup
  - Initial Content as Markdown
- Configuration Options
  - Indentation Style
  - Custom Marked Instance
  - Marked Options
- Verifying Installation
- Common Issues

This guide will walk you through installing and setting up the Markdown extension in your Tiptap editor.

Install the Markdown extension using your preferred package manager:

Add the Markdown extension to your editor:

That's it! Your editor now supports Markdown parsing and serialization.

To load Markdown content when creating the editor:

The Markdown extension accepts several configuration options:

Configure how nested structures (lists, code blocks) are indented in the serialized Markdown:

If you need to use a custom version of marked or pre-configure it:

You can also pass marked options directly to the extension:

See the marked documentation for all available options.

To verify the extension is installed correctly:

If you get an error that @tiptap/markdown cannot be found:

If Markdown isn't being parsed:

If you get TypeScript errors:

**Examples:**

Example 1 (python):
```python
npm install @tiptap/markdown
```

Example 2 (python):
```python
npm install @tiptap/markdown
```

Example 3 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from '@tiptap/markdown'

const editor = new Editor({
  element: document.querySelector('#editor'),
  extensions: [StarterKit, Markdown],
  content: '<p>Hello World!</p>',
})
```

Example 4 (python):
```python
import { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from '@tiptap/markdown'

const editor = new Editor({
  element: document.querySelector('#editor'),
  extensions: [StarterKit, Markdown],
  content: '<p>Hello World!</p>',
})
```

---

## Install Collaboration

**URL:** https://tiptap.dev/docs/collaboration/getting-started/install

**Contents:**
- Install Collaboration
  - Install Tiptap Editor
- Prepare your editor
  - Integrate Yjs and the Collaboration Extension
- Start your Document server
  - Connect to your Document server
  - Set up private registry
  - Adding initial content
  - Initialize Content Properly
- Disable Default Undo/Redo

This guide will get you started with collaborative editing in the Tiptap Editor. If you're already using Tiptap Editor, feel free to skip ahead to Prepare your editor section.

If Tiptap Editor isn't installed yet, run the following command in your CLI for React to install the basic editor and necessary extensions for this example:

Once installed, you can get your Tiptap Editor up and running with this basic setup. Just add the following code snippets to your project:

To introduce team collaboration features into your Tiptap Editor, integrate the Yjs library and Editor Collaboration extension into your frontend. This setup uses Y.Doc, a shared document model, rather than just handling plain text. Afterwards we will connect Y.Doc to the TiptapCollabProvider to synchronize user interactions.

Add the Editor Collaboration extension and Yjs library to your frontend:

Then, update your index.jsx to include these new imports:

Your editor is now almost prepared for collaborative editing!

For collaborative functionality, install the @tiptap-pro/provider package:

Note that you need to follow the instructions here to set up access to the private registry.

Next, configure the provider in your index.jsx file with your server details:

When integrating the Editor in a non-collaborative setting, using the method shown here to set content is perfectly acceptable. However, if you transition to a collaborative environment, you will need to modify how you add initial content as shown after the next headline.

Incorporate the following code to complete the setup:

After following these steps, you should be able to open two different browsers and connect to the same document simultaneously through separate WebSocket connections.

For a clear test of the collaboration features, using two different browsers is recommended to guarantee unique websocket connections.

Upon implementing collaboration in your Tiptap Editor, you might notice that the initial content is repeatedly added each time the editor loads. To prevent this, use the .setContent() method to set the initial content only once.

This ensures the initial content is set only once. To test with new initial content, create a new document by changing the name parameter (e.g., from document.name to document.name2).

If you're integrating collaboration into an editor other than the one provided in this demo, you may need to disable the default Undo/Redo function of your Editor. This is necessary to avoid conflicts with the collaborative history management: You wouldn't want to revert someone else's changes.

This action is only required if your project includes the Tiptap StarterKit or Undo/Redo extension.

Following this guide will set up a basic, yet functional collaborative Tiptap Editor, synchronized through either the Collaboration Cloud or an on-premises backend.

Learn how to secure your collaborative Tiptap editor with JSON Web Tokens (JWTs). The next guide provides step-by-step instructions on creating and managing JWTs for both testing and production, ensuring controlled access with detailed examples. Read more about authentication.

**Examples:**

Example 1 (python):
```python
npm install @tiptap/extension-document @tiptap/extension-paragraph @tiptap/extension-text @tiptap/react
```

Example 2 (python):
```python
npm install @tiptap/extension-document @tiptap/extension-paragraph @tiptap/extension-text @tiptap/react
```

Example 3 (python):
```python
npm install @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-protocols
```

Example 4 (python):
```python
npm install @tiptap/extension-collaboration @tiptap/y-tiptap yjs y-protocols
```

---

## Install the Editor

**URL:** https://tiptap.dev/docs/editor/getting-started/install

**Contents:**
- Install the Editor
  - JavaScript
  - React
  - Next
  - Vue 3
  - Vue 2
  - Nuxt
  - Svelte
  - Alpine.js
  - PHP

Tiptap is framework-agnostic and even works with vanilla JavaScript (if that's your thing). Use the following guides to integrate Tiptap into your JavaScript project.

---

## Integrate AI into your editor

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/ai-generation

**Contents:**
- Integrate AI into your editor
  - More details
  - Was this page helpful?

Integrate AI-powered editor commands and content generation using the AI Generation extension. This extension add advanced AI text and image generation tools directly within your editor interface.

For more detailed information on how to integrate, install, and configure the AI Generation extension, please visit our feature page.

---

## Integrate Comments into your editor

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/comments

**Contents:**
- Integrate Comments into your editor
  - More details
  - Was this page helpful?

Integrate and manage comments within your editor using the Tiptap Comments extension. Create threads and comments in your editor or via REST API.

For more detailed information on how to integrate, install, and configure the Tiptap Comments extension, please visit our feature page.

---

## Integrate Snapshots into your editor

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/snapshot

**Contents:**
- Integrate Snapshots into your editor
  - More details
  - Was this page helpful?

Integrate and manage document revisions using the Snapshot extension. This extension enables tracking of all changes, allowing users to view previous document versions and revert changes as needed.

For more detailed information on how to integrate, install, and configure the Snapshot extension, please visit our feature page.

---

## Integrate the Tiptap Editor

**URL:** https://tiptap.dev/docs/editor/getting-started/overview

**Contents:**
- Integrate the Tiptap Editor
- What is Tiptap?
- Why pick Tiptap?
- Get started faster with ready-made components and templates
  - Templates
  - Components
  - Extend your editor
- Add Version History, Comments or drop in an AI agent
- Editor resources
  - How to set up and configure the Tiptap editor?

Build a rich-text editor that fits your product with exactly the features you need. Tiptap wraps the proven ProseMirror library in a modern, framework-agnostic API and gives you extensions and features for everything.

Tiptap is a headless rich-text editor framework that lets you build a custom editor completely tailored to your and your customers' needs. It's built on top of ProseMirror, a battle-tested library for building rich-text editors on the web. Under the hood Tiptap heavily relies on Events, Commands, and Extensions to provide a flexible and powerful API for building editors.

Want to get started with Tiptap? Follow our installation guide to set up your first editor with Tiptap.

Plug in our library of ready-made React components and full-featured templates to get a polished Editor and customize from there.

Extend your Tiptap Editor with open source or Pro extensions. The Tiptap suite adds more sophisticated features and comes with a 30-day free trial through your Tiptap Cloud dashboard.

How to apply styling to the headless Tiptap Editor

---

## ListKit extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/list-kit

**Contents:**
- ListKit extension
- Install
- Included extensions
  - Nodes
  - Extensions
- Source code
- Using the ListKit extension
  - Was this page helpful?

The ListKit is a collection of all necessary Tiptap list extensions. If you quickly want to setup lists in Tiptap, this extension is for you.

packages/extension-list/

Pass ListKit to the editor to load all included extension at once.

You can configure the included extensions, or even disable a few of them, like shown below.

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
import { ListKit } from '@tiptap/extension-list'

const editor = new Editor({
  extensions: [ListKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { ListKit } from '@tiptap/extension-list'

const editor = new Editor({
  extensions: [ListKit],
})
```

---

## Make your editor collaborative

**URL:** https://tiptap.dev/docs/collaboration/getting-started/overview

**Contents:**
- Make your editor collaborative
- Maintain documents
- Enable advanced features
- Enterprise on-premises solution
- Migrate from Hocuspocus or Collaboration Cloud
- Schema management
  - Was this page helpful?

Collaboration adds real-time collaborative editing to your editor. Presence indicators show who’s active, awareness highlights each user’s cursor and selection, and built-in version history and Comments track every change.

It runs on our open source Hocuspocus backend, syncs content with the Yjs CRDT, and scales from a single demo to thousands of concurrent connections.

Every change is stored as a Yjs update. Use the REST API to fetch JSON or push programmatic edits. Add webhooks for instant notifications and to retrieve all your documents.

Create your own backups of all documents and associated information using our document management API.

Integrate Collaboration and all other Tiptap features into your infrastructure.

Deploy our docker images in your own stack

Scale confidently to millions of users

Custom development and integration support in Chat

Migrating your application from Hocuspocus to either an on-premises solution or the Tiptap Collaboration Cloud involves a simple switch from the HocuspocusProvider to the TiptapCollabProvider, or the other way around.

This doesn't require any other updates to your setup, and the way you interact with the API won't change as well. The TiptapCollabProvider acts as a go-between, managing how your application connects to the server and handles login details.

This migration approach is also applicable when migrating from the Tiptap Collaboration Cloud to an on-premises configuration.

Review the Batch Import endpoint to migrate your documents.

Tiptap enforces strict schema adherence, discarding any elements not defined in the active schema. This can cause issues when clients using different schema versions concurrently edit a document.

For instance, imagine adding a task list feature in an update. Users on the previous schema won't see these task lists, and any added by a user on the new schema will disappear from their view due to schema discrepancies. This occurs because Tiptap synchronizes changes across clients, removing unrecognized elements based on the older schema.

To mitigate these issues, consider implementing Invalid Schema Handling as outlined in the Tiptap Editor docs.

---

## Next.js

**URL:** https://tiptap.dev/docs/editor/getting-started/install/nextjs

**Contents:**
- Next.js
  - Requirements
- Create a project (optional)
  - Install dependencies
- Integrate Tiptap
  - Add it to your app
- Next steps
  - Was this page helpful?

Integrate Tiptap with your Next.js project using this step-by-step guide.

If you already have an existing Next.js project, that's fine too. Just skip this step.

For the purpose of this guide, start a new Next.js project called my-tiptap-project. The following command sets up everything we need to get started.

Now that we have a standard boilerplate set up, we can get Tiptap up and running! For this, we will need to install three packages: @tiptap/react, @tiptap/pm, and @tiptap/starter-kit, which includes all the extensions you need to get started quickly.

If you followed steps 1 and 2, you can now start your project with npm run dev and open http://localhost:3000/ in your favorite browser. This might be different if you're working with an existing project.

To start using Tiptap, you'll need to add a new component to your app. To do so, first create a directory called components/. Now it's time to create our component which we'll call Tiptap. To do this, add the following example code in components/Tiptap.jsx.

Now, let's replace the content of app/page.js (or pages/index.js, if you are using the Pages router) with the following example code to use the Tiptap component in our app.

You should now see Tiptap in your browser. Time to give yourself a pat on the back! :)

**Examples:**

Example 1 (markdown):
```markdown
# create a project
npx create-next-app my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 2 (markdown):
```markdown
# create a project
npx create-next-app my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 3 (python):
```python
npm install @tiptap/react @tiptap/pm @tiptap/starter-kit
```

Example 4 (python):
```python
npm install @tiptap/react @tiptap/pm @tiptap/starter-kit
```

---

## Nuxt

**URL:** https://tiptap.dev/docs/editor/getting-started/install/nuxt

**Contents:**
- Nuxt
  - Requirements
- Create a project (optional)
  - Install the dependencies
- Integrate Tiptap
  - Add it to your app
  - Use v-model (optional)
- Next steps
  - Was this page helpful?

This guide covers how to integrate Tiptap with your Nuxt.js project, complete with code examples.

If you already have a Nuxt.js project, that's fine too. Just skip this step.

For the purpose of this project, start with a fresh Nuxt.js project called my-tiptap-project. The following command sets up everything we need. It asks a lot of questions, but just use what floats your boat or use the defaults.

Enough of the boring boilerplate work. Let's install Tiptap! For the following example, you'll need the @tiptap/vue-3 package with a few components, the @tiptap/pm package, and @tiptap/starter-kit, which has the most common extensions to get started quickly.

If you followed steps 1 and 2, you can now start your project with npm run dev and open http://localhost:8080/ in your favorite browser. This might be different if you're working with an existing project.

To actually start using Tiptap, you'll need to add a new component to your app. Let's call it TiptapEditor and put the following example code in components/TiptapEditor.vue.

This is the fastest way to get Tiptap up and running with Vue. It will give you a very basic version of Tiptap, without any buttons. No worries, you will be able to add more functionality soon.

Now, let's replace the content of pages/index.vue with the following example code to use our new TiptapEditor component in our app.

Note that Tiptap needs to run in the client, not on the server. It's required to wrap the editor in a <client-only> tag. Read more about client-only components.

You should now see Tiptap in your browser. Time to give yourself a pat on the back! :)

You're probably used to binding your data with v-model in forms. This also possible with Tiptap. Here's a working example component, that you can integrate in your project:

**Examples:**

Example 1 (markdown):
```markdown
# create a project
npm init nuxt-app my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 2 (markdown):
```markdown
# create a project
npm init nuxt-app my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 3 (python):
```python
npm install @tiptap/vue-3 @tiptap/pm @tiptap/starter-kit
```

Example 4 (python):
```python
npm install @tiptap/vue-3 @tiptap/pm @tiptap/starter-kit
```

---

## PHP

**URL:** https://tiptap.dev/docs/editor/getting-started/install/php

**Contents:**
- PHP
- Laravel Livewire
  - my-livewire-component.blade.php
  - Hint
  - editor.blade.php
  - index.js
- Next steps
  - Was this page helpful?

You can use Tiptap with Laravel, Livewire, Inertia.js, Alpine.js, Tailwind CSS, and even—yes, you read that right—inside PHP.

We provide an official PHP package to work with Tiptap content. You can transform Tiptap-compatible JSON to HTML and vice versa, sanitize your content, or just modify it.

The .defer modifier is no longer available in Livewire v3, as updating the state is deferred by default. Use the .live modifier if you need to update the state server-side, as it changes.

**Examples:**

Example 1 (typescript):
```typescript
<!--
  In your livewire component you could add an
  autosave method to handle saving the content
  from the editor every 10 seconds if you wanted
-->
<x-editor wire:model="foo" wire:poll.10000ms="autosave"></x-editor>
```

Example 2 (typescript):
```typescript
<!--
  In your livewire component you could add an
  autosave method to handle saving the content
  from the editor every 10 seconds if you wanted
-->
<x-editor wire:model="foo" wire:poll.10000ms="autosave"></x-editor>
```

Example 3 (jsx):
```jsx
<div
  x-data="setupEditor(
    $wire.entangle('{{ $attributes->wire('model')->value() }}').defer
  )"
  x-init="() => init($refs.editor)"
  wire:ignore
  {{ $attributes->whereDoesntStartWith('wire:model') }}
>
  <div x-ref="editor"></div>
</div>
```

Example 4 (jsx):
```jsx
<div
  x-data="setupEditor(
    $wire.entangle('{{ $attributes->wire('model')->value() }}').defer
  )"
  x-init="() => init($refs.editor)"
  wire:ignore
  {{ $attributes->whereDoesntStartWith('wire:model') }}
>
  <div x-ref="editor"></div>
</div>
```

---

## React

**URL:** https://tiptap.dev/docs/editor/getting-started/install/react

**Contents:**
- React
  - Create a React project (optional)
  - Install Tiptap dependencies
- Integrate Tiptap into your React app
  - Add it to your app
- Using the EditorContext
  - Consume the Editor context in child components
- Reacting to Editor state changes
- Use SSR with React and Tiptap
- Optimize your performance

This guide describes how to integrate Tiptap with your React project. We're using Vite, but the workflow should be similar with other setups.

Start with a fresh React project called my-tiptap-project. Vite will set up everything we need.

Next, install the @tiptap/react package, @tiptap/pm (the ProseMirror library), and @tiptap/starter-kit, which includes the most common extensions to get started quickly.

If you followed steps 1 and 2, you can now start your project with npm run dev and open http://localhost:3000 in your browser.

To actually start using Tiptap we need to create a new component. Let's call it Tiptap and add the following example code in src/Tiptap.tsx.

Finally, replace the content of src/App.tsx with our new Tiptap component.

Tiptap provides a React context called EditorContext, that allows you to access the editor instance and its state from anywhere in your component tree. This is particularly useful for building custom toolbars, menus, or other components that need to interact with the editor.

If you use the EditorProvider to set up your Tiptap editor, you can now access your editor instance from any child component using the useCurrentEditor hook.

Important: This won't work if you use the useEditor hook to setup your editor.

You should now see a pretty barebones example of Tiptap in your browser.

To react to editor state changes, you can use the useEditorState hook from @tiptap/react. This hook can be used to fetch information from the editor state without causing re-renders on the editor component or it's children.

Tiptap can be used with server-side rendering (SSR) in React applications. However, to ensure that the editor is only initialized on the client side, you need to use the immediatelyRender option when creating the editor instance to prevent it from rendering on the server.

Here is an example of how to set up Tiptap with SSR in a React component:

We recommend visiting the React Performance Guide to integrate the Tiptap Editor efficiently. This will help you avoid potential issues as your app scales.

**Examples:**

Example 1 (markdown):
```markdown
# create a project with npm
npm create vite@latest my-tiptap-project -- --template react-ts

# OR, create a project with pnpm
pnpm create vite@latest my-tiptap-project --template react-ts

# OR, create a project with yarn
yarn create vite my-tiptap-project --template react-ts

# change directory
cd my-tiptap-project
```

Example 2 (markdown):
```markdown
# create a project with npm
npm create vite@latest my-tiptap-project -- --template react-ts

# OR, create a project with pnpm
pnpm create vite@latest my-tiptap-project --template react-ts

# OR, create a project with yarn
yarn create vite my-tiptap-project --template react-ts

# change directory
cd my-tiptap-project
```

Example 3 (python):
```python
npm install @tiptap/react @tiptap/pm @tiptap/starter-kit
```

Example 4 (python):
```python
npm install @tiptap/react @tiptap/pm @tiptap/starter-kit
```

---

## Runtime configuration

**URL:** https://tiptap.dev/docs/collaboration/operations/configure

**Contents:**
- Runtime configuration
- Settings overview
- Managing settings via API
  - Create or overwrite settings
  - List current settings
  - Retrieve a specific setting
  - Delete a setting
- Server performance metrics
  - Was this page helpful?

Configure runtime settings in Tiptap Collaboration to manage your collaboration environment directly via the REST API.

These settings let you modify secrets, webhook URLs, and more, particularly when adapting to changes in your project requirements or security protocols, without restarting your application.

Several settings can be adjusted dynamically:

The collaboration platform offers a straightforward API for managing these settings. Replace :key with the setting key you wish to update.

Use this call to add or update settings:

Use this call to retrieve a list of all current settings:

Use this call to retrieve the value of a particular setting:

Use this call to delete a setting:

Use the /api/statistics endpoint to gather server performance data, including total document count, peak concurrent connections, total connections over the last 30 days, and lifetime connection counts. Review the metrics page for additional information.

**Examples:**

Example 1 (unknown):
```unknown
curl --location --request PUT 'https://YOUR_APP_ID.collab.tiptap.cloud/api/admin/settings/:key' \
--header 'Authorization: YOUR_SECRET_FROM_SETTINGS_AREA' --header 'Content-Type: text/plain' \
-d 'your value'
```

Example 2 (unknown):
```unknown
curl --location --request PUT 'https://YOUR_APP_ID.collab.tiptap.cloud/api/admin/settings/:key' \
--header 'Authorization: YOUR_SECRET_FROM_SETTINGS_AREA' --header 'Content-Type: text/plain' \
-d 'your value'
```

Example 3 (unknown):
```unknown
curl --location 'https://YOUR_APP_ID.collab.tiptap.cloud/api/admin/settings' \
--header 'Authorization: YOUR_SECRET_FROM_SETTINGS_AREA'
```

Example 4 (unknown):
```unknown
curl --location 'https://YOUR_APP_ID.collab.tiptap.cloud/api/admin/settings' \
--header 'Authorization: YOUR_SECRET_FROM_SETTINGS_AREA'
```

---

## Snapshot Compare Extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/snapshot-compare

**Contents:**
- Snapshot Compare Extension
  - More details
  - Was this page helpful?

Use the Snapshot Compare extension to visualize changes between two document versions. Whether you're collaborating with a team or editing individually, it highlights the differences between snapshots, showing what changed and who made the changes.

For more detailed information on how to integrate, install, and configure the Snapshot Compare extension, please visit our feature page.

---

## Styling the Editor

**URL:** https://tiptap.dev/docs/editor/getting-started/style-editor

**Contents:**
- Styling the Editor
  - Building your own UI
- Style plain HTML
  - Style with CSS modules
- Add custom classes
  - Extensions
  - Editor
- Customize HTML
- Style using Tailwind CSS
  - Using a global CSS file and @apply

Tiptap follows a headless-first approach, which means the core extensions come without any styling or UI components - just pure logic. This gives you complete control over how your editor looks and behaves.

However, we now also offer optional UI components and templates that you can use to accelerate your development.

Get started with UI templates

Our UI templates include pre-built components for common editor features. Download the source code and customize to your needs.

Our UI templates include pre-built components for common editor features. Download the source code and customize to your needs.

If you prefer to build your own UI or need to understand how styling works in Tiptap, here are the available methods

The entire editor is rendered inside a container with the class .tiptap. You can use that to scope your styling to the editor content:

In CSS modules, class names are modified to enable local scoping, which may prevent styles from applying when targeting the .tiptap class. Use global styles or the :global(.tiptap) modifier to ensure styles are applied correctly.

If you're rendering the stored content elsewhere, there won't be a .tiptap container, so you can globally add styling to the relevant HTML tags:

You can control the whole rendering, including adding classes to everything.

Most extensions allow you to add attributes to the rendered HTML through the HTMLAttributes option. You can use that to add a custom class (or any other attribute). That's also very helpful when you work with Tailwind CSS.

The rendered HTML will look like this:

If there are already classes defined by the extensions, your classes will be added.

You can even pass classes to the element that contains the editor:

Or you can customize the markup for extensions. The following example will make a custom bold extension that doesn't render a <strong> tag, but a <b> tag:

You should place your custom extensions in separate files for better organization, but you get the idea.

Since content managed in Tiptap is plain HTML which by default doesn't come with Tailwind CSS classes, you have a few approaches to style your editor:

Even though the Tailwind maintainers recommend not using @apply we think it's the best way to define styles for flexible, user written content like Tiptaps content. You can use a global CSS file and apply Tailwind styles to HTML tags inside your editor. Since nested selectors are now widely supported, you can scope your styles to the .tiptap class:

If you don't want to use a global CSS file, a more complicated approach would be to extend existing components and add Tailwind classes to the HTMLAttributes inside the renderHTML output.

This approach can work but will become more complicated with more complicated extensions or when NodeViews are used so we still would recommend using a global CSS file with @apply for most cases.

The editor works fine with Tailwind CSS, too. Find an example that's styled with the @tailwindcss/typography plugin below.

If you're using TailwindCSS Intellisense, add this snippet to your .vscode/settings.json to enable intellisense support inside Tiptap objects:

**Examples:**

Example 1 (css):
```css
/* Scoped to the editor */
.tiptap p {
  margin: 1em 0;
}
```

Example 2 (css):
```css
/* Scoped to the editor */
.tiptap p {
  margin: 1em 0;
}
```

Example 3 (css):
```css
/* Global styling */
p {
  margin: 1em 0;
}
```

Example 4 (css):
```css
/* Global styling */
p {
  margin: 1em 0;
}
```

---

## Svelte

**URL:** https://tiptap.dev/docs/editor/getting-started/install/svelte

**Contents:**
- Svelte
- Take a shortcut: Svelte REPL with Tiptap
  - Requirements
- Create a project (optional)
  - Install dependencies
- Integrate Tiptap
- Add it to your app
- Next steps
- Setup with Svelte legacy syntax
  - Was this page helpful?

Learn how to integrate Tiptap with your SvelteKit project using this step-by-sep guide. Alternatively, check out our Svelte text editor example.

If you want to jump into it right away, here is a Svelte REPL with Tiptap.

If you already have a SvelteKit project, that's fine too. Just skip this step.

For the purpose of this guide, start with a fresh SvelteKit project called my-tiptap-project. The following commands set up everything we need. It asks a lot of questions, but select your preferred options or use the defaults.

Now that we're done with boilerplate, let's install Tiptap! For the following example you'll need the @tiptap/core package, with a few components, @tiptap/pm, and @tiptap/starter-kit, which includes the most common extensions to get started quickly.

If you followed steps 1 and 2, you can now start your project with npm run dev and open http://localhost:3000/ in your favorite browser. This might be different if you're working with an existing project.

To start using Tiptap, you'll need to add a new component to your app. Let's call it Tiptap and add the following example code in src/lib/Tiptap.svelte.

This is the fastest way to get Tiptap up and running with SvelteKit. It will give you a very basic version of Tiptap, without any buttons. No worries, you will be able to add more functionality soon.

Now, let's replace the content of src/routes/+page.svelte with the following example code to use our new Tiptap component in our app.

Tiptap should now be visible in your browser. Time to give yourself a pat on the back! :)

The example above uses the Svelte runes syntax. If your Svelte app is in legacy mode, use this code for the Tiptap.svelte component instead.

**Examples:**

Example 1 (python):
```python
npm create svelte@latest my-tiptap-project
cd my-tiptap-project
npm install
npm run dev
```

Example 2 (python):
```python
npm create svelte@latest my-tiptap-project
cd my-tiptap-project
npm install
npm run dev
```

Example 3 (python):
```python
npm install @tiptap/core @tiptap/pm @tiptap/starter-kit @tiptap/extension-bubble-menu
```

Example 4 (python):
```python
npm install @tiptap/core @tiptap/pm @tiptap/starter-kit @tiptap/extension-bubble-menu
```

---

## TableKit extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/table-kit

**Contents:**
- TableKit extension
- Install
- Included extensions
  - Nodes
- Source code
- Using the TableKit extension
  - Was this page helpful?

The TableKit is a collection of all necessary Tiptap table extensions. If you quickly want to setup tables in Tiptap, this extension is for you.

packages/extension-table/

Pass TableKit to the editor to load all included extension at once.

You can configure the included extensions, or even disable a few of them, like shown below.

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

const editor = new Editor({
  extensions: [TableKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TableKit } from '@tiptap/extension-table'

const editor = new Editor({
  extensions: [TableKit],
})
```

---

## TextStyleKit extension

**URL:** https://tiptap.dev/docs/editor/extensions/functionality/text-style-kit

**Contents:**
- TextStyleKit extension
- Install
- Included extensions
  - Marks
  - Functionality
- Source code
- Using the TextStyleKit extension
  - Was this page helpful?

The TextStyleKit is a collection of the most common Tiptap text style extensions. If you quickly want to setup styles for your text in Tiptap, this extension is for you.

packages/extension-text-style/

Pass TextStyleKit to the editor to load all included extension at once.

You can configure the included extensions, or even disable a few of them, like shown below.

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
import { TextStyleKit } from '@tiptap/extension-text-style'

const editor = new Editor({
  extensions: [TextStyleKit],
})
```

Example 4 (sql):
```sql
import { Editor } from '@tiptap/core'
import { TextStyleKit } from '@tiptap/extension-text-style'

const editor = new Editor({
  extensions: [TextStyleKit],
})
```

---

## Vanilla JavaScript

**URL:** https://tiptap.dev/docs/editor/getting-started/install/vanilla-javascript

**Contents:**
- Vanilla JavaScript
  - Hint
- Install dependencies
  - Add markup
- Initialize the editor
- Next steps
  - Was this page helpful?

Are you using plain JavaScript or a framework that isn't listed? No worries, we provide everything you need.

If you don't use a bundler like Webpack or Rollup, please follow the CDN guide instead. Since Tiptap is built in a modular way, you will need to use <script type="module"> in your HTML to get our CDN imports working.

For the following example, you will need @tiptap/core (the actual editor), @tiptap/pm (the ProseMirror library), and @tiptap/starter-kit. The StarterKit doesn't include all extensions, only the most common ones.

Add the following HTML where you'd like to mount the editor:

Everything is in place, so let's set up the editor. Add the following code to your JavaScript:

Open your project in the browser to see Tiptap in action. Good work!

**Examples:**

Example 1 (python):
```python
npm install @tiptap/core @tiptap/pm @tiptap/starter-kit
```

Example 2 (python):
```python
npm install @tiptap/core @tiptap/pm @tiptap/starter-kit
```

Example 3 (jsx):
```jsx
<div class="element"></div>
```

Example 4 (jsx):
```jsx
<div class="element"></div>
```

---

## Vue 2

**URL:** https://tiptap.dev/docs/editor/getting-started/install/vue2

**Contents:**
- Vue 2
  - Requirements
- Create a project (optional)
  - Install the dependencies
- Integrate Tiptap
  - Add it to your app
  - Use v-model (optional)
- Next steps
  - Was this page helpful?

This guide details how to integrate Tiptap with your Vue 2 project. Alternatively, check out our Vue text editor example.

If you already have a Vue project, that's fine too. Just skip this step.

For the purpose of this guide, start with a fresh Vue project called my-tiptap-project. The Vue CLI sets up everything we need, just select the default Vue 2 template.

Okay, enough of the boring boilerplate work. Let's finally install Tiptap! For the following example you'll need the @tiptap/vue-2 package, @tiptap/pm (the ProseMirror library) and @tiptap/starter-kit, which includes the most common extensions to get started quickly.

If you followed step 1 and 2, you can now start your project with npm run dev, and open http://localhost:8080 in your favorite browser. This might be different, if you're working with an existing project.

To actually start using Tiptap, you'll need to add a new component to your app. Let's call it Tiptap and put the following example code in components/Tiptap.vue.

This is the fastest way to get Tiptap up and running with Vue. It will give you a very basic version of Tiptap, without any buttons. No worries, you will be able to add more functionality soon.

Now, let's replace the content of src/App.vue with the following example code to use our new Tiptap component in our app.

You should now see Tiptap in your browser. Time to give yourself a pat on the back! :)

You're probably used to bind your data with v-model in forms, that's also possible with Tiptap. Here is a working example component, that you can integrate in your project:

**Examples:**

Example 1 (markdown):
```markdown
# create a project
vue create my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 2 (markdown):
```markdown
# create a project
vue create my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 3 (python):
```python
npm install @tiptap/vue-2 @tiptap/pm @tiptap/starter-kit
```

Example 4 (python):
```python
npm install @tiptap/vue-2 @tiptap/pm @tiptap/starter-kit
```

---

## Vue 3

**URL:** https://tiptap.dev/docs/editor/getting-started/install/vue3

**Contents:**
- Vue 3
  - Requirements
- Create a project (optional)
  - Install the dependencies
- Integrate Tiptap
  - Add it to your app
  - Use v-model (optional)
- Next steps
  - Was this page helpful?

Discover how to integrate Tiptap with your Vue 3 project using this step-by-step guide. Alternatively, check out our Vue text editor example.

If you already have a Vue project, that's fine too. Just skip this step.

For the purpose of this guide, start with a fresh Vue project called my-tiptap-project. The Vue CLI sets up everything we need. Just select the Vue 3 template.

Okay, enough boilerplate work. Let's finally install Tiptap! For the following example, you'll need the @tiptap/vue-3 package, @tiptap/pm (the ProseMirror library), and @tiptap/starter-kit, which includes the most common extensions to get started quickly.

If you followed steps 1 and 2, you can now start your project with npm run dev and open http://localhost:8080 in your favorite browser. This might be different if you're working with an existing project.

To start using Tiptap, you'll need to add a new component to your app. Let's call it Tiptap and put the following example code in components/Tiptap.vue.

This is the fastest way to get Tiptap up and running with Vue. It will give you a very basic version of Tiptap, without any buttons. No worries, you will be able to add more functionality soon.

Alternatively, you can use the Composition API with the useEditor method.

Or feel free to use the new <script setup> syntax.

Now, let's replace the content of src/App.vue with the following example code to use our new Tiptap component in our app.

You should now see Tiptap in your browser. Time to give yourself a pat on the back! :)

You're probably used to binding your data with v-model in forms, that's also possible with Tiptap. Here is how that would work with Tiptap:

**Examples:**

Example 1 (markdown):
```markdown
# create a project
vue create my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 2 (markdown):
```markdown
# create a project
vue create my-tiptap-project

# change directory
cd my-tiptap-project
```

Example 3 (python):
```python
npm install @tiptap/vue-3 @tiptap/pm @tiptap/starter-kit
```

Example 4 (python):
```python
npm install @tiptap/vue-3 @tiptap/pm @tiptap/starter-kit
```

---
