# Tiptap - Nodes

**Pages:** 6

---

## Compare document versions

**URL:** https://tiptap.dev/docs/collaboration/documents/snapshot-compare

**Contents:**
- Compare document versions
- Access the private registry
- Install
- Settings
    - Using mapDiffToDecorations for diff decorations
- Storage
- Commands
  - compareVersions
    - Options
    - Using hydrateUserData to add metadata

Snapshot Compare lets you line-up two document versions and highlights everything that changed. Use it to track edits, review contributions, and restore earlier states.

The Snapshot Compare extension adds extra functionality to the Snapshots by allowing you to visually compare changes made between two versions of a document so you can track what’s been added, removed, or modified. These comparisons are called diffs.

The Snapshot Compare extension is published in Tiptap’s private npm registry. Integrate the extension by following the private registry guide. If you already authenticated your Tiptap account you can go straight to #Install.

Install the extension from our private registry:

You can configure the SnapshotCompare extension with the following options:

Note that you need to provide the user identifier to the TiptapCollabProvider, as this information is used to optimize diffs.

The extension has a default mapping (defaultMapDiffToDecorations) to represent diffs as ProseMirror decorations. For more complex integrations and control, you can customize this mapping with the mapDiffToDecorations option.

Example: Applying custom predefined background colors to inline inserts

The SnapshotCompare storage object contains the following properties:

Use the isPreviewing property to check if the diff view is currently active:

Use the diffs property to access the diffs displayed in the diff view:

The property previousContent is used internally by the extension to restore the content when exiting the diff view. Typically, you do not need to interact with it directly.

Use the compareVersions command to compute and display the differences between two document versions.

You can pass in additional options for more control over the diffing process:

Each diff has an attribution field, which allows you to add additional metadata with the hydrateUserData callback function.

Do note that the userId is populated by the TiptapCollabProvider and should be used to identify the user who made the change. Without the user field provided by the provider, the userId will be null. See more information in the TiptapCollabProvider documentation.

Example: Color-coding diffs based on user

If you need more control over the diffing process, use the onCompare option to receive the result and handle it yourself.

Example: Filtering diffs by user

Use the showDiff command to display the diff within the editor using a change tracking transform (tr). This represents all of the changes made to the document since the last snapshot. You can use this transform to show the diff in the editor.

Typically, you use this command after customizing or filtering diffs with compareVersions, onCompare.

The showDiff command temporarily replaces the current editor content with the diff view, showing the differences between versions. It also stashes the content currently displayed in editor so that you can restore it later.

You can pass additional options to control how the diffs are displayed:

Example: Displaying specific diffs

Use the hideDiff command to hide the diff and restore the previous content.

The Snapshot Compare extension applies classes to the elements of the diff view to help you style inserted and deleted text. See a complete example of how to style the diff view in the code demo at the top of this page.

The extension applies the following attributes to diff elements:

You can target the elements with these data attributes by using CSS selectors.

Here's an example of basic styling for inserted and deleted text:

When styling marks are applied (like bold, italic, or code), the deleted text (the text before the formatting was applied) can have that formatting applied to it. This issue occurs because the deleted text appears inside the HTML tag of the formatting. To prevent this, reset the mark styles inside the deleted content.

First, reset the outer mark styles inside the deleted content.

Then, if there is a mark tag inside the deleted content, re-apply the mark styles to it. This will ensure that the formatting is correctly applied to the deleted content.

When user attribution is available, you can style diffs based on the user who made the change:

When using custom node views, the default diff mapping may not work as expected. You can customize the mapping and render the diffs directly within the custom node view.

Use the extractAttributeChanges helper to extract attribute changes in nodes. This allows you to access the previous and current attributes of a node, making it possible to highlight attribute changes within your custom node views.

When mapping the diffs into decorations yourself, you need to pass the diff as the decoration's spec. This is required for extractAttributeChanges to work correctly.

Example: Customizing a heading node view to display changes

A Diff is an object that represents a change made to the document. It contains the following properties:

A DiffSet is an array of Diff objects, each corresponding to a specific change, like insertion, deletion, or update. You can iterate over the array to inspect individual changes or apply custom logic based on the diff types.

The Attribution object contains metadata about a change. It includes the following properties:

You can extend the Attribution interface to include additional properties:

The ChangeTrackingTransform is a class that records changes made to the document (based on ProseMirror's Transform). It represents a transform whose steps describe all of the changes made to go from one version of the document to another. It has the following properties:

The ChangeTrackingStep is a class that represents a single change made to the document, based on ProseMirror's ReplaceStep class. It has the following property:

Here is the full TypeScript definition for the SnapshotCompare extension:

**Examples:**

Example 1 (python):
```python
npm install @tiptap-pro/extension-snapshot-compare
```

Example 2 (python):
```python
npm install @tiptap-pro/extension-snapshot-compare
```

Example 3 (javascript):
```javascript
const provider = new TiptapCollabProvider({
  // ...
  user: 'your user identifier' // REQUIRED! Note that we use the user identifier to optimize diffs, so it's important to provide it.
})

const editor = new Editor({
  // ...
  extensions: [
    // ...
    SnapshotCompare.configure({
      provider,
    }),
  ],
})
```

Example 4 (javascript):
```javascript
const provider = new TiptapCollabProvider({
  // ...
  user: 'your user identifier' // REQUIRED! Note that we use the user identifier to optimize diffs, so it's important to provide it.
})

const editor = new Editor({
  // ...
  extensions: [
    // ...
    SnapshotCompare.configure({
      provider,
    }),
  ],
})
```

---

## Export custom styles to .docx

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/export-styles

**Contents:**
- Export custom styles to .docx
  - Paragraph style object
  - Run style properties
  - Paragraph style properties
- Tiptap's export default styles
  - Was this page helpful?

When exporting to DOCX, you can define custom styles that will be applied to the exported document. This is useful when you want to have a consistent look and feel across your documents.

In the example above ☝️ we are exporting a document with a custom Heading 1 style. The style is based on the Normal style, has a red color, and uses the Aptos font. The spacing before the paragraph is set to 12pt, and 6pt after it. The line height is set to 1.15.

You can also create custom styling for other elements like Heading 2, Heading 3, List Bullet, List Number, etc. The paragraphStyles array accepts an array of objects with the following properties:

A paragraphStyle object accepts the following properties:

The run object from a paragraphStyle accepts the following properties:

For more advanced styling options and detailed usage you can refer to the IRunStylePropertiesOptions type exposed from our package, or refer to the docx documentation.

The paragraph object from a paragraphStyle accepts the following properties:

For more advanced styling options and detailed usage you can refer to the IParagraphStylePropertiesOptions type exposed from our package, or refer to the docx documentation.

Tiptap offers a sensible default styling for the exported document, but you can override these styles by providing your own custom styles. This allows you to create a consistent look and feel across your documents.

**Examples:**

Example 1 (json):
```json
// Import the ExportDocx extension
import { ExportDocx } from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // Other extensions ...
    ExportDocx.configure({
      onCompleteExport: (result: string | Buffer<ArrayBufferLike> | Blob | Stream) => {}, // required
      styleOverrides: { // Style overrides
        paragraphStyles: [
          // Heading 1 style override
          {
            id: 'Heading1',
            name: 'Heading 1',
            basedOn: 'Normal',
            next: 'Normal',
            quickFormat: true,
            run: {
              font: 'Aptos',
              size: pointsToHalfPoints(16),
              bold: true,
              color: 'FF0000',
            },
            paragraph: {
              spacing: {
                before: pointsToTwips(12),
                after: pointsToTwips(6),
                line: lineHeightToDocx(1.15),
              },
            },
          },
        ]
      }
    }),
    // Other extensions ...
  ],
  // Other editor settings ...
})
```

Example 2 (json):
```json
// Import the ExportDocx extension
import { ExportDocx } from '@tiptap-pro/extension-export-docx'

const editor = new Editor({
  extensions: [
    // Other extensions ...
    ExportDocx.configure({
      onCompleteExport: (result: string | Buffer<ArrayBufferLike> | Blob | Stream) => {}, // required
      styleOverrides: { // Style overrides
        paragraphStyles: [
          // Heading 1 style override
          {
            id: 'Heading1',
            name: 'Heading 1',
            basedOn: 'Normal',
            next: 'Normal',
            quickFormat: true,
            run: {
              font: 'Aptos',
              size: pointsToHalfPoints(16),
              bold: true,
              color: 'FF0000',
            },
            paragraph: {
              spacing: {
                before: pointsToTwips(12),
                after: pointsToTwips(6),
                line: lineHeightToDocx(1.15),
              },
            },
          },
        ]
      }
    }),
    // Other extensions ...
  ],
  // Other editor settings ...
})
```

Example 3 (json):
```json
{
  paragraphStyles: [
    // Normal style (default for most paragraphs)
    {
      id: 'Normal',
      name: 'Normal',
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(11),
      },
      paragraph: {
        spacing: {
          before: 0,
          after: pointsToTwips(10),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // List Paragraph style (used for bullets and numbering)
    {
      id: 'ListParagraph',
      name: 'List Paragraph',
      basedOn: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(11),
      },
      paragraph: {
        spacing: {
          before: 0,
          after: pointsToTwips(2),
          line: lineHeightToDocx(1),
        },
      },
    },
    // Heading 1 style
    {
      id: 'Heading1',
      name: 'Heading 1',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(16),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 2 style
    {
      id: 'Heading2',
      name: 'Heading 2',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(14),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 3 style
    {
      id: 'Heading3',
      name: 'Heading 3',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(13),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 4 style
    {
      id: 'Heading4',
      name: 'Heading 4',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(12),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 5 style
    {
      id: 'Heading5',
      name: 'Heading 5',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(11),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Title style
    {
      id: 'Title',
      name: 'Title',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(22),
        bold: true,
        color: '000000',
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: 0,
          after: 0,
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Subtitle style
    {
      id: 'Subtitle',
      name: 'Subtitle',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(16),
        italics: true,
        color: '666666',
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: 0,
          after: 0,
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Quote style (typically for indented, italic text)
    {
      id: 'Quote',
      name: 'Quote',
      basedOn: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        italics: true,
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: pointsToTwips(10),
          after: pointsToTwips(10),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Intense Quote style (more pronounced indentation)
    {
      id: 'IntenseQuote',
      name: 'Intense Quote',
      basedOn: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        italics: true,
        color: '444444',
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: pointsToTwips(10),
          after: pointsToTwips(10),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // No Spacing style (no extra space before or after paragraphs)
    {
      id: 'NoSpacing',
      name: 'No Spacing',
      basedOn: 'Normal',
      quickFormat: true,
      paragraph: {
        spacing: {
          before: 0,
          after: 0,
          line: lineHeightToDocx(1),
        },
      },
    },
    // Hyperlink style
    {
      id: 'Hyperlink',
      name: 'Hyperlink',
      basedOn: 'Normal',
      run: {
        color: '0563C1',
        underline: {
          type: 'single',
        },
      },
    },
  ],
}
```

Example 4 (json):
```json
{
  paragraphStyles: [
    // Normal style (default for most paragraphs)
    {
      id: 'Normal',
      name: 'Normal',
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(11),
      },
      paragraph: {
        spacing: {
          before: 0,
          after: pointsToTwips(10),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // List Paragraph style (used for bullets and numbering)
    {
      id: 'ListParagraph',
      name: 'List Paragraph',
      basedOn: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(11),
      },
      paragraph: {
        spacing: {
          before: 0,
          after: pointsToTwips(2),
          line: lineHeightToDocx(1),
        },
      },
    },
    // Heading 1 style
    {
      id: 'Heading1',
      name: 'Heading 1',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(16),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 2 style
    {
      id: 'Heading2',
      name: 'Heading 2',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(14),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 3 style
    {
      id: 'Heading3',
      name: 'Heading 3',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(13),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 4 style
    {
      id: 'Heading4',
      name: 'Heading 4',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(12),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Heading 5 style
    {
      id: 'Heading5',
      name: 'Heading 5',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        size: pointsToHalfPoints(11),
        bold: true,
        color: '2E74B5',
      },
      paragraph: {
        spacing: {
          before: pointsToTwips(12),
          after: pointsToTwips(6),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Title style
    {
      id: 'Title',
      name: 'Title',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(22),
        bold: true,
        color: '000000',
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: 0,
          after: 0,
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Subtitle style
    {
      id: 'Subtitle',
      name: 'Subtitle',
      basedOn: 'Normal',
      next: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos Light',
        size: pointsToHalfPoints(16),
        italics: true,
        color: '666666',
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: 0,
          after: 0,
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Quote style (typically for indented, italic text)
    {
      id: 'Quote',
      name: 'Quote',
      basedOn: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        italics: true,
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: pointsToTwips(10),
          after: pointsToTwips(10),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // Intense Quote style (more pronounced indentation)
    {
      id: 'IntenseQuote',
      name: 'Intense Quote',
      basedOn: 'Normal',
      quickFormat: true,
      run: {
        font: 'Aptos',
        italics: true,
        color: '444444',
      },
      paragraph: {
        alignment: AlignmentType.CENTER,
        spacing: {
          before: pointsToTwips(10),
          after: pointsToTwips(10),
          line: lineHeightToDocx(1.15),
        },
      },
    },
    // No Spacing style (no extra space before or after paragraphs)
    {
      id: 'NoSpacing',
      name: 'No Spacing',
      basedOn: 'Normal',
      quickFormat: true,
      paragraph: {
        spacing: {
          before: 0,
          after: 0,
          line: lineHeightToDocx(1),
        },
      },
    },
    // Hyperlink style
    {
      id: 'Hyperlink',
      name: 'Hyperlink',
      basedOn: 'Normal',
      run: {
        color: '0563C1',
        underline: {
          type: 'single',
        },
      },
    },
  ],
}
```

---

## Integrate snapshots

**URL:** https://tiptap.dev/docs/collaboration/documents/snapshot

**Contents:**
- Integrate snapshots
  - Public Demo
- Access the Pro registry
- Install
- Settings
- Autoversioning
- Revert to a version
- Storage
- Commands
- Examples

Document history records every change to your content so you can roll back mistakes, audit edits, or branch a new draft from any point.

This page walks you through installation, configuration, and common tasks for the History extension.

The editor content is shared across all demo visitors.

The Version History extension is published in Tiptap’s private npm registry. Integrate the extension by following the private registry guide. If you already authenticated your Tiptap account you can go straight to #Install.

Note: The @hocuspocus/transformer package is required for transforming Y.js binary into Tiptap JSON. It also requires Y.js installed for collaboration. If you don't have it installed, run npm install yjs in your project. This should happen automatically if you use NPM (as it automatically resolves peer dependencies).

The autoversioning feature automatically creates new versions of your document at regular intervals. This ensures that you have a comprehensive change history without manual intervention.

You can toggle this feature using the toggleVersioning command (default: disabled).

When you enable autoversioning, Tiptap creates new versions at regular intervals (30 seconds by default, only if the document has changed). This can create many versions, so you may want to increase the interval. To customize the interval, you can do the following:

When you revert to a previous version:

Note that reverting only affects the default fragment in the ydoc. When you revert the Tiptap content, the comments don't change (unless you specify a different field in the TiptapCollabProvider).

You can integrate the compare snapshots extension to highlight differences between versions, ensuring you choose the right version to restore.

In this example we retrieve the data of a version update and save it into a variable

In this example, the editor command helps you go back to version 4. When you use this command, it takes you back to how things were in version 4, and it also saves this old version as a new version called 'Revert to version'. This way, you can continue working from version 4, but it's now saved as the latest version.

In this example, when you revert to version 4 of your document, the editor automatically creates two new versions. The first new version captures and saves your document’s state just before reverting, serving as a backup. The second new version restores the document to version 4, allowing you to continue from here as your new starting point.

The examples above directly modify the document and do not provide local-only previews of the version. Therefore, you must create your own frontend solution for this requirement. You can leverage the stateless messaging system of the TiptapCloudProvider to request a specific version from the server.

Start by attaching a listener to the provider:

If you want to unbind the watcher, you can call the returned unbindWatchContent function like this:

Following this setup, you can trigger version.preview requests like so:

To go beyond previews and compare different versions visually, the compare snapshots extension provides an easy way to see the changes between any two versions within the editor.

This function turns the payload from the Collaboration provider into Tiptap JSON content.

This function sets up a watcher on your provider that watches the necessary events to react to version content changes. It also returns a new function that you can use to unwatch those events.

Here is a list of payloads that can be sent or received from the provider:

Request a document revert to a given version with optional title settings.

Creates a new version with an optional title.

This stateless message can be used to retrieve the last saved timestamp.

This stateless message includes information about newly created versions.

This stateless message includes information about a document revert.

**Examples:**

Example 1 (python):
```python
npm install @tiptap-pro/extension-snapshot @hocuspocus/transformer
```

Example 2 (python):
```python
npm install @tiptap-pro/extension-snapshot @hocuspocus/transformer
```

Example 3 (typescript):
```typescript
// Set the interval (in seconds) between autoversions
const ydoc = provider.configuration.document
ydoc.getMap<number>('__tiptapcollab__config').set('intervalSeconds', 900)
```

Example 4 (typescript):
```typescript
// Set the interval (in seconds) between autoversions
const ydoc = provider.configuration.document
ydoc.getMap<number>('__tiptapcollab__config').set('intervalSeconds', 900)
```

---

## Manage Documents with Tiptap

**URL:** https://tiptap.dev/docs/collaboration/documents

**Contents:**
- Manage Documents with Tiptap
- Enterprise on-premises solution
- Integrate documents
  - Note
- Retrieve and manage documents
  - Was this page helpful?

Collaboration Documents form the backbone of Tiptap Collaboration, storing everything from content and comments to versions and metadata using the Yjs format.

Typically, users manage these documents using the REST API or track changes with the Collaboration Webhook, which sends detailed updates. Tiptap converts the documents into HTML or JSON for you, so you don't have to deal directly with the Yjs format.

Integrate Collaboration and all other Tiptap features into your infrastructure.

Deploy our docker images in your own stack

Scale confidently to millions of users

Custom development and integration support in Chat

Integrating documents into your editor and application with Tiptap is straightforward. By adding the Tiptap Collaboration provider to your setup, documents are automatically stored and managed within the Tiptap Collaboration framework.

This integration immediately enables you to use all document features, like storing collaborative documents, managing version histories, using the REST API, and injecting content.

You can easily migrate your documents from our cloud to an on-premises server at a later time.

And now, you are all set to use the document features.

Use the REST API to fetch documents in JSON or HTML format for easy integration with your system. For immediate updates on changes, configure webhooks to receive real-time notifications.

Track changes in documents: The Snapshot extension in Tiptap Collaboration automatically captures and stores snapshots of documents at designated intervals. It also allows for manual versioning, enabling users to track detailed changes and document evolution.

Compare snapshots: The compare snapshots extension lets you visually compare two versions of a document, highlighting changes and their authors, helping you see modifications over time.

Inject content: Update the content of active documents with an Patch Document endpoint, which allows server-side modifications even during active user collaboration.

---

## Preserve images during conversion

**URL:** https://tiptap.dev/docs/conversion/import-export/docx/preserve-images

**Contents:**
- Preserve images during conversion
  - Note
- Import images
  - Callback process
  - Important considerations
  - Server implementation example
  - Was this page helpful?

Some documents that you're importing may include images that you may want to preserve in the converted document.

Tiptap does not provide an image upload service. You will need to implement your own server to handle image uploads.

If you import a DOCX file that has images, the conversion service can include those images in the resulting Tiptap JSON only if you provide an image upload callback URL.

This is a URL endpoint on your server that the conversion service will use to offload images during the import process.

In this configuration, imageUploadCallbackUrl is set to an endpoint (e.g., on your server) that will handle receiving image files. If this is not provided, the importer will strip out images from the document.

When an import is triggered, the conversion service will upload each embedded image to the URL you provided.

This endpoint can be implemented with any web framework or cloud function. The key steps you need to integrate are:

The Tiptap conversion service then takes that URL and inserts it into the Tiptap JSON as the src of an image node.

This example shows a simple server implementation that accepts image uploads & uploads them to an S3 bucket configured by environment variables.

Here is another implementation using bun with no dependencies:

**Examples:**

Example 1 (sql):
```sql
import { Editor } from '@tiptap/core'
 import { Import } from '@tiptap-pro/extension-import-docx'

 const editor = new Editor({
   // ... other editor options,
   extensions: [
     Import.configure({
       appId: '<your-app-id>',
       token: '<your-jwt>',
       imageUploadCallbackUrl: 'https://your-server.com/upload-image'
     })
   ]
 })
```

Example 2 (sql):
```sql
import { Editor } from '@tiptap/core'
 import { Import } from '@tiptap-pro/extension-import-docx'

 const editor = new Editor({
   // ... other editor options,
   extensions: [
     Import.configure({
       appId: '<your-app-id>',
       token: '<your-jwt>',
       imageUploadCallbackUrl: 'https://your-server.com/upload-image'
     })
   ]
 })
```

Example 3 (typescript):
```typescript
import { serve } from '@hono/node-server'
 import { Hono } from 'hono'
 import { Upload } from '@aws-sdk/lib-storage'
 import { S3Client } from '@aws-sdk/client-s3'

 const {
   AWS_ACCESS_KEY_ID,
   AWS_SECRET_ACCESS_KEY,
   AWS_REGION,
   AWS_S3_BUCKET,
   PORT = '3011',
   AWS_ENDPOINT,
   AWS_FORCE_STYLE,
 } = process.env

 if (!AWS_ACCESS_KEY_ID || !AWS_SECRET_ACCESS_KEY || !AWS_S3_BUCKET) {
   console.error('Please provide AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET')
   process.exit(1)
 }

 const s3 = new S3Client({
   credentials: {
     accessKeyId: AWS_ACCESS_KEY_ID,
     secretAccessKey: AWS_SECRET_ACCESS_KEY,
   },

   region: AWS_REGION,
   endpoint: AWS_ENDPOINT,
   forcePathStyle: AWS_FORCE_STYLE === 'true',
 })

 const app = new Hono() as Hono<any>

 app.post('/upload', async (c) => {
   // if you are using v2 import, you need this
   const file = await c.req.blob()

   const filename = c.req.header('File-Name')
   const fileType = c.req.header('Content-Type')
   // end
   // if you are using v1 import, you need this
   const body = await c.req.parseBody()
   const file = body['file']

   const filename = file.name
   const fileType = file.type
   // end

   if (!file) {
     return c.json({ error: 'No file uploaded' }, 400)
   }

   try {
     const data = await new Upload({
       client: s3,
       params: {
         Bucket: AWS_S3_BUCKET,
         Key: filename,
         Body: file,
         ContentType: fileType,
       },
     }).done()

     return c.json({ url: data.Location })
   } catch (error) {
     console.error(error)
     return c.json({ error: 'Failed to upload file' }, 500)
   }
 })

 serve({
   fetch: app.fetch,
   port: Number(PORT) || 3000,
 })
```

Example 4 (typescript):
```typescript
import { serve } from '@hono/node-server'
 import { Hono } from 'hono'
 import { Upload } from '@aws-sdk/lib-storage'
 import { S3Client } from '@aws-sdk/client-s3'

 const {
   AWS_ACCESS_KEY_ID,
   AWS_SECRET_ACCESS_KEY,
   AWS_REGION,
   AWS_S3_BUCKET,
   PORT = '3011',
   AWS_ENDPOINT,
   AWS_FORCE_STYLE,
 } = process.env

 if (!AWS_ACCESS_KEY_ID || !AWS_SECRET_ACCESS_KEY || !AWS_S3_BUCKET) {
   console.error('Please provide AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET')
   process.exit(1)
 }

 const s3 = new S3Client({
   credentials: {
     accessKeyId: AWS_ACCESS_KEY_ID,
     secretAccessKey: AWS_SECRET_ACCESS_KEY,
   },

   region: AWS_REGION,
   endpoint: AWS_ENDPOINT,
   forcePathStyle: AWS_FORCE_STYLE === 'true',
 })

 const app = new Hono() as Hono<any>

 app.post('/upload', async (c) => {
   // if you are using v2 import, you need this
   const file = await c.req.blob()

   const filename = c.req.header('File-Name')
   const fileType = c.req.header('Content-Type')
   // end
   // if you are using v1 import, you need this
   const body = await c.req.parseBody()
   const file = body['file']

   const filename = file.name
   const fileType = file.type
   // end

   if (!file) {
     return c.json({ error: 'No file uploaded' }, 400)
   }

   try {
     const data = await new Upload({
       client: s3,
       params: {
         Bucket: AWS_S3_BUCKET,
         Key: filename,
         Body: file,
         ContentType: fileType,
       },
     }).done()

     return c.json({ url: data.Location })
   } catch (error) {
     console.error(error)
     return c.json({ error: 'Failed to upload file' }, 500)
   }
 })

 serve({
   fetch: app.fetch,
   port: Number(PORT) || 3000,
 })
```

---

## Semantic Search

**URL:** https://tiptap.dev/docs/collaboration/documents/semantic-search

**Contents:**
- Semantic Search
- Live demo
- Get started
  - How it works
  - Perform a search
  - Keeping your API key secret
- Using Retrieval-Augmented Generation (RAG)
  - Was this page helpful?

Tiptap Semantic Search brings AI-native search capabilities to your document library, making it easy to discover relationships and connections across all your documents through contextual understanding provided by large language models.

Searching through large document archives can be challenging, especially if you miss the exact keywords. Semantic Search addresses this by interpreting the intent behind the search query and the contextual meaning within your documents.

The LLM’s interpretation is encoded as numerical representations, known as vectors or embeddings, which capture the semantic meaning of both new and existing content. These embeddings can then be easily compared to retrieve the most relevant documents.

Below is an interactive demo of Tiptap Semantic Search. Type into the editor on the left, and watch as the search results update in real time with the most contextually relevant pages from our public documentation. Discover more details about the demo in our examples.

When you input a query, the following things happen:

We have configured these operations in the background, making the complex process transparent to you as you set up and use this new Tiptap feature. With Tiptap Semantic Search, you can:

This is particularly valuable for knowledge management systems, document retrieval, idea generation, or any application where precise, context-aware search results are critical.

To perform a search, use the search endpoint as described in the REST API documentation.

Please make sure that you handle the requests in your own backend in order to keep your API key secret.

Use RAG to pull relevant information from your library and feed it into large language models, improving the quality of AI-generated content with contextually accurate data. Discover more details about the demo in our examples.

**Examples:**

Example 1 (json):
```json
curl -X POST https://YOUR_APP_ID.collab.tiptap.cloud/api/search \
  -H "Authorization: YOUR_SECRET_FROM_SETTINGS_AREA" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your search terms"}'
```

Example 2 (json):
```json
curl -X POST https://YOUR_APP_ID.collab.tiptap.cloud/api/search \
  -H "Authorization: YOUR_SECRET_FROM_SETTINGS_AREA" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your search terms"}'
```

---
