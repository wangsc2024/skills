# Tiptap - Collaboration

**Pages:** 2

---

## Integrate the Collaboration provider

**URL:** https://tiptap.dev/docs/collaboration/provider/integration

**Contents:**
- Integrate the Collaboration provider
- Set up the provider
  - Set up private registry
  - Note for On-Premises Customers
- Configure the collaboration provider
  - Optimize reconnection timings
- Unsynced changes
- Rate limits
  - Default rate limits (per source IP):
  - Was this page helpful?

Together with the Collaboration backend, providers serve as the backbone for real-time collaborative editing. They establish and manage the communication channels between users, ensuring that updates and changes to documents are synchronized across all participants.

Providers handle the complexities of real-time data exchange, including conflict resolution, network reliability, and user presence awareness.

The TiptapCollabProvider adds advanced features tailored for collaborative environments, such as WebSocket message authentication, debug modes, and flexible connection strategies.

First, install the provider package in your project using:

Note that you need to follow the instructions here to set up access to the private registry.

For a basic setup, connect to the Collaboration backend by specifying the document's name, your app's ID (for cloud setups), or the base URL (for on-premises), along with your JWT.

Depending on your framework, register a callback to the Collaboration backend, such as useEffect() in React or onMounted() in Vue.js.

If you are hosting your collaboration environment on-premises, replace the appId parameter with baseUrl in your provider configuration to connect to your server.

The Tiptap Collaboration provider offers several settings for custom configurations. Review the tables below for all parameters, practical use cases, and key concepts like "awareness".

The provider’s reconnection settings are preset for optimal performance in production settings. If you need to adjust these settings for specific scenarios, you can do so with our delay configurations.

Adjust initial delays, apply exponential backoff, or set maximum wait times to fine-tune your application's reconnection behavior, balancing responsiveness with server efficiency.

Note that these settings can only be configured when creating the TiptapCollabProviderWebsocket instance separately. You'll then need to pass it to the TiptapCollabProvider (as websocketProvider).

The provider maintains an integer that keeps track of the number of unsynced changes. Whenever the server receives a change, it acknowledges it, and the provider decrements the counter.

You should monitor this counter and inform the user when their changes are not yet synced: either before they leave the page or after a certain timeout or number of unsynced changes.

A "change" may be a single character, a single node, or a whole document, depending on your custom use-case, so in general the counter should be at 0 or slightly above, but only for a short period of time (essentially the latency of the connection).

You can get the current value of the counter by accessing TiptapCollabProvider.unsyncedChanges, or by subscribing to the unsyncedChanges event.

To maintain system integrity and protect from misconfigured clients, our infrastructure—including the management API and websocket connections through the TiptapCollabProvider—is subject to rate limits.

If you encounter these limits under normal operation, please email us.

**Examples:**

Example 1 (python):
```python
npm install @tiptap-pro/provider
```

Example 2 (python):
```python
npm install @tiptap-pro/provider
```

Example 3 (jsx):
```jsx
const doc = new Y.Doc()

useEffect(() => {
  const provider = new TiptapCollabProvider({
    name: note.id, // Document identifier
    appId: 'YOUR_APP_ID', // replace with YOUR_APP_ID from Cloud dashboard
    token: 'YOUR_JWT', // Authentication token
    document: doc,
    user: userId,
  })
}, [])
```

Example 4 (jsx):
```jsx
const doc = new Y.Doc()

useEffect(() => {
  const provider = new TiptapCollabProvider({
    name: note.id, // Document identifier
    appId: 'YOUR_APP_ID', // replace with YOUR_APP_ID from Cloud dashboard
    token: 'YOUR_JWT', // Authentication token
    document: doc,
    user: userId,
  })
}, [])
```

---

## Server metrics and statistics

**URL:** https://tiptap.dev/docs/collaboration/operations/metrics

**Contents:**
- Server metrics and statistics
  - Review the postman collection
- Access the API
  - Authentication
  - Document identifiers
- Server statistics endpoint
  - Caution
- Document statistics endpoint
- Server health endpoint
  - Was this page helpful?

The Tiptap Collaboration API offers several endpoints to access real-time statistics and health information for both the server and individual documents. A simplified version of the metrics is also available in the cloud dashboard.

These endpoints help to troubleshoot issues, monitor server performance, or build analytics dashboards for insights into user interactions and system status. Integrating statistics into your monitoring systems allows you to proactively manage your collaboration environment's health.

Experiment with the REST API by visiting our Postman Collection.

The REST API is exposed directly from your Document server at your custom URL:

Authenticate your API requests by including your API secret in the Authorization header. You can find your API secret in the settings of your Tiptap Cloud dashboard.

If your document identifier contains a slash (/), encode it as %2F, e.g., using encodeURIComponent.

This endpoint provides basic statistics about the Tiptap Collaboration server, offering insights into overall activity and usage metrics.

The total number of connections in the last 30 days and the lifetime connection count are presented as strings due to their internal representation as BigInt values.

Example: Server statistics

Retrieve statistics for a specific document by its identifier. Use this endpoint to monitor real-time user engagement with a document.

Example: Statistics of a document named :identifier

Use this call to check liveness, readiness, and cconnectivity to essential components like the database and Redis.

Example: Issue with Redis

Example: No Redis detected

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
GET /api/statistics
```

Example 4 (unknown):
```unknown
GET /api/statistics
```

---
