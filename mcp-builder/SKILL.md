---
name: mcp-builder
description: |
  Build Model Context Protocol (MCP) servers that enable Claude to interact with external services. Creates TypeScript or Python MCP servers with proper tool schemas, error handling, and testing.
  Use when: building MCP servers, integrating APIs, extending Claude's capabilities, or when user mentions MCP, Model Context Protocol, Claude擴展, external tools, MCP server.
  Triggers: "MCP", "Model Context Protocol", "MCP server", "Claude擴展", "external tools", "整合API"
version: 1.0.0
---

# MCP Server Builder

Build production-ready MCP servers following the official Model Context Protocol specification.

## Development Phases

### Phase 1: Research & Planning

1. **Study the target API**
   - Authentication methods (OAuth, API key, JWT)
   - Rate limits and pagination patterns
   - Available endpoints and data models

2. **Design tool strategy**
   - Balance API coverage with workflow tools
   - Group related operations logically
   - Plan for error handling and retries

### Phase 2: Project Setup

#### TypeScript (Recommended)

```bash
mkdir my-mcp-server && cd my-mcp-server
npm init -y
npm install @modelcontextprotocol/sdk zod
npm install -D typescript @types/node
```

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true
  },
  "include": ["src/**/*"]
}
```

#### Python (FastMCP)

```bash
uv init my-mcp-server && cd my-mcp-server
uv add fastmcp
```

### Phase 3: Implementation

#### TypeScript Server Template

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0",
});

// Define tool schema
const SearchSchema = z.object({
  query: z.string().describe("Search query"),
  limit: z.number().optional().default(10).describe("Max results"),
});

// Register tool
server.tool(
  "search",
  "Search for items matching the query",
  SearchSchema.shape,
  async ({ query, limit }) => {
    try {
      const results = await performSearch(query, limit);
      return {
        content: [{ type: "text", text: JSON.stringify(results, null, 2) }],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error: ${error.message}` }],
        isError: true,
      };
    }
  }
);

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

#### Python Server Template

```python
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
async def search(query: str, limit: int = 10) -> str:
    """Search for items matching the query.

    Args:
        query: Search query string
        limit: Maximum number of results to return
    """
    try:
        results = await perform_search(query, limit)
        return json.dumps(results, indent=2)
    except Exception as e:
        raise McpError(f"Search failed: {e}")

if __name__ == "__main__":
    mcp.run()
```

### Phase 4: Tool Design Principles

#### Naming Convention

```
{service}_{action}_{target}

Examples:
- github_create_issue
- slack_send_message
- db_query_users
```

#### Error Handling

```typescript
// Good: Actionable error messages
return {
  content: [{
    type: "text",
    text: `Authentication failed. Please check:
1. API key is set in environment variable API_KEY
2. Key has required permissions: read, write
3. Key has not expired`
  }],
  isError: true,
};

// Bad: Vague error
return { content: [{ type: "text", text: "Error occurred" }], isError: true };
```

#### Response Design

```typescript
// Return focused, relevant data
return {
  content: [{
    type: "text",
    text: JSON.stringify({
      id: item.id,
      name: item.name,
      status: item.status,
      // Omit unnecessary fields
    }, null, 2)
  }]
};
```

### Phase 5: Testing

#### Using MCP Inspector

```bash
# Install inspector
npm install -g @modelcontextprotocol/inspector

# Test your server
mcp-inspector node dist/index.js
```

#### Evaluation Questions

Create 10 test questions that:
- Are complex and realistic
- Can be verified independently
- Cover different tool combinations
- Test error handling paths

## Claude Desktop Configuration

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/dist/index.js"],
      "env": {
        "API_KEY": "your-api-key"
      }
    }
  }
}
```

## Checklist

- [ ] All tools have clear descriptions
- [ ] Input schemas use Zod with `.describe()`
- [ ] Errors are actionable with next steps
- [ ] Responses are focused and relevant
- [ ] No code duplication
- [ ] Consistent naming convention
- [ ] Pagination handled for list operations
- [ ] Rate limiting respected
- [ ] Tests cover main workflows
- [ ] Documentation includes examples

## Reference

- MCP Specification: https://modelcontextprotocol.io
- TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- Python SDK (FastMCP): https://github.com/jlowin/fastmcp
