---
name: context7
description: >
  Fetch up-to-date, version-specific library/framework documentation and code examples via Context7 API.
  Use when the user asks about programming libraries, frameworks, or APIs and needs current documentation
  (not outdated training data). Triggers: "use context7", "使用 context7", "查詢文檔", "查最新文件", or any request for
  library docs, API references, code examples, framework usage, setup/configuration steps. Also use when
  Claude's training data may be outdated for fast-moving libraries like Next.js, React, Vue, Tailwind,
  LangChain, etc. Supports specifying library versions (e.g., "Next.js 15", "React 18").
---

# Context7 - Up-to-date Library Documentation

Fetch current documentation and code examples for any programming library via the Context7 API.

## Workflow

### Step 1: Resolve Library ID

Run `scripts/search_library.py` to find the Context7-compatible library ID:

```bash
python3 scripts/search_library.py "<library_name>" "<user_question>" ["<api_key>"]
```

- `library_name`: The library to search for (e.g., "react", "next.js", "langchain")
- `user_question`: The user's task/question (used for relevance ranking)
- `api_key`: Optional, for higher rate limits

Select the best match by `trustScore` and `verified` status. Prefer results with higher `totalSnippets`.

If the user provides a library ID directly in `/org/project` format, skip this step.

### Step 2: Fetch Documentation

Run `scripts/get_docs.py` with the resolved library ID:

```bash
python3 scripts/get_docs.py "<library_id>" "<query>" ["<api_key>"] ["txt"|"json"]
```

- `library_id`: The Context7 library ID from Step 1 (e.g., `/websites/react_dev`)
- `query`: Be specific. Good: "How to set up authentication with JWT in Express.js". Bad: "auth"
- Format `txt` (default) returns markdown-formatted docs ready for context injection
- Format `json` returns structured JSON with title, content, and source fields

### Step 3: Use the Documentation

Present the fetched documentation to help answer the user's question. Include:
- Relevant code examples from the docs
- Source links for reference
- Version-specific notes if applicable

## Tips

- For version-specific docs, include the version in the query (e.g., "Next.js 14 middleware")
- Or use a versioned library ID: `/vercel/next.js/v15.1.8`
- Limit to 3 API calls per question; use best available result
- The API is free without a key but rate-limited; use an API key from context7.com/dashboard for higher limits

## TLS Note

The Context7 API requires TLS 1.2. The scripts handle this automatically via `ssl.TLSVersion.TLSv1_2`.
