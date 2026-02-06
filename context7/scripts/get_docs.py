#!/usr/bin/env python3
"""Fetch documentation from Context7 for a specific library."""
import sys
import urllib.request
import urllib.parse
import ssl


def get_docs(library_id: str, query: str, api_key: str = "", fmt: str = "txt") -> None:
    """Fetch documentation context for a library from Context7."""
    params = {
        "libraryId": library_id,
        "query": query,
    }
    if fmt:
        params["type"] = fmt

    url = f"https://context7.com/api/v2/context?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "Context7-Skill/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ctx = ssl.create_default_context()
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read().decode()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not data.strip():
        print("No documentation found for this query.")
        sys.exit(1)

    print(data)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: get_docs.py <library_id> <query> [api_key] [format:txt|json]")
        print("  library_id: Context7 library ID (e.g., /websites/react_dev)")
        print("  query: Your question (e.g., 'How to use useState')")
        sys.exit(1)

    lib_id = sys.argv[1]
    q = sys.argv[2]
    key = sys.argv[3] if len(sys.argv) > 3 else ""
    output_fmt = sys.argv[4] if len(sys.argv) > 4 else "txt"
    get_docs(lib_id, q, key, output_fmt)
