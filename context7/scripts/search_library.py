#!/usr/bin/env python3
"""Search for a library on Context7 and return matching library IDs."""
import sys
import json
import urllib.request
import urllib.parse
import ssl


def search_library(library_name: str, query: str = "", api_key: str = "") -> None:
    """Search Context7 for a library by name."""
    params = {"libraryName": library_name}
    if query:
        params["query"] = query

    url = f"https://context7.com/api/v2/libs/search?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "Context7-Skill/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ctx = ssl.create_default_context()
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    results = data.get("results", data) if isinstance(data, dict) else data
    if not results:
        print(json.dumps({"error": "No libraries found", "query": library_name}))
        sys.exit(1)

    # Show top 10 results
    output = []
    for lib in results[:10]:
        output.append({
            "id": lib.get("id", ""),
            "title": lib.get("title", ""),
            "description": lib.get("description", "")[:200],
            "totalSnippets": lib.get("totalSnippets", 0),
            "trustScore": lib.get("trustScore", 0),
            "verified": lib.get("verified", False),
        })

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: search_library.py <library_name> [query] [api_key]")
        sys.exit(1)

    lib_name = sys.argv[1]
    q = sys.argv[2] if len(sys.argv) > 2 else ""
    key = sys.argv[3] if len(sys.argv) > 3 else ""
    search_library(lib_name, q, key)
