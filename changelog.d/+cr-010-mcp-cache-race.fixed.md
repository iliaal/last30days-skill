Prevent concurrent MCP server startups from replacing a complete engine cache with a partial extraction. Each process now stages its own files and serializes cache publication.
