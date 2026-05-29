# Container for the Ashigaru MCP server (optional — you can also just `pip install` on host).
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ashigaru ./ashigaru
RUN pip install --no-cache-dir ".[all]"

# stdio MCP server by default
ENTRYPOINT ["ashigaru-mcp"]
