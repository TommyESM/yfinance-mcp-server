# AGENTS.md - yfinance MCP Server Agent Instructions

## Purpose

This repository provides a Python FastMCP server exposing Yahoo Finance data to
LLM and application workflows. In the HomeLab workspace it primarily supports
StockAI as a Yahoo Finance fallback or secondary market-data integration.

## Operating Rules

- Follow the workspace-level `/opt/gitops/AGENTS.md` when present.
- Keep changes small and compatible with the existing upstream-style package.
- Do not create or switch git branches unless the user explicitly asks.
- Do not commit, push, reset, force-push or rewrite git history unless asked.
- Do not publish packages or change release metadata unless explicitly asked.
- Use `apply_patch` for manual file edits when possible.
- Use English for code comments and developer documentation.

## Repository Boundaries

- This repository owns the MCP server implementation and package metadata.
- HomeLab deployment is handled by infrastructure/runtime configuration outside
  this repository.
- StockAI is a consumer of this service, not part of this codebase.

## Safety Rules

- Do not install or upgrade dependencies without approval.
- Do not change package publishing configuration without approval.
- Do not assume Yahoo Finance data is complete, timely or suitable as the only
  source for trading decisions.
- Do not add secrets or local MCP client credentials to the repository.

## Technology Stack

- Python 3.10+
- FastMCP
- yfinance
- pandas
- MCP transports over stdio and optional HTTP/SSE

## Runtime And Deployment Model

- The server can run locally over stdio for MCP clients.
- The HomeLab infrastructure stack can run it over HTTP for services such as
  StockAI.
- The Docker/Portainer runtime relationship is documented in
  `HomeLab_Infrastructure` and `HomeLab_Docs`.

## Verification Commands

Use focused checks that match the change:

```bash
python -m py_compile main.py __main__.py
python test_mcp.py
```

For documentation-only changes:

```bash
git status --short
```

## Documentation Rules

- Keep `README.md` useful for MCP server setup and usage.
- Put HomeLab-specific service relationships and dependency diagrams in
  `HomeLab_Docs`.
- Do not mix StockAI operational runbooks into this repository.
