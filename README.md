# Langie: Customer Support LangGraph Agent

This repository contains a LangGraph-based AI agent for customer support workflows, as per assignment requirements.

## Files
- `langie_agent.py`: Python implementation of the 11-stage agent with state persistence and mocked MCP clients.
- `langie_config.yaml`: Configuration defining input schema, stages, modes, abilities, and MCP mappings.

## Setup
1. Install dependencies: `pip install langgraph langchain langchain-openai`
2. Set `OPENAI_API_KEY` environment variable.
3. Run `langie_agent.py` in Python 3.8+.

## Demo
- Input: JSON with `customer_name`, `email`, `query`, `priority`, `ticket_id`.
- Output: Logs for all stages, final structured payload.
- See `langie_agent.py` for sample run.

## Notes
- Mocked MCP clients used for demo; replace with `langchain-mcp-adapters` for real ATLAS/COMMON servers.
