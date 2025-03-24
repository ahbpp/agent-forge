# AgentForge

**Note:** It is a very beginning version of a collection of agents which can be helpful for my daily routine. 

A Python package for building and managing language agents based on the LangGraph framework with support for multiple LLM providers.

## Project Overview

AgentForge contains several different LLM agents built with the LangGraph framework. Some of the agents in this repository are based on [langchain-academy](https://github.com/langchain-ai/langchain-academy) modules but have been enhanced with additional features:

### Research Agent
Based on langchain-academy module 4

### Task Maistro
Based on langchain-academy module 5, with the following additional features:
* Additional ideas collection for ideas and notes
* Support for Ollama, Gemini, and DeepSeek models

### PyMongo Agent

Agent for interacting with MongoDB using natural language commands.
See [README](lang_agents/pymongo_agent/README.md) for more details.

**Note:** The `qwen2.5:7b` model deployed with Ollama locally works surprisingly well for these agents.

## Installation

```bash
uv pip install ".[dev]"
```

## License

MIT 