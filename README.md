# Lang Agents

A Python package for building and managing language agents with support for multiple LLM providers.

## Installation

```bash
pip install lang-agents
```

## Quick Start

```python
from lang_agents import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(
        provider="openai",
        model="gpt-4-turbo-preview"
    )
)

response = await agent.run("Your prompt here")
```

## Development

```bash
# Install dev dependencies
pip install ".[dev]"

# Run tests
pytest
```

## License

MIT 