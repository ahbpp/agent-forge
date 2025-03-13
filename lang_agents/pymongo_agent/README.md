# MongoDB Agent

A LangGraph/LangChain-powered agent for interacting with MongoDB using natural language commands.

## Features

- **Natural Language Queries**: Query MongoDB using plain English
- **Read-Only Operations**: Safely execute read-only MongoDB operations (find, aggregate, count, etc.)
- **Query Caching**: Automatically saves queries and results for reproducibility
- **Interactive Mode**: Use the agent in an interactive shell
- **Multiple MongoDB Operations**: Support for various MongoDB operations:
  - List databases
  - List collections in a database
  - Get collection schema
  - Find documents (with query filters)
  - Aggregate pipeline queries
  - Count documents
  - Get distinct values for a field

## Requirements

- Python 3.9+
- MongoDB connection (configured via environment variables)
- OpenAI API Key

## Installation

1. Set up environment variables:
   ```
   PYMONGO_HOST=your_mongodb_host
   PYMONGO_USER=your_mongodb_user
   PYMONGO_PASSWORD=your_mongodb_password
   OPENAI_API_KEY=your_openai_api_key
   ```

2. Install dependencies:
   ```
   uv pip install ".[dev]"
   ```

## Usage

### Command Line Interface

```bash
# Run a single query
python -m lang_agents.pymongo_agent.example "Show me the first 5 documents in the users collection"

# Run in interactive mode
python -m lang_agents.pymongo_agent.example --interactive
```

### As a Library

```python
from langchain_core.messages import HumanMessage
from lang_agents.pymongo_agent.pymongo_agent import mongodb_agent

# Create a human message with your query
message = HumanMessage(content="List all databases")

# Invoke the agent with your message
result = mongodb_agent.invoke({"messages": [message]})

# Print the result
print(result.messages[-1].content)
```

## How It Works

The MongoDB agent uses a LangGraph workflow with three main steps:

1. **Analyze Request**: Takes a natural language query and determines what MongoDB operation to run
2. **Execute Operation**: Runs the determined MongoDB operation with the necessary parameters
3. **Format Response**: Formats the results in a human-readable way and returns it to the user

All queries and results are cached in the `cache` directory for reproducibility.

## Cache Format

The cache stores each query in a JSON file with:
- Timestamp of when the query was run
- Request ID (for tracking)
- Original query (as interpreted by the agent)
- Results of the query

## Limitations

- Read-only operations only (no insert, update, delete)
- Requires proper MongoDB connection details
- Limited to the operations implemented in the tools

## License

This project is open-source and available under the MIT License. 