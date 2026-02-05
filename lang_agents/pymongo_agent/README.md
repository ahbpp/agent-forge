# MongoDB Agent

A LangGraph-powered agent for interacting with MongoDB using natural language commands.

## Features

- **Two query branches:** Simple (single-collection) and Complex (multi-collection with `$lookup` joins)
- **Query validation:** Validates pipelines before execution
- **Automatic retry:** Retries on failures with error context
- **Chainlit UI:** Real-time step visualization

## How It Works

The agent has two execution paths:

### Simple Path (Fast)
For single-collection queries. The agent selects a collection and generates an aggregation pipeline.

### Complex Path
For multi-collection queries requiring `$lookup` joins. Includes validation before execution.

### Agent Graph

```
                                    ┌─────────────────────────────────────────────────┐
                                    │              COMPLEX PATH                        │
                                    │                                                  │
                                    │  ┌──────────────┐    ┌───────────┐              │
                              ┌────►│  │ query_builder│───►│ validator │              │
                              │     │  └──────────────┘    └─────┬─────┘              │
                              │     │         ▲                  │                    │
                              │     │         │            ┌─────┴─────┐              │
                              │     │         │            ▼           ▼              │
                              │     │    (retry if    ┌────────┐  ┌─────────────────┐ │
                              │     │     invalid)    │executor│  │validation_failure│ │
                              │     │                 └────┬───┘  └────────┬────────┘ │
                              │     └──────────────────────│───────────────│──────────┘
                              │                            ▼               ▼
START ──► planner ───────────┼─────────────────────────► END ◄────────────┘
                              │
                              │     ┌─────────────────────────────────────────────────┐
                              │     │              SIMPLE PATH                         │
                              │     │                                                  │
                              └────►│  ┌──────────────┐    ┌───────────────┐          │
                                    │  │simple_handler│───►│simple_executor│──► END   │
                                    │  └──────────────┘    └───────────────┘          │
                                    └─────────────────────────────────────────────────┘
```

### Nodes

| Node | Description |
|------|-------------|
| `planner` | Analyzes query, decides simple/complex/direct_answer path |
| `query_builder` | Generates MongoDB aggregation pipeline with `$lookup` stages |
| `validator` | Validates pipeline (collections exist, field names, etc.) |
| `executor` | Executes validated pipeline |
| `validation_failure` | Handles failures after max validation retries |
| `simple_handler` | Selects collection for simple queries |
| `simple_executor` | Generates and executes single-collection pipeline |


## Installation

1. Set up environment variables in `.env` file
   ```
   PYMONGO_HOST=your_mongodb_host (e.g. mongodb://host.docker.internal:27017/ if running with docker and mongodb://localhost:27017/ if running `lang_agents/pymongo_agent/example.py`)
   PYMONGO_USER=your_mongodb_user (if needed)
   PYMONGO_PASSWORD=your_mongodb_password (if needed)
   OPENAI_API_KEY=your_openai_api_key (if needed)
   LANGSMITH_TRACING=true
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langchain_api_key (it is not necessary, but it is very helpful for debugging)
   LANGSMITH_PROJECT=your_langsmith_project (if needed)
   OLLAMA_HOST="http://host.docker.internal:11434" (if running with docker and use ollama models)
   ```

## Usage


### Create test MongoDB
1. Run MongoDB docker container
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```
2. Create a test database and collection and insert some test data
```bash
python lang_agents/pymongo_agent/mongo_create_db.py
```
3. Run mongosh to check the database (optional)
```bash
docker exec -it mongodb mongosh
```
**mongosh commands:**  
3.1 Switch to target database: `use user_management`  
3.2 List all collections: `show collections`  
3.3 List all documents in the users collection: `db.users.find().pretty()`  
See more: https://www.mongodb.com/docs/manual/mongo/



### Command Line Interface

```bash
# Run a single query
python -m lang_agents.pymongo_agent.example "How many users are from New York?"

# Run in interactive mode
python -m lang_agents.pymongo_agent.example --interactive
```

### Run in docker (`langgraph up`)

```bash
langgraph up
```

See `langgraph.json` for configuration
