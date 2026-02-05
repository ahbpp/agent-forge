# MongoDB Query Assistant 🍃

Welcome to the **MongoDB Query Assistant** powered by LangGraph!

## What can I do?

I help you query your MongoDB database using **natural language**. No need to write complex aggregation pipelines - just describe what you want in plain English.

## Query Types

The agent has two branches:
- **Simple:** Single-collection queries (fast path)
- **Complex:** Multi-collection queries with `$lookup` joins

## Example queries

**Simple queries:**
- "Show me all patients in moleimages collection"
- "Count how many orders are pending"
- "Find images with more than 3 diagnoses"

**Complex queries:**
- "Look in both moleimages and diagnosis collections, find all images for patient 123"
- "Get orders with customer details from customers collection"

## How it works

1. **Planning** - Analyze your request and decide simple vs complex path
2. **Query Building** - Generate MongoDB aggregation pipeline
3. **Validation** - Check pipeline for errors before execution
4. **Execution** - Run query and return results

You'll see each step in real-time, including the pipeline stages and any retries if needed.

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

---

*Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Chainlit](https://chainlit.io)*
