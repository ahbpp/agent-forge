# MongoDB Query Assistant 🍃

Welcome to the **MongoDB Query Assistant** powered by LangGraph!

## What can I do?

I help you query your MongoDB database using **natural language**. No need to write complex aggregation pipelines - just describe what you want in plain English.

## Example queries

- "Show me all users created this month"
- "What are the top 10 products by sales?"
- "Count how many orders are pending"
- "Find customers from New York with more than 5 orders"

## How it works

1. **You ask** a question in natural language
2. **I select** the appropriate collection
3. **I build** a MongoDB aggregation pipeline
4. **I execute** the query and return results

You'll see each step in real-time, including any retries if the query needs adjustment.

---

*Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Chainlit](https://chainlit.io)*
