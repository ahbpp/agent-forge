from logging import Logger
import json
import pprint

from pathlib import Path

from pydantic import BaseModel, Field

from typing import TypedDict, List, Dict, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from lang_agents.pymongo_agent.utils import (
    get_read_mongo_client, 
    get_schema, 
    list_collections,
    aggregate_mongo_doc_to_json_serializable,
    get_model_from_config,
    parse_aggregate_query_tool_call
)
from lang_agents.pymongo_agent.configuration import Configuration, ModelProvider
from dotenv import load_dotenv


load_dotenv()


logger = Logger(__name__)
logger.setLevel("INFO")

# Initialize MongoDB client
mongo_client = get_read_mongo_client()

# Initialize model
model = get_model_from_config(Configuration())

class Collection(TypedDict):
    collection: str

class AggregateQuery(BaseModel):
    query: List[Dict[str, Any]] = Field(description="The PyMongo aggregation pipeline (list of dictionaries)")


def run_aggregate(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = Configuration.from_runnable_config(config)
    
    messages = state['messages']
    last_message = messages[-1]

    tool_call = last_message.tool_calls[0]
    collection_name = tool_call['args']['collection']
    collection = mongo_client[configurable.database][collection_name]

    collection_schema = get_schema(collection)

    system_message = """
    Reflect on the following interaction 
    Create a MongoDB aggregate query for {collection_name} collection 
    You must call the AggregateQuery tool with the aggregate query. 
    Check that all `[`, `]`, `{{`, `}}` are balanced and in the correct order.
    Do not ask any questions or permissions

    Here is the schema for the collection:
    <schema>
    {collection_schema}
    </schema>
    """.format(collection_name=collection_name, collection_schema=collection_schema)
    
    llm_with_tools = model.bind_tools(tools=[AggregateQuery], tool_choice=True)
    response = llm_with_tools.invoke([SystemMessage(content=system_message)]+messages[:-1])
    try:
        query = response.tool_calls[0]["args"]["query"]
    except (IndexError, KeyError) as e:
        _, query = parse_aggregate_query_tool_call(response)
    pprint.pprint(query)
    if isinstance(query, str):
        query = json.loads(query)
    if isinstance(query, dict):
        query = [query]

    # Execute the query
    print(f"Executing query: {query}")
    cursor = collection.aggregate(query)
    result = [aggregate_mongo_doc_to_json_serializable(doc) 
              for doc in cursor]
    pprint.pprint(result)

    content = {
        "query": query,
        "result": result
    }
    content = json.dumps(content, indent=2)

    return {"messages": [{"role": "tool", 
                          "content": content, 
                          "tool_call_id": tool_call['id']}]}


def handle_request(state: MessagesState, 
                   config: RunnableConfig, 
                   store: BaseStore):
    messages = state['messages']
    configurable = Configuration.from_runnable_config(config)

    collections = list_collections(mongo_client, configurable.database)

    system_msg = """You are a MongoDB read-only assistant. 

    Here are the available collections in the database:
    <collections>
    {collections}
    </collections>

    Here are your instructions for reasoning about the user's messages:

    1. First, analyze the user's message to understand their intent
    2. Decide for which collection the user wants to query. Call the Collection tool with the collection name
    3. Do not ask any permission questions. Just call the Collection tool
    """.format(collections=collections)


    response = model.bind_tools([Collection]).invoke([SystemMessage(content=system_msg)]+messages)

    return {"messages": [response]}


builder = StateGraph(MessagesState, config_schema=Configuration)

builder.add_node(handle_request)
builder.add_node(run_aggregate)

builder.add_edge(START, "handle_request")
builder.add_edge("handle_request", "run_aggregate")
builder.add_edge("run_aggregate", END)

# Compile the graph
graph = builder.compile()
