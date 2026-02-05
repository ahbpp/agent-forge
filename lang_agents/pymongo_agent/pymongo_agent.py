import json


from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Literal
from logging import Logger
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import dispatch_custom_event

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from lang_agents.pymongo_agent.utils import (
    get_read_mongo_client, 
    get_schema, 
    list_collections,
    aggregate_mongo_doc_to_json_serializable,
    get_model_from_config,
    parse_aggregate_query_tool_call
)
from lang_agents.pymongo_agent.configuration import Configuration


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
    """
    Invoke LLM to generate MongoDB aggregate query
    Run the query and return the result with retry logic for error handling
    """
    
    configurable = Configuration.from_runnable_config(config)
    max_retries = configurable.max_retry_attempts
    
    messages = state['messages']
    last_message = messages[-1]

    tool_call = last_message.tool_calls[0]
    collection_name = tool_call['args']['collection']
    collection = mongo_client[configurable.database][collection_name]

    collection_schema = get_schema(collection)

    base_system_message = """
    Reflect on the following interaction 
    Create a MongoDB aggregate query for {collection_name} collection 
    You must call the AggregateQuery tool with the `query` argument.
    Check that the tool call is valid and all syntax is correct.
    Check that all `[`, `]`, `{{`, `}}` are balanced and in the correct order.
    Do not ask any questions or permissions

    Here is the schema for the collection:
    <schema>
    {collection_schema}
    </schema>
    """.format(collection_name=collection_name, collection_schema=collection_schema)
    
    llm_with_tools = model.bind_tools(tools=[AggregateQuery], tool_choice=True)
    
    last_error = None
    query = None
    result = []
    
    for attempt in range(max_retries):
        try:
            # Add error context to system message if this is a retry
            if last_error:
                system_message = base_system_message + f"""
                
    IMPORTANT: The previous query attempt failed with the following error:
    <error>
    {last_error}
    </error>
    
    Please fix the query and try again. Attempt {attempt + 1} of {max_retries}.
    """
                print(f"\n🔄 Retry attempt {attempt + 1}/{max_retries} due to error: {last_error}\n")
                
                # Dispatch custom event for UI visibility
                dispatch_custom_event(
                    "retry_attempt",
                    {
                        "type": "retry",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": last_error,
                        "collection": collection_name
                    },
                    config=config
                )
            else:
                system_message = base_system_message
            
            response = llm_with_tools.invoke([SystemMessage(content=system_message)]+messages[:-1])
            
            try:
                query = response.tool_calls[0]["args"]["query"]
            except (IndexError, KeyError) as e:
                # Sometimes not OpenAI models return the tool call in a wrong format
                # parse_aggregate_query_tool_call handles this (not the best solution, but it works)
                _, query = parse_aggregate_query_tool_call(response)
            
            if isinstance(query, str):
                query = json.loads(query)
            if isinstance(query, dict):
                query = [query]

            # Execute the query
            if configurable.run_query:
                logger.info(f"Executing query: {query}")
                print(f"\n🔍 Executing query: {json.dumps(query, indent=2)}\n")
                cursor = collection.aggregate(query)
                result = [aggregate_mongo_doc_to_json_serializable(doc) 
                        for doc in cursor]
                # Success - break out of retry loop
                print(f"\n✅ Query executed successfully. Retrieved {len(result)} documents.\n")
                last_error = None
                break
            else:
                logger.info(f"Query not executed, because run_query is False in the configuration")
                result = []
                break
                
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {str(e)}"
            logger.error(f"Attempt {attempt + 1} failed: {last_error}")
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Attempt {attempt + 1} failed: {last_error}")
    
    # Build response content
    content = {
        "query": query,
        "result": result,
        "count": len(result)
    }
    
    if last_error:
        content["error"] = last_error
        content["retries_exhausted"] = True
        print(f"\n❌ Query failed after {max_retries} attempts. Last error: {last_error}\n")
    
    content = json.dumps(content, indent=2)

    return {"messages": [{"role": "tool", 
                          "content": content, 
                          "tool_call_id": tool_call['id']}]}


def handle_request(state: MessagesState, 
                   config: RunnableConfig, 
                   store: BaseStore):
    """
    Handle the user's request by analyzing the message and executing the appropriate action.
    Handles collection selection for queries or provides direct responses for simple questions.
    """
    messages = state['messages']
    configurable = Configuration.from_runnable_config(config)

    collections = list_collections(mongo_client, configurable.database)
    
    # Build collection info with full schema for better collection selection
    collection_info = []
    for coll_name in collections:
        coll = mongo_client[configurable.database][coll_name]
        schema = get_schema(coll)
        schema_str = json.dumps(schema, indent=2, default=str)
        collection_info.append(f"Collection: {coll_name}\nSchema:\n{schema_str}")
    
    collections_desc = "\n\n".join(collection_info)

    system_msg = """You are a MongoDB read-only assistant that helps users query database collections using natural language.

    Available collections in the database with their full schemas:
    <collections>
    {collections_desc}
    </collections>

    Follow these steps for each user request:
    
    1. Analyze the user's query to determine their information need
    2. For queries requiring database access:
       - Identify the most relevant collection(s) based on the schema described above
       - Call the Collection tool with the appropriate collection name
    3. For general MongoDB questions or schema information:
       - Answer directly without using tools
    4. Always provide concise, accurate responses
    5. For empty results, explain possible reasons

    Remember: You can only perform read operations (find, aggregate, count). Write operations (insert, update, delete) are not permitted.
    """.format(collections_desc=collections_desc)


    response = model.bind_tools([Collection]).invoke([SystemMessage(content=system_msg)]+messages)
    
    # Print selected collection if tool was called
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.get('name') == 'Collection' or 'collection' in tool_call.get('args', {}):
                selected_collection = tool_call['args'].get('collection')
                print(f"\n📁 Selected collection: {selected_collection}\n")

    return {"messages": [response]}


def route_message(state: MessagesState, 
                  config: RunnableConfig, 
                  store: BaseStore) -> Literal[END, "run_aggregate"]: # type: ignore
    """
    Route the message to the appropriate node based on the collection selection.
    """
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        return "run_aggregate"


builder = StateGraph(MessagesState, config_schema=Configuration)

builder.add_node(handle_request)
builder.add_node(run_aggregate)


builder.add_edge(START, "handle_request")
builder.add_conditional_edges("handle_request", route_message)
builder.add_edge("run_aggregate", END)

# Compile the graph
graph = builder.compile()
