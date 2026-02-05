"""
PyMongo Agent with Multi-Collection Query Support

This agent supports:
- Simple single-collection queries (fast path)
- Complex multi-collection queries with $lookup joins
- Query validation before execution
- Automatic retry on failures

Graph Structure:
    START → planner → [conditional routing]
                     ├─► query_builder → validator → [conditional] → executor → END
                     │                       └─► query_builder (retry on invalid)
                     ├─► simple_handler → simple_executor → END
                     └─► END (direct answer)
"""
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import dispatch_custom_event

from langgraph.graph import StateGraph, START, END

from lang_agents.pymongo_agent.state import MultiCollectionState
from lang_agents.pymongo_agent.configuration import Configuration
from lang_agents.pymongo_agent.nodes import (
    planner_node,
    query_builder_node,
    validator_node,
    executor_node
)
from lang_agents.pymongo_agent.utils import (
    get_read_mongo_client, 
    get_schema, 
    list_collections,
    aggregate_mongo_doc_to_json_serializable,
    get_model_from_config,
    parse_aggregate_query_tool_call
)


load_dotenv()

logger = logging.getLogger(__name__)

# Initialize MongoDB client
mongo_client = get_read_mongo_client()

# Initialize model (for simple path)
model = get_model_from_config(Configuration())


# =============================================================================
# Simple Path Nodes (Backwards Compatibility)
# =============================================================================

class Collection(TypedDict):
    collection: str

class AggregateQuery(BaseModel):
    query: List[Dict[str, Any]] = Field(description="The PyMongo aggregation pipeline (list of dictionaries)")


def simple_handler(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Handle simple single-collection queries.
    This is the fast path for queries that don't need joins.
    """
    messages = state.get("messages", [])
    plan = state.get("plan", {})
    configurable = Configuration.from_runnable_config(config)
    
    primary_collection = plan.get("primary_collection", "") if plan else ""
    
    if not primary_collection:
        # Fallback: ask LLM to select collection
        collections = list_collections(mongo_client, configurable.database)
        collection_schemas = {}
        collection_info = []
        for coll_name in collections:
            coll = mongo_client[configurable.database][coll_name]
            schema = get_schema(coll)
            collection_schemas[coll_name] = schema
            schema_str = json.dumps(schema, indent=2, default=str)
            collection_info.append(f"Collection: {coll_name}\nSchema:\n{schema_str}")
        
        collections_desc = "\n\n".join(collection_info)
        
        system_msg = """You are a MongoDB read-only assistant. Select the appropriate collection for the query.

Available collections:
<collections>
{collections_desc}
</collections>

Call the Collection tool with the collection name.
""".format(collections_desc=collections_desc)
        
        response = model.bind_tools([Collection]).invoke([SystemMessage(content=system_msg)] + messages)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            primary_collection = tool_call.get('args', {}).get('collection', '')
            
            dispatch_custom_event(
                "collection_selected",
                {
                    "type": "collection_selected",
                    "collection": primary_collection
                },
                config=config
            )
        
        return {
            "primary_collection": primary_collection,
            "messages": [response]
        }
    
    dispatch_custom_event(
        "collection_selected",
        {
            "type": "collection_selected",
            "collection": primary_collection
        },
        config=config
    )
    
    return {"primary_collection": primary_collection}


def simple_executor(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Execute a simple single-collection query.
    Generates and runs aggregation pipeline with retry logic.
    """
    configurable = Configuration.from_runnable_config(config)
    max_retries = configurable.max_retry_attempts
    
    messages = state.get("messages", [])
    primary_collection = state.get("primary_collection", "")
    
    if not primary_collection:
        return {
            "messages": [{"role": "assistant", "content": "Error: No collection selected"}],
            "errors": ["No collection selected"]
        }
    
    collection = mongo_client[configurable.database][primary_collection]
    collection_schema = get_schema(collection)
    
    base_system_message = """
    Create a MongoDB aggregate query for {collection_name} collection.
    You must call the AggregateQuery tool with the `query` argument.
    
    Here is the schema for the collection:
    <schema>
    {collection_schema}
    </schema>
    
    Guidelines:
    - Always include a $limit stage (default 100 if not specified)
    - Use correct field names from the schema
    """.format(collection_name=primary_collection, collection_schema=json.dumps(collection_schema, indent=2))
    
    llm_with_tools = model.bind_tools(tools=[AggregateQuery], tool_choice=True)
    
    last_error = None
    query = None
    result = []
    
    for attempt in range(max_retries):
        try:
            if last_error:
                system_message = base_system_message + f"""
                
IMPORTANT: The previous query attempt failed with the following error:
<error>
{last_error}
</error>

Please fix the query and try again. Attempt {attempt + 1} of {max_retries}.
"""
                dispatch_custom_event(
                    "retry_attempt",
                    {
                        "type": "retry",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": last_error,
                        "collection": primary_collection
                    },
                    config=config
                )
            else:
                system_message = base_system_message
            
            response = llm_with_tools.invoke([SystemMessage(content=system_message)] + messages)
            
            try:
                query = response.tool_calls[0]["args"]["query"]
            except (IndexError, KeyError):
                _, query = parse_aggregate_query_tool_call(response)
            
            if isinstance(query, str):
                query = json.loads(query)
            if isinstance(query, dict):
                query = [query]
            
            dispatch_custom_event(
                "query_generated",
                {
                    "type": "query_generated",
                    "query": query,
                    "collection": primary_collection,
                    "attempt": attempt + 1
                },
                config=config
            )
            
            if configurable.run_query:
                dispatch_custom_event(
                    "query_executing",
                    {
                        "type": "query_executing",
                        "collection": primary_collection
                    },
                    config=config
                )
                
                cursor = collection.aggregate(query)
                result = [aggregate_mongo_doc_to_json_serializable(doc) for doc in cursor]
                
                dispatch_custom_event(
                    "query_success",
                    {
                        "type": "query_success",
                        "collection": primary_collection,
                        "count": len(result)
                    },
                    config=config
                )
                
                last_error = None
                break
            else:
                dispatch_custom_event(
                    "query_skipped",
                    {"type": "query_skipped", "reason": "run_query is False"},
                    config=config
                )
                result = []
                break
                
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {str(e)}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
    
    content = {
        "query": query,
        "collection": primary_collection,
        "result": result,
        "count": len(result)
    }
    
    if last_error:
        content["error"] = last_error
        content["retries_exhausted"] = True
        dispatch_custom_event(
            "query_failed",
            {
                "type": "query_failed",
                "error": last_error,
                "attempts": max_retries,
                "collection": primary_collection
            },
            config=config
        )
    
    return {
        "query_result": result,
        "result_count": len(result),
        "pipeline": query,
        "messages": [{"role": "assistant", "content": json.dumps(content, indent=2)}]
    }


# =============================================================================
# Routing Functions
# =============================================================================

def route_after_plan(state: MultiCollectionState) -> Literal["query_builder", "simple_handler", "__end__"]:
    """Route based on query type from planner."""
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "simple") if plan else "simple"
    
    if query_type == "direct_answer":
        return END
    elif query_type == "complex":
        return "query_builder"
    else:  # simple
        return "simple_handler"


def route_after_validation(state: MultiCollectionState) -> Literal["executor", "query_builder", "validation_failure"]:
    """Route based on validation result."""
    is_valid = state.get("is_valid", False)
    validation_attempts = state.get("validation_attempts", 0)
    validation_errors = state.get("validation_errors", [])
    max_attempts = 3
    
    logger.info(f"route_after_validation: is_valid={is_valid}, attempts={validation_attempts}, errors={len(validation_errors)}")
    
    if is_valid:
        logger.info("Routing to executor")
        return "executor"
    elif validation_attempts < max_attempts:
        logger.info(f"Routing to query_builder for retry (attempt {validation_attempts + 1})")
        return "query_builder"
    else:
        logger.warning(f"Max validation attempts ({max_attempts}) reached, routing to failure handler")
        return "validation_failure"


def validation_failure_handler(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Handle validation failures after max retries.
    Returns an error message to the user.
    """
    validation_errors = state.get("validation_errors", [])
    validation_attempts = state.get("validation_attempts", 0)
    primary_collection = state.get("primary_collection", "")
    pipeline = state.get("pipeline", [])
    
    # Build error details
    error_details = []
    for err in validation_errors:
        if err.get("error_type") == "error":
            error_details.append(f"- {err.get('message', 'Unknown error')}")
    
    error_content = {
        "error": f"Failed to generate valid query after {validation_attempts} attempts",
        "validation_errors": error_details,
        "collection": primary_collection,
        "last_pipeline": pipeline,
        "suggestion": "Try simplifying your query or being more specific about what you want"
    }
    
    dispatch_custom_event(
        "complex_query_failed",
        {
            "type": "complex_query_failed",
            "attempts": validation_attempts,
            "errors": error_details
        },
        config=config
    )
    
    logger.error(f"Complex query failed after {validation_attempts} attempts: {error_details}")
    
    return {
        "query_result": [],
        "result_count": 0,
        "errors": error_details,
        "messages": [{"role": "assistant", "content": json.dumps(error_content, indent=2)}]
    }


def route_after_simple_handler(state: MultiCollectionState) -> Literal["simple_executor", "__end__"]:
    """Route after simple handler - check if collection was selected."""
    primary_collection = state.get("primary_collection")
    messages = state.get("messages", [])
    
    # Check if last message has tool calls (collection selected)
    if primary_collection:
        return "simple_executor"
    
    # Check messages for tool calls
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "simple_executor"
    
    return END


# =============================================================================
# Build the Graph
# =============================================================================

builder = StateGraph(MultiCollectionState, config_schema=Configuration)

# Add nodes
builder.add_node("planner", planner_node)
builder.add_node("query_builder", query_builder_node)
builder.add_node("validator", validator_node)
builder.add_node("executor", executor_node)
builder.add_node("simple_handler", simple_handler)
builder.add_node("simple_executor", simple_executor)
builder.add_node("validation_failure", validation_failure_handler)

# Add edges
builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", route_after_plan)
builder.add_edge("query_builder", "validator")
builder.add_conditional_edges("validator", route_after_validation)
builder.add_edge("executor", END)
builder.add_edge("validation_failure", END)
builder.add_conditional_edges("simple_handler", route_after_simple_handler)
builder.add_edge("simple_executor", END)

# Compile the graph
graph = builder.compile()
