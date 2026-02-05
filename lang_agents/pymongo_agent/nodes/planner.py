"""
Planner node for the PyMongo Agent.
Analyzes user requests and decides the execution strategy.
"""
import json
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import dispatch_custom_event

from lang_agents.pymongo_agent.state import MultiCollectionState
from lang_agents.pymongo_agent.tools import QueryPlanTool
from lang_agents.pymongo_agent.configuration import Configuration
from lang_agents.pymongo_agent.utils import (
    get_read_mongo_client,
    get_schema,
    list_collections,
    get_model_from_config
)


logger = logging.getLogger(__name__)

# Initialize MongoDB client
mongo_client = get_read_mongo_client()


def planner_node(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Analyze the user's request and create an execution plan.
    
    Decides between:
    - simple: Single collection query (fast path)
    - complex: Multi-collection query with $lookup joins
    - direct_answer: No database query needed
    """
    configurable = Configuration.from_runnable_config(config)
    model = get_model_from_config(configurable)
    
    messages = state.get("messages", [])
    
    # Get available collections and schemas
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
    
    system_message = """You are a MongoDB query planner. Analyze the user's request and create an execution plan.

Available collections in the database with their schemas:
<collections>
{collections_desc}
</collections>

Your task is to decide the best approach:

1. **simple** - Use when the query only needs ONE collection with no joins.
   Examples:
   - "Show me all users" → simple
   - "Count orders" → simple  
   - "Find patient by ID in moleimages collection" → simple
   - "How many images with diagnosis_count > 0" → simple (if diagnosis_count is a field in the collection)

2. **complex** - Use when the query needs to JOIN data from MULTIPLE collections.
   Use this when:
   - User mentions "both collections", "join", or "lookup"
   - User wants data from collection A combined with details from collection B
   - User needs to cross-reference documents between collections
   Examples:
   - "Look in both moleimages and diagnosis collections" → complex
   - "Find images AND their diagnosis records from the diagnoses collection" → complex
   - "Get orders with customer details from customers collection" → complex
   
   For complex queries, you MUST specify:
   - primary_collection: The main collection to start from
   - lookups: List of join specifications with from_collection, local_field, foreign_field, as_field

3. **direct_answer** - Use when you can answer without querying the database.
   Example: "What collections exist?", "What fields does the users collection have?"

IMPORTANT GUIDELINES:
- If user mentions a SINGLE collection AND the query can be answered from that collection's fields alone → simple
- If user explicitly mentions "both collections" or wants to combine data from multiple collections → complex
- If a collection already HAS aggregated fields (like diagnosis_count, attached_diagnoses), a join may not be needed → simple
- When user asks for related data that exists in another collection (not just IDs/counts) → complex

Call the QueryPlanTool with your execution plan.
""".format(collections_desc=collections_desc)

    llm_with_tools = model.bind_tools([QueryPlanTool], tool_choice="required")
    
    dispatch_custom_event(
        "planning_started",
        {
            "type": "planning_started",
            "collections_count": len(collections)
        },
        config=config
    )
    
    response = llm_with_tools.invoke([SystemMessage(content=system_message)] + messages)
    
    # Extract plan from tool call
    plan = None
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call.get("args", {})
        
        plan = {
            "query_type": args.get("query_type", "simple"),
            "reasoning": args.get("reasoning", ""),
            "primary_collection": args.get("primary_collection"),
            "lookups": args.get("lookups", []),
            "output_description": args.get("output_description", ""),
            "direct_response": args.get("direct_response")
        }
        
        logger.info(f"Plan created: {plan['query_type']} - {plan['reasoning']}")
        
        dispatch_custom_event(
            "plan_created",
            {
                "type": "plan_created",
                "query_type": plan["query_type"],
                "reasoning": plan["reasoning"],
                "primary_collection": plan.get("primary_collection"),
                "lookups_count": len(plan.get("lookups", [])),
                "lookups": plan.get("lookups", [])
            },
            config=config
        )
    
    # For direct_answer, add response to messages
    new_messages = []
    if plan and plan["query_type"] == "direct_answer" and plan.get("direct_response"):
        new_messages = [{"role": "assistant", "content": plan["direct_response"]}]
    
    # Extract primary_collection at state level for consistency
    primary_collection = plan.get("primary_collection") if plan else None
    
    logger.info(f"Planner returning: query_type={plan.get('query_type') if plan else None}, "
                f"primary_collection={primary_collection}, "
                f"lookups_count={len(plan.get('lookups', [])) if plan else 0}")
    
    return {
        "plan": plan,
        "primary_collection": primary_collection,  # Set at state level for downstream nodes
        "available_collections": collections,
        "collection_schemas": collection_schemas,
        "messages": new_messages if new_messages else [],
        "validation_attempts": 0,
        "validation_errors": [],  # Reset validation errors
        "pipeline": None,  # Reset pipeline from previous runs
        "is_valid": False,  # Reset validation state
        "errors": []
    }
