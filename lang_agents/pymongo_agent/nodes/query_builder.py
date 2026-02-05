"""
Query Builder node for the PyMongo Agent.
Generates MongoDB aggregation pipelines with $lookup for multi-collection queries.
"""
import json
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import dispatch_custom_event

from lang_agents.pymongo_agent.state import MultiCollectionState
from lang_agents.pymongo_agent.tools import AggregationPipelineTool
from lang_agents.pymongo_agent.configuration import Configuration
from lang_agents.pymongo_agent.utils import get_model_from_config


logger = logging.getLogger(__name__)


def query_builder_node(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Generate a MongoDB aggregation pipeline based on the execution plan.
    Creates pipelines with $lookup stages for joining collections.
    """
    configurable = Configuration.from_runnable_config(config)
    model = get_model_from_config(configurable)
    
    messages = state.get("messages", [])
    plan = state.get("plan", {})
    collection_schemas = state.get("collection_schemas", {})
    validation_errors = state.get("validation_errors", [])
    validation_attempts = state.get("validation_attempts", 0)
    
    primary_collection = plan.get("primary_collection", "") if plan else ""
    lookups = plan.get("lookups", []) if plan else []
    output_description = plan.get("output_description", "") if plan else ""
    
    logger.info(f"Query builder: primary_collection={primary_collection}, lookups={len(lookups)}")
    
    if not primary_collection:
        error_msg = "No primary collection specified in plan"
        logger.error(error_msg)
        return {
            "pipeline": [],
            "primary_collection": "",
            "validation_errors": [{"stage_index": -1, "error_type": "error", "message": error_msg, "suggestion": "Ensure planner sets primary_collection"}],
            "is_valid": False
        }
    
    # Get primary collection schema
    primary_schema = collection_schemas.get(primary_collection, {})
    primary_schema_str = json.dumps(primary_schema, indent=2, default=str)
    
    # Build lookup context
    lookup_context = ""
    if lookups:
        lookup_parts = []
        for i, lookup in enumerate(lookups, 1):
            from_coll = lookup.get("from_collection", "")
            from_schema = collection_schemas.get(from_coll, {})
            from_schema_str = json.dumps(from_schema, indent=2, default=str)
            lookup_parts.append(f"""
Lookup {i}:
  - Join with: {from_coll}
  - Local field: {lookup.get('local_field', '')}
  - Foreign field: {lookup.get('foreign_field', '')}
  - Output as: {lookup.get('as_field', '')}
  - Unwind: {lookup.get('unwind', False)}
  - Schema of {from_coll}:
{from_schema_str}
""")
        lookup_context = "\n".join(lookup_parts)
    
    # Build error context if this is a retry
    error_context = ""
    if validation_errors and validation_attempts > 0:
        error_parts = []
        for err in validation_errors:
            error_parts.append(f"- Stage {err.get('stage_index', '?')}: {err.get('message', '')} (Suggestion: {err.get('suggestion', '')})")
        error_context = f"""
IMPORTANT: The previous query attempt had validation errors:
{chr(10).join(error_parts)}

Please fix these issues in the new pipeline.
Attempt {validation_attempts + 1}.
"""
    
    system_message = """You are a MongoDB aggregation pipeline builder. Create a complete aggregation pipeline.

Primary Collection: {primary_collection}
Primary Collection Schema:
{primary_schema_str}

{lookup_section}

Expected Output: {output_description}

{error_context}

Guidelines for building the pipeline:
1. Start with $match to filter the primary collection (if filtering is needed)
2. Add $lookup stages to join with other collections
3. Use $addFields to compute derived fields (like counts using $size)
4. Add $sort for ordering results
5. Add $limit to restrict the number of results (default to 100 if not specified)
6. Use $project to select only needed fields for the final output

$lookup syntax:
{{
    "$lookup": {{
        "from": "collection_name",
        "localField": "field_in_current_collection",
        "foreignField": "field_in_foreign_collection",
        "as": "output_array_name"
    }}
}}

If unwind is needed (to flatten the array):
{{ "$unwind": {{ "path": "$output_array_name", "preserveNullAndEmptyArrays": true }} }}

IMPORTANT:
- Use correct field names from the schemas provided
- Ensure all stages are valid MongoDB aggregation stages
- Always include a $limit stage to prevent returning too many documents
- Return the pipeline as a JSON array of stage objects

Call the AggregationPipelineTool with the complete pipeline.
""".format(
        primary_collection=primary_collection,
        primary_schema_str=primary_schema_str,
        lookup_section=f"Lookups to perform:\n{lookup_context}" if lookup_context else "No lookups needed (single collection query)",
        output_description=output_description,
        error_context=error_context
    )
    
    llm_with_tools = model.bind_tools([AggregationPipelineTool], tool_choice="required")
    
    dispatch_custom_event(
        "query_building_started",
        {
            "type": "query_building_started",
            "primary_collection": primary_collection,
            "lookups_count": len(lookups),
            "is_retry": validation_attempts > 0,
            "attempt": validation_attempts + 1
        },
        config=config
    )
    
    # Try to generate pipeline with retries for LLM errors
    max_llm_retries = 2
    pipeline = []
    last_error = None
    
    for llm_attempt in range(max_llm_retries):
        try:
            if last_error:
                # Add error context to prompt for retry
                retry_system_message = system_message + f"""

CRITICAL: The previous attempt to generate a pipeline failed with error:
{last_error}

Please ensure you call the AggregationPipelineTool with a valid pipeline array.
"""
                response = llm_with_tools.invoke([SystemMessage(content=retry_system_message)] + messages)
            else:
                response = llm_with_tools.invoke([SystemMessage(content=system_message)] + messages)
            
            # Extract pipeline from tool call
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {})
                pipeline = args.get("pipeline", [])
                
                # Handle case where pipeline is a string
                if isinstance(pipeline, str):
                    try:
                        pipeline = json.loads(pipeline)
                    except json.JSONDecodeError as e:
                        last_error = f"Failed to parse pipeline JSON: {e}"
                        logger.error(last_error)
                        continue
                
                # Validate we got a non-empty list
                if not pipeline or not isinstance(pipeline, list):
                    last_error = f"Pipeline is empty or not a list: {type(pipeline)}"
                    logger.error(last_error)
                    continue
                
                # Success!
                logger.info(f"Generated pipeline with {len(pipeline)} stages (attempt {llm_attempt + 1})")
                last_error = None
                break
            else:
                last_error = "No tool calls in LLM response"
                logger.error(last_error)
                continue
                
        except Exception as e:
            last_error = f"LLM error: {type(e).__name__}: {str(e)}"
            logger.error(last_error)
            continue
    
    # If we still have an error after retries, create a validation error
    if last_error:
        logger.error(f"Failed to generate pipeline after {max_llm_retries} attempts: {last_error}")
        return {
            "pipeline": [],
            "primary_collection": primary_collection,
            "validation_errors": [{
                "stage_index": -1,
                "error_type": "error",
                "message": f"Failed to generate pipeline: {last_error}",
                "suggestion": "Try rephrasing your query or simplifying the request"
            }],
            "is_valid": False
        }
    
    dispatch_custom_event(
        "query_built",
        {
            "type": "query_built",
            "primary_collection": primary_collection,
            "pipeline": pipeline,
            "stages_count": len(pipeline)
        },
        config=config
    )
    
    return {
        "pipeline": pipeline,
        "primary_collection": primary_collection,
        "validation_errors": [],  # Clear previous errors
        "is_valid": False  # Will be set by validator
    }
