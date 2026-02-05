"""
Executor node for the PyMongo Agent.
Executes validated MongoDB aggregation pipelines with retry logic.
"""
import json
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import dispatch_custom_event

from lang_agents.pymongo_agent.state import MultiCollectionState
from lang_agents.pymongo_agent.configuration import Configuration
from lang_agents.pymongo_agent.utils import (
    get_read_mongo_client,
    aggregate_mongo_doc_to_json_serializable
)


logger = logging.getLogger(__name__)

# Initialize MongoDB client
mongo_client = get_read_mongo_client()


def executor_node(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Execute the validated MongoDB aggregation pipeline.
    Handles retries on execution errors.
    """
    configurable = Configuration.from_runnable_config(config)
    max_retries = configurable.max_retry_attempts
    
    pipeline = state.get("pipeline", [])
    primary_collection = state.get("primary_collection", "")
    
    logger.info(f"Executor: primary_collection={primary_collection}, pipeline_stages={len(pipeline) if pipeline else 0}")
    logger.debug(f"Pipeline: {json.dumps(pipeline, indent=2, default=str) if pipeline else 'None'}")
    
    if not pipeline or not primary_collection:
        error_msg = "Missing pipeline or primary collection"
        logger.error(error_msg)
        return {
            "query_result": [],
            "result_count": 0,
            "errors": [error_msg],
            "messages": [{"role": "assistant", "content": f"Error: {error_msg}"}]
        }
    
    collection = mongo_client[configurable.database][primary_collection]
    
    dispatch_custom_event(
        "execution_started",
        {
            "type": "execution_started",
            "collection": primary_collection,
            "pipeline_stages": len(pipeline)
        },
        config=config
    )
    
    last_error = None
    result = []
    total_stages = len(pipeline)
    
    for attempt in range(max_retries):
        try:
            if last_error:
                dispatch_custom_event(
                    "execution_retry",
                    {
                        "type": "execution_retry",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": last_error
                    },
                    config=config
                )
                logger.warning(f"Execution retry {attempt + 1}/{max_retries}: {last_error}")
            
            dispatch_custom_event(
                "query_executing",
                {
                    "type": "query_executing",
                    "collection": primary_collection,
                    "attempt": attempt + 1
                },
                config=config
            )
            
            if configurable.run_query:
                logger.info(f"Executing pipeline on {primary_collection}")
                
                # Show the full pipeline being executed
                dispatch_custom_event(
                    "executing_pipeline",
                    {
                        "type": "executing_pipeline",
                        "collection": primary_collection,
                        "pipeline": pipeline,
                        "stages_count": total_stages
                    },
                    config=config
                )
                
                cursor = collection.aggregate(pipeline)
                result = [aggregate_mongo_doc_to_json_serializable(doc) for doc in cursor]
                
                dispatch_custom_event(
                    "execution_success",
                    {
                        "type": "execution_success",
                        "collection": primary_collection,
                        "count": len(result),
                        "sample": result[:3] if result else []  # First 3 for preview
                    },
                    config=config
                )
                
                logger.info(f"Query executed successfully. Retrieved {len(result)} documents.")
                last_error = None
                break
            else:
                dispatch_custom_event(
                    "query_skipped",
                    {
                        "type": "query_skipped",
                        "reason": "run_query is False in configuration"
                    },
                    config=config
                )
                logger.info("Query not executed, run_query is False")
                result = []
                break
                
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Execution attempt {attempt + 1} failed: {last_error}")
    
    # Build response content
    content = {
        "query": pipeline,
        "collection": primary_collection,
        "result": result,
        "count": len(result)
    }
    
    errors = state.get("errors", [])
    
    if last_error:
        content["error"] = last_error
        content["retries_exhausted"] = True
        errors.append(last_error)
        
        dispatch_custom_event(
            "execution_failed",
            {
                "type": "execution_failed",
                "error": last_error,
                "attempts": max_retries,
                "collection": primary_collection
            },
            config=config
        )
        logger.error(f"Execution failed after {max_retries} attempts: {last_error}")
    
    content_str = json.dumps(content, indent=2)
    
    return {
        "query_result": result,
        "result_count": len(result),
        "errors": errors,
        "messages": [{"role": "assistant", "content": content_str}]
    }
