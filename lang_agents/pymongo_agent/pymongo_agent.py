import uuid
import os
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict, List, Dict, Any, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from lang_agents.pymongo_agent.utils import get_read_mongo_client, print_schema

from dotenv import load_dotenv

load_dotenv()

# Create cache directory if it doesn't exist
CACHE_DIR = Path("lang_agents/pymongo_agent/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MongoDB client
mongo_client = get_read_mongo_client()

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define state type for our MongoDB agent
class MongoDBAgentState(MessagesState):
    """State for the MongoDB agent."""
    db_info: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    results: Optional[Any] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# MongoDB Tools
@tool
def list_databases() -> List[str]:
    """List all available databases in MongoDB."""
    databases = mongo_client.list_database_names()
    return databases

@tool
def list_collections(database: str) -> List[str]:
    """List all collections in a specific database."""
    try:
        db = mongo_client[database]
        collections = db.list_collection_names()
        return collections
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_collection_schema(database: str, collection: str) -> Dict:
    """Get the schema of a specific collection."""
    try:
        db = mongo_client[database]
        coll = db[collection]
        sample_doc = coll.find_one()
        
        # Convert MongoDB document to serializable format
        import json
        from bson import json_util
        schema_str = json.loads(json_util.dumps(sample_doc))
        
        return {"schema": schema_str}
    except Exception as e:
        return {"error": str(e)}

@tool
def run_find_query(database: str, collection: str, query: Dict, limit: int = 10) -> Dict:
    """Run a find query on a MongoDB collection."""
    try:
        db = mongo_client[database]
        coll = db[collection]
        
        # Execute the query
        results = list(coll.find(query).limit(limit))
        
        # Convert MongoDB cursor to serializable format
        import json
        from bson import json_util
        results_json = json.loads(json_util.dumps(results))
        
        return {
            "query": query,
            "results": results_json,
            "count": len(results_json)
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def run_aggregate_query(database: str, collection: str, pipeline: List[Dict], limit: int = 10) -> Dict:
    """Run an aggregate query on a MongoDB collection."""
    try:
        db = mongo_client[database]
        coll = db[collection]
        
        # Add a limit stage if not already in pipeline
        has_limit = any('$limit' in stage for stage in pipeline)
        if not has_limit:
            pipeline.append({"$limit": limit})
        
        # Execute the aggregation
        results = list(coll.aggregate(pipeline))
        
        # Convert MongoDB cursor to serializable format
        import json
        from bson import json_util
        results_json = json.loads(json_util.dumps(results))
        
        return {
            "pipeline": pipeline,
            "results": results_json,
            "count": len(results_json)
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def run_count_query(database: str, collection: str, query: Dict) -> Dict:
    """Count documents in a MongoDB collection based on a query."""
    try:
        db = mongo_client[database]
        coll = db[collection]
        
        # Execute the count
        count = coll.count_documents(query)
        
        return {
            "query": query,
            "count": count
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def run_distinct_query(database: str, collection: str, field: str, query: Dict = None) -> Dict:
    """Get distinct values for a field in a MongoDB collection."""
    try:
        db = mongo_client[database]
        coll = db[collection]
        
        # Execute the distinct query
        if query:
            distinct_values = coll.distinct(field, query)
        else:
            distinct_values = coll.distinct(field)
        
        # Convert to serializable format
        import json
        from bson import json_util
        values_json = json.loads(json_util.dumps(distinct_values))
        
        return {
            "field": field,
            "query": query,
            "values": values_json,
            "count": len(values_json)
        }
    except Exception as e:
        return {"error": str(e)}

# Function to cache results
def save_to_cache(state: MongoDBAgentState) -> None:
    """Save query and results to cache for reproducibility."""
    if not state.query or not state.results:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_file = CACHE_DIR / f"query_{timestamp}_{state.request_id}.json"
    
    cache_data = {
        "timestamp": timestamp,
        "request_id": state.request_id,
        "query": state.query,
        "results": state.results
    }
    
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

# Graph nodes
def analyze_request(state: MongoDBAgentState) -> MongoDBAgentState:
    """Analyze the user's request and determine what MongoDB operations to perform."""
    messages = state.messages.copy()
    
    system_message = """You are a MongoDB query assistant. Your job is to:
1. Analyze the user's request related to MongoDB
2. Determine what MongoDB operation is needed (list databases, list collections, find, aggregate, etc.)
3. Specify the exact parameters needed for the MongoDB operation

Return a JSON object with the following structure:
{
    "operation": "one of [list_databases, list_collections, get_collection_schema, run_find_query, run_aggregate_query, run_count_query, run_distinct_query]",
    "parameters": {
        // Parameters required for the operation
    },
    "explanation": "Brief explanation of what you're about to do"
}
"""
    
    messages.insert(0, SystemMessage(content=system_message))
    response = model.invoke(messages)
    
    try:
        import json
        analysis = json.loads(response.content)
        state.query = analysis
        return state
    except Exception as e:
        # If parsing fails, return the error
        state.query = {"error": f"Failed to parse analysis: {str(e)}"}
        return state

def execute_operation(state: MongoDBAgentState) -> MongoDBAgentState:
    """Execute the MongoDB operation based on the analysis."""
    if not state.query or "error" in state.query:
        return state
    
    operation = state.query.get("operation")
    parameters = state.query.get("parameters", {})
    
    # Map operations to tools
    operation_map = {
        "list_databases": list_databases,
        "list_collections": list_collections,
        "get_collection_schema": get_collection_schema,
        "run_find_query": run_find_query,
        "run_aggregate_query": run_aggregate_query,
        "run_count_query": run_count_query,
        "run_distinct_query": run_distinct_query
    }
    
    if operation in operation_map:
        try:
            # Execute the operation
            results = operation_map[operation](**parameters)
            state.results = results
            
            # Save to cache
            save_to_cache(state)
            
            return state
        except Exception as e:
            state.results = {"error": f"Failed to execute operation: {str(e)}"}
            return state
    else:
        state.results = {"error": f"Unknown operation: {operation}"}
        return state

def format_response(state: MongoDBAgentState) -> MongoDBAgentState:
    """Format the results for the user."""
    messages = state.messages.copy()
    
    system_message = """You are a MongoDB query assistant. Your job is to:
1. Format the MongoDB query results in a clear, readable way for the user
2. Explain what the results mean in the context of their original request
3. Mention that the results and query have been cached for reproducibility

Be concise but comprehensive in your explanation.
"""
    
    # Add the operation and results to context
    context = f"""
MongoDB Operation: {state.query.get('operation', 'Unknown')}
Explanation: {state.query.get('explanation', 'No explanation provided')}

Results:
{json.dumps(state.results, indent=2)}

Request ID: {state.request_id}
"""
    
    messages.append(HumanMessage(content=f"Please format these MongoDB results for me: {context}"))
    messages.insert(0, SystemMessage(content=system_message))
    
    response = model.invoke(messages)
    state.messages.append(response)
    
    return state

# Create the graph
def create_mongodb_agent():
    """Create and return the MongoDB agent graph."""
    # Define the graph
    workflow = StateGraph(MongoDBAgentState)
    
    # Add nodes
    workflow.add_node("analyze_request", analyze_request)
    workflow.add_node("execute_operation", execute_operation)
    workflow.add_node("format_response", format_response)
    
    # Define edges
    workflow.add_edge(START, "analyze_request")
    workflow.add_edge("analyze_request", "execute_operation")
    workflow.add_edge("execute_operation", "format_response")
    workflow.add_edge("format_response", END)
    
    # Compile the graph
    mongodb_agent = workflow.compile()
    
    return mongodb_agent

# Create the agent
mongodb_agent = create_mongodb_agent()

# Example usage
if __name__ == "__main__":
    # Example usage
    input_message = HumanMessage(content="Show me the top 5 documents from the users collection in the app database")
    result = mongodb_agent.invoke({"messages": [input_message]})
    
    # Print the final response
    print(result.messages[-1].content)

