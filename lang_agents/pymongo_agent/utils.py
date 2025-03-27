import os
import pymongo
import datetime
import re
import json
from collections import defaultdict

from bson.objectid import ObjectId
from bson.typings import _DocumentType
from pymongo.synchronous.collection import Collection

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

from lang_agents.pymongo_agent.configuration import Configuration, ModelProvider


CACHE_DIR = ".cache/pymongo_agent_cache"
SCHEMA_CACHE_DIR = os.path.join(CACHE_DIR, "schemas")


def get_read_mongo_client() -> pymongo.MongoClient:
    return pymongo.MongoClient(
        host=os.getenv("PYMONGO_HOST"),
        username=os.getenv("PYMONGO_USER"),
        password=os.getenv("PYMONGO_PASSWORD"),
        authSource="admin",
        read_preference=pymongo.ReadPreference.SECONDARY
    )


def get_schema_recursive(doc: dict) -> dict:
    for key, value in doc.items():
        if isinstance(value, dict):
            doc[key] = get_schema_recursive(value)
    return doc


def get_schema(collection: Collection, sample_size: int = 50) -> _DocumentType:
    """
    Generate a schema dictionary from a list of MongoDB sample documents.
    The schema maps each field (using dot notation for nested fields) to a set of type names.
    
    Args:
        collection (Collection): The MongoDB collection to analyze.
        sample_size (int): The number of documents to sample from the collection.
    
    Returns:
        dict: A dictionary with field paths as keys and sets of type names as values.
    """
    schema_name = f"{collection.name}_schema.json"
    schema_path = os.path.join(SCHEMA_CACHE_DIR, schema_name)
    if os.path.exists(schema_path):
        print(f"Loading schema from cache: {schema_path}")
        with open(schema_path, "r") as f:
            return json.load(f)
    samples = collection.aggregate([{"$sample": {"size": sample_size}}])
    schema = defaultdict(set)
    
    def analyze_doc(doc, path=""):
        for key, value in doc.items():
            # Create the full path for the field (e.g., "mole.mole_id")
            key_path = f"{path}.{key}" if path else key
            # Record the type name of the current value
            schema[key_path].add(type(value).__name__)
            
            # If the value is a dictionary, recurse to get its fields.
            if isinstance(value, dict):
                analyze_doc(value, key_path)
            # If it's a list, check each item in the list.
            elif isinstance(value, list):
                for item in value:
                    # For dictionaries inside the list, recurse into them.
                    if isinstance(item, dict):
                        analyze_doc(item, key_path)
                    else:
                        schema[key_path].add(type(item).__name__)
    
    # Process each document in the sample set.
    for doc in samples:
        analyze_doc(doc)

    # make set json serializable
    schema = {k: list(v) for k, v in schema.items()}
    # sort keys alphabetically
    schema = dict(sorted(schema.items()))

    if not os.path.exists(SCHEMA_CACHE_DIR):
        os.makedirs(SCHEMA_CACHE_DIR)
    try:
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=4)
    except Exception as e:
        print(f"Error saving schema to cache: {str(e)}")
        if os.path.exists(schema_path):
            os.remove(schema_path)
        raise e
    return schema



def list_collections(mongo_client: pymongo.MongoClient, database: str) -> list[str]:
    try:
        db = mongo_client[database]
        return db.list_collection_names()
    except Exception as e:
        raise Exception(f"Error listing collections: {str(e)}")


def aggregate_mongo_doc_to_json_serializable(doc):
    """
    Convert a MongoDB document to a JSON serializable format.
    """
    if isinstance(doc, dict):
        return {k: aggregate_mongo_doc_to_json_serializable(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [aggregate_mongo_doc_to_json_serializable(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, datetime.datetime):
        return str(doc)
    return doc


def get_model_from_config(configurable: Configuration) -> BaseChatModel:
    """Initialize the appropriate model based on the configuration.
    
    Args:
        configurable: Configuration object containing model settings
        
    Returns:
        Initialized chat model
    """
    # Use init_chat_model to initialize the model based on provider
    model_provider = configurable.model_provider
    if model_provider == ModelProvider.OLLAMA:
        return ChatOllama(model=configurable.model_name, 
                          base_url=os.getenv("OLLAMA_HOST"),
                          temperature=configurable.temperature)
    
    model_provider = model_provider.value if isinstance(model_provider, ModelProvider) else model_provider
    return init_chat_model(
        model=configurable.model_name,
        model_provider=model_provider,
        temperature=configurable.temperature
    )


def parse_aggregate_query_tool_call(model_output: AIMessage):
    """
    Parses the model output to extract the AggregateQuery tool call.
    model_output format:
    <tool_call>
    {"name": "AggregateQuery", "arguments": [{...}, {...}]}
    </tool_call>

    Args:
        model_output: The raw string output from the language model.

    Returns:
        A tuple containing the tool name and arguments (as a Python list of dictionaries),
        or (None, None) if the tool call cannot be parsed.
    """
    model_output = model_output.content
    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", model_output)
    if tool_call_match:
        try:
            tool_call_str = tool_call_match.group(1)
            tool_call_data = json.loads(tool_call_str)
            tool_name = tool_call_data.get("name")
            tool_arguments = tool_call_data.get("arguments")
            if tool_name == "AggregateQuery":
                return tool_name, tool_arguments
            else:
                return None, None
        except json.JSONDecodeError:
            return None, None
    else:
        return None, None
