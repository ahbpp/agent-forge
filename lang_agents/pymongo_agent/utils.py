import os
import pymongo
import datetime
import re
import json

from bson.objectid import ObjectId
from bson.typings import _DocumentType
from pymongo.synchronous.collection import Collection

from lang_agents.pymongo_agent.configuration import Configuration, ModelProvider
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model


def get_read_mongo_client() -> pymongo.MongoClient:
    return pymongo.MongoClient(
        host=os.getenv("PYMONGO_HOST"),
        username=os.getenv("PYMONGO_USER"),
        password=os.getenv("PYMONGO_PASSWORD"),
        authSource="admin",
        read_preference=pymongo.ReadPreference.SECONDARY
    )


def get_schema(collection: Collection) -> _DocumentType:
    schema = collection.find_one()
    assert schema is not None, "Collection is empty"
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

    Args:
        model_output: The raw string output from the language model.

    Returns:
        A tuple containing the tool name and arguments (as a Python list of dictionaries),
        or (None, None) if the tool call cannot be parsed.
    """
    model_output = model_output.content
    print(model_output)
    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", model_output)
    print(tool_call_match)
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