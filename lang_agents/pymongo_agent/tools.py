"""
Tool definitions for the PyMongo Agent with multi-collection support.
These are Pydantic models used for LLM structured output.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class LookupSpecTool(BaseModel):
    """Specification for a MongoDB $lookup stage"""
    from_collection: str = Field(description="The collection to join with")
    local_field: str = Field(description="Field in the primary collection to match")
    foreign_field: str = Field(description="Field in the foreign collection to match")
    as_field: str = Field(description="Name for the output array field containing joined documents")
    unwind: bool = Field(
        default=False, 
        description="Whether to flatten the joined array (use when expecting single match)"
    )


class QueryPlanTool(BaseModel):
    """Tool for planner to create execution plan"""
    query_type: Literal["simple", "complex", "direct_answer"] = Field(
        description=(
            "Type of query: "
            "'simple' for single collection queries (PREFERRED - faster), "
            "'complex' ONLY when explicitly joining multiple collections, "
            "'direct_answer' for no query needed"
        )
    )
    reasoning: str = Field(description="Brief explanation of why this query type was chosen")
    
    # For queries
    primary_collection: Optional[str] = Field(
        default=None,
        description="The main collection to query from (required for simple and complex)"
    )
    
    # For complex queries with joins
    lookups: List[LookupSpecTool] = Field(
        default_factory=list,
        description="List of $lookup operations for joining collections. Leave empty for simple queries!"
    )
    
    # What user wants
    output_description: str = Field(
        default="",
        description="Description of the expected output format and content"
    )
    
    # For direct answers
    direct_response: Optional[str] = Field(
        default=None,
        description="Direct response text when no database query is needed"
    )


class AggregationPipelineTool(BaseModel):
    """Tool for query_builder to create MongoDB aggregation pipeline"""
    pipeline: List[Dict[str, Any]] = Field(
        description="Complete MongoDB aggregation pipeline including $match, $lookup, $addFields, $sort, $limit, $project stages"
    )


class SimpleCollectionTool(BaseModel):
    """Tool for simple single-collection selection (backwards compatibility)"""
    collection: str = Field(description="The collection name to query")


class SimpleAggregateQueryTool(BaseModel):
    """Tool for simple aggregation query (backwards compatibility)"""
    query: List[Dict[str, Any]] = Field(
        description="The PyMongo aggregation pipeline (list of dictionaries)"
    )
