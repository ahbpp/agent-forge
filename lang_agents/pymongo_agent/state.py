"""
State definitions for the PyMongo Agent with multi-collection support.
"""
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import MessagesState


class LookupSpec(TypedDict):
    """Specification for a MongoDB $lookup stage"""
    from_collection: str          # Collection to join with
    local_field: str              # Field in primary/current collection
    foreign_field: str            # Field in foreign collection
    as_field: str                 # Output array field name
    unwind: bool                  # Whether to $unwind (flatten) the results


class QueryPlan(TypedDict):
    """Query execution plan created by the planner"""
    query_type: Literal["simple", "complex", "direct_answer"]
    reasoning: str                # Why this query type was chosen
    
    # For simple and complex queries
    primary_collection: Optional[str]
    
    # For complex queries with joins
    lookups: List[LookupSpec]
    
    # Description of expected output
    output_description: str
    
    # For direct answers (no query needed)
    direct_response: Optional[str]


class ValidationError(TypedDict):
    """Validation error details"""
    stage_index: int              # Which pipeline stage has the error (-1 for general)
    error_type: Literal["error", "warning"]
    message: str
    suggestion: str


class MultiCollectionState(MessagesState):
    """
    Extended state for multi-collection queries.
    Inherits messages from MessagesState.
    """
    # Query plan from planner
    plan: Optional[QueryPlan]
    
    # Available collections and their schemas (set once at start)
    available_collections: List[str]
    collection_schemas: Dict[str, Dict[str, Any]]
    
    # Generated pipeline from query_builder
    pipeline: Optional[List[Dict[str, Any]]]
    primary_collection: Optional[str]
    
    # Validation state
    is_valid: bool
    validation_errors: List[ValidationError]
    validation_attempts: int
    
    # Execution results
    query_result: Optional[List[Dict[str, Any]]]
    result_count: int
    
    # Error tracking
    errors: List[str]
