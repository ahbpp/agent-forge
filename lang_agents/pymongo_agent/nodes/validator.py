"""
Validator node for the PyMongo Agent.
Validates MongoDB aggregation pipelines before execution.
"""
import logging
from typing import Dict, Any, List, Tuple
from difflib import get_close_matches

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import dispatch_custom_event

from lang_agents.pymongo_agent.state import MultiCollectionState, ValidationError


logger = logging.getLogger(__name__)


def extract_field_names(schema: Dict[str, Any]) -> List[str]:
    """Extract all field names from a collection schema."""
    if not schema:
        return []
    
    fields = []
    for field_name in schema.keys():
        # Get base field name (before the dot)
        base_field = field_name.split(".")[0]
        if base_field not in fields:
            fields.append(base_field)
        fields.append(field_name)
    
    return fields


def find_similar_field(field: str, available_fields: List[str]) -> str:
    """Find similar field name for suggestions."""
    if not available_fields:
        return "unknown"
    
    matches = get_close_matches(field, available_fields, n=1, cutoff=0.6)
    return matches[0] if matches else available_fields[0]


def validate_lookup(
    stage_index: int,
    lookup: Dict[str, Any],
    available_collections: List[str],
    schemas: Dict[str, Dict]
) -> List[ValidationError]:
    """Validate $lookup stage."""
    errors = []
    
    from_collection = lookup.get("from")
    foreign_field = lookup.get("foreignField")
    as_field = lookup.get("as")
    
    # Check collection exists
    if from_collection and from_collection not in available_collections:
        errors.append({
            "stage_index": stage_index,
            "error_type": "error",
            "message": f"Collection '{from_collection}' does not exist",
            "suggestion": f"Available collections: {', '.join(available_collections[:5])}"
        })
    
    # Check foreign field exists in target collection schema
    # Note: This is a WARNING not an error because schema sampling may not capture all fields
    if from_collection and from_collection in schemas:
        schema_fields = extract_field_names(schemas[from_collection])
        if foreign_field and foreign_field not in schema_fields:
            similar = find_similar_field(foreign_field, schema_fields)
            errors.append({
                "stage_index": stage_index,
                "error_type": "warning",  # Changed from error - schema may be incomplete
                "message": f"Field '{foreign_field}' not found in sampled schema of '{from_collection}'",
                "suggestion": f"Did you mean '{similar}'? Available fields: {', '.join(schema_fields[:10])}"
            })
    
    # Check as_field is provided
    if not as_field:
        errors.append({
            "stage_index": stage_index,
            "error_type": "error",
            "message": "$lookup missing 'as' field",
            "suggestion": "Add 'as' field to specify output array name"
        })
    
    return errors


def validate_match(
    stage_index: int,
    match: Dict[str, Any],
    collection: str,
    schemas: Dict[str, Dict]
) -> List[ValidationError]:
    """Validate $match stage field references."""
    errors = []
    
    if collection not in schemas:
        return errors
    
    schema_fields = extract_field_names(schemas[collection])
    
    def check_fields(obj: Dict, prefix: str = ""):
        for field, value in obj.items():
            # Skip operators like $and, $or, $expr, $regex, etc.
            if field.startswith("$"):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            check_fields(item, prefix)
                elif isinstance(value, dict):
                    check_fields(value, prefix)
                continue
            
            # Handle nested field notation (e.g., "diagnoses.result")
            base_field = field.split(".")[0]
            
            # Skip fields that are outputs from previous $lookup stages
            # These won't be in the original schema
            if base_field not in schema_fields and not prefix:
                similar = find_similar_field(base_field, schema_fields)
                errors.append({
                    "stage_index": stage_index,
                    "error_type": "warning",  # Warning because it might be a $lookup result
                    "message": f"Field '{field}' not found in original '{collection}' schema",
                    "suggestion": f"If this is not a $lookup result, did you mean '{similar}'?"
                })
    
    check_fields(match)
    return errors


def validate_pipeline(
    pipeline: List[Dict[str, Any]],
    primary_collection: str,
    available_collections: List[str],
    schemas: Dict[str, Dict]
) -> Tuple[bool, List[ValidationError]]:
    """
    Validate MongoDB aggregation pipeline.
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors: List[ValidationError] = []
    
    if not pipeline:
        errors.append({
            "stage_index": -1,
            "error_type": "error",
            "message": "Pipeline is empty",
            "suggestion": "Generate a valid aggregation pipeline"
        })
        return False, errors
    
    has_limit = False
    lookup_outputs = set()  # Track fields created by $lookup
    
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict):
            errors.append({
                "stage_index": i,
                "error_type": "error",
                "message": f"Stage {i} is not a valid object",
                "suggestion": "Each stage must be a JSON object with a single operator key"
            })
            continue
        
        if len(stage) != 1:
            errors.append({
                "stage_index": i,
                "error_type": "warning",
                "message": f"Stage {i} has multiple keys, expected single operator",
                "suggestion": "Each stage should have exactly one operator key"
            })
        
        stage_name = list(stage.keys())[0]
        stage_value = stage[stage_name]
        
        # Validate $lookup stages
        if stage_name == "$lookup":
            lookup_errors = validate_lookup(i, stage_value, available_collections, schemas)
            errors.extend(lookup_errors)
            
            # Track lookup output field
            if stage_value.get("as"):
                lookup_outputs.add(stage_value["as"])
        
        # Validate $match stages (only for first $match before lookups)
        elif stage_name == "$match" and not lookup_outputs:
            match_errors = validate_match(i, stage_value, primary_collection, schemas)
            errors.extend(match_errors)
        
        # Track $limit
        elif stage_name == "$limit":
            has_limit = True
            if not isinstance(stage_value, int) or stage_value <= 0:
                errors.append({
                    "stage_index": i,
                    "error_type": "error",
                    "message": "$limit value must be a positive integer",
                    "suggestion": f"Change $limit to a positive integer, got: {stage_value}"
                })
        
        # Validate $sort
        elif stage_name == "$sort":
            if not isinstance(stage_value, dict):
                errors.append({
                    "stage_index": i,
                    "error_type": "error",
                    "message": "$sort value must be an object",
                    "suggestion": "Use format: {\"field\": 1} for ascending or {\"field\": -1} for descending"
                })
        
        # Validate $unwind
        elif stage_name == "$unwind":
            unwind_path = stage_value if isinstance(stage_value, str) else stage_value.get("path", "")
            if unwind_path and unwind_path.startswith("$"):
                field_name = unwind_path[1:]  # Remove $ prefix
                # Check if it's a lookup output or original field
                if field_name not in lookup_outputs:
                    schema_fields = extract_field_names(schemas.get(primary_collection, {}))
                    if field_name not in schema_fields:
                        errors.append({
                            "stage_index": i,
                            "error_type": "warning",
                            "message": f"$unwind path '{unwind_path}' not found in schema or lookup outputs",
                            "suggestion": f"Lookup outputs available: {lookup_outputs or 'none yet'}"
                        })
    
    # Check for missing $limit
    if not has_limit:
        errors.append({
            "stage_index": -1,
            "error_type": "warning",
            "message": "No $limit stage found",
            "suggestion": "Add $limit to prevent returning too many documents"
        })
    
    # Determine if valid (only errors, not warnings)
    is_valid = not any(e["error_type"] == "error" for e in errors)
    
    return is_valid, errors


def validator_node(state: MultiCollectionState, config: RunnableConfig) -> dict:
    """
    Validate the generated pipeline before execution.
    """
    pipeline = state.get("pipeline", [])
    primary_collection = state.get("primary_collection", "")
    available_collections = state.get("available_collections", [])
    schemas = state.get("collection_schemas", {})
    validation_attempts = state.get("validation_attempts", 0)
    
    dispatch_custom_event(
        "validation_started",
        {
            "type": "validation_started",
            "pipeline_stages": len(pipeline),
            "primary_collection": primary_collection,
            "attempt": validation_attempts + 1
        },
        config=config
    )
    
    is_valid, errors = validate_pipeline(
        pipeline=pipeline,
        primary_collection=primary_collection,
        available_collections=available_collections,
        schemas=schemas
    )
    
    if is_valid:
        warnings = [e for e in errors if e["error_type"] == "warning"]
        
        dispatch_custom_event(
            "validation_passed",
            {
                "type": "validation_passed",
                "warnings_count": len(warnings),
                "warnings": warnings
            },
            config=config
        )
        
        logger.info(f"Validation passed with {len(warnings)} warnings")
        
        return {
            "is_valid": True,
            "validation_errors": warnings,  # Keep warnings for reference
            "validation_attempts": validation_attempts + 1
        }
    else:
        # Log the actual errors for debugging
        error_details = [e for e in errors if e["error_type"] == "error"]
        for err in error_details:
            logger.warning(f"Validation error at stage {err.get('stage_index', '?')}: {err.get('message', '')} - Suggestion: {err.get('suggestion', '')}")
        
        dispatch_custom_event(
            "validation_failed",
            {
                "type": "validation_failed",
                "errors": errors,
                "attempt": validation_attempts + 1,
                "max_attempts": 3
            },
            config=config
        )
        
        logger.warning(f"Validation failed with {len(error_details)} errors (total issues: {len(errors)})")
        
        return {
            "is_valid": False,
            "validation_errors": errors,
            "validation_attempts": validation_attempts + 1
        }
