"""
Chainlit UI for PyMongo Agent

Run with: chainlit run ui/chainlit_app.py -w
"""
import json
from dataclasses import asdict
import chainlit as cl

from lang_agents.pymongo_agent.pymongo_agent import graph
from lang_agents.pymongo_agent.configuration import Configuration


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    cl.user_session.set("history", [])
    
    # Display welcome message
    await cl.Message(
        content="👋 **Welcome to MongoDB Query Assistant!**\n\n"
                "I can help you query your MongoDB database using natural language.\n\n"
                "Try asking something like:\n"
                "- *Show me all users*\n"
                "- *What are the top 5 orders by amount?*\n"
                "- *Count documents in each collection*"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages"""
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})
    
    # Convert dataclass to dict for config
    config_dict = asdict(Configuration())
    # Convert enum to string value
    config_dict["model_provider"] = config_dict["model_provider"].value
    config = {"configurable": config_dict}
    
    # Create main processing step
    async with cl.Step(name="🔍 Processing Query", type="run") as main_step:
        main_step.input = message.content
        
        # Track steps for nested display
        planner_step = None
        query_builder_step = None
        validator_step = None
        executor_step = None
        simple_handler_step = None
        simple_executor_step = None
        validation_failure_step = None
        
        try:
            async for event in graph.astream_events(
                {"messages": history},
                config=config,
                version="v2"
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")
                
                # Handle node starts
                if event_type == "on_chain_start":
                    if event_name == "planner":
                        planner_step = cl.Step(
                            name="🧠 Planning Query",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await planner_step.__aenter__()
                    
                    elif event_name == "query_builder":
                        query_builder_step = cl.Step(
                            name="🔧 Building Query",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await query_builder_step.__aenter__()
                    
                    elif event_name == "validator":
                        validator_step = cl.Step(
                            name="✅ Validating Query",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await validator_step.__aenter__()
                    
                    elif event_name == "executor":
                        executor_step = cl.Step(
                            name="⚡ Executing Query",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await executor_step.__aenter__()
                    
                    elif event_name == "simple_handler":
                        simple_handler_step = cl.Step(
                            name="📁 Selecting Collection",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await simple_handler_step.__aenter__()
                    
                    elif event_name == "simple_executor":
                        simple_executor_step = cl.Step(
                            name="🔄 Building & Executing Query",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await simple_executor_step.__aenter__()
                    
                    elif event_name == "validation_failure":
                        validation_failure_step = cl.Step(
                            name="❌ Query Generation Failed",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await validation_failure_step.__aenter__()
                
                # Handle node ends
                elif event_type == "on_chain_end":
                    if event_name == "planner" and planner_step:
                        output = event.get("data", {}).get("output", {})
                        plan = output.get("plan", {})
                        if plan:
                            query_type = plan.get("query_type", "unknown")
                            reasoning = plan.get("reasoning", "")
                            planner_step.output = f"**Type:** {query_type}\n**Reasoning:** {reasoning}"
                        await planner_step.__aexit__(None, None, None)
                    
                    elif event_name == "query_builder" and query_builder_step:
                        output = event.get("data", {}).get("output", {})
                        pipeline = output.get("pipeline", [])
                        query_builder_step.output = f"Built pipeline with **{len(pipeline)}** stages"
                        await query_builder_step.__aexit__(None, None, None)
                    
                    elif event_name == "validator" and validator_step:
                        output = event.get("data", {}).get("output", {})
                        is_valid = output.get("is_valid", False)
                        errors = output.get("validation_errors", [])
                        if is_valid:
                            validator_step.output = "✅ Query validated successfully"
                        else:
                            error_msgs = [e.get("message", "") for e in errors if e.get("error_type") == "error"]
                            validator_step.output = f"❌ Validation failed: {'; '.join(error_msgs)}"
                        await validator_step.__aexit__(None, None, None)
                    
                    elif event_name == "executor" and executor_step:
                        output = event.get("data", {}).get("output", {})
                        count = output.get("result_count", 0)
                        errors = output.get("errors", [])
                        if errors:
                            executor_step.output = f"❌ Error: {errors[-1]}"
                        else:
                            executor_step.output = f"✅ Retrieved **{count}** documents"
                        await executor_step.__aexit__(None, None, None)
                    
                    elif event_name == "simple_handler" and simple_handler_step:
                        output = event.get("data", {}).get("output", {})
                        collection = output.get("primary_collection", "")
                        if collection:
                            simple_handler_step.output = f"Selected: **{collection}**"
                        else:
                            simple_handler_step.output = "Direct response"
                        await simple_handler_step.__aexit__(None, None, None)
                    
                    elif event_name == "simple_executor" and simple_executor_step:
                        output = event.get("data", {}).get("output", {})
                        count = output.get("result_count", 0)
                        pipeline = output.get("pipeline", [])
                        simple_executor_step.output = f"✅ Retrieved **{count}** documents\n\n```json\n{json.dumps(pipeline, indent=2)}\n```"
                        await simple_executor_step.__aexit__(None, None, None)
                    
                    elif event_name == "validation_failure" and validation_failure_step:
                        output = event.get("data", {}).get("output", {})
                        errors = output.get("errors", [])
                        validation_failure_step.output = f"❌ Failed after max retries\n\n**Errors:**\n" + "\n".join(errors[:5])
                        await validation_failure_step.__aexit__(None, None, None)
                
                # Handle custom events (for detailed step visibility)
                elif event_type == "on_custom_event":
                    custom_event_name = event.get("name", "")
                    event_data = event.get("data", {})
                    
                    # Get the current active step for parenting
                    active_step = executor_step or simple_executor_step or query_builder_step or main_step
                    
                    # Planning events
                    if custom_event_name == "planning_started":
                        pass  # Handled by node start
                    
                    elif custom_event_name == "plan_created":
                        query_type = event_data.get("query_type", "")
                        lookups_count = event_data.get("lookups_count", 0)
                        reasoning = event_data.get("reasoning", "")
                        if planner_step:
                            planner_step.output = f"**Type:** {query_type}\n**Joins:** {lookups_count}\n**Reasoning:** {reasoning}"
                    
                    # Query builder events
                    elif custom_event_name == "query_building_started":
                        is_retry = event_data.get("is_retry", False)
                        attempt = event_data.get("attempt", 1)
                        if query_builder_step and is_retry:
                            query_builder_step.name = f"🔧 Building Query (Attempt {attempt})"
                    
                    elif custom_event_name == "query_built":
                        pipeline = event_data.get("pipeline", [])
                        collection = event_data.get("primary_collection", "")
                        async with cl.Step(
                            name="📝 Generated Pipeline",
                            type="tool",
                            parent_id=query_builder_step.id if query_builder_step else main_step.id
                        ) as pipeline_step:
                            pipeline_step.output = f"**Collection:** `{collection}`\n\n**Pipeline:**\n```json\n{json.dumps(pipeline, indent=2)}\n```"
                    
                    # Validation events
                    elif custom_event_name == "validation_started":
                        stages = event_data.get("pipeline_stages", 0)
                        if validator_step:
                            validator_step.output = f"Checking {stages} pipeline stages..."
                    
                    elif custom_event_name == "validation_passed":
                        warnings = event_data.get("warnings", [])
                        if validator_step:
                            if warnings:
                                validator_step.output = f"✅ Valid (with {len(warnings)} warnings)"
                            else:
                                validator_step.output = "✅ Query validated successfully"
                    
                    elif custom_event_name == "validation_failed":
                        errors = event_data.get("errors", [])
                        attempt = event_data.get("attempt", 1)
                        max_attempts = event_data.get("max_attempts", 3)
                        async with cl.Step(
                            name=f"⚠️ Validation Failed (Attempt {attempt}/{max_attempts})",
                            type="tool",
                            parent_id=validator_step.id if validator_step else main_step.id
                        ) as err_step:
                            error_text = "\n".join([f"- {e.get('message', '')}: {e.get('suggestion', '')}" for e in errors])
                            err_step.output = f"**Errors found:**\n{error_text}\n\nRetrying..."
                    
                    # Execution events
                    elif custom_event_name == "execution_started":
                        collection = event_data.get("collection", "")
                        if executor_step:
                            executor_step.output = f"Running on `{collection}`..."
                    
                    elif custom_event_name == "execution_success":
                        count = event_data.get("count", 0)
                        async with cl.Step(
                            name="✅ Query Completed",
                            type="tool",
                            parent_id=executor_step.id if executor_step else main_step.id
                        ) as success_step:
                            success_step.output = f"Retrieved **{count}** documents"
                    
                    elif custom_event_name == "execution_failed":
                        error = event_data.get("error", "Unknown")
                        attempts = event_data.get("attempts", 0)
                        async with cl.Step(
                            name="❌ Execution Failed",
                            type="tool",
                            parent_id=executor_step.id if executor_step else main_step.id
                        ) as fail_step:
                            fail_step.output = f"**Failed after {attempts} attempts**\n\n**Error:** {error}"
                    
                    elif custom_event_name == "execution_retry":
                        attempt = event_data.get("attempt", 0)
                        max_retries = event_data.get("max_retries", 3)
                        error = event_data.get("error", "")
                        async with cl.Step(
                            name=f"🔄 Retry {attempt}/{max_retries}",
                            type="tool",
                            parent_id=executor_step.id if executor_step else main_step.id
                        ) as retry_step:
                            retry_step.output = f"**Error:** {error}\n\nRetrying..."
                    
                    # Complex query failed after max retries
                    elif custom_event_name == "complex_query_failed":
                        attempts = event_data.get("attempts", 0)
                        errors = event_data.get("errors", [])
                        async with cl.Step(
                            name="❌ Complex Query Failed",
                            type="tool",
                            parent_id=validation_failure_step.id if validation_failure_step else main_step.id
                        ) as fail_step:
                            error_text = "\n".join(errors[:5]) if errors else "Unknown error"
                            fail_step.output = f"**Failed after {attempts} attempts**\n\n**Errors:**\n{error_text}"
                    
                    # Collection selected (simple path)
                    elif custom_event_name == "collection_selected":
                        collection = event_data.get("collection", "unknown")
                        if simple_handler_step:
                            simple_handler_step.output = f"Selected: **{collection}**"
                    
                    # Query generated (simple path)
                    elif custom_event_name == "query_generated":
                        query = event_data.get("query", [])
                        collection = event_data.get("collection", "")
                        attempt = event_data.get("attempt", 1)
                        
                        step_name = "📝 Generated Query" if attempt == 1 else f"📝 Generated Query (Attempt {attempt})"
                        async with cl.Step(
                            name=step_name,
                            type="tool",
                            parent_id=simple_executor_step.id if simple_executor_step else main_step.id
                        ) as query_step:
                            query_step.output = f"**Collection:** `{collection}`\n\n**Pipeline:**\n```json\n{json.dumps(query, indent=2)}\n```"
                    
                    # Query executing (simple path)
                    elif custom_event_name == "query_executing":
                        collection = event_data.get("collection", "")
                        async with cl.Step(
                            name="⚡ Executing Query",
                            type="tool",
                            parent_id=simple_executor_step.id if simple_executor_step else main_step.id
                        ) as exec_step:
                            exec_step.output = f"Running aggregation on `{collection}`..."
                    
                    # Query success (simple path)
                    elif custom_event_name == "query_success":
                        count = event_data.get("count", 0)
                        async with cl.Step(
                            name="✅ Query Completed",
                            type="tool",
                            parent_id=simple_executor_step.id if simple_executor_step else main_step.id
                        ) as success_step:
                            success_step.output = f"Retrieved **{count}** documents"
                    
                    # Query failed (simple path)
                    elif custom_event_name == "query_failed":
                        error = event_data.get("error", "Unknown error")
                        attempts = event_data.get("attempts", 0)
                        async with cl.Step(
                            name="❌ Query Failed",
                            type="tool",
                            parent_id=simple_executor_step.id if simple_executor_step else main_step.id
                        ) as fail_step:
                            fail_step.output = f"**Failed after {attempts} attempts**\n\n**Error:** {error}"
                    
                    # Retry attempt (simple path)
                    elif custom_event_name == "retry_attempt":
                        attempt = event_data.get("attempt", 0)
                        max_retries = event_data.get("max_retries", 3)
                        error = event_data.get("error", "Unknown error")
                        async with cl.Step(
                            name=f"🔄 Retry Attempt {attempt}/{max_retries}",
                            type="tool",
                            parent_id=simple_executor_step.id if simple_executor_step else main_step.id
                        ) as retry_step:
                            retry_step.output = f"**Previous error:** {error}\n\nGenerating new query..."
        
        except Exception as e:
            main_step.output = f"❌ Error: {str(e)}"
            raise
    
    # Get final response by invoking the graph
    result = await graph.ainvoke({"messages": history}, config=config)
    final_messages = result.get("messages", [])
    
    if final_messages:
        last_message = final_messages[-1]
        
        # Extract content based on message type
        if hasattr(last_message, 'content'):
            response_content = last_message.content
        else:
            response_content = last_message.get("content", "")
        
        # Try to parse and format tool response
        try:
            data = json.loads(response_content)
            if "query" in data and "result" in data:
                # This is an aggregate result
                query = data.get("query")
                results = data.get("result", [])
                count = data.get("count", 0)
                error = data.get("error")
                
                if error:
                    formatted_response = f"❌ **Query Failed**\n\n**Error:** {error}"
                else:
                    formatted_response = f"✅ **Query Results** ({count} documents)\n\n"
                    formatted_response += f"**Query:**\n```json\n{json.dumps(query, indent=2)}\n```\n\n"
                    
                    if results:
                        # Limit displayed results
                        display_results = results[:10]
                        formatted_response += f"**Results:**\n```json\n{json.dumps(display_results, indent=2)}\n```"
                        if len(results) > 10:
                            formatted_response += f"\n\n*...and {len(results) - 10} more documents*"
                    else:
                        formatted_response += "📭 **No matching documents found in the database.**\n\n"
                        formatted_response += "Try adjusting your search parameters"
                
                response_content = formatted_response
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            pass
        
        # Update history
        history.append({"role": "assistant", "content": response_content})
        cl.user_session.set("history", history)
        
        # Send final response
        await cl.Message(content=response_content).send()


@cl.on_chat_end
async def on_chat_end():
    """Clean up on chat end"""
    cl.user_session.set("history", [])


# Optional: Add settings panel for configuration
@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates"""
    # You can add configurable settings here
    pass
