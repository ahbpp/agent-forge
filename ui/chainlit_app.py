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
        
        collection_step = None
        aggregate_step = None
        current_attempt = 0
        
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
                    if event_name == "handle_request":
                        collection_step = cl.Step(
                            name="📁 Selecting Collection",
                            type="tool",
                            parent_id=main_step.id
                        )
                        await collection_step.__aenter__()
                        
                    elif event_name == "run_aggregate":
                        aggregate_step = cl.Step(
                            name="🔄 Building & Executing Query",
                            type="tool", 
                            parent_id=main_step.id
                        )
                        await aggregate_step.__aenter__()
                
                # Handle node ends
                elif event_type == "on_chain_end":
                    if event_name == "handle_request" and collection_step:
                        output = event.get("data", {}).get("output", {})
                        messages = output.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            # Check if collection was selected
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    coll = tc.get('args', {}).get('collection')
                                    if coll:
                                        collection_step.output = f"Selected: **{coll}**"
                            else:
                                # Direct response without query
                                collection_step.output = "Direct response (no collection needed)"
                        await collection_step.__aexit__(None, None, None)
                        
                    elif event_name == "run_aggregate" and aggregate_step:
                        output = event.get("data", {}).get("output", {})
                        messages = output.get("messages", [])
                        if messages:
                            content = messages[-1].get("content", "{}")
                            try:
                                data = json.loads(content)
                                query = data.get("query")
                                count = data.get("count", 0)
                                error = data.get("error")
                                
                                if error:
                                    aggregate_step.output = f"❌ **Error:** {error}"
                                else:
                                    aggregate_step.output = (
                                        f"✅ **Success!** Retrieved {count} documents\n\n"
                                        f"```json\n{json.dumps(query, indent=2)}\n```"
                                    )
                            except json.JSONDecodeError:
                                aggregate_step.output = content
                        await aggregate_step.__aexit__(None, None, None)
                
                # Handle custom events (for retry visibility)
                elif event_type == "on_custom_event":
                    event_name = event.get("name", "")
                    event_data = event.get("data", {})
                    
                    if event_name == "retry_attempt" or event_data.get("type") == "retry":
                        attempt = event_data.get("attempt", 0)
                        max_retries = event_data.get("max_retries", 3)
                        error = event_data.get("error", "Unknown error")
                        async with cl.Step(
                            name=f"🔄 Retry Attempt {attempt}/{max_retries}",
                            type="tool",
                            parent_id=aggregate_step.id if aggregate_step else main_step.id
                        ) as retry_step:
                            retry_step.output = f"**Error:** {error}\n\nRetrying..."
        
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
                        formatted_response += "*No documents found*"
                
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
