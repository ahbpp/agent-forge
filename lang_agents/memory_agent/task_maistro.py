import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from lang_agents.memory_agent import configuration

## Utilities 

# Inspect the tool calls for Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Idea", "Profile")
    """
    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Check if there are any patches
                if call['args']['patches']:
                    if call['args']['patches'][0]['op'] == 'remove':
                        changes.append({
                            'type': 'remove',
                            'doc_id': call['args']['json_doc_id'],
                            'planned_edits': call['args']['planned_edits']
                        })
                    else:
                        changes.append({
                            'type': 'update',
                            'doc_id': call['args']['json_doc_id'],
                            'planned_edits': call['args']['planned_edits'],
                            'value': call['args']['patches'][0]['value']
                        })
                else:
                    # Handle case where no changes were needed
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        elif change['type'] == 'remove':
            result_parts.append(
                f"Document {change['doc_id']} removed:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts), changes


def get_model_from_config(config_obj: configuration.Configuration) -> BaseChatModel:
    """Initialize the appropriate model based on the configuration.
    
    Args:
        config_obj: Configuration object containing model settings
        
    Returns:
        Initialized chat model
    """
    # Use init_chat_model to initialize the model based on provider
    model_provider = config_obj.model_provider
    if model_provider == configuration.ModelProvider.OLLAMA:
        return ChatOllama(model=config_obj.model_name, 
                          base_url="http://host.docker.internal:11434",
                          temperature=config_obj.temperature)
    
    model_provider = model_provider.value if isinstance(model_provider, configuration.ModelProvider) else model_provider
    return init_chat_model(
        model=config_obj.model_name,
        model_provider=model_provider,
        temperature=config_obj.temperature
    )

## Schema definitions

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).")
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

# Idea schema
class Idea(BaseModel):
    """Schema for storing creative ideas"""
    idea: str = Field(description="The idea or concept title")
    description: Optional[str] = Field(
        description="A longer description of the idea with more context"
    )
    solutions: list[str] = Field(
        description="List of possible implementations or ways to develop this idea",
        min_items=1,
        default_factory=list
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Estimated difficulty level to implement this idea",
    )
    time_to_implement: Optional[int] = Field(
        description="Estimated time to implement this idea (minutes)",
        default=None
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the idea",
        default="not started"
    )


# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions', 'idea']

## Prompts 

# TODO: Imprve this prompt. It works not well for Gemini and DeepSeek.
# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """{task_maistro_role} 

You have a long term memory which keeps track of four things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. The user's Ideas list
4. General instructions for updating the ToDo or Ideas list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here is the current Ideas List (may be empty if no ideas have been added yet):
<ideas>
{ideas}
</ideas>

Here are the current user-specified preferences for updating the ToDo or Ideas list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. First, carefully analyze the user's message to understand their intent and any implicit or explicit information they've shared.

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If ideas or concepts are mentioned, update the Ideas list by calling UpdateMemory tool with type `idea`
- If the user has specified preferences for how to update the ToDo or Ideas list, update the instructions by calling UpdateMemory tool with type `instructions`
- If the user asks only to retrieve information from the memory, return the information based on current User Profile, ToDo List, and Ideas List

4. For complex messages, prioritize identifying multiple relevant pieces of information that might belong in different memory categories.

5. Tell the user that you have updated your memory, if you did

6. Err on the side of updating the todo list. No need to ask for explicit permission.

7. End after you perform the action requested by the user. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made.
"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo or Ideas list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo or Ideas list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

## Node definitions

def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    task_maistro_role = configurable.task_maistro_role

    # Initialize the model based on configuration
    model = get_model_from_config(configurable)

   # Retrieve profile memory from the store
    namespace = ("profile", todo_category, user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve people memory from the store
    namespace = ("todo", todo_category, user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve ideas memory from the store
    namespace = ("idea", todo_category, user_id)
    memories = store.search(namespace)
    ideas = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", todo_category, user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(task_maistro_role=task_maistro_role, 
                                             user_profile=user_profile, 
                                             todo=todo, 
                                             ideas=ideas, 
                                             instructions=instructions)

    # TODO: check that there is no bug here.
    # Respond using memory as well as the chat history
    # The issue is that we're comparing an enum instance with an enum class
    # We need to compare the enum values directly
    if (configurable.model_provider == configuration.ModelProvider.GEMINI or
        configurable.model_provider == configuration.ModelProvider.OLLAMA):
        response = model.bind_tools([UpdateMemory]).invoke([SystemMessage(content=system_msg)]+state["messages"])
    else:
        response = model.bind_tools([UpdateMemory], 
                                    parallel_tool_calls=False
                                    ).invoke([SystemMessage(content=system_msg)]+state["messages"])
        
    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Initialize the model based on configuration
    model = get_model_from_config(configurable)
    
    # Create the extractor with the dynamically initialized model
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )

    # Define the namespace for the memories
    namespace = ("profile", todo_category, user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Invoke the extractor
    result = profile_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Initialize the model based on configuration
    model = get_model_from_config(configurable)

    # Define the namespace for the memories
    namespace = ("todo", todo_category, user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor for updating the ToDo list 
    todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice=tool_name,
    enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})
    todo_update_msg, changes_to_apply = extract_tool_info(spy.called_tools, tool_name)


    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    for change in changes_to_apply:
        if change['type'] == 'remove':
            try:
                store.delete(namespace, change['doc_id'])
            except Exception as e:
                print(e)
        
    # Respond to the tool call made in task_mAIstro, confirming the update    
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id": tool_calls[0]['id']}]}

def update_ideas(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the ideas collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Initialize the model based on configuration
    model = get_model_from_config(configurable)

    # Define the namespace for the ideas
    namespace = ("idea", todo_category, user_id)

    # Retrieve the most recent ideas for context
    existing_items = store.search(namespace)

    # Format the existing ideas for the Trustcall extractor
    tool_name = "Idea"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor for updating the ideas list 
    idea_extractor = create_extractor(
    model,
    tools=[Idea],
    tool_choice=tool_name,
    enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = idea_extractor.invoke({"messages": updated_messages, 
                                   "existing": existing_memories})
    idea_update_msg, changes_to_apply = extract_tool_info(spy.called_tools, tool_name)


    # Save the ideas from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    for change in changes_to_apply:
        if change['type'] == 'remove':
            try:
                store.delete(namespace, change['doc_id'])
            except Exception as e:
                print(e)
        
    # Respond to the tool call made in task_mAIstro, confirming the update    
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    return {"messages": [{"role": "tool", "content": idea_update_msg, "tool_call_id": tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    
    # Initialize the model based on configuration
    model = get_model_from_config(configurable)
    
    namespace = ("instructions", todo_category, user_id)

    existing_memory = store.get(namespace, "user_instructions")
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_todos", "update_instructions", "update_profile", "update_ideas"]: # type: ignore

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        elif tool_call['args']['update_type'] == "idea":
            return "update_ideas"
        else:
            raise ValueError

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(task_mAIstro)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)
builder.add_node(update_ideas)

# Define the flow 
builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")
builder.add_edge("update_ideas", "task_mAIstro")

# Compile the graph
graph = builder.compile()