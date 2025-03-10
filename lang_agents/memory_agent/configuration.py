import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from enum import Enum
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass


class ModelProvider(Enum):
    OPENAI = "openai"
    GEMINI = "google_genai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    user_id: str = "default-user"
    todo_category: str = "general" 
    task_maistro_role: str = "You are a helpful task management assistant. You help you create, organize, and manage the user's ToDo list."
    model_provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4o"
    temperature: float = 0.0

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        
        # Convert model_provider string to Enum if it exists
        if "model_provider" in values and values["model_provider"] is not None:
            if isinstance(values["model_provider"], str):
                try:
                    values["model_provider"] = ModelProvider(values["model_provider"])
                except ValueError:
                    raise ValueError(f"Invalid model provider: {values['model_provider']}")
        
        return cls(**{k: v for k, v in values.items() if v})