from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    system_prompt: str
    temperature: float = 0.7
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    web_search_enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    document_ids: list[str] = Field(default_factory=list)
