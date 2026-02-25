from pydantic import BaseModel, Field
from typing import Optional
import uuid


class ExecuteRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class StepTrace(BaseModel):
    module: str
    prompt: Optional[str] = None
    response: Optional[str] = None


class ExecuteResponse(BaseModel):
    status: str = "ok"
    error: Optional[str] = None
    response: str = ""
    steps: list[StepTrace] = []


class TeamMember(BaseModel):
    name: str
    id: str


class TeamInfo(BaseModel):
    team_name: str
    students: list[TeamMember]


class AgentInfo(BaseModel):
    description: str
    purpose: str
    prompt_templates: list[str]
    examples: list[dict]


class BakeStatus(BaseModel):
    session_id: str
    active: bool
    current_step: Optional[str] = None
    next_step: Optional[str] = None
    time_remaining_minutes: Optional[float] = None
    steps: list[dict] = []
