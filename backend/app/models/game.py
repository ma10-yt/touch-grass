from typing import List, Optional, Literal
from pydantic import AliasChoices, BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from app.models.location_context import LocationContext
from app.models.target import Target

class GameState(str, Enum):
    WAITING = "WAITING"
    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"


class TargetChangeVote(BaseModel):
    proposal_id: str
    target_id: str
    target_name: str
    requested_by: str
    eligible_player_ids: list[str]
    approved_player_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "rejected", "approved", "unavailable"]
    message: Optional[str] = None
    replacement_target: Optional[Target] = None
    revision: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TargetChangeRequest(BaseModel):
    player_id: str


class TargetChangeBallot(BaseModel):
    player_id: str
    approve: bool


class Game(BaseModel):
    id: str = Field(alias="_id")
    join_code: str
    creator_id: str
    status: GameState = GameState.WAITING
    targets: List[Target] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    location_context: Optional[LocationContext] = None
    location_context_lat: Optional[float] = None
    location_context_lon: Optional[float] = None
    location_context_updated_at: Optional[datetime] = None
    context_candidate_lat: Optional[float] = None
    context_candidate_lon: Optional[float] = None
    context_candidate_count: int = 0
    target_change_vote: Optional[TargetChangeVote] = None

class GameCreate(BaseModel):
    creator_name: str
    lat: Optional[float] = None
    lon: Optional[float] = None

class GameResponse(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "_id"))
    join_code: str
    creator_id: str
    status: GameState
    targets: List[Target]
    created_at: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location_context: Optional[LocationContext] = None
    target_change_vote: Optional[TargetChangeVote] = None
