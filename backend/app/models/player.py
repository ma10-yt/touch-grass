from typing import List, Optional
from pydantic import AliasChoices, BaseModel, Field
from datetime import datetime, timezone

class Player(BaseModel):
    id: str = Field(alias="_id")
    game_id: str
    display_name: str
    latest_lat: Optional[float] = None
    latest_lon: Optional[float] = None
    last_location_update: Optional[datetime] = None
    location_accuracy_m: Optional[float] = None
    movement_meters: float = 0.0
    movement_last_lat: Optional[float] = None
    movement_last_lon: Optional[float] = None
    movement_last_sample_at: Optional[datetime] = None
    movement_last_accuracy_m: Optional[float] = None
    movement_candidate_lat: Optional[float] = None
    movement_candidate_lon: Optional[float] = None
    movement_candidate_count: int = 0
    score: int = 0
    completed_targets: List[str] = Field(default_factory=list)
    join_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: Optional[datetime] = None

class PlayerJoin(BaseModel):
    display_name: str
    join_code: str

class LocationUpdate(BaseModel):
    lat: float = Field(
        validation_alias=AliasChoices("lat", "latitude"),
        ge=-90,
        le=90,
    )
    lon: float = Field(
        validation_alias=AliasChoices("lon", "longitude"),
        ge=-180,
        le=180,
    )
    accuracy_m: float = Field(
        validation_alias=AliasChoices("accuracy_m", "accuracy"),
        gt=0,
    )
    timestamp: datetime

class PlayerResponse(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "_id"))
    game_id: str
    display_name: str
    score: int
    completed_targets: List[str]
    movement_meters: float = 0.0
