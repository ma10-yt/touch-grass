from typing import Literal, Optional
from pydantic import BaseModel, Field

EnvironmentType = Literal["urban", "suburban", "rural", "park_nature", "unknown"]

class TargetCandidate(BaseModel):
    class_name: str
    score: int
    reasons: list[str] = Field(default_factory=list)


class LocationContext(BaseModel):
    locality: Optional[str] = None
    region: Optional[str] = None
    environment_type: EnvironmentType = "unknown"
    nearby_categories: list[str] = Field(default_factory=list)
    nearby_feature_counts: dict[str, int] = Field(default_factory=dict)
    target_candidates: list[TargetCandidate] = Field(default_factory=list)
    context_mode: Literal["contextual", "general"] = "general"
    context_provider: Optional[str] = None
    context_status: Literal["available", "partial", "unavailable"] = "unavailable"
    notice: Optional[str] = None
