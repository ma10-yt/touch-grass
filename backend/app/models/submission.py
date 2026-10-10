from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Submission(BaseModel):
    id: str = Field(alias="_id")
    game_id: str
    player_id: str
    target_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    verified: bool
    confidence: Optional[float] = None
    message: str
    distance_meters: Optional[float] = None

class SubmissionResult(BaseModel):
    success: bool
    target: str
    verified: bool
    confidence: Optional[float] = None
    message: str
    completed_targets: int = 0
    total_targets: int = 0
    progress_percent: int = 0
    demo_mode: bool = False
