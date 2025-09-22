"""
Pydantic schemas for simulation engine data models.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Valid event types for global events."""
    WORK = "work"
    SOCIAL = "social"
    PERSONAL = "personal"


class EventStatus(str, Enum):
    """Valid processing statuses for events."""
    UNPROCESSED = "unprocessed"
    PROCESSED = "processed"


class ImpactLevel(str, Enum):
    """Valid impact levels for event effects."""
    INCREASE = "increase"
    DECREASE = "decrease"
    NEUTRAL = "neutral"


class MoodImpact(str, Enum):
    """Valid mood impact types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TrendDirection(str, Enum):
    """Valid trend directions for state changes."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class LogLevel(str, Enum):
    """Valid log levels."""
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class GlobalEventBase(BaseModel):
    """Base schema for GlobalEvents."""
    event_type: EventType = Field(..., description="Event category")
    summary: str = Field(..., min_length=1, max_length=1000, description="Event description")
    intensity: Optional[int] = Field(None, ge=1, le=10, description="Event intensity (1-10)")
    impact_mood: Optional[MoodImpact] = Field(None, description="Expected mood impact")
    impact_energy: Optional[ImpactLevel] = Field(None, description="Expected energy impact")
    impact_stress: Optional[ImpactLevel] = Field(None, description="Expected stress impact")


class GlobalEventCreate(GlobalEventBase):
    """Schema for creating a new global event."""
    pass


class GlobalEventUpdate(BaseModel):
    """Schema for updating an existing global event."""
    status: Optional[EventStatus] = None
    processed_at: Optional[datetime] = None
    emotional_reaction: Optional[str] = Field(None, description="Ava's emotional reaction")
    chosen_action: Optional[str] = Field(None, description="Ava's chosen action")
    internal_thoughts: Optional[str] = Field(None, description="Ava's internal thoughts")
    consciousness_raw_response: Optional[str] = Field(None, description="Raw LLM response")


class GlobalEvent(GlobalEventBase):
    """Complete schema for GlobalEvents with all fields."""
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    timestamp: datetime
    status: EventStatus
    processed_at: Optional[datetime] = None
    created_by: str
    emotional_reaction: Optional[str] = None
    chosen_action: Optional[str] = None
    internal_thoughts: Optional[str] = None
    consciousness_raw_response: Optional[str] = None


class AvaGlobalStateBase(BaseModel):
    """Base schema for AvaGlobalState."""
    trait_name: str = Field(..., min_length=1, max_length=50, description="Trait identifier")
    value: str = Field(..., min_length=1, max_length=100, description="Trait value")
    numeric_value: Optional[int] = Field(None, ge=0, le=100, description="Numeric value (0-100)")
    change_reason: Optional[str] = Field(None, max_length=200, description="Reason for change")
    trend: Optional[TrendDirection] = Field(None, description="Recent trend")
    min_value: Optional[int] = Field(0, ge=0, description="Minimum allowed value")
    max_value: Optional[int] = Field(100, le=100, description="Maximum allowed value")


class AvaGlobalStateCreate(AvaGlobalStateBase):
    """Schema for creating a new global state trait."""
    pass


class AvaGlobalStateUpdate(BaseModel):
    """Schema for updating an existing global state trait."""
    value: Optional[str] = Field(None, min_length=1, max_length=100)
    numeric_value: Optional[int] = Field(None, ge=0, le=100)
    change_reason: Optional[str] = Field(None, max_length=200)
    trend: Optional[TrendDirection] = None
    last_event_id: Optional[str] = None


class AvaGlobalState(AvaGlobalStateBase):
    """Complete schema for AvaGlobalState with all fields."""
    model_config = ConfigDict(from_attributes=True)

    state_id: int
    last_updated: datetime
    last_event_id: Optional[str] = None


class SimulationLogBase(BaseModel):
    """Base schema for SimulationLog."""
    level: LogLevel = Field(..., description="Log level")
    component: str = Field(..., min_length=1, max_length=50, description="System component")
    message: str = Field(..., min_length=1, description="Log message")
    event_id: Optional[str] = Field(None, description="Related event ID")
    user_id: Optional[str] = Field(None, description="Related user ID")
    extra_data: Optional[str] = Field(None, description="Additional metadata as JSON")


class SimulationLogCreate(SimulationLogBase):
    """Schema for creating a new simulation log entry."""
    pass


class SimulationLog(SimulationLogBase):
    """Complete schema for SimulationLog with all fields."""
    model_config = ConfigDict(from_attributes=True)

    log_id: int
    timestamp: datetime


class SimulationConfigBase(BaseModel):
    """Base schema for SimulationConfig."""
    key: str = Field(..., min_length=1, max_length=100, description="Configuration key")
    value: str = Field(..., min_length=1, max_length=500, description="Configuration value")
    description: Optional[str] = Field(None, description="Configuration description")
    category: str = Field("general", max_length=50, description="Configuration category")
    is_active: bool = Field(True, description="Whether configuration is active")


class SimulationConfigCreate(SimulationConfigBase):
    """Schema for creating a new simulation configuration."""
    pass


class SimulationConfigUpdate(BaseModel):
    """Schema for updating an existing simulation configuration."""
    value: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class SimulationConfig(SimulationConfigBase):
    """Complete schema for SimulationConfig with all fields."""
    model_config = ConfigDict(from_attributes=True)

    config_id: int
    last_updated: datetime


# Aggregated response schemas for API endpoints
class SimulationSummary(BaseModel):
    """Summary of simulation engine status."""
    total_events: int
    unprocessed_events: int
    last_event_time: Optional[datetime]
    active_traits: int
    simulation_uptime: Optional[str]


class EventGenerationStats(BaseModel):
    """Statistics about event generation."""
    events_today: int
    events_this_week: int
    avg_events_per_day: float
    most_common_type: Optional[str]
    least_common_type: Optional[str]