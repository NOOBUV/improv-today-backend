"""
Pydantic schemas for journal entry validation and API responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime, date
from enum import Enum


class JournalStatus(str, Enum):
    """Allowed journal entry status values"""
    DRAFT = "draft"
    APPROVED = "approved"
    POSTED = "posted"


class EmotionalTheme(str, Enum):
    """Allowed emotional theme values"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    CHAOTIC = "chaotic"


class JournalEntryBase(BaseModel):
    """Base schema for journal entries"""
    entry_date: date = Field(..., description="Date this journal entry is for")
    content: str = Field(..., min_length=10, max_length=2000, description="Journal entry content")
    status: JournalStatus = Field(default=JournalStatus.DRAFT, description="Entry status")
    emotional_theme: Optional[EmotionalTheme] = Field(None, description="Dominant emotional theme")
    admin_notes: Optional[str] = Field(None, max_length=1000, description="Admin notes")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate journal content meets quality standards"""
        if not v.strip():
            raise ValueError("Content cannot be empty")

        # Check for basic sentence structure
        if not any(punct in v for punct in '.!?'):
            raise ValueError("Content should contain proper punctuation")

        return v.strip()


class JournalEntryCreate(JournalEntryBase):
    """Schema for creating journal entries"""
    events_processed: Optional[int] = Field(None, ge=0, description="Number of events processed")
    metadata_json: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class JournalEntryUpdate(BaseModel):
    """Schema for updating journal entries"""
    content: Optional[str] = Field(None, min_length=10, max_length=2000)
    status: Optional[JournalStatus] = None
    admin_notes: Optional[str] = Field(None, max_length=1000)
    reviewed_by: Optional[str] = Field(None, max_length=100)

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate content if provided"""
        if v is not None:
            if not v.strip():
                raise ValueError("Content cannot be empty")
            return v.strip()
        return v


class JournalEntryResponse(JournalEntryBase):
    """Schema for journal entry API responses"""
    entry_id: str
    events_processed: Optional[int] = None
    generated_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    published_at: Optional[datetime] = None
    character_count: Optional[int] = None
    readability_score: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata_json: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class JournalEntryListResponse(BaseModel):
    """Schema for paginated journal entry lists"""
    entries: List[JournalEntryResponse]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool


class JournalGenerationRequest(BaseModel):
    """Schema for manual journal generation requests"""
    target_date: Optional[date] = Field(None, description="Date to generate journal for (defaults to today)")
    force_regenerate: bool = Field(False, description="Force regeneration even if entry exists")
    admin_user: Optional[str] = Field(None, description="Admin user requesting generation")


class JournalGenerationResponse(BaseModel):
    """Schema for journal generation results"""
    success: bool
    entry_id: Optional[str] = None
    target_date: date
    events_processed: int
    generation_duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    created_new: bool = False


class JournalStatsResponse(BaseModel):
    """Schema for journal statistics"""
    total_entries: int
    draft_count: int
    approved_count: int
    posted_count: int
    avg_events_per_entry: float
    most_common_theme: Optional[str] = None
    recent_generation_success_rate: float
    last_generated: Optional[datetime] = None


class DailyContextResponse(BaseModel):
    """Schema for daily context aggregation results"""
    target_date: date
    event_count: int
    significant_events: List[Dict[str, Any]]
    emotional_state: Dict[str, Any]
    dominant_emotion: str
    emotional_arc: str


class JournalTemplateBase(BaseModel):
    """Base schema for journal templates"""
    name: str = Field(..., min_length=1, max_length=100)
    emotional_theme: EmotionalTheme
    template_content: str = Field(..., min_length=10, max_length=1000)
    is_active: bool = Field(True)
    notes: Optional[str] = Field(None, max_length=500)


class JournalTemplateCreate(JournalTemplateBase):
    """Schema for creating journal templates"""
    pass


class JournalTemplateUpdate(BaseModel):
    """Schema for updating journal templates"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    emotional_theme: Optional[EmotionalTheme] = None
    template_content: Optional[str] = Field(None, min_length=10, max_length=1000)
    is_active: Optional[bool] = None
    notes: Optional[str] = Field(None, max_length=500)


class JournalTemplateResponse(JournalTemplateBase):
    """Schema for journal template API responses"""
    template_id: int
    usage_count: int
    created_at: datetime
    last_used: Optional[datetime] = None
    avg_engagement_score: Optional[str] = None

    class Config:
        from_attributes = True


class GenerationLogResponse(BaseModel):
    """Schema for journal generation log responses"""
    log_id: int
    target_date: date
    status: str
    events_found: int
    events_processed: int
    generation_duration_ms: Optional[int] = None
    llm_model_used: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    celery_task_id: Optional[str] = None
    triggered_by: str

    class Config:
        from_attributes = True