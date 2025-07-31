from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.core.database import get_db
from app.models.user import User
from app.models.session import Session as SessionModel, SessionTranscript
from app.models.vocabulary import VocabularyUsage

router = APIRouter()

# Request/Response Models
class SessionStartRequest(BaseModel):
    session_type: str = "practice"  # practice, assessment, daily
    topic: Optional[str] = None
    user_id: Optional[int] = None  # For anonymous users, we'll create one

class SessionStartResponse(BaseModel):
    session_id: int
    user_id: int
    status: str
    start_time: datetime
    message: str

class SessionEndRequest(BaseModel):
    session_id: int
    transcript_data: Optional[Dict[str, Any]] = None
    analysis_data: Optional[Dict[str, Any]] = None

class SessionEndResponse(BaseModel):
    session_id: int
    duration_seconds: int
    word_count: int
    vocabulary_used_count: int
    overall_score: Optional[float]
    analysis_summary: Dict[str, Any]
    success: bool

class TranscriptSaveRequest(BaseModel):
    session_id: int
    original_text: Optional[str] = None
    cleaned_text: str
    confidence_score: Optional[float] = None
    corrections_made: Optional[list] = None
    detected_vocabulary_level: Optional[str] = None

@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest, db: Session = Depends(get_db)):
    """
    Start a new conversation session
    Creates user if needed (for first-time anonymous users)
    """
    try:
        user_id = request.user_id
        
        # Create anonymous user if none provided
        if not user_id:
            new_user = User(
                is_anonymous=True,
                assessment_completed=False
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            user_id = new_user.id
        
        # Create new session
        session = SessionModel(
            user_id=user_id,
            session_type=request.session_type,
            topic=request.topic,
            status="active",
            start_time=datetime.utcnow(),
            word_count=0,
            vocabulary_used_count=0
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return SessionStartResponse(
            session_id=session.id,
            user_id=user_id,
            status=session.status,
            start_time=session.start_time,
            message=f"Session started successfully. Session ID: {session.id}"
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@router.post("/end", response_model=SessionEndResponse)
async def end_session(request: SessionEndRequest, db: Session = Depends(get_db)):
    """
    End a conversation session and calculate metrics
    """
    try:
        # Get session
        session = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "active":
            raise HTTPException(status_code=400, detail="Session is not active")
        
        # Calculate session metrics
        end_time = datetime.utcnow()
        duration_seconds = int((end_time - session.start_time).total_seconds())
        
        # Get word count from transcripts
        total_word_count = db.query(func.sum(func.array_length(func.string_to_array(SessionTranscript.cleaned_text, ' '), 1)))\
            .filter(SessionTranscript.session_id == request.session_id).scalar() or 0
        
        # Get vocabulary usage count
        vocab_usage_count = db.query(func.count(VocabularyUsage.id))\
            .filter(VocabularyUsage.session_id == request.session_id).scalar() or 0
        
        # Update session
        session.end_time = end_time
        session.duration_seconds = duration_seconds
        session.word_count = int(total_word_count)
        session.vocabulary_used_count = vocab_usage_count
        session.status = "completed"
        session.analysis_data = request.analysis_data
        
        # Calculate overall score (basic implementation)
        if duration_seconds > 0 and total_word_count > 0:
            # Simple scoring based on words per minute and vocabulary usage
            words_per_minute = (total_word_count / duration_seconds) * 60
            vocab_score = min(100, (vocab_usage_count / max(1, total_word_count)) * 1000)
            fluency_score = min(100, words_per_minute * 10)  # Rough estimate
            session.overall_score = (vocab_score + fluency_score) / 2
        
        db.commit()
        
        # Prepare analysis summary
        analysis_summary = {
            "duration_minutes": round(duration_seconds / 60, 1),
            "words_per_minute": round((total_word_count / max(1, duration_seconds)) * 60, 1),
            "vocabulary_words_used": vocab_usage_count,
            "total_words": int(total_word_count),
            "session_type": session.session_type,
            "topic": session.topic
        }
        
        if request.analysis_data:
            analysis_summary.update(request.analysis_data)
        
        return SessionEndResponse(
            session_id=session.id,
            duration_seconds=duration_seconds,
            word_count=int(total_word_count),
            vocabulary_used_count=vocab_usage_count,
            overall_score=session.overall_score,
            analysis_summary=analysis_summary,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@router.post("/transcript", response_model=dict)
async def save_transcript(request: TranscriptSaveRequest, db: Session = Depends(get_db)):
    """
    Save transcript data for a session
    """
    try:
        # Verify session exists and is active
        session = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create transcript record
        transcript = SessionTranscript(
            session_id=request.session_id,
            original_text=request.original_text,
            cleaned_text=request.cleaned_text,
            confidence_score=request.confidence_score,
            corrections_made=request.corrections_made,
            detected_vocabulary_level=request.detected_vocabulary_level,
            word_complexity_score=None,  # Can be calculated later
            grammar_score=None  # Can be calculated later
        )
        
        db.add(transcript)
        db.commit()
        db.refresh(transcript)
        
        return {
            "transcript_id": transcript.id,
            "session_id": request.session_id,
            "word_count": len(request.cleaned_text.split()) if request.cleaned_text else 0,
            "success": True,
            "message": "Transcript saved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save transcript: {str(e)}")

@router.get("/{session_id}", response_model=dict)
async def get_session(session_id: int, db: Session = Depends(get_db)):
    """
    Get session details and analysis
    """
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get transcripts
        transcripts = db.query(SessionTranscript)\
            .filter(SessionTranscript.session_id == session_id)\
            .order_by(SessionTranscript.created_at)\
            .all()
        
        # Get vocabulary usage
        vocabulary_usage = db.query(VocabularyUsage)\
            .filter(VocabularyUsage.session_id == session_id)\
            .all()
        
        return {
            "session": {
                "id": session.id,
                "user_id": session.user_id,
                "session_type": session.session_type,
                "topic": session.topic,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration_seconds": session.duration_seconds,
                "word_count": session.word_count,
                "vocabulary_used_count": session.vocabulary_used_count,
                "fluency_score": session.fluency_score,
                "overall_score": session.overall_score,
                "analysis_data": session.analysis_data
            },
            "transcripts": [
                {
                    "id": t.id,
                    "original_text": t.original_text,
                    "cleaned_text": t.cleaned_text,
                    "confidence_score": t.confidence_score,
                    "corrections_made": t.corrections_made,
                    "detected_vocabulary_level": t.detected_vocabulary_level,
                    "created_at": t.created_at
                } for t in transcripts
            ],
            "vocabulary_usage": [
                {
                    "word": v.word,
                    "used_correctly": v.used_correctly,
                    "context_sentence": v.context_sentence,
                    "usage_score": v.usage_score,
                    "feedback": v.feedback
                } for v in vocabulary_usage
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.get("/user/{user_id}/history", response_model=dict)
async def get_user_session_history(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """
    Get user's session history
    """
    try:
        sessions = db.query(SessionModel)\
            .filter(SessionModel.user_id == user_id)\
            .order_by(SessionModel.start_time.desc())\
            .limit(limit)\
            .all()
        
        return {
            "user_id": user_id,
            "sessions": [
                {
                    "id": s.id,
                    "session_type": s.session_type,
                    "topic": s.topic,
                    "status": s.status,
                    "start_time": s.start_time,
                    "duration_seconds": s.duration_seconds,
                    "word_count": s.word_count,
                    "overall_score": s.overall_score
                } for s in sessions
            ],
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")