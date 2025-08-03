import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, conversation, conversation_v2, conversation_test, vocabulary, feedback, sessions, websocket
from app.core.config import settings
from app.services.conversation_state_manager import conversation_state_manager

app = FastAPI(title="Improv Today API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(conversation.router, prefix="/api/conversation", tags=["conversation"])

# Enhanced conversation API with WebSocket integration
app.include_router(conversation_v2.router, prefix="/api/v2", tags=["conversation-v2", "realtime"])

app.include_router(vocabulary.router, prefix="/api/vocabulary", tags=["vocabulary"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])

# WebSocket and real-time conversation endpoints
app.include_router(websocket.router, prefix="/api", tags=["websocket", "realtime"])

# Test endpoints for WebSocket system validation
app.include_router(conversation_test.router, prefix="/api", tags=["testing", "websocket-test"])

@app.get("/")
async def root():
    return {"message": "Improv Today API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger = logging.getLogger("app.startup")
    logger.info("Initializing ImprovToday backend services...")
    
    # Initialize conversation state manager
    logger.info("Conversation state manager initialized")
    
    # Log available endpoints
    logger.info("WebSocket endpoints available:")
    logger.info("  - /api/ws/conversations/{conversation_id} - Main WebSocket endpoint")
    logger.info("  - /api/v2/conversations - Enhanced conversation API")
    logger.info("  - /api/test/conversations - Test endpoints for validation")
    
    logger.info("ImprovToday backend startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on application shutdown"""
    logger = logging.getLogger("app.shutdown")
    logger.info("Shutting down ImprovToday backend services...")
    
    # Cleanup would go here if needed
    logger.info("ImprovToday backend shutdown complete")