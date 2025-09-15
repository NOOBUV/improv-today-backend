import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from app.api import auth, conversation, vocabulary, feedback, sessions, ava, subscriptions
from app.api.simulation import admin as simulation_admin
from app.core.config import settings
from app.middleware.subscription_middleware import SubscriptionMiddleware

app = FastAPI(
    title="Improv Today API",
    version="1.0.0",
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# CORS middleware
# CORS origins from env (comma-separated), fallback to common localhost dev origins
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000")
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add subscription middleware for handling subscription-required responses
app.add_middleware(SubscriptionMiddleware)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(conversation.router, prefix="/api/conversation", tags=["conversation"])
app.include_router(ava.router, prefix="/api/ava", tags=["ava"])

app.include_router(vocabulary.router, prefix="/api/vocabulary", tags=["vocabulary"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])
app.include_router(subscriptions.router, prefix="/api", tags=["subscriptions"])

# Simulation engine admin routes
app.include_router(simulation_admin.router, prefix="/api/simulation", tags=["simulation"])


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
    
    # Import models to ensure they're registered with Base
    from app.models import conversation_v2, user, session, vocabulary, ava_state
    # Models imported for SQLAlchemy registration
    
    # Initialize database tables
    from app.core.database import create_tables, check_connection, ensure_dev_sqlite_columns
    if check_connection():
        # In production, rely on Alembic migrations
        if settings.is_development:
            create_tables()
            # Dev convenience: add columns for SQLite if missing
            ensure_dev_sqlite_columns()
        logger.info("Database tables initialized")
    else:
        logger.error("Database connection failed")
    
    # API endpoints ready
    logger.info("HTTP API endpoints available:")
    logger.info("  - /api/conversation - Main conversation endpoint")
    logger.info("  - /api/sessions - Session management")
    logger.info("  - /api/auth - Authentication")
    
    logger.info("ImprovToday backend startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on application shutdown"""
    logger = logging.getLogger("app.shutdown")
    logger.info("Shutting down ImprovToday backend services...")
    
    # Cleanup would go here if needed
    logger.info("ImprovToday backend shutdown complete")