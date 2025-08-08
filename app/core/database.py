from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Determine database type and configure engine accordingly
def create_database_engine():
    """Create database engine with appropriate configuration based on database type"""
    database_url = settings.database_url
    # Normalize to psycopg3 driver if using Postgres without explicit driver
    if database_url.startswith('postgresql://') and '+psycopg' not in database_url:
        database_url = database_url.replace('postgresql://', 'postgresql+psycopg://', 1)
    
    if database_url.startswith('sqlite'):
        # SQLite configuration
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},  # Required for SQLite
            echo=settings.debug,
        )
    else:
        # PostgreSQL configuration
        engine = create_engine(
            database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_pre_ping=True,  # Verify connections before use
            echo=settings.debug,  # Log SQL queries in debug mode
        )
    
    return engine

# Create engine
engine = create_database_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all tables in the database.

    Note: This is a best-effort bootstrap for dev SQLite. For Postgres or when
    using Alembic, you should run migrations to add new columns (e.g., anon_uuid).
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully (metadata-based)")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def check_connection():
    """Check database connection health"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def _sqlite_table_has_column(connection, table_name: str, column_name: str) -> bool:
    try:
        result = connection.execute(text(f"PRAGMA table_info({table_name})"))
        for row in result:
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            if str(row[1]) == column_name:
                return True
        return False
    except Exception:
        return False


def ensure_dev_sqlite_columns():
    """Best-effort: ensure new columns exist in dev SQLite.

    This is only for local dev convenience. For production, use Alembic.
    """
    try:
        if not settings.database_url.startswith('sqlite'):
            return
        with engine.begin() as connection:
            # users.anon_uuid
            if not _sqlite_table_has_column(connection, 'users', 'anon_uuid'):
                connection.execute(text("ALTER TABLE users ADD COLUMN anon_uuid VARCHAR"))
            # sessions.personality
            if not _sqlite_table_has_column(connection, 'sessions', 'personality'):
                connection.execute(text("ALTER TABLE sessions ADD COLUMN personality VARCHAR"))
            # sessions.last_message_at
            if not _sqlite_table_has_column(connection, 'sessions', 'last_message_at'):
                connection.execute(text("ALTER TABLE sessions ADD COLUMN last_message_at TIMESTAMP"))
            logger.info("Ensured dev SQLite columns exist (anon_uuid, personality, last_message_at)")
    except Exception as e:
        logger.warning(f"Could not ensure dev SQLite columns: {e}")