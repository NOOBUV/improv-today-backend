import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./improv_today.db")
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key")
    
    class Config:
        env_file = ".env"

settings = Settings()