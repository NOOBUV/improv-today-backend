import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Gemini Configuration (Clara's conversation model, via OpenAI-compatible endpoint)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Supabase Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "")
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Security
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    
    # Auth0 Configuration
    auth0_domain: str = os.getenv("AUTH0_DOMAIN", "")
    auth0_audience: str = os.getenv("AUTH0_AUDIENCE", "")
    auth0_issuer: str = os.getenv("AUTH0_ISSUER", "")
    auth0_client_secret: str = os.getenv("AUTH0_CLIENT_SECRET", "")
    jwt_algorithms: str = os.getenv("JWT_ALGORITHMS", "RS256")
    auth0_mgmt_client_id: str = os.getenv("AUTH0_MGMT_CLIENT_ID", "")
    auth0_mgmt_client_secret: str = os.getenv("AUTH0_MGMT_CLIENT_SECRET", "")
    
    # Application Settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # Database Connection Pool Settings
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    db_pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    
    # Stripe Configuration
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_publishable_key: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # Superadmin Configuration
    superadmin_emails: str = os.getenv("SUPERADMIN_EMAILS", "")

    @property
    def superadmin_emails_list(self) -> list[str]:
        """Parse comma-separated superadmin emails into a list"""
        if not self.superadmin_emails:
            return []
        return [email.strip() for email in self.superadmin_emails.split(",")]

    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# Fail fast at import: a production boot with either of these missing is a broken
# deploy, not a degraded one (the default JWT secret forges tokens; Clara without
# an API key answers with canned fallback lines).
if settings.is_production:
    # Tripwire for the unauthenticated dev bypass in app/auth/dependencies.py: it
    # gates on `is_development`, so if that property is ever loosened (e.g. to
    # "not production") a production boot dies here instead of quietly serving
    # every request as dev@localhost.
    if settings.is_development:
        raise RuntimeError(
            "Dev auth bypass would be active in production: is_development must "
            'mean environment == "development" and nothing else'
        )
    if settings.jwt_secret == "your-secret-key-change-in-production":
        raise RuntimeError("JWT_SECRET must be set in production")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in production")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY must be set in production")