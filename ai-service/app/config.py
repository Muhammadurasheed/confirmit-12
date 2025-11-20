from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Server
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    # Google Gemini
    GEMINI_API_KEY: str

    # Firebase
    FIREBASE_PROJECT_ID: str
    FIREBASE_PRIVATE_KEY: Optional[str] = None
    FIREBASE_CLIENT_EMAIL: Optional[str] = None

    # Cloudinary
    CLOUDINARY_URL: str

    # Model Configuration
    DEFAULT_MODEL: str = "gemini-1.5-flash"
    MAX_CONCURRENT_AGENTS: int = 5
    AGENT_TIMEOUT_SECONDS: int = 30

    # Forensics
    ELA_QUALITY: int = 95
    FORENSIC_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
