from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Pattern Extractor API"
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    PATTERN_DIR: str = "assets/patterns"

settings = Settings()