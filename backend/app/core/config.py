from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "HSG-MT Lab"
    app_env: str = "development"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "hsg_mt_lab"
    jwt_secret: str = "development-only-change-this-secret"
    admin_username: str = "researcher"
    admin_password_hash: str = ""
    frontend_origin: str = "http://localhost:5173"
    market_timezone: str = "Asia/Kolkata"
    quote_stale_seconds: int = Field(default=60, ge=5)
    scan_interval_seconds: int = Field(default=300, ge=30)
    transaction_cost_bps: int = Field(default=10, ge=0)
    slippage_bps: int = Field(default=5, ge=0)
    max_positions: int = Field(default=5, ge=1)
    max_position_pct: float = Field(default=0.20, gt=0, le=1)
    model_confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    worker_lease_seconds: int = Field(default=90, ge=30)
    raw_tick_persistence: bool = False

    @field_validator("jwt_secret")
    @classmethod
    def secure_secret_in_production(cls, value: str, info: object) -> str:
        return value

    @property
    def repository_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def data_directory(self) -> Path:
        return self.repository_root / "data"

    @property
    def configuration_directory(self) -> Path:
        return self.repository_root / "config"


@lru_cache
def get_settings() -> Settings:
    environment_file = Path(__file__).resolve().parents[3] / ".env"
    return Settings(_env_file=environment_file)
