""" Application settings """

from functools import lru_cache
from pathlib import Path
from pydantic import Field,model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config =SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",extra="ignore")

    data_dir: Path =Path("data")
    preferred_languages: list[str] = Field(default_factory=lambda: ["en"])
    max_video_duration_s: int = 4 * 60 * 60
    youtube_proxy_url: str | None = None
    youtube_cookies_file: Path | None = None

    # segmentation
    segment_window_s: float= 45.0
    segment_overlap_s: float= 10.0
    segment_max_chars: int = 2000
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_segment_window(self)-> "Settings":
        if not 0<= self.segment_overlap_s <self.segment_window_s:   
            raise ValueError("SEGMENT_OVERLAP_S must be >= 0 and smaller than SEGMENT_WINDOW_S")
        return self

@lru_cache
def get_settings()-> Settings:
    return Settings()