from __future__ import annotations

import logging, re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs,urlparse
from rag.models import VideoMetadata, RawTranscript,TranscriptLine
from rag.core.errors import (
    InvalidURLError,
    TransientError,
    VideoUnavailableError,
    YouTubeBlockedError,
)

log= logging.getLogger(__name__)
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_PATH_PREFIXES = ("shorts", "embed", "live", "v", "e")

def parse_video_id(url_or_id: str)->str:
    """Return the 11-character video id from any common YouTube URL form, or a bare id."""
    s= (url_or_id or "").strip()
    if _ID_RE.match(s):
        return s    
    if "://" not in s:
        s = "https://" + s
    try:
        u = urlparse(s)
    except ValueError as e:
        raise InvalidURLError(f"Not a valid URL: {url_or_id!r}") from e

    host= (u.hostname or "").lower()
    candidate:str| None =None
    if host in ("youtu.be", "www.youtu.be"):
        candidate = u.path.lstrip("/").split("/")[0]
    elif host in _YT_HOSTS:
        if u.path in ("/watch", "/watch/"):
            candidate = (parse_qs(u.query).get("v") or [None])[0]
        else:
            parts = [p for p in u.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
                candidate = parts[1]
    else:
        raise InvalidURLError(f"Not a YouTube URL: {url_or_id!r}")

    if candidate and _ID_RE.match(candidate):
        return candidate
    raise InvalidURLError(f"Could not find a video id in {url_or_id!r}")

def canonical_url(video_id: str)-> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# metadata
class MetadataFetcher(Protocol):
    def fetch(self, video_id: str)-> VideoMetadata: ...

def validate_metadata(meta: VideoMetadata,max_duration_s: int)-> None:
    """Reject videos the pipeline cannot or should not process."""
    if meta.is_live:
        raise VideoUnavailableError("Live streams and upcoming premieres are not supported.")
    if meta.age_limit >= 18:
        raise VideoUnavailableError("This video is age-restricted.")
    if meta.availability in ("private", "needs_auth", "subscriber_only", "premium_only"):
        raise VideoUnavailableError(f"This video is not publicly available ({meta.availability}).")
    if meta.duration_s <= 0:
        raise VideoUnavailableError("This video has no duration (it may be a live stream).")
    if meta.duration_s > max_duration_s:
        raise VideoUnavailableError(
            f"Video is {meta.duration_s / 3600:.1f} h long; the limit is "
            f"{max_duration_s / 3600:.1f} h (MAX_VIDEO_DURATION_S)."
        )

