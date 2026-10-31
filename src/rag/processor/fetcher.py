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

def _trim_ytdlp_message(message: str)-> str:
    msg= message.split("; please report this issue")[0]
    return msg.removeprefix("ERROR: ").strip()

class _SilentLogger:
    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def info(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp warning: %s", msg)

    def error(self, msg: str) -> None:
        log.debug("yt-dlp error: %s", msg)

def classify_ytdlp_error(message: str)-> Exception:
    """"Mapping a yt dlp error message to a permanent error """
    message= _trim_ytdlp_message(message)
    m= message.lower()
    if "not a bot" in m or "http error 429" in m or "too many requests" in m:
        return YouTubeBlockedError(message.strip()[:200])
    if "private video" in m:
        return VideoUnavailableError("This video is private.")
    if "confirm your age" in m or "age-restricted" in m or "age restricted" in m:
        return VideoUnavailableError("This video is age-restricted.")
    if "members-only" in m or "join this channel" in m:
        return VideoUnavailableError("This video is members-only.")
    if (
        "video unavailable" in m
        or "has been removed" in m
        or "is not available" in m
        or "does not exist" in m
        or "terminated" in m
    ):
        return VideoUnavailableError("This video is unavailable.")
    return TransientError(f"Could not fetch video metadata: {message.strip()[:300]}")

def _ydl_base_opts(proxy_url: str |None, cookies_file:Path| None)-> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "logger": _SilentLogger(),
    }
    if proxy_url:
        opts["proxy"]= proxy_url
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)
    return opts

class YtDlpMetadataFetcher:
    def __init__(self,proxy_url: str|None =None, cookies_file: Path|None=None)-> None:
        self._opts =_ydl_base_opts(proxy_url,cookies_file)

    def fetch(self, video_id: str)-> VideoMetadata:
        import yt_dlp

        try:
            with yt_dlp.YoutubeDL(self._opts) as ydl:
                info = ydl.extract_info(canonical_url(video_id),download=False)
        except yt_dlp.utils.DownloadError as e:
            raise classify_ytdlp_error(str(e)) from e
        except OSError as e:
            raise TransientError(f"Network error while fetching metadata: {e}") from e
        return metadata_from_info(video_id, info or {})

def metadata_from_info(video_id: str,info: dict) ->VideoMetadata:
    live_status = info.get("live_status")
    return VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        title=info.get("title") or video_id,
        channel=info.get("channel") or info.get("uploader") or "",
        duration_s=float(info.get("duration") or 0),
        language=info.get("language"),
        is_live=bool(info.get("is_live")) or live_status in ("is_live", "is_upcoming"),
        age_limit=int(info.get("age_limit") or 0),
        availability=info.get("availability"),
    )