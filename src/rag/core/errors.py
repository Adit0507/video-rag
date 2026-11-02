class VideoRAGError(Exception):
    """Base class. `str(err)` is safe to show to the user."""


class PermanentError(VideoRAGError):
    """Retrying will not help."""


class TransientError(VideoRAGError):
    """Worth retrying later."""


class InvalidURLError(PermanentError):
    pass


class VideoUnavailableError(PermanentError):
    """Private, removed, members-only, age-restricted, live, or too long."""


class YouTubeBlockedError(PermanentError):
    """YouTube refused the request from this IP. Retrying from the same IP rarely helps."""

    def __init__(self, detail: str = "") -> None:
        msg = (
            "YouTube blocked the request from this machine's IP address. "
            "This is common on cloud servers; set YOUTUBE_PROXY_URL (and for metadata, "
            "YOUTUBE_COOKIES_FILE) or run from a home connection."
        )
        super().__init__(f"{msg} ({detail})" if detail else msg)


class NoTranscriptError(PermanentError):
    pass
