""""Turns raw captions into clean, non overalapping transcript lines """

from __future__ import annotations
import html
import re
from rag.models import TranscriptLine

_TAG_RE = re.compile(r"<[^>]+>")
_NON_SPEECH_RE = re.compile(r"\[[^\]]*\]|\((?:music|applause|laughter|laughs|inaudible|silence)[^)]*\)", re.I)
_MUSIC_RE = re.compile(r"[♪♫]+")
_SPEAKER_RE = re.compile(r"(^|\s)>>+\s*")
_WS_RE = re.compile(r"\s+")

MIN_WORDS = 3
MIN_DURATION_S = 1.0
MIN_OVERLAP_WORDS = 3

def normalize_text(text: str)-> str:
    t = html.unescape(text)
    t = _TAG_RE.sub(" ", t)
    t = _NON_SPEECH_RE.sub(" ", t)
    t = _MUSIC_RE.sub(" ", t)
    t = _SPEAKER_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()

def _strip_rolling_overlap(prev:str, cur: str)-> str:

    if cur == prev:
        return ""
    pw, cw = prev.split(), cur.split()
    for n in range(min(len(pw), len(cw)), MIN_OVERLAP_WORDS - 1, -1):
        if [w.lower() for w in pw[-n:]] == [w.lower() for w in cw[:n]]:
            return " ".join(cw[n:])
    return cur

def _is_fragment(line: TranscriptLine)-> bool:
    return len(line.text.split()) < MIN_WORDS or (line.end - line.start) < MIN_DURATION_S

def clean(lines: list[TranscriptLine]) ->list[TranscriptLine]:
    out: list[TranscriptLine] =[]

    # normalizing text, drop empties, strip rolling repeats
    for line in sorted(lines, key=lambda x: x.start):
        text =normalize_text(line.text)
        if out and text:
            text =_strip_rolling_overlap(out[-1].text, text)
            if not text:
                out[-1] = out[-1].model_copy(update={"end": max(out[-1].end, line.end)})
                continue
        if text:
            out.append(TranscriptLine(start=line.start, end=line.end, text=text))

    # clamp overlapping ends
    for i in range(len(out) - 1):
        nxt = out[i + 1].start
        if out[i].end > nxt >= out[i].start:
            out[i] = out[i].model_copy(update={"end": nxt})

    # merge fragments into prev line
    merged: list[TranscriptLine] = []
    pending:TranscriptLine |None =None
    for line in out:
        if pending is not None:
            line = TranscriptLine(start=pending.start, end=line.end, text=f"{pending.text} {line.text}")
            pending = None
        if _is_fragment(line):
            if merged:
                prev = merged[-1]
                merged[-1] = TranscriptLine(start=prev.start, end=max(prev.end, line.end), text=f"{prev.text} {line.text}")
            else:
                pending =line
            continue
        merged.append(line)

    if pending is not None:
        merged.append(pending)
    return merged
            