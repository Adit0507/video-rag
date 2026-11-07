"""
    command line entry point
 """

from __future__ import annotations
import argparse 
import asyncio
import sys

from rag.core.config import get_settings
from rag.core.errors import VideoRAGError
from rag.core.logging import setup_logging
from rag.models import VideoDocument
from rag.processor.fetcher import parse_video_id
from rag.store import artifacts as A
from rag.wiring import build_artifacts,build_pipeline

def _ts(seconds:float)-> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

async def _cmd_process(url: str)-> int:
    settings =get_settings()
    pipeline = build_pipeline(settings)

    async def progress(stage: str, p:float)-> None:
        print(f"  [{p:4.0%}] {stage}", file=sys.stderr)

    doc= await pipeline.run(url, on_progress=progress)
    path = build_artifacts(settings).path(doc.video_id, A.DOCUMENT)
    print(f"\n{doc.title}  ({doc.channel}, {_ts(doc.duration_s)})")
    print(f"transcript: {doc.transcript_source}, language {doc.language}")
    print(f"segments:   {len(doc.segments)}")
    print(f"saved:      {path}")
    return 0

def _cmd_show(video_id: str,limit: int)-> int:
    store =build_artifacts(get_settings)
    vid =parse_video_id(video_id)

    if not store.exists(vid, A.DOCUMENT):
        print(f"No document for {vid}. Run `rag process` first.", file=sys.stderr)
        return 1

    doc= store.load(vid, A.DOCUMENT,VideoDocument)    
    print(f"{doc.title}  [{doc.transcript_source}, {len(doc.segments)} segments]\n")
    for seg in doc.segments[:limit]:
        text = seg.text if len(seg.text) <= 160 else seg.text[:157] + "..."
        print(f"[S{seg.index}] {_ts(seg.start)}-{_ts(seg.end)}  {text}")
    if len(doc.segments) > limit:
        print(f"... {len(doc.segments) - limit} more")
    return 0 

def main(argv: list[str]|None=None) ->int:
    parser= argparse.ArgumentParser(prog="rag")
    sub =parser.add_subparsers(dest="cmd",required=True)
    p =sub.add_parser("process", help="process a YT URL into a VideoDocument")
    p.add_argument("url")
    s = sub.add_parser("show", help="print a processed video's segments")
    s.add_argument("video_id", help="video id or URL")
    s.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    setup_logging(get_settings().log_level)
    try:
        if args.cmd == "process":
            return asyncio.run(_cmd_process(args.url))
        if args.cmd == "show":
            return _cmd_show(args.video_id, args.limit)
    except VideoRAGError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1

if __name__== "__main__":
    sys.exit(main())