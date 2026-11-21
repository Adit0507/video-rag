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
from rag.wiring import build_artifacts,build_pipeline, build_indexer

def _ts(seconds:float)-> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _short(text: str,n:int= 160)-> str:
    return text if len(text)<= n else text[:n-3]+ "..."

def _load_doc(video_id: str)-> VideoDocument |None:
    store =build_artifacts(get_settings())
    vid= parse_video_id(video_id)
    if not store.exists(vid, A.DOCUMENT):
        print(f"No document for {vid}. Run `rag process` first.", file=sys.stderr)
        return None
    return store.load(vid, A.DOCUMENT,VideoDocument)

async def _cmd_process(url: str, index: bool)-> int:
    settings =get_settings()
    pipeline = build_pipeline(settings)

    async def progress(stage: str, p:float)-> None:
        print(f"  [{p:4.0%}] {stage}", file=sys.stderr)

    doc= await pipeline.run(url, on_progress=progress)
    if index:
        print("  [ 85%] indexing", file=sys.stderr)
        await asyncio.to_thread(build_indexer(settings).index, doc)
    path = build_artifacts(settings).path(doc.video_id, A.DOCUMENT)
    print(f"\n{doc.title}  ({doc.channel}, {_ts(doc.duration_s)})")
    print(f"transcript: {doc.transcript_source}, language {doc.language}")
    print(f"segments:   {len(doc.segments)}{'  (indexed)' if index else ''}")
    print(f"saved:      {path}")
    return 0

def _cmd_show(video_id: str,limit: int):
    doc= _load_doc(video_id)
    if doc is None:
        return  1
    print(f"{doc.title}  [{doc.transcript_source}, {len(doc.segments)} segments]")
    for seg in doc.segments[:limit]:
        print(f"[S{seg.index}] {_ts(seg.start)}-{_ts(seg.end)}  {_short(seg.text)}")

    if len(doc.segments)> limit:
        print(f"... {len(doc.segments)- limit} more")
    return 0

def _cmd_index(video_id: str)-> int:
    doc= _load_doc(video_id)
    if doc is None:
        return 1

    build_indexer(get_settings()).index(doc)
    print(f"indexed {doc.video_id}: {len(doc.segments)} segments")
    return 0

def _cmd_reindex()-> int:
    setings= get_settings()
    store= build_artifacts(setings)
    ids= store.list_documents()
    if not ids:
        print("No saved docs to reindex")
        return 0

    indexer =build_indexer(setings)
    for vid in ids:
        doc= store.load(vid, A.DOCUMENT, VideoDocument)
        indexer.index(doc)
        print(f"  {vid}: {len(doc.segments)} segments")

    print(f"reindexed {len(ids)} video(s) with {setings.embedding_model}")
    return 0

def _cmd_search(video_id: str| None, query: str, k:int)-> int:
    vid =parse_video_id(video_id) if video_id else None
    hits= build_indexer(get_settings()).search(query,k=k, video_id=vid)
    if not hits:
        print("No results. Is the video indexed? Try `videorag index <video_id>`.")
        return 0

    for h in hits: 
        where=""if vid else f"{h.video_id} "
        print(f"{h.score:.3f}  {where}[S{h.index}] {_ts(h.start)}-{_ts(h.end)}  {_short(h.text, 140)}")
    return 0

def main(argv: list[str]|None=None) ->int:
    parser= argparse.ArgumentParser(prog="rag")
    sub =parser.add_subparsers(dest="cmd",required=True)
    p = sub.add_parser("process", help="process a YouTube URL and index it")
    p.add_argument("url")
    p.add_argument("--no-index", action="store_true", help="stop after saving the document")

    s = sub.add_parser("show", help="print a processed video's segments")
    s.add_argument("video_id", help="video id or URL")
    s.add_argument("--limit", type=int, default=20)

    i = sub.add_parser("index", help="(re)index one saved document")
    i.add_argument("video_id")

    sub.add_parser("reindex", help="re-embed every saved document with the current model") 
    q = sub.add_parser("search", help="search a video's segments")
    q.add_argument("video_id", nargs="?", help="omit with --all")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--all", action="store_true", help="search across the whole library")

    args = parser.parse_args(argv)
    setup_logging(get_settings().log_level)
    try:
        if args.cmd == "process":
            return asyncio.run(_cmd_process(args.url, index=not args.no_index))
        if args.cmd == "show":
            return _cmd_show(args.video_id, args.limit)
        if args.cmd == "index":
            return _cmd_index(args.video_id)
        if args.cmd == "reindex":
            return _cmd_reindex()
        if args.cmd == "search":
            if args.all == bool(args.video_id):
                parser.error("give a video_id, or --all to search the whole library (not both)")
            return _cmd_search(None if args.all else args.video_id, args.query, args.k)
    except VideoRAGError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1

if __name__== "__main__":
    sys.exit(main())