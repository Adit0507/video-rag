""" Groups clean transcript lines into retrievable segments """

from __future__ import annotations
from typing import Protocol
from rag.models import Segment,TranscriptLine

class Segmenter(Protocol):
    @property
    def name(self)-> str:
        """ identifies algorithm and its parameters """
        ...

    def segment(self, video_ids: str,lines: list[TranscriptLine])-> list[Segment]: ...

class FixedWindowSegmenter:
    def __init__(self,window_s:float=45.0, overlap_s: float= 10.0,max_chars:int= 2000)-> None:
        if window_s<= 0:
            raise ValueError("window_s must be positive")
        if not 0 <=overlap_s < window_s:
            raise ValueError("overlap_s must be >= 0 and smaller than window_s")
        if max_chars <=0:
            raise ValueError("max_chars must be positive")
        self.window_s = window_s
        self.overlap_s = overlap_s
        self.max_chars = max_chars

    @property
    def name(self)-> str:
        return f"fixed-window:{self.window_s:g}s/{self.overlap_s:g}s/{self.max_chars}c"

    def segment(self, video_id:str, lines: list[TranscriptLine])-> list[Segment]:
        segments: list[Segment]= []
        n =len(lines)
        i= 0
        while i<n:
            window_start =lines[i].start
            window_end =window_start+ self.window_s
            # growing window
            last = i
            chars = len(lines[i].text)
            while last+1 <n and lines[last +1].start <window_end:
                nxt =len(lines[last + 1].text) + 1
                if chars +nxt >self.max_chars:
                    break
                chars +=nxt
                last+=1

            segments.append(Segment(
                    id=f"{video_id}:{len(segments)}",
                    index=len(segments),
                    start=lines[i].start,
                    end=max(l.end for l in lines[i : last + 1]),
                    text=" ".join(l.text for l in lines[i : last + 1]),
            ))
            if last== n -1:
                break

            step_at =window_end -self.overlap_s
            k= i+1
            while k<=last and lines[k].start <step_at:
                k +=1
            i = min(max(k, i + 1), last + 1)
        return segments
