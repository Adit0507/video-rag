""" Per video artifacts on disk 
writes are atomic, so a crash mid write never leaves a half written artifct
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel  

M= TypeVar("M",bound=BaseModel)
METADATA="metadata.json"
SEGMENTS="segments.json"
SUMMARY="summary.json"
DOCUMENT="document.json"

class ArtifactStore(BaseModel):
    def __init__(self, data_dir:Path)->None:
        self.root=Path(data_dir)/ "videos"

    def video_dir(self, video_id: str)-> Path:
        return self.root/ video_id

    def path(self, video_id: str, name: str)-> Path:
        return self.video_dir(video_id)/ name

    def exists(self,video_id: str, name: str)-> bool:
        return self.path(video_id, name).is_file()

    def save(self, video_id: str, name: str,obj: BaseModel)-> Path:
        target= self.path(video_id, name)
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd,"w", encoding="utf-8") as f:
                f.write(obj.model_dump_json(indent=2))
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        return target

    def load(self, video_id:str, name:str, model: type[M])->M:
        return model.model_validate_json(self.path(video_id, name).read_text(encoding="utf-8"))
    def load_if_exists(self, video_id:str, name:str, model: type[M])->M | None:
        return self.load(self, video_id,name, model) if self.exists(video_id,name) else None

    def delete(self, video_id:str, name: str)-> None:
        self.path(video_id, name).unlink(missing_ok=True)
    def delete_video(self,video_id: str)-> None:
        shutil.rmtree(self.video_dir(video_id),ignore_errors=True)

    def list_documents(self)-> list[str]:
        if not self.root.is_dir():
            return []

        return sorted(p.parent.name for p in self.root.glob(f"*/{DOCUMENT}"))