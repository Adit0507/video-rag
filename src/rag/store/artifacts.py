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


