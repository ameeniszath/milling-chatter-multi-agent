"""Agent 1 - data acquisition.

Watches a folder (in the course demo this was a Google Drive folder polled every minute by n8n)
for new test-cut recordings, reads each one once and hands it on as a Recording.
"""
from __future__ import annotations

from pathlib import Path

from .data import Recording, load_mat


class AcquisitionAgent:
    name = "agent1_acquisition"

    def __init__(self, watch_dir, pattern: str = "*.mat"):
        self.watch_dir = Path(watch_dir)
        self.pattern = pattern
        self.seen: set[str] = set()

    def poll(self) -> list[Recording]:
        """Return recordings that appeared since the last poll (oldest name first)."""
        new = []
        for p in sorted(self.watch_dir.glob(self.pattern), key=lambda q: _natural_key(q.name)):
            if p.name in self.seen:
                continue
            self.seen.add(p.name)
            new.append(load_mat(p))
        return new


def _natural_key(name: str):
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]
