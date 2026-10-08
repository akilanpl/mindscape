"""Explicit partitioned SQLite storage; evaluation retrieval disabled by default."""
import json
from pathlib import Path
import sqlite3
import uuid


class EpisodicMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, partition TEXT, payload TEXT)")

    def store_episode(self, problem, trajectory, outcome, success, metadata, partition="train"):
        if partition not in ("train", "validation", "test", "ood_test", "demo"):
            raise ValueError("Unknown memory partition")
        key = uuid.uuid4().hex
        payload = dict(problem=problem, trajectory=trajectory, outcome=outcome,
                       success=success, metadata=metadata, partition=partition)
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO episodes VALUES (?, ?, ?)", (key, partition, json.dumps(payload)))
        return key

    def get(self, key, partition="train"):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM episodes WHERE id=? AND partition=?", (key, partition)).fetchone()
        return json.loads(row[0]) if row else None

    def retrieve(self, partition="train", limit=10):
        if type(limit) is not int or limit < 0:
            raise ValueError("Invalid retrieval limit")
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id,payload FROM episodes WHERE partition=? ORDER BY rowid DESC LIMIT ?", (partition, limit)).fetchall()
        return [(key, json.loads(value)) for key, value in rows]

    def clear(self, partition="train"):
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM episodes WHERE partition=?", (partition,))
