"""Conservative shared rate reservations and explicitly attributed response cache."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import time

from .dataset import fingerprint
from .storage import _publish, IntegrityError, ImmutableConflictError


class RateLimiter:
    def __init__(self, path, policy, *, clock=time.time, sleep=time.sleep):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rpm, self.tpm = policy["requests_per_minute"], policy["tokens_per_minute"]
        self.clock, self.sleep = clock, sleep
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS reservations (at REAL, tokens INTEGER, requests INTEGER)")

    def reserve(self, tokens=0, requests=1):
        if (type(tokens) is not int or type(requests) is not int or tokens < 0 or requests < 0
                or tokens > self.tpm or requests > self.rpm):
            raise ValueError("one request exceeds configured rate budget")
        while True:
            now = self.clock()
            with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM reservations WHERE at<=?", (now-60,))
                rows = db.execute("SELECT at,tokens,requests FROM reservations ORDER BY at").fetchall()
                if sum(r[1] for r in rows)+tokens <= self.tpm and sum(r[2] for r in rows)+requests <= self.rpm:
                    db.execute("INSERT INTO reservations VALUES (?,?,?)", (now, tokens, requests))
                    return
                wait = max(0.001, rows[0][0]+60-now) if rows else 60
            self.sleep(min(wait, 60))


class ResponseCache:
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def key(spec, messages_sha256):
        return fingerprint({"model": spec, "messages_sha256": messages_sha256})

    def get(self, key):
        path = self.root / (key+".json")
        if not path.exists():
            return None
        if path.is_symlink():
            raise IntegrityError("cache symlink is not allowed")
        value = json.loads(path.read_bytes())
        envelope = dict(value)
        claimed = envelope.pop("envelope_sha256", None)
        if (fingerprint(envelope) != claimed or value.get("cache_key") != key
                or fingerprint(value["response"]) != value.get("response_sha256")):
            raise IntegrityError("cache content/binding hash changed")
        return value

    def put(self, key, response, origin):
        self.root.mkdir(parents=True, exist_ok=True)
        value = {"cache_key": key, "response": response, "response_sha256": fingerprint(response),
                 "origin": origin, "independent_observation": False}
        value["envelope_sha256"] = fingerprint(value)
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)+"\n").encode()
        path = self.root / (key+".json")
        if path.exists():
            self.get(key)  # Another completed run may have won publication.
            return False
        try:
            _publish(path, data)
        except (FileExistsError, ImmutableConflictError):
            self.get(key)
            return False
        return True
