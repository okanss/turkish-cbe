"""Durable call journals, conservative resumption and append-only raw records.

A local checkpoint cannot make a remote API exactly once: a process can stop
after the provider accepted a request but before its response was persisted.
Such attempts become ambiguous and require an explicit, audited decision before
another call. Completed raw responses can instead be normalized without calling
the provider again. POSIX ``fcntl`` locks support macOS and Linux.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - documented unsupported platform
    fcntl = None


STORAGE_VERSION = "call-journal-v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_TERMINAL = frozenset({"succeeded", "cached", "permanent_failed"})


class StorageError(RuntimeError):
    """A durable journal cannot safely perform the requested operation."""


class ImmutableConflictError(StorageError):
    """A write would change a previously recorded immutable value."""


class IntegrityError(StorageError):
    """A persisted file differs from its checkpointed byte hash."""


class RunLockedError(StorageError):
    """Another process already owns this run's lifetime lock."""


class AmbiguousCallError(StorageError):
    """An uncertain earlier provider call blocks automatic repetition."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _json_bytes(value: dict) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("journal records must be dictionaries")
    return (_canonical(value) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_dir(path: Path) -> None:
    """Persist each newly created directory entry through its existing parent."""
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _fsync_dir(directory.parent)


def _publish(path: Path, payload: bytes) -> dict:
    """Publish a complete file exclusively; an identical replay is harmless."""
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ImmutableConflictError(f"immutable record already exists: {path}") from None
        _fsync_dir(path.parent)
    finally:
        # Only this invocation's unpublished temporary file is removed. Raw
        # request/response/error files are never rewritten or deleted.
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": _sha(payload), "bytes": len(payload)}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
 task_id TEXT PRIMARY KEY, request_key TEXT NOT NULL UNIQUE, task_json TEXT NOT NULL,
 status TEXT NOT NULL, current_attempt_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
 attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
 status TEXT NOT NULL, request_sha256 TEXT NOT NULL,
 response_sha256 TEXT, error_sha256 TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_task ON attempts(task_id, created_at);
CREATE TABLE IF NOT EXISTS normalized_outbox (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
 attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
 record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resolutions (
 resolution_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
 attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id), decision TEXT NOT NULL,
 record_sha256 TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


class RunStore:
    """Own one run's lock and checkpoint for the duration of a ``with`` block.

``root`` is the project directory. Raw records live below
``results/raw/<run_id>/``; the mutable SQLite checkpoint lives in ``logs/``.
``finish`` commits a normalized outbox entry atomically with task completion,
then appends it to ``results/normalized/<run_id>.jsonl``. On resume an incomplete
last JSONL line is completed by appending its missing suffix, without changing
any existing byte. Unexpected or altered normalized bytes are rejected.
"""

    def __init__(self, root: Path, run_id: str, plan: dict):
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id requires 1-96 letters, digits, dots, underscores or hyphens")
        if not isinstance(plan, dict):
            raise ValueError("plan must be a dictionary")
        self.root = Path(root)
        self.run_id = run_id
        self.plan = json.loads(_canonical(plan))
        self.plan_sha256 = _sha(_canonical(self.plan).encode("utf-8"))
        self.raw_dir = self.root / "results" / "raw" / run_id
        self.attempts_dir = self.raw_dir / "attempts"
        self.manifest_path = self.raw_dir / "manifest.json"
        self.checkpoint_path = self.root / "logs" / (run_id + ".sqlite")
        self.normalized_path = self.root / "results" / "normalized" / (run_id + ".jsonl")
        self.lock_path = self.root / "logs" / (run_id + ".lock")
        self._lock = None
        self._db = None
        self._last_normalized_seq = 0
        self._normalized_bytes = 0

    def __enter__(self):
        if self._db is not None or self._lock is not None:
            raise StorageError("RunStore is already open")
        if fcntl is None:
            raise StorageError("RunStore requires POSIX fcntl locks (macOS or Linux)")
        _ensure_dir(self.lock_path.parent)
        self._lock = self.lock_path.open("a+b")
        try:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunLockedError(f"run is already locked: {self.run_id}") from None
            _ensure_dir(self.attempts_dir)
            _ensure_dir(self.normalized_path.parent)
            self._manifest()
            self._db = sqlite3.connect(self.checkpoint_path)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript(_SCHEMA)
            _fsync_dir(self.checkpoint_path.parent)
            expected = {"storage_version": STORAGE_VERSION, "run_id": self.run_id,
                        "plan_sha256": self.plan_sha256}
            with self._transaction():
                for key, value in expected.items():
                    prior = self._db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                    if prior is not None and prior[0] != value:
                        raise ImmutableConflictError(f"checkpoint {key} differs from the supplied run")
                    self._db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, value))
            self._reconcile()
            self._sync_normalized()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock is not None:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None

    def _require_open(self):
        if self._db is None:
            raise StorageError("RunStore must be used inside its context manager")
        return self._db

    @contextmanager
    def _transaction(self):
        connection = self._require_open()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def _manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_bytes())
            except (ValueError, OSError) as error:
                raise IntegrityError("run manifest is unreadable") from error
            if (not isinstance(manifest, dict) or manifest.get("storage_version") != STORAGE_VERSION
                    or manifest.get("run_id") != self.run_id
                    or manifest.get("plan_sha256") != self.plan_sha256
                    or manifest.get("plan") != self.plan):
                raise ImmutableConflictError("existing run manifest has a different plan")
            return
        manifest = {"storage_version": STORAGE_VERSION, "run_id": self.run_id,
                    "plan_sha256": self.plan_sha256, "plan": self.plan, "created_at": _now()}
        _publish(self.manifest_path, _json_bytes(manifest))

    def _attempt(self, attempt_id: str):
        row = self._require_open().execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            raise StorageError("unknown attempt_id")
        return row

    def attempt_path(self, attempt_id: str, kind: str = "response") -> Path:
        self._attempt(attempt_id)
        if kind not in {"request", "response", "error"}:
            raise ValueError("kind must be request, response or error")
        return self.attempts_dir / attempt_id / (kind + ".json")

    def _read_record(self, attempt_id: str, kind: str) -> dict | None:
        row = self._attempt(attempt_id)
        path = self.attempt_path(attempt_id, kind)
        expected = row[kind + "_sha256"]
        if not path.exists():
            if expected is not None and row["status"] in _TERMINAL:
                raise IntegrityError(f"terminal raw {kind} record is missing")
            return None
        if expected is None:
            raise IntegrityError(f"unanchored raw {kind} record")
        payload = path.read_bytes()
        if _sha(payload) != expected:
            raise IntegrityError(f"raw {kind} SHA-256 mismatch for {attempt_id}")
        try:
            value = json.loads(payload)
        except ValueError as error:
            raise IntegrityError(f"raw {kind} is not valid JSON") from error
        if not isinstance(value, dict) or _json_bytes(value) != payload:
            raise IntegrityError(f"raw {kind} is not canonical JSON")
        return value

    def get_request(self, attempt_id: str) -> dict | None:
        return self._read_record(attempt_id, "request")

    def get_response(self, attempt_id: str) -> dict | None:
        return self._read_record(attempt_id, "response")

    def get_error(self, attempt_id: str) -> dict | None:
        return self._read_record(attempt_id, "error")

    def _reconcile(self) -> None:
        for resolution in self._db.execute("SELECT resolution_id FROM resolutions"):
            self.get_resolution(resolution[0])
        rows = self._db.execute("SELECT * FROM tasks WHERE status IN ('inflight','ambiguous')").fetchall()
        for task in rows:
            attempt_id = task["current_attempt_id"]
            request = self.get_request(attempt_id)
            response = self.get_response(attempt_id)
            error = self.get_error(attempt_id)
            if request is not None and response is not None:
                # A fully published response is recoverable without a new call.
                continue
            if (request is not None and error is not None
                    and error.get("retryable") is True and error.get("ambiguous") is False):
                # The provider explicitly rejected this request. A crash after
                # persisting that fact does not turn it into an uncertain call.
                with self._transaction():
                    self._db.execute("UPDATE tasks SET status='retryable',updated_at=? WHERE task_id=?",
                                     (_now(), task["task_id"]))
                    self._db.execute("UPDATE attempts SET status='retryable',updated_at=? WHERE attempt_id=?",
                                     (_now(), attempt_id))
                continue
            if task["status"] == "inflight":
                with self._transaction():
                    self._db.execute("UPDATE tasks SET status='ambiguous',updated_at=? WHERE task_id=?",
                                     (_now(), task["task_id"]))
                    self._db.execute("UPDATE attempts SET status='ambiguous',updated_at=? WHERE attempt_id=?",
                                     (_now(), attempt_id))

    def register_tasks(self, tasks: list[dict]) -> None:
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list of dictionaries")
        prepared, seen = [], set()
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("each task must be a dictionary")
            task_id, request_key = task.get("task_id"), task.get("request_key")
            for value in (task_id, request_key):
                if (not isinstance(value, str) or not value or value != value.strip()
                        or len(value) > 512 or any(ord(char) < 32 for char in value)):
                    raise ValueError("task_id and request_key must be nonempty trimmed identifiers")
            if task_id in seen:
                raise ValueError("task list contains a duplicate task_id")
            seen.add(task_id)
            prepared.append((task_id, request_key, _canonical(task)))
        with self._transaction():
            for task_id, request_key, encoded in prepared:
                prior = self._db.execute("SELECT task_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                if prior is not None:
                    if prior[0] != encoded:
                        raise ImmutableConflictError("registered task payload changed")
                    continue
                other = self._db.execute("SELECT task_id FROM tasks WHERE request_key=?", (request_key,)).fetchone()
                if other is not None:
                    raise ImmutableConflictError("request_key already belongs to another task")
                timestamp = _now()
                self._db.execute("INSERT INTO tasks VALUES (?,?,?,'pending',NULL,?,?)",
                                 (task_id, request_key, encoded, timestamp, timestamp))

    def claim(self, task_id: str, request: dict) -> str | None:
        """Persist an inflight attempt and raw request before the provider call."""
        payload = _json_bytes(request)
        request_sha = _sha(payload)
        with self._transaction():
            task = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if task is None:
                raise StorageError("unknown task_id")
            if task["current_attempt_id"] is not None:
                prior = self._attempt(task["current_attempt_id"])
                if prior["request_sha256"] != request_sha:
                    raise ImmutableConflictError("request payload differs from the previous attempt")
            if task["status"] in _TERMINAL:
                return None
            if task["status"] in {"inflight", "ambiguous"}:
                raise AmbiguousCallError("an earlier attempt must be recovered or explicitly resolved")
            if task["status"] not in {"pending", "retryable"}:
                raise StorageError("task is not claimable")
            attempt_id = "call-" + uuid.uuid4().hex
            timestamp = _now()
            self._db.execute("INSERT INTO attempts VALUES (?,?, 'inflight',?,NULL,NULL,?,?)",
                             (attempt_id, task_id, request_sha, timestamp, timestamp))
            self._db.execute("UPDATE tasks SET status='inflight',current_attempt_id=?,updated_at=? WHERE task_id=?",
                             (attempt_id, timestamp, task_id))
        directory = self.attempts_dir / attempt_id
        directory.mkdir()
        _fsync_dir(self.attempts_dir)
        try:
            _publish(directory / "request.json", payload)
        except BaseException:
            with self._transaction():
                self._db.execute("UPDATE tasks SET status='ambiguous',updated_at=? WHERE task_id=?", (_now(), task_id))
                self._db.execute("UPDATE attempts SET status='ambiguous',updated_at=? WHERE attempt_id=?", (_now(), attempt_id))
            raise
        return attempt_id

    def _write_record(self, attempt_id: str, kind: str, value: dict) -> dict:
        payload = _json_bytes(value)
        sha = _sha(payload)
        with self._transaction():
            attempt = self._attempt(attempt_id)
            prior = attempt[kind + "_sha256"]
            if prior is not None and prior != sha:
                raise ImmutableConflictError(f"immutable raw {kind} cannot be changed")
            if prior is None and attempt["status"] not in {"inflight", "ambiguous"}:
                raise StorageError("a closed attempt cannot receive a new raw record")
            # Anchor the expected bytes durably before publishing the final file.
            self._db.execute(f"UPDATE attempts SET {kind}_sha256=?,updated_at=? WHERE attempt_id=?",
                             (sha, _now(), attempt_id))
        return _publish(self.attempt_path(attempt_id, kind), payload)

    def write_response(self, attempt_id: str, response: dict) -> dict:
        return self._write_record(attempt_id, "response", response)

    def write_error(self, attempt_id: str, error: dict) -> dict:
        return self._write_record(attempt_id, "error", error)

    def _normalized_record(self, task_id: str, attempt_id: str, normalized: dict) -> tuple[str, str]:
        if not isinstance(normalized, dict):
            raise ValueError("normalized result must be a dictionary")
        record = dict(normalized)
        for key, value in (("task_id", task_id), ("attempt_id", attempt_id), ("run_id", self.run_id)):
            if key in record and record[key] != value:
                raise ImmutableConflictError(f"normalized {key} does not match its task")
            record[key] = value
        encoded = _canonical(record)
        return encoded, _sha((encoded + "\n").encode("utf-8"))

    def _insert_outbox(self, task_id: str, attempt_id: str, encoded: str, sha: str) -> None:
        prior = self._db.execute("SELECT * FROM normalized_outbox WHERE task_id=?", (task_id,)).fetchone()
        if prior is not None:
            if prior["attempt_id"] != attempt_id or prior["record_json"] != encoded or prior["record_sha256"] != sha:
                raise ImmutableConflictError("a different normalized result already exists for this task")
            return
        self._db.execute("INSERT INTO normalized_outbox(task_id,attempt_id,record_json,record_sha256) VALUES (?,?,?,?)",
                         (task_id, attempt_id, encoded, sha))

    def finish(self, task_id: str, attempt_id: str, normalized: dict, status: str = "succeeded") -> None:
        if status not in _TERMINAL:
            raise ValueError("terminal status must be succeeded, cached or permanent_failed")
        encoded, sha = self._normalized_record(task_id, attempt_id, normalized)
        attempt = self._attempt(attempt_id)
        if attempt["task_id"] != task_id:
            raise ImmutableConflictError("attempt does not belong to task")
        if self.get_request(attempt_id) is None:
            raise StorageError("completion requires a durable raw request")
        response = self.get_response(attempt_id)
        error = self.get_error(attempt_id)
        if status in {"succeeded", "cached"} and response is None:
            raise StorageError("successful completion requires a durable raw response")
        if status == "permanent_failed" and error is None and response is None:
            raise StorageError("permanent failure requires a durable raw error or response")
        with self._transaction():
            task = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if task["current_attempt_id"] != attempt_id:
                raise ImmutableConflictError("attempt is not the current task attempt")
            if task["status"] in _TERMINAL and task["status"] != status:
                raise ImmutableConflictError("terminal task status cannot change")
            allowed = {"inflight", "ambiguous", *tuple(_TERMINAL)}
            if status == "permanent_failed":
                allowed.add("retryable")
            if task["status"] not in allowed:
                raise StorageError("task cannot be finished in its current state")
            self._insert_outbox(task_id, attempt_id, encoded, sha)
            timestamp = _now()
            self._db.execute("UPDATE tasks SET status=?,updated_at=? WHERE task_id=?", (status, timestamp, task_id))
            self._db.execute("UPDATE attempts SET status=?,updated_at=? WHERE attempt_id=?", (status, timestamp, attempt_id))
        self._append_normalized()

    def set_retry(self, task_id: str, attempt_id: str, error: dict) -> None:
        """Record a known rejected call; caller must classify retry safety first."""
        if (not isinstance(error, dict) or error.get("retryable") is not True
                or error.get("ambiguous") is not False):
            raise StorageError("automatic retry requires an explicitly retryable, unambiguous rejection")
        attempt = self._attempt(attempt_id)
        if attempt["task_id"] != task_id:
            raise ImmutableConflictError("attempt does not belong to task")
        if self.get_response(attempt_id) is not None:
            raise StorageError("a completed raw response must be normalized, not retried")
        self.write_error(attempt_id, error)
        with self._transaction():
            task = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if task["current_attempt_id"] != attempt_id or task["status"] not in {"inflight", "retryable"}:
                raise StorageError("only the current known rejected inflight attempt can be retried")
            timestamp = _now()
            self._db.execute("UPDATE tasks SET status='retryable',updated_at=? WHERE task_id=?", (timestamp, task_id))
            self._db.execute("UPDATE attempts SET status='retryable',updated_at=? WHERE attempt_id=?", (timestamp, attempt_id))

    def mark_ambiguous(self, task_id: str, attempt_id: str, error: dict) -> None:
        """Expose an uncertain provider outcome immediately without retrying."""
        attempt = self._attempt(attempt_id)
        if attempt["task_id"] != task_id:
            raise ImmutableConflictError("attempt does not belong to task")
        self.write_error(attempt_id, error)
        with self._transaction():
            task = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if task["current_attempt_id"] != attempt_id or task["status"] not in {"inflight", "ambiguous"}:
                raise StorageError("only the current inflight attempt can be marked ambiguous")
            timestamp = _now()
            self._db.execute("UPDATE tasks SET status='ambiguous',updated_at=? WHERE task_id=?", (timestamp, task_id))
            self._db.execute("UPDATE attempts SET status='ambiguous',updated_at=? WHERE attempt_id=?", (timestamp, attempt_id))

    def resolve_ambiguous(self, task_id: str, decision: str, reason: str,
                          ack_duplicate_risk: bool = False) -> str:
        if decision not in {"retry", "abandon"}:
            raise ValueError("decision must be retry or abandon")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("an explicit resolution reason is required")
        if type(ack_duplicate_risk) is not bool or (decision == "retry" and not ack_duplicate_risk):
            raise StorageError("retry requires explicit acknowledgment of possible duplicate billing")
        task = self.get_task(task_id)
        if task["status"] != "ambiguous":
            raise StorageError("only an ambiguous task may be explicitly resolved")
        if task["recoverable"]:
            raise StorageError("a persisted response can be recovered without another provider call")
        resolution_id = "resolution-" + uuid.uuid4().hex
        attempt_id = task["current_attempt_id"]
        timestamp = _now()
        record = {"resolution_id": resolution_id, "task_id": task_id, "attempt_id": attempt_id,
                  "run_id": self.run_id, "decision": decision, "reason": reason,
                  "ack_duplicate_risk": ack_duplicate_risk, "created_at": timestamp}
        directory = self.raw_dir / "resolutions"
        _ensure_dir(directory)
        published = _publish(directory / (resolution_id + ".json"), _json_bytes(record))
        with self._transaction():
            self._db.execute("INSERT INTO resolutions VALUES (?,?,?,?,?,?)",
                             (resolution_id, task_id, attempt_id, decision, published["sha256"], timestamp))
            status = "retryable" if decision == "retry" else "permanent_failed"
            self._db.execute("UPDATE tasks SET status=?,updated_at=? WHERE task_id=?", (status, timestamp, task_id))
            self._db.execute("UPDATE attempts SET status=?,updated_at=? WHERE attempt_id=?",
                             ("resolved_" + decision, timestamp, attempt_id))
            if decision == "abandon":
                normalized = {"status": status, "error": {"code": "ambiguous_call_abandoned", "reason": reason},
                              "resolution_id": resolution_id}
                encoded, sha = self._normalized_record(task_id, attempt_id, normalized)
                self._insert_outbox(task_id, attempt_id, encoded, sha)
        self._append_normalized()
        return resolution_id

    def get_resolution(self, resolution_id: str) -> dict:
        row = self._require_open().execute("SELECT * FROM resolutions WHERE resolution_id=?", (resolution_id,)).fetchone()
        if row is None:
            raise StorageError("unknown resolution_id")
        path = self.raw_dir / "resolutions" / (resolution_id + ".json")
        if not path.is_file():
            raise IntegrityError("immutable ambiguous-call resolution record is missing")
        payload = path.read_bytes()
        if _sha(payload) != row["record_sha256"]:
            raise IntegrityError("ambiguous-call resolution SHA-256 mismatch")
        try:
            record = json.loads(payload)
        except ValueError as error:
            raise IntegrityError("ambiguous-call resolution is not JSON") from error
        if not isinstance(record, dict) or _json_bytes(record) != payload:
            raise IntegrityError("ambiguous-call resolution is not canonical JSON")
        return record

    def _sync_normalized(self) -> None:
        """Validate existing append-only bytes and complete any partial suffix."""
        if not self.normalized_path.exists():
            descriptor = os.open(self.normalized_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
            _fsync_dir(self.normalized_path.parent)
        self._last_normalized_seq = 0
        self._normalized_bytes = 0
        missing = False
        with self.normalized_path.open("rb") as existing:
            for row in self._db.execute("SELECT * FROM normalized_outbox ORDER BY seq"):
                expected = (row["record_json"] + "\n").encode("utf-8")
                if _sha(expected) != row["record_sha256"]:
                    raise IntegrityError("normalized checkpoint record hash mismatch")
                actual = b"" if missing else existing.read(len(expected))
                if not expected.startswith(actual):
                    raise IntegrityError("append-only normalized output differs from its checkpoint")
                if len(actual) < len(expected):
                    missing = True
                    self._append_bytes(expected[len(actual):])
                self._last_normalized_seq = row["seq"]
                self._normalized_bytes += len(expected)
            if not missing and existing.read(1):
                raise IntegrityError("normalized output contains uncheckpointed bytes")

    def _append_bytes(self, payload: bytes) -> None:
        descriptor = os.open(self.normalized_path, os.O_WRONLY | os.O_APPEND)
        with os.fdopen(descriptor, "ab", buffering=0) as handle:
            view = memoryview(payload)
            while view:
                written = handle.write(view)
                if written is None or written <= 0:
                    raise StorageError("normalized append made no progress")
                view = view[written:]
            os.fsync(handle.fileno())

    def _append_normalized(self) -> None:
        if self.normalized_path.stat().st_size != self._normalized_bytes:
            # A failed previous append may have persisted only a prefix.
            self._sync_normalized()
            return
        for row in self._db.execute("SELECT * FROM normalized_outbox WHERE seq>? ORDER BY seq",
                                    (self._last_normalized_seq,)):
            payload = (row["record_json"] + "\n").encode("utf-8")
            if _sha(payload) != row["record_sha256"]:
                raise IntegrityError("normalized checkpoint record hash mismatch")
            self._append_bytes(payload)
            self._last_normalized_seq = row["seq"]
            self._normalized_bytes += len(payload)

    def get_task(self, task_id: str) -> dict:
        row = self._require_open().execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise StorageError("unknown task_id")
        result = json.loads(row["task_json"])
        result.update(status=row["status"], current_attempt_id=row["current_attempt_id"],
                      attempt_id=row["current_attempt_id"], created_at=row["created_at"], updated_at=row["updated_at"],
                      recoverable=False, response_sha256=None, error_sha256=None, normalized=None)
        attempts = self._db.execute("SELECT * FROM attempts WHERE task_id=? ORDER BY rowid", (task_id,)).fetchall()
        result["attempts"] = [dict(attempt) for attempt in attempts]
        if row["current_attempt_id"] is not None:
            attempt = self._attempt(row["current_attempt_id"])
            result["response_sha256"] = attempt["response_sha256"]
            result["error_sha256"] = attempt["error_sha256"]
            result["recoverable"] = (self.get_request(attempt["attempt_id"]) is not None
                                     and self.get_response(attempt["attempt_id"]) is not None)
        normalized = self._db.execute("SELECT record_json FROM normalized_outbox WHERE task_id=?", (task_id,)).fetchone()
        if normalized is not None:
            result["normalized"] = json.loads(normalized[0])
        return result

    def pending_tasks(self) -> list[dict]:
        ids = self._require_open().execute(
            "SELECT task_id FROM tasks WHERE status NOT IN ('succeeded','cached','permanent_failed') ORDER BY rowid").fetchall()
        return [self.get_task(row[0]) for row in ids]

    def summary(self) -> dict:
        connection = self._require_open()
        statuses = {row[0]: row[1] for row in connection.execute("SELECT status,count(*) FROM tasks GROUP BY status")}
        published = {"request": 0, "response": 0, "error": 0}
        for attempt in connection.execute("SELECT * FROM attempts"):
            for kind in published:
                if (attempt[kind + "_sha256"] is not None
                        and (self.attempts_dir / attempt["attempt_id"] / (kind + ".json")).is_file()):
                    published[kind] += 1
        return {"run_id": self.run_id, "plan_sha256": self.plan_sha256,
                "tasks": sum(statuses.values()), "statuses": statuses,
                "attempts": connection.execute("SELECT count(*) FROM attempts").fetchone()[0],
                "raw_requests": published["request"], "raw_responses": published["response"],
                "raw_errors": published["error"],
                "normalized_records": connection.execute("SELECT count(*) FROM normalized_outbox").fetchone()[0],
                "resolutions": connection.execute("SELECT count(*) FROM resolutions").fetchone()[0]}
