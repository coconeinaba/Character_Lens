"""Per-image transactional persistence. Original files are never modified."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from domain import dumps, now


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        with self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                  id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, path_key TEXT UNIQUE,
                  crop TEXT, added TEXT NOT NULL, hidden INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS runs (
                  id INTEGER PRIMARY KEY, asset_id INTEGER NOT NULL REFERENCES assets(id),
                  cache_key TEXT NOT NULL UNIQUE, source_sha TEXT NOT NULL,
                  crop TEXT, config TEXT NOT NULL, status TEXT NOT NULL,
                  observation TEXT, result TEXT, error TEXT NOT NULL DEFAULT '',
                  created TEXT NOT NULL, updated TEXT NOT NULL,
                  review_name TEXT NOT NULL DEFAULT '', review_note TEXT NOT NULL DEFAULT '',
                  reviewed INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS runs_asset ON runs(asset_id,id);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            columns = {r[1] for r in self.db.execute("PRAGMA table_info(assets)")}
            if "path_key" not in columns:
                self.db.execute("ALTER TABLE assets ADD COLUMN path_key TEXT")
            for row in self.db.execute("SELECT id,path FROM assets WHERE path_key IS NULL").fetchall():
                self.db.execute("UPDATE assets SET path_key=? WHERE id=?", (os.path.normcase(row[1]), row[0]))
            self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS asset_path_key ON assets(path_key)")
            self.db.execute("UPDATE runs SET status='paused',error='前回の終了時に解析が中断されました。再開できます。',updated=? WHERE status='running'", (now(),))

    def close(self):
        with self.lock:
            self.db.close()

    def get_setting(self, key, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, dumps(value)))

    def add(self, paths):
        ids = []
        with self.lock, self.db:
            for path in paths:
                path = str(Path(path).resolve())
                key = os.path.normcase(path)
                self.db.execute("INSERT INTO assets(path,path_key,added) VALUES(?,?,?) ON CONFLICT(path_key) DO UPDATE SET hidden=0,path=excluded.path", (path, key, now()))
                ids.append(self.db.execute("SELECT id FROM assets WHERE path_key=?", (key,)).fetchone()[0])
        return ids

    def hide(self, ids):
        with self.lock, self.db:
            self.db.executemany("UPDATE assets SET hidden=1 WHERE id=?", [(i,) for i in ids])

    def hide_all(self):
        with self.lock, self.db:
            return self.db.execute("UPDATE assets SET hidden=1 WHERE hidden=0").rowcount

    def set_crop(self, asset_id, crop):
        with self.lock, self.db:
            self.db.execute("UPDATE assets SET crop=? WHERE id=?", (dumps(crop) if crop else None, asset_id))

    @staticmethod
    def decode(row):
        if row is None:
            return None
        row = dict(row)
        for key in ("crop", "run_crop", "config", "observation", "result"):
            if row.get(key) is not None:
                row[key] = json.loads(row[key])
        return row

    def assets(self):
        with self.lock:
            rows = self.db.execute("""SELECT a.*,r.id AS run_id,r.status,r.result,r.error,r.updated,
                r.crop AS run_crop,r.source_sha,r.config,r.observation,r.review_name,r.review_note,r.reviewed
                FROM assets a LEFT JOIN runs r ON r.id=(SELECT id FROM runs WHERE asset_id=a.id ORDER BY updated DESC,id DESC LIMIT 1)
                WHERE a.hidden=0 ORDER BY a.id""").fetchall()
        return [self.decode(row) for row in rows]

    def asset(self, asset_id):
        with self.lock:
            return self.decode(self.db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone())

    def run(self, run_id):
        with self.lock:
            return self.decode(self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())

    def find(self, cache_key):
        with self.lock:
            return self.decode(self.db.execute("SELECT * FROM runs WHERE cache_key=?", (cache_key,)).fetchone())

    def compatible(self, asset_id, source_sha, crop, config):
        with self.lock:
            row = self.db.execute("SELECT * FROM runs WHERE asset_id=? AND source_sha=? AND crop IS ? AND config=? ORDER BY id DESC LIMIT 1", (asset_id, source_sha, dumps(crop) if crop else None, dumps(config))).fetchone()
        return self.decode(row)

    def begin(self, asset_id, cache_key, source_sha, crop, config):
        with self.lock, self.db:
            row = self.db.execute("SELECT id,status FROM runs WHERE cache_key=?", (cache_key,)).fetchone()
            if row:
                self.db.execute("UPDATE runs SET status='running',error='',updated=? WHERE id=?", (now(), row[0]))
                return row[0]
            cur = self.db.execute("INSERT INTO runs(asset_id,cache_key,source_sha,crop,config,status,created,updated) VALUES(?,?,?,?,?,'running',?,?)", (asset_id, cache_key, source_sha, dumps(crop) if crop else None, dumps(config), now(), now()))
            return cur.lastrowid

    def touch(self, run_id):
        with self.lock, self.db:
            self.db.execute("UPDATE runs SET updated=? WHERE id=?", (now(), run_id))

    def observation(self, run_id, observation):
        with self.lock, self.db:
            self.db.execute("UPDATE runs SET observation=?,updated=? WHERE id=?", (dumps(observation), now(), run_id))

    def finish(self, run_id, result):
        with self.lock, self.db:
            self.db.execute("UPDATE runs SET result=?,status='done',error='',updated=? WHERE id=?", (dumps(result), now(), run_id))

    def fail(self, run_id, error, paused=False):
        with self.lock, self.db:
            self.db.execute("UPDATE runs SET status=?,error=?,updated=? WHERE id=?", ("paused" if paused else "error", str(error)[:3000], now(), run_id))

    def review(self, run_id, name, note, reviewed):
        with self.lock, self.db:
            self.db.execute("UPDATE runs SET review_name=?,review_note=?,reviewed=? WHERE id=?", (name.strip(), note.strip(), int(reviewed), run_id))

    def history(self, asset_id):
        with self.lock:
            rows = self.db.execute("SELECT * FROM runs WHERE asset_id=? ORDER BY id DESC", (asset_id,)).fetchall()
        return [self.decode(row) for row in rows]

    def validate_destination(self, destination):
        destination = Path(destination)
        protected = {self.path.resolve(), Path(str(self.path) + "-wal").resolve(), Path(str(self.path) + "-shm").resolve()}
        if destination.resolve() in protected:
            raise ValueError("保存先に実行中のデータベースは指定できません。")
        with self.lock:
            if any(destination.resolve() == Path(row[0]).resolve() for row in self.db.execute("SELECT path FROM assets")):
                raise ValueError("保存先に元画像は指定できません。")

    def backup(self, destination):
        self.validate_destination(destination)
        with self.lock:
            target = sqlite3.connect(destination)
            try:
                self.db.backup(target)
            finally:
                target.close()

