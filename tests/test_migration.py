from __future__ import annotations

import copy
from contextlib import closing
import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration import backup_sqlite, check_ollama, download_verified, history_lock, pull_ollama, relink_images, restore_history, sha256
from setup_wizard import ready_to_start
from storage import Store


class DownloadResponse(io.BytesIO):
    def geturl(self):
        return "https://example.com/model.onnx"


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "model.onnx"
        self.bytes = b"verified test model bytes"
        self.expected = {"size": len(self.bytes), "sha256": hashlib.sha256(self.bytes).hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def test_verified_download_is_atomic_and_repeat_skips_network(self):
        self.path.write_bytes(b"old model")
        opened = []
        def opener(request, timeout):
            opened.append(request.full_url)
            return DownloadResponse(self.bytes)
        self.assertTrue(download_verified("https://example.com/model.onnx", self.path, self.expected, opener=opener))
        self.assertEqual(self.path.read_bytes(), self.bytes)
        self.assertFalse(download_verified("https://example.com/model.onnx", self.path, self.expected, opener=opener))
        self.assertEqual(len(opened), 1)

    def test_bad_hash_short_and_oversize_downloads_preserve_old_file(self):
        for content in (b"?" * len(self.bytes), self.bytes[:-1], self.bytes + b"extra"):
            with self.subTest(content=content):
                self.path.write_bytes(b"old model")
                with self.assertRaises(ValueError):
                    download_verified("https://example.com/model.onnx", self.path, self.expected, opener=lambda *a, **k: DownloadResponse(content))
                self.assertEqual(self.path.read_bytes(), b"old model")
                self.assertFalse(list(self.root.glob("*.part")))

    def test_transport_failure_preserves_old_file_and_cleans_partial(self):
        class Broken(DownloadResponse):
            def read(self, size):
                raise OSError("connection lost")
        self.path.write_bytes(b"old model")
        with self.assertRaises(OSError):
            download_verified("https://example.com/model.onnx", self.path, self.expected, opener=lambda *a, **k: Broken(b""))
        self.assertEqual(self.path.read_bytes(), b"old model")
        self.assertFalse(list(self.root.glob("*.part")))

    def test_insecure_redirect_is_rejected(self):
        class Insecure(DownloadResponse):
            def geturl(self):
                return "http://example.com/model.onnx"
        with self.assertRaises(ValueError):
            download_verified("https://example.com/model.onnx", self.path, self.expected, opener=lambda *a, **k: Insecure(self.bytes))
        self.assertFalse(self.path.exists())


class ModelRestoreTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"ollama": {"model": "test-vision:latest", "digest": "a" * 64}}

    def test_pull_payload_and_required_success(self):
        calls = []
        def request(route, payload, timeout):
            calls.append((route, payload, timeout))
            return io.BytesIO(b'{"status":"pulling manifest"}\n{"status":"success"}\n')
        with patch("migration.local_request", request), patch("migration.check_ollama", side_effect=[{"installed": False}, {"installed": True, "matches": True}]):
            pull_ollama(self.profile)
        self.assertEqual(calls[0][0], "/api/pull")
        self.assertEqual(calls[0][1], {"model": "test-vision:latest", "stream": True})
        with patch("migration.local_request", return_value=io.BytesIO(b'{"status":"pulling manifest"}\n')), patch("migration.check_ollama", return_value={"installed": False}):
            with self.assertRaises(RuntimeError):
                pull_ollama(self.profile)

    def test_existing_ollama_model_is_preserved_even_when_version_differs(self):
        with patch("migration.check_ollama", return_value={"installed": True, "matches": False}), patch("migration.local_request") as request:
            pull_ollama(self.profile)
        request.assert_not_called()

    def test_model_digest_difference_is_reported_without_claiming_match(self):
        def request(route, payload=None, timeout=15):
            data = {"models": [{"name": "test-vision:latest", "digest": "b" * 64}]} if route == "/api/tags" else {"capabilities": ["vision"]}
            return io.BytesIO(json.dumps(data).encode())
        with patch("migration.local_request", request):
            result = check_ollama(self.profile)
        self.assertTrue(result["installed"])
        self.assertFalse(result["matches"])

    def test_cloud_model_is_not_accepted(self):
        data = {"models": [{"name": "test-vision:latest", "digest": "a" * 64, "remote_host": "remote"}]}
        with patch("migration.local_request", return_value=io.BytesIO(json.dumps(data).encode())):
            with self.assertRaises(ValueError):
                check_ollama(self.profile)

    def test_missing_parts_cannot_be_bypassed_by_version_override(self):
        ready = {"imports_ok": True, "tagger": True, "ollama": {"installed": True, "matches": True}, "python_matches": True, "packages": [{"matches": True}]}
        self.assertTrue(ready_to_start(ready))
        # Initial preparation no longer requires an Ollama connection or a specific model.
        different = copy.deepcopy(ready)
        different["ollama"]["matches"] = False
        self.assertTrue(ready_to_start(different))
        self.assertTrue(ready_to_start(different, True))
        for field in ("imports_ok", "tagger"):
            missing = copy.deepcopy(ready)
            missing[field] = False
            self.assertFalse(ready_to_start(missing, True))
        missing = copy.deepcopy(ready)
        missing["ollama"]["installed"] = False
        self.assertTrue(ready_to_start(missing, True))


class HistoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.original = self.root / "old" / "image.png"
        self.content = b"identical source image"
        self.hash = hashlib.sha256(self.content).hexdigest()
        source = self.root / "source.sqlite3"
        store = Store(source)
        self.asset_id = store.add([self.original])[0]
        run = store.begin(self.asset_id, "run-key", self.hash, None, {})
        store.finish(run, {"provenance": "preserve this result"})
        store.review(run, "review name", "review note", True)
        store.set_setting("ui", {"model": "test-vision"})
        store.close()
        (self.bundle / "migration").mkdir()
        self.backup = self.bundle / "migration" / "history.sqlite3"
        backup_sqlite(source, self.backup)
        (self.bundle / "migration" / "history_manifest.json").write_text(json.dumps({"size": self.backup.stat().st_size, "sha256": sha256(self.backup)}))
        self.data = self.root / "new-data"

    def tearDown(self):
        self.temp.cleanup()

    def test_history_restore_preserves_results_reviews_and_settings(self):
        restore_history(self.bundle, self.data)
        with closing(sqlite3.connect(self.data / "character_lens.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT review_name,review_note,reviewed FROM runs").fetchone(), ("review name", "review note", 1))
            self.assertEqual(json.loads(db.execute("SELECT value FROM settings").fetchone()[0]), {"model": "test-vision"})
            self.assertEqual(json.loads(db.execute("SELECT result FROM runs").fetchone()[0]), {"provenance": "preserve this result"})

    def test_restore_refuses_existing_history(self):
        restore_history(self.bundle, self.data)
        before = sha256(self.data / "character_lens.sqlite3")
        with self.assertRaises(ValueError):
            restore_history(self.bundle, self.data)
        self.assertEqual(sha256(self.data / "character_lens.sqlite3"), before)

    def test_corrupt_bundled_history_is_rejected(self):
        self.backup.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            restore_history(self.bundle, self.data)
        self.assertFalse((self.data / "character_lens.sqlite3").exists())

    def test_running_app_locks_history_restore(self):
        with history_lock(self.data):
            with self.assertRaises(RuntimeError):
                restore_history(self.bundle, self.data)

    def test_relink_requires_name_and_hash_and_preserves_original_images(self):
        restore_history(self.bundle, self.data)
        images = self.root / "new-images"
        images.mkdir()
        image = images / "image.png"
        image.write_bytes(self.content)
        result = relink_images(self.data, images)
        self.assertEqual(result["changed"], 1)
        self.assertTrue(Path(result["backup"]).exists())
        self.assertEqual(image.read_bytes(), self.content)
        with closing(sqlite3.connect(self.data / "character_lens.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT path FROM assets").fetchone()[0], str(image.resolve()))
            self.assertEqual(db.execute("SELECT reviewed FROM runs").fetchone()[0], 1)
        with closing(sqlite3.connect(result["backup"])) as db:
            self.assertEqual(db.execute("SELECT path FROM assets").fetchone()[0], str(self.original.resolve()))

    def test_same_name_wrong_content_is_not_relinked(self):
        restore_history(self.bundle, self.data)
        images = self.root / "new-images"
        images.mkdir()
        (images / "image.png").write_bytes(b"wrong content")
        result = relink_images(self.data, images)
        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(result["unresolved"]), 1)

    def test_ambiguous_duplicate_is_not_relinked(self):
        restore_history(self.bundle, self.data)
        images = self.root / "new-images"
        for subfolder in ("a", "b"):
            path = images / subfolder
            path.mkdir(parents=True)
            (path / "image.png").write_bytes(self.content)
        result = relink_images(self.data, images)
        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(result["unresolved"]), 1)

    def test_running_app_blocks_relink_and_preserves_database(self):
        restore_history(self.bundle, self.data)
        with history_lock(self.data):
            with self.assertRaises(RuntimeError):
                relink_images(self.data, self.root)

