from __future__ import annotations

import base64
import copy
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from domain import AXES, ValidationError, build_config, contract_bundle, parse_reply, prepare_image, score_result, validate_weights
from settings import load_settings, save_settings
from engine import BatchEngine
from exporter import export_csv, export_html, export_json, spreadsheet_text
from ollama_client import Cancelled, OllamaClient, OllamaError, local_endpoint
from storage import Store


def observation():
    return {"summary": "人物の検証画像", "subject_count": 1, "features": {key: "観察した特徴" for key in AXES}, "limitations": []}


def answer(name="テスト候補"):
    return {"identification": "candidate", "overall_reason": "固有の装飾が似ているという推定", "candidates": [{"name": name, "display_name": name, "work": "検証作品", "variant": "不明", "confidence": "medium", "rationale": "特徴に一致と相違があります", "matches": ["装飾"], "differences": ["衣装"], "axes": [{"axis": key, "score": 75, "reason": "見える特徴に基づく仮評価"} for key in AXES]}], "limitations": ["参照画像なし"]}


class FakeClient:
    def __init__(self, cancel=None):
        self.calls = []
        self.fail_second = False
        self.cancel_after_observe = False
        self.event = cancel or threading.Event()
        self.mutate = None
        self.schemas = []
        self.answer_name = "テスト候補"
        self.result_override = None

    def cancel(self):
        self.event.set()

    def verify_model(self, config):
        pass

    def structured(self, config, schema, prompt, encoded_image, progress=None, validator=None):
        stage = "observe" if "features" in schema["properties"] else "identify"
        self.calls.append((stage, prompt, encoded_image))
        self.schemas.append(copy.deepcopy(schema))
        if stage == "identify" and self.fail_second:
            raise OllamaError("検証用の一時エラー")
        if stage == "observe" and self.cancel_after_observe:
            self.event.set()
        result = observation() if stage == "observe" else (copy.deepcopy(self.result_override) if self.result_override else answer(self.answer_name))
        result = parse_reply(json.dumps(result), schema)
        if self.mutate and stage == "identify":
            self.mutate()
        if validator:
            result = validator(result)
        return result, {"attempts": 1}


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.schema = contract_bundle()["result_schema"]

    def test_unknown_is_valid_without_candidates(self):
        result = {"identification": "unknown", "overall_reason": "情報不足", "candidates": [], "limitations": []}
        self.assertEqual(score_result(parse_reply(json.dumps(result), self.schema))["candidates"], [])

    def test_range_boolean_and_string_scores_are_rejected(self):
        for value in (True, "90", 101, -1, 90.5):
            with self.subTest(value=value):
                data = answer()
                data["candidates"][0]["axes"][0]["score"] = value
                with self.assertRaises(ValidationError):
                    parse_reply(json.dumps(data), self.schema)

    def test_partial_fenced_duplicate_and_nonfinite_json_rejected(self):
        for text in ('{"identification":', '```json\n{}\n```', '{"x":1,"x":2}', '{"x":NaN}'):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                parse_reply(text, self.schema)

    def test_duplicate_axes_rejected(self):
        data = answer()
        data["candidates"][0]["axes"][0]["axis"] = "face"
        with self.assertRaises(ValidationError):
            score_result(data)

    def test_unknown_with_candidate_rejected(self):
        data = answer()
        data["identification"] = "unknown"
        with self.assertRaises(ValidationError):
            score_result(data)

    def test_claimed_reference_comparison_and_unknown_work_rejected(self):
        data = answer()
        data["candidates"][0]["rationale"] = "公式画像と比較すると髪が似ています。"
        with self.assertRaises(ValidationError):
            score_result(data)
        data = answer()
        data["candidates"][0]["work"] = "不明"
        with self.assertRaises(ValidationError):
            score_result(data)

    def test_weights_are_computed_not_model_supplied(self):
        data = answer()
        for axis in data["candidates"][0]["axes"]:
            axis["score"] = 100 if axis["axis"] == "distinctive" else 0
        scored = score_result(data)["candidates"][0]
        self.assertEqual(scored["estimated_similarity"], 30)
        self.assertEqual(scored["coverage"], 100)

    def test_unseen_axes_not_counted_as_zero_and_low_coverage_has_no_total(self):
        data = answer()
        for axis in data["candidates"][0]["axes"]:
            axis["score"] = 100 if axis["axis"] == "distinctive" else None
        scored = score_result(data)["candidates"][0]
        self.assertIsNone(scored["estimated_similarity"])
        self.assertEqual(scored["coverage"], 30)
        data["candidates"][0]["axes"][0]["score"] = 100
        scored = score_result(data)["candidates"][0]
        self.assertEqual(scored["estimated_similarity"], 100)
        self.assertEqual(scored["coverage"], 50)

    def test_ip_endpoints_and_credentials_rejected(self):
        self.assertEqual(local_endpoint("http://localhost:12345"), ("127.0.0.1", 12345))
        self.assertEqual(local_endpoint("http://[::1]:11434"), ("::1", 11434))
        self.assertEqual(local_endpoint("http://192.168.1.2"), ("192.168.1.2", 11434))
        for url in ("https://127.0.0.1:11434", "http://ollama.com", "http://user:pass@127.0.0.1", "http://127.0.0.1/path", "http://127.0.0.1?x=1", "http://127.0.0.1:abc", "not a url"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_endpoint(url)

    def test_target_name_required(self):
        with self.assertRaises(ValueError):
            build_config("test", "digest", "target", " ")

    def test_custom_weights_are_saved_and_used(self):
        weights = {key: 1 for key in AXES}
        weights["hair"] = 90
        data = answer()
        for axis in data["candidates"][0]["axes"]:
            axis["score"] = 100 if axis["axis"] == "hair" else 0
        self.assertEqual(score_result(data, weights)["candidates"][0]["estimated_similarity"], 95)
        with self.assertRaises(ValueError):
            validate_weights({})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "setting.json"
            saved = save_settings({"weights": weights, "ollama_profiles": [{"name": "LAN", "endpoint": "http://192.168.1.2:11434"}], "active_profile": "LAN"}, path)
            self.assertEqual(saved["weights"], weights)
            self.assertEqual(load_settings(path)["ollama_profiles"][0]["endpoint"], "http://192.168.1.2:11434")


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_rotation_metadata_removed_and_source_unchanged(self):
        path = self.root / "CharacterSecret.jpg"
        im = Image.new("RGB", (40, 20), "red")
        exif = Image.Exif()
        exif[274] = 6
        exif[270] = "CharacterSecret"
        im.save(path, exif=exif)
        original = path.read_bytes()
        result = prepare_image(path)
        self.assertEqual(result["original_size"], [20, 40])
        with Image.open(io.BytesIO(base64.b64decode(result["image"]))) as sent:
            self.assertFalse(sent.getexif())
        self.assertEqual(path.read_bytes(), original)
        self.assertNotIn("CharacterSecret", json.dumps(result))

    def test_crop_bounds_and_resize(self):
        path = self.root / "a.png"
        Image.new("RGB", (200, 100), "blue").save(path)
        result = prepare_image(path, 50, [0, 0, 100, 100])
        self.assertEqual(result["sent_size"], [50, 50])
        for crop in ([0, 0, 0, 1], [-1, 0, 2, 2], [0, 0, 201, 1], [1.0, 0, 2, 2]):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                prepare_image(path, crop=crop)

    def test_transparency_white_background_and_animation_first_frame(self):
        path = self.root / "transparent.png"
        Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(path)
        result = prepare_image(path)
        with Image.open(io.BytesIO(base64.b64decode(result["image"]))) as sent:
            self.assertGreater(min(sent.getpixel((10, 10))), 245)
        gif = self.root / "animated.gif"
        Image.new("RGB", (20, 20), "red").save(gif, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")])
        result = prepare_image(gif)
        self.assertEqual(result["frames"], 2)
        with Image.open(io.BytesIO(base64.b64decode(result["image"]))) as sent:
            self.assertGreater(sent.getpixel((10, 10))[0], 200)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "SECRET_CharacterName.png"
        Image.new("RGB", (40, 60), "red").save(self.path)
        self.store = Store(self.root / "state.sqlite3")
        self.asset_id = self.store.add([self.path])[0]
        self.config = build_config("fake-vision", "digest")
        self.events = []
        self.cancel = threading.Event()
        self.client = FakeClient(self.cancel)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def run_engine(self, force=False):
        engine = BatchEngine(self.store, "http://127.0.0.1:11434", self.config, lambda k, p: self.events.append((k, p)), self.cancel, self.client)
        engine.run(self.store.assets(), force)
        return self.store.assets()[0]

    def test_complete_two_stages_and_preserve_original_filename(self):
        row = self.run_engine()
        self.assertEqual(row["status"], "done")
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(row["result"]["provenance"]["file_name"], self.path.name)
        self.assertNotIn("SECRET", " ".join(c[1] for c in self.client.calls))
        self.assertEqual(row["result"]["candidates"][0]["estimated_similarity"], 75)

    def test_tagger_no_candidate_still_completes_free_llm_evaluation(self):
        self.config["tagger"] = {"threshold": 0.85}
        tagged = {"candidates": [], "top_below_threshold": [{"tag": "weak_guess", "score": 0.2}], "threshold": 0.85}
        with patch("tagger.CharacterTagger") as constructor:
            constructor.return_value.predict.return_value = tagged
            row = self.run_engine()
        self.assertEqual([c[0] for c in self.client.calls], ["observe", "identify"])
        self.assertTrue(row["result"]["tagger_candidates_empty"])
        self.assertEqual(row["result"]["tagger"], tagged)

    def test_tagger_bounds_candidates_and_hides_hints_during_observation(self):
        self.config["tagger"] = {"threshold": 0.85}
        self.client.answer_name = "accepted_character"
        tagged = {"candidates": [{"tag": "accepted_character", "score": 0.99}], "threshold": 0.85}
        with patch("tagger.CharacterTagger") as constructor:
            constructor.return_value.predict.return_value = tagged
            row = self.run_engine()
        self.assertEqual(row["status"], "done")
        self.assertNotIn("accepted_character", self.client.calls[0][1])
        candidates_schema = self.client.schemas[1]["properties"]["candidates"]
        self.assertEqual(candidates_schema["maxItems"], 1)
        self.assertEqual(candidates_schema["items"]["properties"]["name"]["enum"], ["accepted_character"])
        with self.assertRaises(ValidationError):
            parse_reply(json.dumps(answer("made_up_character")), self.client.schemas[1])
        duplicated = answer("accepted_character")
        duplicated["candidates"] *= 2
        with self.assertRaises(ValidationError):
            parse_reply(json.dumps(duplicated), self.client.schemas[1])

    def test_tagger_does_not_block_explicit_target_evaluation(self):
        self.config["tagger"] = {"threshold": 0.85}
        self.config.update(mode="target", target="指定名")
        with patch("tagger.CharacterTagger") as constructor:
            constructor.return_value.predict.return_value = {"candidates": [], "threshold": 0.85}
            row = self.run_engine()
        self.assertEqual(row["status"], "done")
        self.assertEqual(len(self.client.calls), 2)

    def test_dictionary_fixes_label_contract_without_biasing_blind_observation(self):
        self.config["tagger"] = {"threshold": 0.85}
        self.client.result_override = answer("sonoda_umi")
        self.client.result_override["candidates"][0].update(display_name="園田海未", work="ラブライブ！")
        with patch("tagger.CharacterTagger") as constructor:
            constructor.return_value.predict.return_value = {"candidates": [{"tag": "sonoda_umi", "score": 0.99}], "threshold": 0.85}
            row = self.run_engine()
        self.assertNotIn("園田海未", self.client.calls[0][1])
        self.assertIn("園田海未", self.client.calls[1][1])
        self.assertEqual(row["result"]["candidates"][0]["label_source"], "dictionary")
        data = answer("sonoda_umi")
        data["candidates"][0].update(display_name="架空の名前", work="誤った作品")
        with self.assertRaises(ValidationError):
            parse_reply(json.dumps(data), self.client.schemas[1])

    def test_unregistered_tag_keeps_model_translation_out_of_primary_name(self):
        self.config["tagger"] = {"threshold": 0.85}
        self.client.result_override = answer("unregistered_character")
        self.client.result_override["candidates"][0]["display_name"] = "推測した日本語名"
        with patch("tagger.CharacterTagger") as constructor:
            constructor.return_value.predict.return_value = {"candidates": [{"tag": "unregistered_character", "score": 0.99}], "threshold": 0.85}
            row = self.run_engine()
        candidate = row["result"]["candidates"][0]
        self.assertEqual(candidate["display_name"], "unregistered_character")
        self.assertEqual(candidate["model_labels"]["display_name"], "推測した日本語名")

    def test_tagger_revision_invalidates_completed_cache(self):
        self.config["tagger"] = {"threshold": 0.85, "revision": "first"}
        with patch("tagger.CharacterTagger") as constructor:
            constructor.return_value.predict.return_value = {"candidates": [], "threshold": 0.85}
            first = self.run_engine()["run_id"]
            self.config["tagger"]["revision"] = "second"
            self.assertNotEqual(self.run_engine()["run_id"], first)

    def test_completed_is_cached_and_manual_rerun_becomes_latest_cache(self):
        first = self.run_engine()["run_id"]
        self.run_engine()
        self.assertEqual(len(self.client.calls), 2)
        last = self.run_engine(force=True)["run_id"]
        self.assertNotEqual(first, last)
        self.assertEqual(self.run_engine()["run_id"], last)
        self.assertEqual(len(self.client.calls), 4)

    def test_second_stage_error_resumes_from_saved_observation(self):
        self.client.fail_second = True
        row = self.run_engine()
        self.assertEqual(row["status"], "error")
        self.assertIsNotNone(row["observation"])
        self.client.fail_second = False
        row = self.run_engine()
        self.assertEqual(row["status"], "done")
        self.assertEqual([c[0] for c in self.client.calls], ["observe", "identify", "identify"])

    def test_cancellation_saves_observation_and_resumes(self):
        self.client.cancel_after_observe = True
        row = self.run_engine()
        self.assertEqual(row["status"], "paused")
        self.cancel.clear()
        self.client.cancel_after_observe = False
        row = self.run_engine()
        self.assertEqual(row["status"], "done")
        self.assertEqual([c[0] for c in self.client.calls], ["observe", "identify"])

    def test_model_prompt_crop_and_source_invalidate_cache(self):
        self.run_engine()
        self.config["model_digest"] = "new-digest"
        self.run_engine()
        self.config["bundle"]["observe"] += "\nnew rule"
        self.run_engine()
        self.store.set_crop(self.asset_id, [0, 0, 20, 20])
        self.run_engine()
        Image.new("RGB", (40, 60), "blue").save(self.path)
        self.run_engine()
        self.assertEqual(len(self.client.calls), 10)
        self.assertEqual(len(self.store.history(self.asset_id)), 5)

    def test_changed_source_during_inference_not_accepted(self):
        self.client.mutate = lambda: Image.new("RGB", (40, 60), "blue").save(self.path)
        row = self.run_engine()
        self.assertEqual(row["status"], "error")
        self.assertIn("変更", row["error"])
        self.assertIsNone(row["result"])

    def test_broken_image_is_recorded_and_next_image_continues(self):
        self.path.write_bytes(b"not an image")
        other = self.root / "good.png"
        Image.new("RGB", (30, 30), "green").save(other)
        self.store.add([other])
        self.run_engine()
        self.assertEqual([r["status"] for r in self.store.assets()], ["error", "done"])

    def test_target_mode_hides_target_during_observation(self):
        self.config = build_config("fake-vision", "digest", "target", "指定名テスト")
        self.run_engine()
        self.assertNotIn("指定名テスト", self.client.calls[0][1])
        self.assertIn("指定名テスト", self.client.calls[1][1])

    def test_restart_recovery_and_backup(self):
        run_id = self.store.begin(self.asset_id, "pending-test", "sha", None, self.config)
        self.store.close()
        self.store = Store(self.root / "state.sqlite3")
        self.assertEqual(self.store.run(run_id)["status"], "paused")
        backup = self.root / "backup.sqlite3"
        self.store.backup(backup)
        saved = Store(backup)
        try:
            self.assertEqual(Path(saved.asset(self.asset_id)["path"]).resolve(), self.path.resolve())
        finally:
            saved.close()

    def test_backup_cannot_overwrite_hidden_original_image(self):
        before = self.path.read_bytes()
        self.store.hide([self.asset_id])
        with self.assertRaises(ValueError):
            self.store.backup(self.path)
        with self.assertRaises(ValueError):
            self.store.validate_destination(self.path)
        for suffix in ("", "-wal", "-shm"):
            with self.assertRaises(ValueError):
                self.store.validate_destination(str(self.store.path) + suffix)
        self.assertEqual(self.path.read_bytes(), before)

    def test_review_preserves_ai_and_hidden_assets_reappear(self):
        row = self.run_engine()
        original = row["result"]
        self.store.review(row["run_id"], "修正名", "メモ", True)
        self.assertEqual(self.store.run(row["run_id"])["result"], original)
        self.store.hide([self.asset_id])
        self.assertFalse(self.store.assets())
        self.assertTrue(self.path.exists())
        self.store.add([self.path])
        self.assertTrue(self.store.assets()[0]["reviewed"])

    def test_reports_escape_html_csv_and_roundtrip_json(self):
        row = self.run_engine()
        row["result"]["candidates"][0]["name"] = '<script>alert("x")</script>'
        row["review_name"] = '=HYPERLINK("https://example.com")'
        export_html([row], self.root / "report.html")
        html = (self.root / "report.html").read_text(encoding="utf-8")
        self.assertNotIn('<script>alert', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("data:image/jpeg;base64,", html)
        export_csv([row], self.root / "report.csv")
        self.assertIn("'=HYPERLINK", (self.root / "report.csv").read_text(encoding="utf-8-sig"))
        export_json([row], self.root / "report.json")
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["images"][0]["result"], row["result"])

    def test_formula_escapes_whitespace_and_control_prefixes(self):
        for value in ("=SUM(1)", " +cmd", "-2+3", "@name", "\ttext", "\ntext"):
            self.assertTrue(spreadsheet_text(value).startswith("'"))


class ServerFixture:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.do_POST()
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(raw) if raw else None
                fixture.requests.append((self.path, payload))
                fixture.entered.set()
                try:
                    fixture.handler(self, payload, fixture)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *_):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)


def stream(handler, packets):
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.end_headers()
    for packet in packets:
        handler.wfile.write((json.dumps(packet, ensure_ascii=False) + "\n").encode("utf-8"))
        handler.wfile.flush()


class HttpTests(unittest.TestCase):
    def test_chunked_json_contract_and_private_image_payload(self):
        def handler(h, payload, fixture):
            text = json.dumps(observation(), ensure_ascii=False)
            stream(h, [{"message": {"content": text[:17]}, "done": False}, {"message": {"content": text[17:]}, "done": True, "done_reason": "stop"}])
        with ServerFixture(handler) as server:
            client = OllamaClient(server.url)
            config = build_config("fake", "digest")
            result, metrics = client.structured(config, config["bundle"]["observation_schema"], "観察", "BASE64")
            self.assertEqual(result, observation())
            payload = server.requests[0][1]
            self.assertFalse(payload["think"])
            self.assertEqual(payload["messages"][1]["images"], ["BASE64"])
            self.assertIsInstance(payload["format"], dict)
            self.assertEqual(metrics["response_field"], "content")

    def test_complete_json_only_in_alternate_field_is_validated(self):
        def handler(h, payload, fixture):
            stream(h, [{"message": {"thinking": json.dumps(observation()), "content": ""}, "done": True}])
        with ServerFixture(handler) as server:
            config = build_config("fake", "digest")
            result, metrics = OllamaClient(server.url).structured(config, config["bundle"]["observation_schema"], "観察", "BASE64")
            self.assertEqual(result, observation())
            self.assertEqual(metrics["response_field"], "thinking_json")

    def test_alternate_reasoning_text_is_not_extracted_as_json(self):
        def handler(h, payload, fixture):
            stream(h, [{"message": {"thinking": "analysis first " + json.dumps(observation())}, "done": True}])
        with ServerFixture(handler) as server:
            config = build_config("fake", "digest")
            with self.assertRaises(OllamaError):
                OllamaClient(server.url).structured(config, config["bundle"]["observation_schema"], "観察", "BASE64")
            self.assertEqual(len(server.requests), 2)

    def test_invalid_contract_retried_only_once(self):
        def handler(h, payload, fixture):
            stream(h, [{"message": {"content": "{}"}, "done": True}])
        with ServerFixture(handler) as server:
            config = build_config("fake", "digest")
            with self.assertRaises(OllamaError):
                OllamaClient(server.url).structured(config, config["bundle"]["observation_schema"], "観察", "BASE64")
            self.assertEqual(len(server.requests), 2)

    def test_truncated_and_length_terminated_streams_rejected(self):
        for reason in ("missing_done", "length"):
            def handler(h, payload, fixture):
                stream(h, [{"message": {"content": "{}"}, "done": reason == "length", "done_reason": reason}])
            with self.subTest(reason=reason), ServerFixture(handler) as server:
                with self.assertRaises(OllamaError):
                    OllamaClient(server.url).request("/api/chat", {"stream": True})

    def test_cancel_interrupts_waiting_headers(self):
        def handler(h, payload, fixture):
            fixture.release.wait(5)
            stream(h, [{"message": {"content": "{}"}, "done": True}])
        with ServerFixture(handler) as server:
            client = OllamaClient(server.url)
            outcome = []
            def run():
                try:
                    client.request("/api/chat", {"stream": True}, timeout=30)
                except Exception as exc:
                    outcome.append(exc)
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(server.entered.wait(2))
            started = time.monotonic()
            client.cancel()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertLess(time.monotonic() - started, 2)
            self.assertIsInstance(outcome[0], Cancelled)

    def test_deadline_interrupts_no_response(self):
        def handler(h, payload, fixture):
            fixture.release.wait(5)
        with ServerFixture(handler) as server:
            started = time.monotonic()
            cancelled = threading.Event()
            client = OllamaClient(server.url, cancelled)
            with self.assertRaises(OllamaError):
                client.request("/api/chat", {"stream": True}, timeout=0.2)
            self.assertLess(time.monotonic() - started, 2)
            self.assertFalse(cancelled.is_set())

    def test_stream_packets_reset_the_inactivity_limit(self):
        def handler(h, payload, fixture):
            text = json.dumps(observation(), ensure_ascii=False)
            h.send_response(200)
            h.send_header("Content-Type", "application/x-ndjson")
            h.end_headers()
            for index, chunk in enumerate((text[:20], text[20:60], text[60:])):
                h.wfile.write((json.dumps({"message": {"content": chunk}, "done": index == 2}) + "\n").encode())
                h.wfile.flush()
                if index < 2:
                    time.sleep(0.12)
        with ServerFixture(handler) as server:
            config = build_config("fake", "digest", timeout=120)
            config["timeout"] = 0.2
            result, _metrics = OllamaClient(server.url).structured(config, config["bundle"]["observation_schema"], "観察", "BASE64")
            self.assertEqual(result, observation())

    def test_cloud_and_nonvision_models_are_filtered(self):
        def handler(h, payload, fixture):
            data = {"models": [{"name": "local", "digest": "d", "capabilities": ["vision"]}, {"name": "remote-cloud", "capabilities": ["vision"]}, {"name": "remote-proxy", "remote_host": "example.com", "capabilities": ["vision"]}, {"name": "text-only", "capabilities": ["completion"]}]}
            h.send_response(200)
            h.end_headers()
            h.wfile.write(json.dumps(data).encode())
        with ServerFixture(handler) as server:
            self.assertEqual([m["name"] for m in OllamaClient(server.url).models()], ["local"])


if __name__ == "__main__":
    unittest.main()

