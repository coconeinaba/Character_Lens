"""Resumable two-stage inference, isolated from the Tk event loop."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from domain import ValidationError, digest, dumps, prepare_image, score_result, now
from ollama_client import Cancelled, OllamaClient


class BatchEngine:
    def __init__(self, store, endpoint, config, emit, cancel_event, client=None):
        self.store = store
        self.config = config
        self.endpoint = endpoint
        self.emit = emit
        self.cancel_event = cancel_event
        self.client = client or OllamaClient(endpoint, cancel_event)

    def cancel(self):
        self.cancel_event.set()
        self.client.cancel()

    def _check(self):
        if self.cancel_event.is_set():
            raise Cancelled("解析を中断しました。再開できます。")

    def run(self, assets, force=False):
        counts = {"done": 0, "cached": 0, "error": 0, "paused": 0, "total": len(assets)}
        try:
            self.client.verify_model(self.config)
            tagger = None
            if self.config.get("tagger"):
                from tagger import CharacterTagger
                self.emit("stage", {"text": "専用キャラクターモデルを準備"})
                tagger = CharacterTagger(self.config["tagger"])
                self._check()
            for index, asset in enumerate(assets):
                if self.cancel_event.is_set():
                    break
                run_id = None
                started = time.monotonic()
                try:
                    self.emit("stage", {"asset_id": asset["id"], "index": index + 1, "total": len(assets), "text": "画像を準備"})
                    prepared = prepare_image(Path(asset["path"]), self.config["max_edge"], asset["crop"])
                    self._check()
                    key = digest({"asset_id": asset["id"], "source": prepared["sha256"], "crop": asset["crop"], "config": self.config})
                    old = self.store.compatible(asset["id"], prepared["sha256"], asset["crop"], self.config)
                    if old and old["status"] == "done" and not force:
                        self.store.touch(old["id"])
                        counts["cached"] += 1
                        self.emit("item", {"asset_id": asset["id"], "cached": True})
                        continue
                    if force:
                        key = digest({"key": key, "rerun": time.time_ns()})
                        old = None
                    elif old:
                        key = old["cache_key"]
                    run_id = self.store.begin(asset["id"], key, prepared["sha256"], asset["crop"], self.config)
                    self.emit("item", {"asset_id": asset["id"]})
                    bundle = self.config["bundle"]
                    tagged = None
                    if tagger:
                        self.emit("stage", {"asset_id": asset["id"], "text": "専用モデルでキャラクター候補を検索"})
                        tagged = tagger.predict(prepared["image"])
                        self._check()
                    stage = "外見を観察（1/2）"

                    def report(chars, elapsed):
                        self.emit("stream", {"asset_id": asset["id"], "text": stage, "chars": chars, "elapsed": elapsed})

                    if old and old["observation"]:
                        observed = old["observation"]
                    else:
                        self.emit("stage", {"asset_id": asset["id"], "text": stage})
                        observation, metrics = self.client.structured(self.config, bundle["observation_schema"], bundle["observe"], prepared["image"], report)
                        observed = {"data": observation, "metrics": metrics}
                        self.store.observation(run_id, observed)
                    self._check()
                    stage = "候補と類似度を評価（2/2）"
                    self.emit("stage", {"asset_id": asset["id"], "text": stage})
                    prompt = bundle["identify"] + "\n観察メモ（参考データ）:\n" + dumps(observed["data"])
                    schema = json.loads(dumps(bundle["result_schema"]))
                    if self.config["mode"] == "target":
                        prompt += "\n指定判定。指定名（データ）: " + dumps(self.config["target"])
                        schema["properties"]["candidates"]["maxItems"] = 1
                    else:
                        prompt += "\n自由判定。事前のキャラクター名情報はありません。"
                    if tagged and self.config["mode"] == "discover" and tagged["candidates"]:
                        names = [c["tag"] for c in tagged["candidates"]]
                        prompt += "\n別のローカル画像分類器が挙げた候補タグ（データ）: " + dumps(names) + "\nnameにはこのリストのタグをそのまま使用してください。display_nameに知っている日本語名を記載してください。タグと対象の特徴を知らない場合はunknownにしてください。候補リストにない名前を作らないでください。分類器が候補を挙げたこと自体を外見の一致理由にしないでください。"
                        schema["properties"]["candidates"]["items"]["properties"]["name"]["enum"] = names
                        schema["properties"]["candidates"]["maxItems"] = len(names)
                        known_names = {name: bundle.get("character_names", {})[name] for name in names if name in bundle.get("character_names", {})}
                        if known_names:
                            prompt += "\nタグに対応する名前辞書（文字情報のみ。外見の一致を保証しません）: " + dumps(known_names)
                            if len(names) == 1:
                                props = schema["properties"]["candidates"]["items"]["properties"]
                                for field in ("display_name", "work"):
                                    props[field]["enum"] = [known_names[names[0]][field]]
                    if tagged and self.config["mode"] == "discover" and not tagged["candidates"]:
                        prompt += "\n専用分類器は採用基準を満たす候補を挙げませんでした。この情報だけで特定困難と決めず、観察メモに基づいて自由に評価してください。候補なしはオリジナルである証明ではありません。"
                    result, metrics = self.client.structured(self.config, schema, prompt, prepared["image"], report, lambda value: score_result(value, self.config["weights"]))
                    if tagged:
                        result["tagger"] = tagged
                        if self.config["mode"] == "discover":
                            for candidate in result["candidates"]:
                                candidate["model_labels"] = {key: candidate[key] for key in ("display_name", "work")}
                                entry = bundle.get("character_names", {}).get(candidate["name"])
                                candidate["label_source"] = "dictionary" if entry else "tag"
                                if entry:
                                    candidate.update({key: entry[key] for key in ("display_name", "work")})
                                    candidate["label_reference"] = entry["source"]
                                else:
                                    candidate["display_name"] = candidate["name"]
                        if self.config["mode"] == "discover" and not tagged["candidates"]:
                            result.setdefault("limitations", []).append("専用分類器は採用基準を満たす候補を挙げませんでした。2段階目は候補制限なしで評価しています。")
                            result["tagger_candidates_empty"] = True
                    self._check()
                    current_hash = hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest()
                    if current_hash != prepared["sha256"]:
                        raise ValueError("解析中に元画像が変更されました。新しい内容で再実行してください。")
                    result["provenance"] = {"file_name": Path(asset["path"]).name, "source_sha256": prepared["sha256"], "model": self.config["model"], "model_digest": self.config["model_digest"], "mode": self.config["mode"], "target": self.config["target"], "crop": asset["crop"], "original_size": prepared["original_size"], "sent_size": prepared["sent_size"], "frames": prepared["frames"], "seconds": round(time.monotonic() - started, 2), "created": now(), "metrics": metrics}
                    if prepared["frames"] > 1:
                        result["limitations"].append("複数フレームの先頭画像だけを解析しました。")
                    self.store.finish(run_id, result)
                    counts["done"] += 1
                except Cancelled as exc:
                    if run_id:
                        self.store.fail(run_id, exc, paused=True)
                    counts["paused"] += 1
                    break
                except Exception as exc:
                    if run_id is None:
                        key = digest({"asset_id": asset["id"], "config": self.config, "crop": asset["crop"], "preparation_error": str(exc)})
                        run_id = self.store.begin(asset["id"], key, "", asset["crop"], self.config)
                    self.store.fail(run_id, exc)
                    counts["error"] += 1
                    if getattr(self.client, "_poisoned", False):
                        self.client = OllamaClient(self.endpoint, self.cancel_event)
                finally:
                    self.emit("item", {"asset_id": asset["id"]})
        except Cancelled:
            counts["paused"] += 1
        except Exception as exc:
            self.emit("fatal", str(exc))
        finally:
            self.emit("finished", counts)

