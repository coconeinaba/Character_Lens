"""Optional local WD ViT tagger. Inference never downloads or sends images."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

from PIL import Image

from domain import ROOT

MODEL_DIR = ROOT / "models" / "wd-vit-tagger-v3"
REPO = "SmilingWolf/wd-vit-tagger-v3"


def availability():
    if not (MODEL_DIR / "model.onnx").is_file() or not (MODEL_DIR / "selected_tags.csv").is_file() or not (MODEL_DIR / "manifest.json").is_file():
        return False, "専用モデル未配置"
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("numpy") is None:
        return False, "ONNX Runtime / NumPyが必要です"
    return True, "WDキャラクター判定を使用"


def signature():
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {"repo": REPO, "revision": manifest["revision"], "files": {name: value["sha256"] for name, value in manifest["files"].items() if name in ("model.onnx", "selected_tags.csv")}, "threshold": 0.85, "max_candidates": 3, "preprocessing": "square-white-bgr-bicubic-v1"}


class CharacterTagger:
    def __init__(self, expected):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        self.expected = expected
        for filename, sha in expected["files"].items():
            with (MODEL_DIR / filename).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != sha:
                raise ValueError(f"専用モデルのファイルが変更されています: {filename}")
        with (MODEL_DIR / "selected_tags.csv").open(encoding="utf-8", newline="") as source:
            self.labels = list(csv.DictReader(source))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.edge = int(self.input.shape[1])

    def predict(self, encoded_image):
        with Image.open(io.BytesIO(base64.b64decode(encoded_image))) as source:
            image = source.convert("RGB")
        edge = max(image.size)
        padded = Image.new("RGB", (edge, edge), "white")
        padded.paste(image, ((edge - image.width) // 2, (edge - image.height) // 2))
        resized = padded.resize((self.edge, self.edge), Image.Resampling.BICUBIC)
        pixels = self.np.asarray(resized, dtype=self.np.float32)[:, :, ::-1].copy()[None, ...]
        values = self.session.run(None, {self.input.name: pixels})[0][0]
        if len(values) != len(self.labels):
            raise ValueError("専用モデルとタグ一覧の要素数が一致しません。")
        scores = [{"tag": row["name"], "score": round(float(values[i]), 6)} for i, row in enumerate(self.labels) if row["category"] == "4"]
        scores.sort(key=lambda item: item["score"], reverse=True)
        selected = [r for r in scores if r["score"] >= self.expected["threshold"]][:self.expected["max_candidates"]]
        return {"model": REPO, "candidates": selected, "top_below_threshold": scores[:3] if not selected else [], "threshold": self.expected["threshold"], "note": "スコアは分類器の出力です。見た目の類似度・校正済み確率ではありません。対応タグがないキャラクターは検出できません。"}

