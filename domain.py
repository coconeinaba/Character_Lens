"""Character Lens contracts, image preparation and deterministic scoring."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
AXES = {"hair": "髪", "face": "顔・目", "costume": "衣装", "distinctive": "固有の目印", "colors": "配色", "body": "体形・種族的特徴"}
WEIGHTS = {"hair": 20, "face": 20, "costume": 15, "distinctive": 30, "colors": 5, "body": 10}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
STATUS_LABELS = {"pending": "未処理", "running": "解析中", "done": "完了", "error": "失敗", "paused": "中断"}
CONFIDENCE_LABELS = {"high": "高（モデル推定）", "medium": "中（モデル推定）", "low": "低（モデル推定）"}
IDENTIFICATION_LABELS = {"candidate": "候補あり", "ambiguous": "複数候補で曖昧", "unknown": "特定困難", "no_character": "キャラクターを確認できず"}


class ValidationError(ValueError):
    pass


def candidate_label(candidate):
    canonical = candidate.get("name", "")
    translated = candidate.get("display_name", "")
    return f"{translated} ({canonical})" if translated and translated != canonical else canonical


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def contract_bundle() -> dict:
    bundle = {name: (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8") for name in ("system", "observe", "identify")}
    bundle["observation_schema"] = json.loads((ROOT / "schemas" / "observation.json").read_text(encoding="utf-8"))
    bundle["result_schema"] = json.loads((ROOT / "schemas" / "result.json").read_text(encoding="utf-8"))
    bundle["character_names"] = json.loads((ROOT / "character_names.json").read_text(encoding="utf-8"))
    validate_schema(bundle["character_names"], {"type": "object"})
    for tag, entry in bundle["character_names"].items():
        validate_schema(entry, {"type": "object", "required": ["display_name", "work", "source"], "additionalProperties": False, "properties": {key: {"type": "string", "minLength": 1, "maxLength": 500} for key in ("display_name", "work", "source")}})
    # Reject malformed edits before any image is sent.
    for key in ("observation_schema", "result_schema"):
        if not isinstance(bundle[key], dict) or bundle[key].get("type") != "object":
            raise ValidationError(f"{key}: JSON Schemaの最上位はobjectにしてください。")
    return bundle


def validate_schema(value, schema: dict, path: str = "$", depth: int = 0) -> None:
    """Validate the deliberately small JSON Schema vocabulary shipped with this app."""
    if depth > 30:
        raise ValidationError("JSONの階層が深すぎます。")
    kind = schema.get("type")
    if isinstance(kind, list):
        for option in kind:
            try:
                validate_schema(value, {**schema, "type": option}, path, depth + 1)
                return
            except ValidationError:
                pass
        raise ValidationError(f"{path}: 型が一致しません。")
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str), "integer": type(value) is int, "number": type(value) in (int, float) and math.isfinite(value), "boolean": type(value) is bool, "null": value is None}
    if kind not in matches or not matches[kind]:
        raise ValidationError(f"{path}: {kind}型が必要です。")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: 許可された値ではありません。")
    if kind == "object":
        props = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValidationError(f"{path}: 必須項目がありません: {', '.join(sorted(missing))}")
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ValidationError(f"{path}: 未定義の項目があります。")
        for key in value.keys() & props.keys():
            validate_schema(value[key], props[key], f"{path}.{key}", depth + 1)
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000):
            raise ValidationError(f"{path}: 要素数が範囲外です。")
        for i, item in enumerate(value):
            validate_schema(item, schema["items"], f"{path}[{i}]", depth + 1)
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 10000):
            raise ValidationError(f"{path}: 文字数が範囲外です。")
    elif kind in ("integer", "number"):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValidationError(f"{path}: 数値が範囲外です。")


def parse_reply(text: str, schema: dict) -> dict:
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValidationError(f"JSONのキーが重複しています: {key}")
            obj[key] = value
        return obj
    try:
        result = json.loads(text, object_pairs_hook=unique, parse_constant=lambda s: (_ for _ in ()).throw(ValidationError(f"不正な数値: {s}")))
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"JSONを読み取れません: {exc}") from exc
    validate_schema(result, schema)
    return result


def validate_weights(weights: dict) -> dict:
    if not isinstance(weights, dict) or set(weights) != set(AXES):
        raise ValueError("重みは6つの評価軸をすべて指定してください。")
    result = {}
    for key in AXES:
        value = weights[key]
        if type(value) is not int or not 0 <= value <= 100:
            raise ValueError("重みは0から100の整数で指定してください。")
        result[key] = value
    if not sum(result.values()):
        raise ValueError("少なくとも1つの重みを1以上にしてください。")
    return result


def score_result(result: dict, weights=None) -> dict:
    """Scores are feature ratings, never probabilities; missing evidence remains unknown."""
    result = json.loads(dumps(result))
    # A reference-free run must never claim that an official reference was inspected.
    narrative = dumps(result)
    if re.search(r"(?:公式|参照)(?:の)?画像(?:と比較|では|の(?:髪|顔|傷|衣装|色|背景|ドレス|王冠)|で確認)", narrative):
        raise ValidationError("提供されていない公式画像を比較したという説明が含まれています。")
    weights = validate_weights(weights or WEIGHTS)
    minimum_coverage = math.ceil(sum(weights.values()) / 2)
    candidates = result["candidates"]
    if result["identification"] in ("unknown", "no_character") and candidates:
        raise ValidationError("特定困難・人物なしの結果に候補が含まれています。")
    if result["identification"] in ("candidate", "ambiguous") and not candidates:
        raise ValidationError("候補ありの結果に候補がありません。")
    names = set()
    for candidate in candidates:
        if not candidate["name"].strip():
            raise ValidationError("キャラクター名が空です。")
        if candidate["work"].strip().casefold() in ("", "不明", "未知", "unknown", "不詳", "不確か"):
            raise ValidationError("出典作品を挙げられない候補があります。知識不足の場合はunknownにしてください。")
        key = (candidate["name"].casefold().strip(), candidate["work"].casefold().strip())
        if key in names:
            raise ValidationError("同じ候補が重複しています。")
        names.add(key)
        axes = candidate["axes"]
        if {a["axis"] for a in axes} != set(AXES) or len(axes) != len(AXES):
            raise ValidationError("評価軸は6種類を1回ずつ含めてください。")
        total = weighted = 0
        for axis in axes:
            if axis["score"] is not None:
                weight = weights[axis["axis"]]
                weighted += axis["score"] * weight
                total += weight
        candidate["coverage"] = round(total * 100 / sum(weights.values()))
        candidate["estimated_similarity"] = int(weighted / total + 0.5) if total >= minimum_coverage else None
        candidate["score_note"] = "暫定の特徴評価。公式画像との実測値・正答確率ではありません。"
    result["scoring_version"] = "features-v2"
    result["weights"] = weights
    result["reference_basis"] = "モデルが学習したキャラクター像。公式画像との照合なし。"
    return result


def prepare_image(path: Path, max_edge: int = 1280, crop: list | None = None) -> dict:
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("画像が100MBを超えています。小さいコピーを選択してください。")
    raw = path.read_bytes()
    if len(raw) > 100 * 1024 * 1024:
        raise ValueError("画像が100MBを超えています。小さいコピーを選択してください。")
    fingerprint = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 50_000_000:
            raise ValueError("画像が5000万画素を超えています。小さいコピーを選択してください。")
        frames = getattr(source, "n_frames", 1)
        source.seek(0)
        oriented = ImageOps.exif_transpose(source)
        size = oriented.size
        rgba = oriented.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")
        canvas.alpha_composite(rgba)
        rgb = canvas.convert("RGB")
    if crop is not None:
        if len(crop) != 4 or not all(type(v) is int for v in crop):
            raise ValueError("切り抜き範囲が不正です。")
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= rgb.width and 0 <= y0 < y1 <= rgb.height):
            raise ValueError("画像サイズと切り抜き範囲が一致しません。範囲を設定し直してください。")
        rgb = rgb.crop(crop)
    rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    rgb.save(output, "JPEG", quality=92)
    return {"image": base64.b64encode(output.getvalue()).decode("ascii"), "sha256": fingerprint, "original_size": list(size), "sent_size": list(rgb.size), "frames": frames}


def display_image(path: str | Path, size: tuple[int, int], crop=None) -> Image.Image:
    with Image.open(path) as source:
        if source.width * source.height > 50_000_000:
            raise ValueError("画像が5000万画素を超えています。")
        im = ImageOps.exif_transpose(source).convert("RGBA")
    if crop:
        im = im.crop(crop)
    background = Image.new("RGBA", im.size, "white")
    background.alpha_composite(im)
    im = background.convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    return im


def build_config(model: str, model_digest: str, mode="discover", target="", max_edge=1280, timeout=300, weights=None) -> dict:
    if mode not in ("discover", "target"):
        raise ValueError("判定モードが不正です。")
    if mode == "target" and not target.strip():
        raise ValueError("指定するキャラクター名を入力してください。")
    return {"app_version": VERSION, "model": model, "model_digest": model_digest, "mode": mode, "target": target.strip() if mode == "target" else "", "max_edge": int(max_edge), "timeout": int(timeout), "num_ctx": 16384, "num_predict": 5500, "temperature": 0, "seed": 42, "bundle": contract_bundle(), "weights": validate_weights(weights or WEIGHTS)}

