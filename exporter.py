"""Self-contained, escaped reports and spreadsheet-safe CSV."""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import os
import tempfile
from pathlib import Path

from domain import AXES, CONFIDENCE_LABELS, IDENTIFICATION_LABELS, STATUS_LABELS, VERSION, candidate_label, display_image, now


def atomic_text(destination, text, encoding="utf-8"):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=".character-lens-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def spreadsheet_text(value):
    text = str(value if value is not None else "")
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        text = "'" + text
    return text


def tagger_summary(result):
    tagged = result.get("tagger")
    if not tagged:
        return "専用モデル未使用"
    tags = " / ".join(f"{c['tag']} ({c['score']:.3f})" for c in tagged["candidates"])
    return f"{tags or '採用候補なし'} / 採用基準 {tagged['threshold']:.2f}。分類出力は類似度・正答確率ではありません。"


def export_json(rows, destination):
    atomic_text(destination, json.dumps({"application": "Character Lens", "version": VERSION, "exported": now(), "notice": "公式画像との照合なし。点数は暫定の推定類似度です。", "images": rows}, ensure_ascii=False, indent=2, allow_nan=False))


def export_csv(rows, destination):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["ファイル名", "元画像パス", "状態", "候補順位", "キャラクター候補", "作品", "推定類似度（確率ではない）", "評価できた重み（%）", "確かさ（自己申告）", "一致点", "相違点", "理由", "モデル", "人の修正名", "確認済み", "確認メモ", "エラー", "専用モデルの候補と分類出力"])
    for row in rows:
        result = row.get("result") or {}
        candidates = result.get("candidates") or [{}]
        for rank, candidate in enumerate(candidates, 1):
            values = [Path(row["path"]).name, row["path"], STATUS_LABELS.get(row.get("status"), "未処理"), rank if candidate else "", candidate_label(candidate), candidate.get("work", ""), candidate.get("estimated_similarity"), candidate.get("coverage"), CONFIDENCE_LABELS.get(candidate.get("confidence"), ""), " / ".join(candidate.get("matches", [])), " / ".join(candidate.get("differences", [])), candidate.get("rationale", result.get("overall_reason", "")), (row.get("config") or {}).get("model", ""), row.get("review_name"), "済" if row.get("reviewed") else "", row.get("review_note"), row.get("error")]
            writer.writerow([spreadsheet_text(value) for value in values + [tagger_summary(result)]])
    atomic_text(destination, output.getvalue(), "utf-8-sig")


def export_html(rows, destination):
    esc = lambda value: html.escape(str(value if value is not None else ""), quote=True)
    cards = []
    for row in rows:
        picture = "<p>画像を読み込めません</p>"
        try:
            current_sha = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            if row.get("source_sha") and row["source_sha"] != current_sha:
                picture = "<p>元画像が解析時から変更されたため画像表示を省略しました。</p>"
            else:
                buffer = io.BytesIO()
                display_image(row["path"], (360, 420), row.get("run_crop")).save(buffer, "JPEG", quality=80)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                picture = f'<img src="data:image/jpeg;base64,{encoded}" alt="解析対象画像">'
        except (OSError, ValueError):
            pass
        result = row.get("result") or {}
        details = []
        for candidate in result.get("candidates", []):
            axes = "".join(f'<tr><td>{esc(AXES[a["axis"]])}</td><td>{esc(a["score"] if a["score"] is not None else "評価不能")}</td><td>{esc(a["reason"])}</td></tr>' for a in candidate["axes"])
            score = candidate.get("estimated_similarity")
            details.append(f'<section><h3>{esc(candidate_label(candidate))} <small>{esc(candidate["work"])}</small></h3><p>推定類似度: <strong>{esc(score if score is not None else "評価不能")}</strong> / 100　評価範囲: {candidate["coverage"]}%　確かさ: {esc(CONFIDENCE_LABELS[candidate["confidence"]])}</p><p>{esc(candidate["rationale"])}</p><p>一致点: {esc(" / ".join(candidate["matches"]))}</p><p>相違点: {esc(" / ".join(candidate["differences"]))}</p><table><tr><th>特徴</th><th>点数</th><th>理由</th></tr>{axes}</table></section>')
        details.append(f'<p class="muted">専用モデル: {esc(tagger_summary(result))}</p>')
        cards.append(f'<article><div>{picture}</div><div><h2>{esc(Path(row["path"]).name)}</h2><p>{esc(STATUS_LABELS.get(row.get("status"), "未処理"))} / {esc(IDENTIFICATION_LABELS.get(result.get("identification"), ""))}</p><p>{esc(result.get("overall_reason", ""))}</p>{"".join(details)}<p class="muted">制約: {esc(" / ".join(result.get("limitations", [])))}</p><p>人の修正名: {esc(row.get("review_name"))}　確認済み: {"はい" if row.get("reviewed") else "いいえ"}<br>{esc(row.get("review_note"))}</p><p class="muted">モデル: {esc((row.get("config") or {}).get("model", ""))}<br>{esc(row.get("error", ""))}</p></div></article>')
    document = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'"><title>Character Lens 解析レポート</title><style>body{font:15px/1.7 system-ui,sans-serif;background:#eef2f6;color:#17263b;margin:0;padding:30px;max-width:1350px;margin:auto}h1{margin:0}h2{font-size:19px;overflow-wrap:anywhere}h3{font-size:18px}small,.muted{color:#52637a}article{display:grid;grid-template-columns:360px 1fr;gap:28px;background:white;padding:24px;margin:24px 0;border-radius:14px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dde4ed;padding:8px;text-align:left;vertical-align:top}section{margin-bottom:26px}p{overflow-wrap:anywhere}@media(max-width:800px){article{grid-template-columns:1fr}body{padding:12px}}</style><h1>Character Lens</h1><p>公式画像との照合なし。点数はモデルの知識に基づく暫定評価で、正答確率や公式画像との一致率ではありません。</p><p>作成日時: ' + esc(now()) + '</p>' + "".join(cards) + '</html>'
    atomic_text(destination, document)

