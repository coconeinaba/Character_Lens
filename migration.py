"""Standard-library-only preparation, verified downloads and history relocation."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_FOLDER = Path("models") / "wd-vit-tagger-v3"
LOCAL_OLLAMA = "http://127.0.0.1:11434"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_profile(root=ROOT):
    profile = json.loads((Path(root) / "migration_profile.json").read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("移行情報の形式に対応していません。")
    model = profile["ollama"]
    if not re.fullmatch(r"[a-zA-Z0-9_.:/-]+", model["model"]) or "cloud" in model["model"].lower():
        raise ValueError("移行するローカルモデルの名前が不正です。")
    if not re.fullmatch(r"[a-f0-9]{64}", model["digest"]):
        raise ValueError("モデル識別子が不正です。")
    return profile


def package_status(profile):
    rows = []
    for entry in profile["packages"]:
        try:
            version = importlib.metadata.version(entry["distribution"])
            installed = importlib.util.find_spec(entry["module"]) is not None
        except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
            installed, version = False, "未導入"
        rows.append({**entry, "installed": installed, "actual": version, "matches": installed and version_at_least(version, entry["version"])})
    return rows


def version_at_least(actual, required):
    """Compare the numeric package prefixes without importing optional packages."""
    def numbers(value):
        found = re.findall(r"\d+", str(value))
        return tuple(int(part) for part in found) if found else ()
    actual, required = numbers(actual), numbers(required)
    return bool(actual) and actual + (0,) * max(0, len(required) - len(actual)) >= required + (0,) * max(0, len(actual) - len(required))


def python_console():
    path = Path(sys.executable)
    console = path.with_name("python.exe")
    return str(console if path.name.lower() == "pythonw.exe" and console.exists() else path)


def hidden_process_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def install_packages(root=ROOT, progress=lambda text: None):
    # Uses this interpreter; no shell, virtual environment or batch launcher.
    command = [python_console(), "-m", "pip", "install", "--user", "--only-binary=:all:", "--disable-pip-version-check", "--no-input", "--retries", "2", "--timeout", "30", "--index-url", "https://pypi.org/simple", "-r", str(Path(root) / "requirements-migration.txt")]
    progress("PyPIから必要なライブラリを取得します。既存の別バージョンのPythonは削除しません。")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **hidden_process_options())
    tail = []
    for raw in iter(process.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").strip()
        tail = (tail + [line])[-12:]
        progress(line[:800])
    code = process.wait()
    if code:
        raise RuntimeError("ライブラリの導入に失敗しました。\n" + "\n".join(tail))
    import site
    site.addsitedir(site.getusersitepackages())
    importlib.invalidate_caches()


def probe_packages():
    result = subprocess.run([python_console(), "-c", "import PIL, numpy, onnxruntime; print('OK')"], capture_output=True, timeout=60, **hidden_process_options())
    if result.returncode:
        raise RuntimeError("ライブラリを読み込めません。\n" + result.stderr.decode("utf-8", "replace")[-1500:])


def model_manifest(root=ROOT):
    directory = Path(root) / MODEL_FOLDER
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if value.get("repo") != "SmilingWolf/wd-vit-tagger-v3" or not re.fullmatch(r"[a-f0-9]{40}", value.get("revision", "")):
        raise ValueError("専用モデルの配布元または版が不正です。")
    required = {"model.onnx", "selected_tags.csv", "README.md"}
    if set(value["files"]) != required:
        raise ValueError("専用モデルのファイル一覧が不正です。")
    for item in value["files"].values():
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")) or type(item.get("size")) is not int or not 0 < item["size"] <= 600_000_000:
            raise ValueError("専用モデルの検証情報が不正です。")
    return directory, value


def matches_file(path, expected):
    return Path(path).is_file() and Path(path).stat().st_size == expected["size"] and sha256(path) == expected["sha256"]


def check_tagger(root=ROOT):
    directory, manifest = model_manifest(root)
    return all(matches_file(directory / name, item) for name, item in manifest["files"].items())


def download_verified(url, destination, expected, progress=lambda text: None, opener=None):
    """Only publish complete, hash-verified bytes; keep an old file on failure."""
    destination = Path(destination)
    if matches_file(destination, expected):
        progress(destination.name + " は検証済みのため再利用します。")
        return False
    if not url.startswith("https://"):
        raise ValueError("ダウンロード元にはHTTPSが必要です。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < expected["size"] + 32 * 1024 * 1024:
        raise RuntimeError("専用モデルの取得に必要なディスク空き容量がありません。")
    fd, temporary = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=destination.parent)
    opener = opener or urllib.request.urlopen
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CharacterLens-Migration/1.0"})
        with os.fdopen(fd, "wb") as output:
            with opener(request, timeout=60) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("安全でないダウンロード先への転送を拒否しました。")
                received, last_update = 0, 0.0
                hasher = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > expected["size"]:
                        raise ValueError("受信サイズが記録された値を超えました。")
                    output.write(block)
                    hasher.update(block)
                    if time.monotonic() - last_update > 0.4:
                        progress(f"{destination.name}: {received / 1024**2:.1f} / {expected['size'] / 1024**2:.1f} MB")
                        last_update = time.monotonic()
                if received != expected["size"] or hasher.hexdigest() != expected["sha256"]:
                    raise ValueError("ダウンロードのサイズまたはSHA-256が一致しません。再取得してください。")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        progress(destination.name + " の取得とSHA-256検証が完了しました。")
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def download_tagger(root=ROOT, progress=lambda text: None):
    directory, manifest = model_manifest(root)
    for name, item in manifest["files"].items():
        url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}?download=true"
        download_verified(url, directory / name, item, progress)


def local_request(route, payload=None, timeout=15):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(LOCAL_OLLAMA + route, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def check_ollama(profile):
    with local_request("/api/tags") as response:
        data = response.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError("Ollamaのモデル一覧が大きすぎます。")
    wanted = profile["ollama"]
    model = next((r for r in json.loads(data).get("models", []) if r.get("name") == wanted["model"]), None)
    if model is None:
        return {"installed": False, "matches": False, "actual": "モデル未取得"}
    if model.get("remote_host") or model.get("remote_model"):
        raise ValueError("クラウドモデルは移行対象にできません。")
    with local_request("/api/show", {"model": wanted["model"]}) as response:
        info = json.loads(response.read(4_000_000))
    if "vision" not in info.get("capabilities", []) or info.get("remote_host") or info.get("remote_model"):
        raise ValueError("選択モデルは画像対応のローカルモデルではありません。")
    return {"installed": True, "matches": model.get("digest") == wanted["digest"], "actual": model.get("digest", "不明")}


def pull_ollama(profile, progress=lambda text: None):
    wanted = profile["ollama"]["model"]
    existing = check_ollama(profile)
    if existing["installed"]:
        progress("既存モデルを再利用します。" + ("移行元と同じ版です。" if existing["matches"] else "版が異なるため、上書きせずに保持します。版違いの確認欄を確認してください。"))
        return
    progress(f"Ollamaで {wanted} を取得します。数GB以上をダウンロードする場合があります。")
    # Ollama handles layer reuse when this operation is retried.
    with local_request("/api/pull", {"model": wanted, "stream": True}, timeout=120) as response:
        finished, last_update = False, 0.0
        while line := response.readline(1_000_001):
            if len(line) > 1_000_000:
                raise ValueError("Ollamaの応答が上限を超えました。")
            packet = json.loads(line)
            if packet.get("error"):
                raise RuntimeError(str(packet["error"]))
            if time.monotonic() - last_update > 0.4 or packet.get("status") == "success":
                text = packet.get("status", "取得中")
                if packet.get("total"):
                    text += f"  {packet.get('completed', 0) / 1024**3:.2f} / {packet['total'] / 1024**3:.2f} GB"
                progress(text)
                last_update = time.monotonic()
            if packet.get("status") == "success":
                finished = True
                break
    if not finished:
        raise RuntimeError("モデルの取得が途中で終了しました。同じボタンで再試行できます。")
    status = check_ollama(profile)
    if not status["installed"]:
        raise RuntimeError("取得後に画像モデルを確認できませんでした。")
    progress("モデルを取得しました。" + ("移行元と同じ版です。" if status["matches"] else "移行元と版が異なります。確認画面を確認してください。"))


@contextlib.contextmanager
def history_lock(data_dir):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".instance.lock").open("a+b")
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("本体アプリが起動中です。本体を終了してから履歴の操作を行ってください。") from None
    try:
        yield
    finally:
        handle.close()


def backup_sqlite(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("バックアップ先には未使用のファイル名を指定してください。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original:
        with contextlib.closing(sqlite3.connect(destination)) as target:
            original.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("履歴データの整合性検査に失敗しました。")


def restore_history(root, data_dir):
    source = Path(root) / "migration" / "history.sqlite3"
    manifest = json.loads((Path(root) / "migration" / "history_manifest.json").read_text(encoding="utf-8"))
    if not matches_file(source, manifest):
        raise ValueError("同梱履歴のサイズまたはSHA-256が一致しません。")
    destination = Path(data_dir) / "character_lens.sqlite3"
    with history_lock(data_dir):
        if destination.exists():
            raise ValueError("この保存先には履歴が既にあります。既存の履歴には上書きしません。")
        temporary = destination.with_name(".restore-" + str(time.time_ns()) + ".sqlite3")
        try:
            backup_sqlite(source, temporary)
            with contextlib.closing(sqlite3.connect(temporary)) as db:
                if not {"assets", "runs", "settings"} <= {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
                    raise ValueError("Character Lensの履歴ではありません。")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def relink_images(data_dir, folder, progress=lambda text: None):
    """Relink only hash-matching images, with a pre-change SQLite backup."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("画像フォルダーがありません。")
    with history_lock(data_dir):
        database = Path(data_dir) / "character_lens.sqlite3"
        if not database.exists():
            raise ValueError("先に履歴を復元してください。")
        with contextlib.closing(sqlite3.connect(database)) as db:
            rows = db.execute("SELECT a.id,a.path,(SELECT source_sha FROM runs WHERE asset_id=a.id AND source_sha!='' ORDER BY updated DESC,id DESC LIMIT 1) FROM assets a").fetchall()
            wanted = {Path(r[1]).name.casefold() for r in rows}
            index = {}
            for count, path in enumerate(folder.rglob("*"), 1):
                if count > 100_000:
                    raise ValueError("フォルダーが大きすぎます。対象画像を含む小さいフォルダーを選択してください。")
                if path.is_file() and path.name.casefold() in wanted:
                    index.setdefault(path.name.casefold(), []).append(path)
                if count % 250 == 0:
                    progress(f"画像を検索中: {count}項目")
            used = {os.path.normcase(str(Path(r[1]).resolve())): r[0] for r in rows}
            changed, unresolved, unchanged = [], [], 0
            for asset_id, old, expected in rows:
                old_path = Path(old)
                if expected and old_path.is_file() and sha256(old_path) == expected:
                    unchanged += 1
                    continue
                matches = [p for p in index.get(old_path.name.casefold(), []) if expected and sha256(p) == expected]
                if len(matches) != 1:
                    unresolved.append({"file": old_path.name, "reason": "同名・同内容の画像が1件に定まりません" if expected else "解析時のSHA-256が未記録"})
                    continue
                new = str(matches[0].resolve())
                key = os.path.normcase(new)
                if key in used and used[key] != asset_id:
                    unresolved.append({"file": old_path.name, "reason": "移動先画像が既に別の項目に登録されています"})
                    continue
                used[key] = asset_id
                changed.append((new, key, asset_id))
            backup = None
            if changed:
                backup = Path(data_dir) / "backups" / f"before_relink_{time.time_ns()}.sqlite3"
                backup_sqlite(database, backup)
                with db:
                    db.executemany("UPDATE assets SET path=?,path_key=? WHERE id=?", changed)
            return {"changed": len(changed), "unchanged": unchanged, "unresolved": unresolved, "backup": str(backup) if backup else None}

