"""Ollama HTTP client with cancellable, inactivity-bounded streaming."""
from __future__ import annotations

import http.client
import ipaddress
import json
import queue
import socket
import threading
import time
from urllib.parse import urlsplit

from domain import ValidationError, parse_reply


class Cancelled(Exception):
    pass


class OllamaError(RuntimeError):
    pass


class OutputLengthError(OllamaError):
    """The model stopped because the generation limit was reached."""
    pass


def local_endpoint(url: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(url.strip())
        host, port = parsed.hostname, parsed.port or 11434
        valid = host == "localhost" or bool(host and ipaddress.ip_address(host))
    except ValueError:
        valid = False
    if not valid or parsed.scheme != "http" or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("接続先はHTTPのIPアドレスまたはlocalhostだけを指定してください。例: http://192.168.1.20:11434")
    return ("127.0.0.1" if host == "localhost" else host), port


class OllamaClient:
    def __init__(self, endpoint="http://127.0.0.1:11434", cancel_event=None):
        self.host, self.port = local_endpoint(endpoint)
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._lock = threading.Lock()
        self._socket = None

    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled("解析を中断しました。未完了の画像は再開できます。")

    def _disconnect(self):
        with self._lock:
            sock = self._socket
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self):
        self.cancel_event.set()
        self._disconnect()

    def request(self, route, payload=None, timeout=15, progress=None):
        """Keep cancellation responsive even if Windows is blocked inside socket.recv."""
        self._check_cancel()
        if getattr(self, "_poisoned", False):
            raise OllamaError("タイムアウトした接続は再利用できません。再接続してください。")
        responses = queue.Queue(maxsize=1)
        def transport():
            try:
                responses.put((True, self._request(route, payload, timeout, progress)))
            except Exception as exc:
                responses.put((False, exc))
        thread = threading.Thread(target=transport, daemon=True)
        thread.start()
        self._last_activity = time.monotonic()
        while True:
            self._check_cancel()
            if time.monotonic() - self._last_activity >= timeout:
                # Do not reuse this client while a timed-out transport unwinds.
                self._poisoned = True
                self._disconnect()
                raise OllamaError(f"応答が{timeout}秒間止まりました。待ち時間を増やすか小さいモデルで再実行してください。")
            try:
                succeeded, result = responses.get(timeout=0.05)
            except queue.Empty:
                continue
            if succeeded:
                return result
            raise result

    def _request(self, route, payload=None, timeout=15, progress=None):
        self._check_cancel()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        started = time.monotonic()
        try:
            conn.connect()
            with self._lock:
                self._socket = conn.sock
            self._check_cancel()
            conn.sock.settimeout(timeout)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
            conn.request("POST" if body is not None else "GET", route, body, {"Content-Type": "application/json", "Accept": "application/json"})
            response = conn.getresponse()
            self._last_activity = time.monotonic()
            if not 200 <= response.status < 300:
                message = response.read(8192).decode("utf-8", "replace")
                raise OllamaError(f"Ollama HTTP {response.status}: {message[:600]}")
            if not payload or not payload.get("stream"):
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise OllamaError("Ollamaの応答が大きすぎます。")
                self._check_cancel()
                return json.loads(raw)
            chunks = []
            thinking_chunks = []
            chars = 0
            done = None
            last_progress = 0.0
            while True:
                self._check_cancel()
                line = response.readline(1_000_001)
                if not line:
                    break
                if len(line) > 1_000_000:
                    raise OllamaError("Ollamaの1応答行が上限を超えました。")
                if not line.strip():
                    continue
                self._last_activity = time.monotonic()
                packet = json.loads(line)
                if packet.get("error"):
                    raise OllamaError(str(packet["error"]))
                content = packet.get("message", {}).get("content", "")
                thinking = packet.get("message", {}).get("thinking", "")
                if not isinstance(content, str) or not isinstance(thinking, str):
                    raise OllamaError("Ollamaの応答形式が不正です。")
                chunks.append(content)
                thinking_chunks.append(thinking)
                chars += len(content) + len(thinking)
                if chars > 1_000_000:
                    raise OllamaError("出力が安全上限を超えました。候補数や説明を短くして再実行してください。")
                current = time.monotonic()
                if progress and current - last_progress >= 0.4:
                    progress(chars, current - started)
                    last_progress = current
                if packet.get("done"):
                    done = packet
                    break
            self._check_cancel()
            if not done:
                raise OllamaError("応答が途中で切れました。未完成の結果は採用しません。")
            if done.get("done_reason") == "length":
                raise OutputLengthError("出力が長さ上限で終了しました。")
            return {"text": "".join(chunks), "alternate_text": "".join(thinking_chunks), "metrics": {k: done.get(k) for k in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count")}}
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self._check_cancel()
            if isinstance(exc, (TimeoutError, socket.timeout)):
                raise OllamaError(f"応答が{timeout}秒間止まりました。待ち時間を増やすか小さいモデルで再実行してください。") from exc
            raise OllamaError(f"Ollamaとの通信に失敗しました: {exc}") from exc
        finally:
            with self._lock:
                self._socket = None
            conn.close()

    def models(self):
        rows = self.request("/api/tags").get("models", [])
        result = []
        for row in rows:
            name = row.get("name", "")
            if "cloud" in name.lower() or row.get("remote_host") or row.get("remote_model"):
                continue
            info = row
            caps = row.get("capabilities")
            if not caps:
                info = self.request("/api/show", {"model": name})
                caps = info.get("capabilities", [])
            if "vision" in caps and not info.get("remote_host") and not info.get("remote_model"):
                result.append({"name": name, "digest": row.get("digest", ""), "size": row.get("size", 0), "capabilities": caps})
        return result

    def verify_model(self, config):
        models = self.models()
        match = next((m for m in models if m["name"] == config["model"]), None)
        if not match:
            raise OllamaError("指定モデルはこのPCの画像対応モデル一覧にありません。接続確認で選び直してください。")
        if match["digest"] != config["model_digest"]:
            raise OllamaError("モデルが保存時から更新されています。新しい解析を作成してください。")

    def structured(self, config, schema, prompt, encoded_image, progress=None, validator=None):
        messages = [{"role": "system", "content": config["bundle"]["system"]}, {"role": "user", "content": prompt, "images": [encoded_image]}]
        errors = []
        last_error = None
        max_attempts = 4
        for attempt in range(max_attempts):
            self._check_cancel()
            request_messages = [dict(m) for m in messages]
            if attempt:
                if isinstance(last_error, OutputLengthError):
                    request_messages[-1]["content"] += "\n前回の回答は出力上限で途中終了しました。画像内で見える各キャラクターを省略せず、各説明・理由を短くして、指定SchemaのJSONを最後まで出力してください。"
                else:
                    request_messages[-1]["content"] += "\n前回の回答は形式検査に失敗しました。説明文やMarkdownを付けず、Schemaの型・必須項目を守り、評価軸は6種類を重複なく出力してください。同じキャラクターは1件にまとめてください。"
            base_context = max(1024, int(config.get("num_ctx", 32768)))
            base_predict = int(config.get("num_predict", 12000))
            if base_predict < 0:
                num_predict = base_predict
            else:
                num_predict = min(32768, max(1, base_predict) * (2 ** attempt))
            num_ctx = min(65536, max(base_context, num_predict + 4096 if num_predict > 0 else base_context))
            payload = {"model": config["model"], "messages": request_messages, "format": schema, "stream": True, "think": False, "keep_alive": "5m", "options": {"temperature": config["temperature"], "seed": config["seed"], "num_ctx": num_ctx, "num_predict": num_predict}}
            try:
                response = self.request("/api/chat", payload, config["timeout"], progress)
            except OutputLengthError as exc:
                last_error = exc
                errors.append(str(exc))
                continue
            try:
                source = "content"
                reply = response["text"]
                # Some local builds put the entire schema-constrained JSON in thinking.
                # Accept that field only as a complete validated JSON document, never extract
                # a fragment from reasoning or expose the raw alternate field in the UI.
                if not reply.strip():
                    reply = response.get("alternate_text", "")
                    source = "thinking_json"
                result = parse_reply(reply, schema)
                if validator:
                    result = validator(result)
                return result, {"attempts": attempt + 1, "response_field": source, **response["metrics"]}
            except ValidationError as exc:
                last_error = exc
                errors.append(str(exc))
        if isinstance(last_error, OutputLengthError):
            raise OllamaError("出力上限を拡張して4回試行しましたが、モデルが完成したJSONを返せませんでした。") from last_error
        raise OllamaError(f"回答が形式検査に{max_attempts}回失敗しました: {errors[-1] if errors else '不明な検証エラー'}")
