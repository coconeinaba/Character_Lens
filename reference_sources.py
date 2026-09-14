"""Registered-URL reference image collection with strict network allowlisting."""
from __future__ import annotations

import hashlib
import html
import ipaddress
import mimetypes
import re
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from domain import now


class ReferenceError(ValueError):
    pass


GENERIC_HEADINGS = {"character", "characters", "キャラクター", "character image", "キャラクター character"}
BLOCKED_IMAGE_PARTS = ("logo", "ロゴ", "banner", "bnr", "favicon", "icon", "sprite")


def _host(value: str) -> str:
    return value.strip().rstrip(".").lower()


def _assert_public_host(host: str, resolve: bool) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            raise ReferenceError("登録URLにローカル・予約済みIPアドレスは指定できません。")
        return
    if not resolve:
        return
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ReferenceError(f"登録URLのホストを確認できません: {host}") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or any(not ipaddress.ip_address(value).is_global for value in addresses):
        raise ReferenceError("登録URLの接続先にローカル・予約済みIPアドレスが含まれています。")


def validate_reference_url(value: str, allowed_hosts=None, resolve=False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceError("参照URLを入力してください。")
    raw = value.strip()
    parsed = urlsplit(raw)
    host = _host(parsed.hostname or "")
    if parsed.scheme.lower() != "https":
        raise ReferenceError("参照URLはHTTPSだけを使用できます。")
    if not host or parsed.username or parsed.password:
        raise ReferenceError("参照URLにホスト名と認証情報なしの形式を指定してください。")
    hosts = {_host(item) for item in (allowed_hosts or [])}
    if hosts and host not in hosts:
        raise ReferenceError("登録されていないドメインへの参照は許可されていません。")
    _assert_public_host(host, resolve)
    return urlunsplit(("https", host, parsed.path or "/", parsed.query, ""))


class _RedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_reference_url(newurl, self.allowed_hosts, resolve=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, allowed_hosts: set[str], timeout=45):
    normalized = validate_reference_url(url, allowed_hosts, resolve=True)
    opener = build_opener(_RedirectHandler(allowed_hosts))
    request = Request(normalized, headers={
        "User-Agent": "CharacterLens-RegisteredReferences/1.0",
        "Accept": "text/html,application/xhtml+xml,image/*",
    })
    try:
        response = opener.open(request, timeout=timeout)
    except Exception as exc:
        raise ReferenceError(f"参照URLを取得できません: {normalized}") from exc
    with response:
        final_url = validate_reference_url(response.geturl(), allowed_hosts, resolve=True)
        data = response.read()
        content_type = response.headers.get_content_type().lower()
        charset = response.headers.get_content_charset() or "utf-8"
    return {"url": final_url, "data": data, "content_type": content_type, "charset": charset}


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = []
        self.headings = []
        self.links = []
        self.images = []
        self._title = False
        self._heading = None
        self._link = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        tag = tag.lower()
        if tag == "title":
            self._title = True
        elif tag in ("h1", "h2", "h3"):
            self._heading = [tag, []]
        elif tag == "a" and values.get("href"):
            self._link = [values["href"], []]
        elif tag == "img":
            for key in ("src", "data-src", "data-original", "data-lazy-src"):
                if values.get(key):
                    self.images.append((values[key], values.get("alt", "")))
            if values.get("srcset"):
                self.images.append((values["srcset"].split(",")[0].strip().split(" ")[0], values.get("alt", "")))
        elif tag == "meta" and values.get("content") and values.get("property", "").lower() in ("og:image", "twitter:image"):
            self.images.append((values["content"], ""))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._title = False
        elif tag in ("h1", "h2", "h3") and self._heading:
            self.headings.append(" ".join(self._heading[1]))
            self._heading = None
        elif tag == "a" and self._link:
            self.links.append((self._link[0], " ".join(self._link[1])))
            self._link = None

    def handle_data(self, data):
        value = " ".join(data.split())
        if not value:
            return
        if self._title:
            self.title.append(value)
        if self._heading:
            self._heading[1].append(value)
        if self._link:
            self._link[1].append(value)


def _parse_page(payload):
    parser = _PageParser()
    try:
        parser.feed(payload["data"].decode(payload["charset"], "replace"))
    except (LookupError, UnicodeError):
        parser.feed(payload["data"].decode("utf-8", "replace"))
    return parser


def _clean(value):
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _title_parts(parser):
    return [_clean(part) for part in re.split(r"[|｜／/]", " ".join(parser.title)) if _clean(part)]


def _page_name(parser, fallback=""):
    for value in parser.headings:
        value = _clean(value)
        if value and value.casefold() not in GENERIC_HEADINGS:
            return value
    parts = _title_parts(parser)
    for value in parts:
        if value.casefold() not in GENERIC_HEADINGS and "読売テレビ" not in value:
            return value
    return _clean(fallback) or "登録参照キャラクター"


def _work_name(parser, fallback=""):
    parts = _title_parts(parser)
    for value in reversed(parts):
        if value.casefold() not in GENERIC_HEADINGS and "読売テレビ" not in value and value != _page_name(parser):
            return value
    return _clean(fallback) or "参照元記載作品"


def _is_index_page(url, parser):
    path = urlsplit(url).path.rstrip("/")
    last_segment = path.rsplit("/", 1)[-1].casefold()
    if last_segment in {"character", "characters", "キャラクター"}:
        return True
    return any(_clean(value).casefold() in GENERIC_HEADINGS for value in parser.headings)


def _allowed_child(url, base_url, allowed_hosts):
    try:
        candidate = validate_reference_url(urljoin(base_url, url), allowed_hosts, resolve=False)
    except ReferenceError:
        return None
    base = urlsplit(base_url)
    target = urlsplit(candidate)
    prefix = (base.path.rstrip("/") + "/") if base.path.rstrip("/") else "/"
    if target.netloc != base.netloc or not target.path.startswith(prefix) or target.path.rstrip("/") == base.path.rstrip("/"):
        return None
    return candidate


def _image_url(value, page_url, allowed_hosts):
    try:
        candidate = validate_reference_url(urljoin(page_url, value), allowed_hosts, resolve=False)
    except ReferenceError:
        return None
    path = urlsplit(candidate).path.lower()
    if path.endswith((".svg", ".ico")) or any(part in path for part in BLOCKED_IMAGE_PARTS):
        return None
    return candidate


def _cache_image(image_url, page_url, cache_dir, allowed_hosts):
    payload = _fetch(image_url, allowed_hosts)
    if not payload["content_type"].startswith("image/") and not re.search(r"\.(?:png|jpe?g|webp|bmp|gif|tiff?)$", urlsplit(image_url).path, re.I):
        raise ReferenceError("画像URLではないため参照対象から除外しました。")
    sha = hashlib.sha256(payload["data"]).hexdigest()
    extension = Path(urlsplit(image_url).path).suffix.lower()
    if not extension or len(extension) > 8:
        extension = mimetypes.guess_extension(payload["content_type"]) or ".img"
    destination = cache_dir / f"{sha}{extension}"
    if not destination.exists():
        destination.write_bytes(payload["data"])
    return {"url": payload["url"], "page_url": page_url, "path": str(destination), "sha256": sha, "fetched_at": now()}


def _add_page(group_map, page_url, parser, fallback_name, cache_dir, allowed_hosts, warnings):
    name = _page_name(parser, fallback_name)
    work = _work_name(parser)
    key = (name.casefold(), work.casefold())
    group = group_map.setdefault(key, {"name": name, "display_name": name, "work": work, "page_url": page_url, "images": []})
    seen = {image["url"] for image in group["images"]}
    for raw_url, _alt in parser.images:
        image_url = _image_url(raw_url, page_url, allowed_hosts)
        if not image_url or image_url in seen:
            continue
        try:
            image = _cache_image(image_url, page_url, cache_dir, allowed_hosts)
        except ReferenceError as exc:
            warnings.append(f"{name}: {exc}")
            continue
        group["images"].append(image)
        seen.add(image_url)


def _metadata(catalog):
    groups = []
    for group in catalog:
        groups.append({
            "name": group["name"],
            "display_name": group["display_name"],
            "work": group["work"],
            "page_url": group["page_url"],
            "images": [{"url": item["url"], "page_url": item["page_url"], "sha256": item["sha256"]} for item in group["images"]],
        })
    return groups


def provenance_metadata(catalog):
    """Return export-safe reference metadata including retrieval timestamps."""
    groups = []
    for group in catalog or []:
        groups.append({
            "name": group["name"],
            "display_name": group["display_name"],
            "work": group["work"],
            "page_url": group["page_url"],
            "images": [{
                "url": item["url"],
                "page_url": item["page_url"],
                "sha256": item["sha256"],
                "fetched_at": item.get("fetched_at"),
            } for item in group["images"]],
        })
    return groups


def normalize_profiles(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ReferenceError("参照URLプロファイルの形式が不正です。")
    profiles = []
    names = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ReferenceError("参照URLプロファイルの形式が不正です。")
        name = _clean(raw.get("name"))
        urls = []
        for item in raw.get("urls", []):
            url = validate_reference_url(item)
            if url not in urls:
                urls.append(url)
        if not name or not urls or name in names:
            raise ReferenceError("参照URLプロファイル名とURLは空欄・重複なしで指定してください。")
        names.add(name)
        profiles.append({"name": name, "urls": urls})
    return profiles


def collect_reference_catalog(profile, cache_dir, check=None):
    profiles = normalize_profiles([profile])
    if not profiles:
        raise ReferenceError("参照URLプロファイルが空です。")
    profile = profiles[0]
    allowed_hosts = {_host(_host(urlsplit(url).hostname or "")) for url in profile["urls"]}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    group_map = {}
    warnings = []
    visited_pages = set()
    for seed in profile["urls"]:
        if check:
            check()
        payload = _fetch(seed, allowed_hosts)
        content_type = payload["content_type"]
        if content_type.startswith("image/"):
            name = Path(urlsplit(seed).path).stem or profile["name"]
            group_map.setdefault((name.casefold(), profile["name"].casefold()), {"name": name, "display_name": name, "work": profile["name"], "page_url": seed, "images": []})
            group = group_map[(name.casefold(), profile["name"].casefold())]
            try:
                group["images"].append(_cache_image(seed, seed, cache_dir, allowed_hosts))
            except ReferenceError as exc:
                warnings.append(f"{name}: {exc}")
            continue
        parser = _parse_page(payload)
        page_url = payload["url"]
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)
        if _is_index_page(page_url, parser):
            for href, label in parser.links:
                child = _allowed_child(href, page_url, allowed_hosts)
                if not child or child in visited_pages:
                    continue
                if check:
                    check()
                try:
                    child_payload = _fetch(child, allowed_hosts)
                    if child_payload["content_type"].startswith("image/"):
                        continue
                    child_parser = _parse_page(child_payload)
                    visited_pages.add(child_payload["url"])
                    _add_page(group_map, child_payload["url"], child_parser, _clean(label), cache_dir, allowed_hosts, warnings)
                except ReferenceError as exc:
                    warnings.append(f"{child}: {exc}")
        else:
            _add_page(group_map, page_url, parser, profile["name"], cache_dir, allowed_hosts, warnings)
    groups = [group for group in group_map.values() if group["images"]]
    if not groups:
        raise ReferenceError("登録URLからキャラクター画像を取得できませんでした。")
    return {"groups": groups, "metadata": _metadata(groups), "warnings": warnings}
