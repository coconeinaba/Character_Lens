"""First-run UI: standard library and Tkinter only, before importing Pillow."""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import sys
import tempfile
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from migration import ROOT, MODEL_FOLDER, check_ollama, check_tagger, download_tagger, install_packages, load_profile, package_status, probe_packages, pull_ollama, relink_images, restore_history, sha256


def needs_setup(root=ROOT, data_dir=None):
    profile = load_profile(root)
    packages = package_status(profile)
    data_dir = Path(data_dir) if data_dir is not None else Path(root) / "data"
    try:
        marker = json.loads((data_dir / "setup_complete.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    changed = marker.get("python", "").split(".")[:2] < ["3", "12"]
    return changed or not all(row["matches"] for row in packages) or not all((Path(root) / MODEL_FOLDER / name).is_file() for name in ("model.onnx", "selected_tags.csv"))


def inspect_environment(root=ROOT):
    profile = load_profile(root)
    packages = package_status(profile)
    imports_ok, import_error = False, ""
    if all(row["installed"] for row in packages):
        try:
            probe_packages()
            imports_ok = True
        except Exception as exc:
            import_error = str(exc)
    try:
        tagger = check_tagger(root)
    except Exception:
        tagger = False
    return {"python": platform.python_version(), "python_matches": tuple(map(int, platform.python_version().split(".")[:2])) >= (3, 12), "packages": packages, "imports_ok": imports_ok, "import_error": import_error, "tagger": tagger}


def ready_to_start(snapshot, allow_different=False):
    if not snapshot:
        return False
    runnable = snapshot["imports_ok"] and snapshot["tagger"]
    same = snapshot["python_matches"] and all(row["matches"] for row in snapshot["packages"])
    return runnable and (same or allow_different)


class SetupWizard(tk.Tk):
    def __init__(self, root, data_dir, setup_only=False):
        super().__init__()
        self.root_folder, self.data_dir = Path(root), Path(data_dir)
        self.profile = load_profile(root)
        self.setup_only = setup_only
        self.title("Character Lens — 初回準備・別PCへの移行")
        self.geometry("980x840")
        self.minsize(880, 760)
        self.events = queue.Queue()
        self.busy = False
        self.last_error = None
        self.accepted = False
        self.snapshot = None
        self.buttons = []
        self.allow_different = tk.BooleanVar(value=False)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Yu Gothic UI", 10))
        style.configure("TButton", font=("Yu Gothic UI", 10), padding=7)
        header = tk.Frame(self, bg="#18304b", padx=22, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="Character Lens  初回準備", fg="white", bg="#18304b", font=("Yu Gothic UI", 21, "bold")).pack(anchor="w")
        tk.Label(header, text="モデルはこのPCで取得します。画像の外部送信は行いません。", fg="white", bg="#18304b", font=("Yu Gothic UI", 10)).pack(anchor="w")
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="① 状態を確認 → ② 不足分を取得 → ③ 必要なら履歴を復元 → アプリを開始", wraplength=850).pack(anchor="w", pady=(0, 8))
        self.table = ttk.Treeview(frame, columns=("item", "state"), show="headings", height=9, selectmode="none")
        self.table.heading("item", text="項目")
        self.table.heading("state", text="このPCの状態")
        self.table.column("item", width=200, stretch=False)
        self.table.column("state", width=610)
        self.table.pack(fill="x")
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=10)
        self.button(actions, "状態を再確認", lambda: self.start(self.check)).pack(side="left")
        self.button(actions, "ライブラリを導入", lambda: self.start(self.install)).pack(side="left", padx=5)
        self.button(actions, "専用モデルを取得（約379MB）", lambda: self.start(self.download)).pack(side="left", padx=5)
        self.button(actions, "Ollamaモデルを取得", lambda: self.start(self.pull)).pack(side="left")
        official = ttk.Frame(frame)
        official.pack(fill="x", pady=(0, 10))
        ttk.Button(official, text="Ollamaの公式ダウンロード", command=lambda: webbrowser.open("https://ollama.com/download/windows")).pack(side="left")
        ttk.Button(official, text="Pythonの公式ダウンロード", command=lambda: webbrowser.open("https://www.python.org/downloads/windows/")).pack(side="left", padx=6)
        ttk.Label(frame, text=f"Ollama本体は公式インストーラーで導入し、起動してください。取得対象: {self.profile['ollama']['model']}（移行元のサイズ 約{self.profile['ollama']['size'] / 1024**3:.1f}GB）。\nダウンロードにはインターネット接続と空き容量が必要です。取得済みファイルは検証して再利用します。", wraplength=850).pack(anchor="w")
        history = ttk.LabelFrame(frame, text="履歴と画像の復元（任意）", padding=8)
        history.pack(fill="x", pady=10)
        self.button(history, "同梱の履歴を復元", lambda: self.start(self.restore)).pack(side="left")
        self.button(history, "移行先の画像フォルダーを指定", self.choose_folder).pack(side="left", padx=8)
        ttk.Label(history, text="元画像は別途コピーしてください。\n既存履歴への上書きは行いません。", wraplength=320).pack(side="left")
        ttk.Checkbutton(frame, text="版の違いを確認し、このPCの版で開始する（同一環境の復元ではありません）", variable=self.allow_different, command=self.update_continue).pack(anchor="w", pady=(0, 6))
        log_frame = ttk.Frame(frame)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=7, wrap="word", font=("Yu Gothic UI", 9), state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        footer = ttk.Frame(frame)
        footer.pack(side="bottom", fill="x", pady=(9, 0), before=log_frame)
        self.status = tk.StringVar(value="状態を確認しています…")
        ttk.Label(footer, textvariable=self.status, wraplength=600).pack(side="left")
        self.continue_button = ttk.Button(footer, text="準備を完了して閉じる" if setup_only else "アプリを開始", command=self.finish, state="disabled")
        self.continue_button.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(80, self.drain)
        self.after(150, lambda: self.start(self.check))

    def button(self, parent, text, command):
        widget = ttk.Button(parent, text=text, command=command)
        self.buttons.append(widget)
        return widget

    def emit(self, text):
        self.events.put(("log", text))

    def start(self, operation):
        if self.busy:
            return
        self.busy = True
        self.last_error = None
        self.status.set("処理中です。完了までこの画面を開いたままにしてください。")
        for widget in self.buttons:
            widget.configure(state="disabled")
        self.update_continue()
        def worker():
            try:
                operation()
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("finished", None))
        threading.Thread(target=worker, daemon=True).start()

    def check(self):
        self.events.put(("snapshot", inspect_environment(self.root_folder)))

    def install(self):
        install_packages(self.root_folder, self.emit)
        self.check()

    def download(self):
        download_tagger(self.root_folder, self.emit)
        self.check()

    def pull(self):
        pull_ollama(self.profile, self.emit)
        self.check()

    def restore(self):
        restore_history(self.root_folder, self.data_dir)
        self.emit("同梱の履歴を復元しました。画像の場所が変わった場合は、次に画像フォルダーを指定してください。")

    def choose_folder(self):
        if self.busy:
            return
        folder = filedialog.askdirectory(parent=self, title="別途コピーした画像のフォルダーを選択（サブフォルダーも検索）")
        if folder:
            self.start(lambda: self.relink(folder))

    def relink(self, folder):
        result = relink_images(self.data_dir, folder, self.emit)
        self.emit(f"画像の場所: 更新{result['changed']}件 / 変更不要{result['unchanged']}件 / 未解決{len(result['unresolved'])}件")
        for row in result["unresolved"][:15]:
            self.emit(row["file"] + ": " + row["reason"])
        if result["backup"]:
            self.emit("変更前の履歴を保存: " + result["backup"])

    def render_snapshot(self, value):
        self.snapshot = value
        self.table.delete(*self.table.get_children())
        def add(name, text):
            self.table.insert("", "end", values=(name, text))
        add("Python", value["python"] + (" / 必要条件を満たしています" if value["python_matches"] else " / Python 3.12以上が必要です"))
        for row in value["packages"]:
            add(row["distribution"], row["actual"] + (" / 必要な版以上" if row["matches"] else " / 必要な版 " + row["version"]))
        add("専用キャラクターモデル", "SHA-256一致・利用可能" if value["tagger"] else "未取得または検証不一致 → 専用モデルを取得")
        if value["import_error"]:
            self.append_log(value["import_error"])
        self.update_continue()

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", str(text) + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 250:
            self.log.delete("1.0", f"{lines - 200}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def update_continue(self):
        enabled = not self.busy and ready_to_start(self.snapshot, self.allow_different.get())
        self.continue_button.configure(state="normal" if enabled else "disabled")

    def drain(self):
        for _ in range(30):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "snapshot":
                self.render_snapshot(payload)
            elif kind in ("log", "error"):
                self.append_log(("処理できませんでした: " if kind == "error" else "") + payload)
                if kind == "error":
                    self.last_error = payload
                    self.snapshot = None
            elif kind == "finished":
                self.busy = False
                for widget in self.buttons:
                    widget.configure(state="normal")
                self.status.set("準備できています。履歴の復元は任意です。" if ready_to_start(self.snapshot, self.allow_different.get()) else "不足項目を準備し、状態を再確認してください。")
                if self.last_error:
                    self.status.set("処理できませんでした。上の記録を確認してから再試行してください。")
                self.update_continue()
        self.after(80, self.drain)

    def finish(self):
        if not self.busy and ready_to_start(self.snapshot, self.allow_different.get()):
            marker = {"python": self.snapshot["python"], "packages": {r["distribution"]: r["actual"] for r in self.snapshot["packages"]}, "different_versions_accepted": self.allow_different.get()}
            temporary = None
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".setup-", suffix=".tmp", dir=self.data_dir, delete=False) as output:
                    temporary = Path(output.name)
                    json.dump(marker, output, ensure_ascii=False, indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.data_dir / "setup_complete.json")
            except OSError as exc:
                self.append_log("準備情報を保存できません。書き込み可能なフォルダーへ展開してください。\n" + str(exc))
                return
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
            self.accepted = True
            self.destroy()

    def close(self):
        if self.busy:
            self.status.set("処理中です。導入・取得が完了してから閉じてください。")
        else:
            self.destroy()


def boot():
    if sys.version_info < (3, 12) or sys.maxsize <= 2**32:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Pythonを確認してください", "Python 3.12以上の64bit版が必要です。移行元と合わせる場合はPython 3.14.6（64bit）を使用してください。", parent=root)
        root.destroy()
        return False
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    options, _ = parser.parse_known_args()
    if "--help" in sys.argv or "-h" in sys.argv:
        return True
    if not options.setup and not options.setup_only and not needs_setup(ROOT, options.data_dir):
        return True
    wizard = SetupWizard(ROOT, options.data_dir, options.setup_only)
    wizard.mainloop()
    sys.argv[:] = [arg for arg in sys.argv if arg not in ("--setup", "--setup-only")]
    return wizard.accepted and not options.setup_only

