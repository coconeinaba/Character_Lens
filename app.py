"""Character Lens: local-only character inspection. Launch this file with Python."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

if __name__ == "__main__":
    from setup_wizard import boot
    try:
        prepared = boot()
    except Exception as exc:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("初回準備を開けません", str(exc) + "\nZIPをフォルダーごと展開し直し、移行手順を確認してください。", parent=error_root)
        error_root.destroy()
        raise SystemExit(1)
    if not prepared:
        raise SystemExit(0)

try:
    from PIL import Image, ImageOps, ImageTk
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Pillowが必要です", "画像表示にPillowが必要です。\nREADME.mdの準備手順を確認してください。")
    raise SystemExit(1)

from domain import AXES, CONFIDENCE_LABELS, EXTENSIONS, IDENTIFICATION_LABELS, ROOT, STATUS_LABELS, VERSION, WEIGHTS, build_config, candidate_label, display_image
from engine import BatchEngine
from exporter import export_csv, export_html, export_json
from ollama_client import OllamaClient
from storage import Store
from settings import load_settings, save_settings as save_file_settings
from tagger import availability as tagger_availability, signature as tagger_signature

BG = "#edf2f7"
INK = "#18304b"
TEAL = "#087e83"
MUTED = "#52677d"
PAGE_SIZE = 150


def readonly_text(parent):
    frame = ttk.Frame(parent)
    text = tk.Text(frame, wrap="word", font=("Yu Gothic UI", 10), bg="white", fg=INK, relief="flat", padx=14, pady=12, state="disabled", width=35, height=12)
    bar = ttk.Scrollbar(frame, command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    text.tag_configure("title", font=("Yu Gothic UI", 13, "bold"), foreground=INK, spacing1=8, spacing3=6)
    text.tag_configure("heading", font=("Yu Gothic UI", 10, "bold"), foreground=TEAL, spacing1=10, spacing3=4)
    text.tag_configure("muted", foreground=MUTED)
    text.tag_configure("warning", foreground="#a05719")
    return frame, text


class CropDialog(tk.Toplevel):
    def __init__(self, parent, path, on_save):
        super().__init__(parent)
        self.title("人物・顔の範囲を指定")
        self.transient(parent)
        self.on_save = on_save
        with Image.open(path) as source:
            self.original_size = ImageOps.exif_transpose(source).size
        im = display_image(path, (900, 630))
        self.scale_x = self.original_size[0] / im.width
        self.scale_y = self.original_size[1] / im.height
        self.photo = ImageTk.PhotoImage(im, master=self)
        ttk.Label(self, text="解析したい範囲をドラッグしてください。元画像は変更しません。", padding=12).pack()
        self.canvas = tk.Canvas(self, width=im.width, height=im.height, highlightthickness=0, background="white", cursor="crosshair")
        self.canvas.pack(padx=12)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.rect = None
        self.start = None
        self.selection = None
        self.canvas.bind("<ButtonPress-1>", self.down)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.up)
        self.label = ttk.Label(self, text="範囲未選択", padding=8)
        self.label.pack()
        actions = ttk.Frame(self, padding=12)
        actions.pack(fill="x")
        ttk.Button(actions, text="この範囲を使用", command=self.save, style="Accent.TButton").pack(side="right")
        ttk.Button(actions, text="キャンセル", command=self.destroy).pack(side="right", padx=8)
        self.grab_set()

    def point(self, event):
        return (max(0, min(self.photo.width(), event.x)), max(0, min(self.photo.height(), event.y)))

    def down(self, event):
        self.start = self.point(event)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(*self.start, *self.start, outline="#00b6b7", width=3)

    def drag(self, event):
        if self.start:
            self.canvas.coords(self.rect, *self.start, *self.point(event))

    def up(self, event):
        if not self.start:
            return
        x0, y0 = self.start
        x1, y1 = self.point(event)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 8 or bottom - top < 8:
            self.selection = None
            self.label.configure(text="もう少し広い範囲を選択してください。")
            return
        self.selection = [round(left * self.scale_x), round(top * self.scale_y), round(right * self.scale_x), round(bottom * self.scale_y)]
        self.label.configure(text=f"選択範囲: {self.selection[2] - self.selection[0]} × {self.selection[3] - self.selection[1]} px")

    def save(self):
        if self.selection:
            self.on_save(self.selection)
            self.destroy()


class CharacterLens(tk.Tk):
    def __init__(self, data_dir, initial_files=()):
        super().__init__()
        self.title("Character Lens — ローカル キャラクター識別")
        self.geometry("1480x940")
        self.minsize(1160, 760)
        self.configure(background=BG)
        self.data_dir = Path(data_dir)
        self.store = Store(self.data_dir / "character_lens.sqlite3")
        self.events = queue.Queue()
        self.busy = False
        self.refreshing = False
        self.closing = False
        self.engine = None
        self.cancel_event = threading.Event()
        self.worker = None
        self.background_jobs = []
        self.models = []
        self.rows = []
        self.filtered = []
        self.page = 0
        self.selected_id = None
        self.selected_run_id = None
        self.photo = None
        self.thumb_images = {}
        self.thumb_cache = OrderedDict()
        self.thumb_lock = threading.Lock()
        self.thumb_epoch = 0
        self.preview_epoch = 0
        self.preview_key = None
        self.preview_item = None
        self.history_crop = None
        self.showing_history = False
        self.sort_key = "id"
        self.sort_reverse = False
        self.completed_batch = set()
        self.started_at = None
        self.current_stage = ""
        self.locked_widgets = []
        self._style()
        self._variables()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_rows()
        if initial_files:
            self.store.add([p for p in initial_files if Path(p).is_file() and Path(p).suffix.lower() in EXTENSIONS])
            self.refresh_rows()
        self.after(80, self.drain_events)
        self.after(200, self.refresh_models)
        self.after(1000, self.tick)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Yu Gothic UI", 10), background=BG, foreground=INK)
        style.configure("TButton", padding=(12, 7))
        style.configure("Accent.TButton", background=TEAL, foreground="white", padding=(15, 8))
        style.map("Accent.TButton", background=[("active", "#096b70"), ("disabled", "#b1bec8")])
        style.configure("TEntry", fieldbackground="white", padding=4)
        style.configure("TCombobox", fieldbackground="white", padding=4)
        style.configure("Treeview", background="white", fieldbackground="white", rowheight=48, borderwidth=0)
        style.configure("Treeview.Heading", font=("Yu Gothic UI", 9, "bold"), padding=8)
        style.map("Treeview", background=[("selected", "#d4edef")], foreground=[("selected", INK)])
        style.configure("History.Treeview", rowheight=28)
        style.configure("TNotebook.Tab", padding=(14, 8))
        style.configure("White.TFrame", background="white")

    def _variables(self):
        saved = self.store.get_setting("ui", {})
        self.file_settings = load_settings()
        self.profiles = self.file_settings["ollama_profiles"]
        self.profile = tk.StringVar(value=self.file_settings["active_profile"])
        selected = next(p for p in self.profiles if p["name"] == self.profile.get())
        self.endpoint = tk.StringVar(value=selected["endpoint"])
        self.model = tk.StringVar(value=saved.get("model", "qwen3-vl:latest"))
        self.mode = tk.StringVar(value=saved.get("mode", "自由判定"))
        self.target = tk.StringVar(value=saved.get("target", ""))
        self.edge = tk.StringVar(value=str(saved.get("edge", 1280)))
        self.timeout = tk.StringVar(value=str(saved.get("timeout", 300)))
        self.weight_vars = {key: tk.IntVar(value=self.file_settings["weights"][key]) for key in AXES}
        self.weight_text = {key: tk.StringVar(value=str(self.weight_vars[key].get())) for key in AXES}
        self.recursive = tk.BooleanVar(value=False)
        self.filter = tk.StringVar(value="すべて")
        self.search = tk.StringVar()
        self.force = tk.BooleanVar(value=False)
        ready, _ = tagger_availability()
        self.use_tagger = tk.BooleanVar(value=saved.get("use_tagger", ready) and ready)
        self.connection_text = tk.StringVar(value="Ollamaを確認しています…")
        self.status = tk.StringVar(value="画像を追加して、解析を開始してください。")
        self.summary = tk.StringVar()
        self.pager = tk.StringVar()
        self.review_name = tk.StringVar()
        self.reviewed = tk.BooleanVar()
        self.preview_caption = tk.StringVar(value="画像を選択すると、ここに表示します。")

    def _lock(self, widget, normal="normal"):
        self.locked_widgets.append((widget, normal))
        return widget

    def _build(self):
        header = tk.Frame(self, background=INK, padx=22, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="Character Lens", font=("Yu Gothic UI", 22, "bold"), fg="white", bg=INK).pack(side="left")
        tk.Label(header, text="画像からキャラクターの候補と、似ている理由を探す", font=("Yu Gothic UI", 10), fg="#c9d8e7", bg=INK).pack(side="left", padx=22)
        tk.Label(header, text="OLLAMA", fg="#99e1dc", bg=INK, font=("Segoe UI", 10, "bold")).pack(side="right")
        settings = ttk.Frame(self, padding=(16, 10))
        settings.pack(fill="x")
        ttk.Label(settings, text="接続先").grid(row=0, column=0, padx=(0, 6))
        combo = self._lock(ttk.Combobox(settings, textvariable=self.profile, values=[p["name"] for p in self.profiles], width=12, state="readonly"), "readonly")
        combo.grid(row=0, column=1)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.select_profile())
        self._lock(ttk.Entry(settings, textvariable=self.endpoint, width=28)).grid(row=0, column=2, padx=5)
        self._lock(ttk.Button(settings, text="保存", command=self.save_profile)).grid(row=0, column=3)
        self._lock(ttk.Button(settings, text="新規", command=self.add_profile)).grid(row=0, column=4, padx=(4, 0))
        self.refresh_btn = self._lock(ttk.Button(settings, text="接続・モデル確認", command=self.refresh_models))
        self.refresh_btn.grid(row=0, column=5, padx=8)
        ttk.Label(settings, text="画像モデル").grid(row=0, column=6, padx=8)
        self.model_combo = self._lock(ttk.Combobox(settings, textvariable=self.model, width=24, state="readonly"), "readonly")
        self.model_combo.grid(row=0, column=7)
        ttk.Label(settings, textvariable=self.connection_text, foreground=MUTED).grid(row=0, column=8, sticky="w", padx=12)
        self._lock(ttk.Button(settings, text="初回準備・移行", command=self.open_setup)).grid(row=0, column=9, padx=5)
        modebar = ttk.Frame(self, padding=(16, 0, 16, 9))
        modebar.pack(fill="x")
        modecombo = self._lock(ttk.Combobox(modebar, textvariable=self.mode, values=["自由判定", "指定キャラクターを評価"], width=23, state="readonly"), "readonly")
        modecombo.pack(side="left")
        modecombo.bind("<<ComboboxSelected>>", lambda e: self.update_target_state())
        self.target_entry = ttk.Entry(modebar, textvariable=self.target, width=27)
        self.target_entry.pack(side="left", padx=8)
        ttk.Label(modebar, text="画像長辺").pack(side="left", padx=(12, 4))
        self._lock(ttk.Combobox(modebar, textvariable=self.edge, values=[768, 1024, 1280, 1600, 2048], width=6, state="readonly"), "readonly").pack(side="left")
        ttk.Label(modebar, text="応答が止まった場合の上限秒").pack(side="left", padx=(12, 4))
        self._lock(ttk.Combobox(modebar, textvariable=self.timeout, values=[120, 300, 600, 900], width=6, state="readonly"), "readonly").pack(side="left")
        self._lock(ttk.Checkbutton(modebar, text="専用モデルで候補を絞る", variable=self.use_tagger)).pack(side="left", padx=12)
        self.update_target_state()
        weight_frame = ttk.LabelFrame(self, text="似ている度の重み（0〜100。未確認の軸は点数に含めません）", padding=(16, 5))
        weight_frame.pack(fill="x", padx=16, pady=(0, 7))
        for column, (key, label) in enumerate(AXES.items()):
            group = ttk.Frame(weight_frame)
            group.grid(row=0, column=column, sticky="ew", padx=5)
            ttk.Label(group, text=label).pack(side="left")
            scale = self._lock(ttk.Scale(group, from_=0, to=100, variable=self.weight_vars[key], command=lambda _v, k=key: self.weight_text[k].set(str(self.weight_vars[k].get()))))
            scale.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(group, textvariable=self.weight_text[key], width=3).pack(side="left")
            weight_frame.columnconfigure(column, weight=1)
        self._lock(ttk.Button(weight_frame, text="標準に戻す", command=self.reset_weights)).grid(row=0, column=len(AXES), padx=7)
        notice = tk.Label(self, text="公式画像との照合は行いません。点数はモデルの知識に基づく推定で、正答確率・公式画像との一致率ではありません。", anchor="w", bg="#fff3dd", fg="#805623", padx=16, pady=8, font=("Yu Gothic UI", 9))
        notice.pack(fill="x")
        toolbar = ttk.Frame(self, padding=(16, 10))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="画像を追加", command=self.add_files).pack(side="left")
        ttk.Button(toolbar, text="フォルダーを追加", command=self.add_folder).pack(side="left", padx=6)
        ttk.Checkbutton(toolbar, text="サブフォルダーも", variable=self.recursive).pack(side="left", padx=(0, 12))
        self._lock(ttk.Button(toolbar, text="一覧から外す", command=self.hide_selected)).pack(side="left")
        self._lock(ttk.Button(toolbar, text="一覧を初期化", command=self.reset_list)).pack(side="left", padx=6)
        export_button = ttk.Menubutton(toolbar, text="結果を書き出す", direction="below")
        menu = tk.Menu(export_button, tearoff=False)
        for kind in ("HTML", "CSV", "JSON"):
            menu.add_command(label=f"{kind}（絞り込み中の全画像）", command=lambda k=kind: self.export(k))
        menu.add_separator()
        menu.add_command(label="履歴データをバックアップ", command=self.backup)
        export_button["menu"] = menu
        export_button.pack(side="right")
        ttk.Label(toolbar, textvariable=self.summary, foreground=MUTED).pack(side="right", padx=15)
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=16)
        left = ttk.Frame(panes)
        center = ttk.Frame(panes, style="White.TFrame")
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(center, weight=3)
        panes.add(right, weight=4)
        searchbar = ttk.Frame(left)
        searchbar.pack(fill="x", pady=(0, 8))
        ttk.Entry(searchbar, textvariable=self.search, width=18).pack(side="left", fill="x", expand=True)
        filters = ttk.Combobox(searchbar, textvariable=self.filter, values=["すべて", "未処理", "完了", "失敗", "中断", "候補なし", "要確認", "確認済み"], state="readonly", width=10)
        filters.pack(side="left", padx=(6, 0))
        filters.bind("<<ComboboxSelected>>", lambda e: self.filter_changed())
        self.search.trace_add("write", lambda *_: self.filter_changed())
        # Six rows are the minimum height.  The list takes remaining space,
        # while the pager below keeps its own visible height.
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=("state", "score"), show="tree headings", selectmode="extended", height=6)
        self.tree.heading("#0", text="画像 / 第一候補", command=lambda: self.sort_by("name"))
        self.tree.heading("state", text="状態", command=lambda: self.sort_by("status"))
        self.tree.heading("score", text="推定点", command=lambda: self.sort_by("score"))
        self.tree.column("#0", width=230, minwidth=160, stretch=True)
        self.tree.column("state", width=66, minwidth=60, stretch=False)
        self.tree.column("score", width=56, minwidth=50, stretch=False)
        self.tree.tag_configure("error", foreground="#a84035")
        self.tree.tag_configure("done", foreground=INK)
        self.tree.tag_configure("running", foreground=TEAL)
        scrollbar = ttk.Scrollbar(tree_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.select_image)
        pages = ttk.Frame(left, padding=(0, 6))
        pages.pack(fill="x")
        ttk.Button(pages, text="前へ", width=5, command=lambda: self.turn_page(-1)).pack(side="left")
        ttk.Label(pages, textvariable=self.pager).pack(side="left", padx=6)
        ttk.Button(pages, text="次へ", width=5, command=lambda: self.turn_page(1)).pack(side="right")
        self.preview = tk.Canvas(center, background="white", highlightthickness=0, width=330, height=200)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", self.resize_preview)
        self.preview_timer = None
        # Caption changes must not resize the canvas and trigger another image load.
        caption_frame = ttk.Frame(center, height=88)
        caption_frame.pack(fill="x")
        caption_frame.pack_propagate(False)
        ttk.Label(caption_frame, textvariable=self.preview_caption, wraplength=330, padding=8, background="white", foreground=MUTED, anchor="nw").pack(fill="both", expand=True)
        cropbar = ttk.Frame(center, padding=8)
        cropbar.pack(fill="x")
        self._lock(ttk.Button(cropbar, text="範囲を指定", command=self.crop)).pack(side="left")
        self._lock(ttk.Button(cropbar, text="全体に戻す", command=lambda: self.save_crop(None))).pack(side="left", padx=5)
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)
        evaluation, self.detail_text = readonly_text(notebook)
        observation, self.observation_text = readonly_text(notebook)
        notebook.add(evaluation, text="候補・評価")
        notebook.add(observation, text="観察・記録")
        historyframe = ttk.Frame(notebook, padding=8)
        notebook.add(historyframe, text="過去の解析")
        ttk.Label(historyframe, text="行を選ぶと、その時の結果を表示します。", wraplength=340).pack(anchor="w", pady=6)
        self.history_tree = ttk.Treeview(historyframe, columns=("time", "model", "state"), show="headings", style="History.Treeview", selectmode="browse")
        for col, label, width in (("time", "日時", 130), ("model", "モデル", 145), ("state", "状態", 65)):
            self.history_tree.heading(col, text=label)
            self.history_tree.column(col, width=width)
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<<TreeviewSelect>>", self.select_history)
        review = ttk.LabelFrame(right, text="人による確認（AIの結果は保持）", padding=8)
        review.pack(fill="x", pady=(8, 0))
        ttk.Label(review, text="修正名").grid(row=0, column=0, sticky="w")
        ttk.Entry(review, textvariable=self.review_name).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Checkbutton(review, text="確認済み", variable=self.reviewed).grid(row=0, column=2)
        ttk.Label(review, text="メモ").grid(row=1, column=0, sticky="nw", pady=6)
        self.review_note = tk.Text(review, height=2, width=25, font=("Yu Gothic UI", 9), wrap="word", relief="solid", borderwidth=1)
        self.review_note.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(review, text="保存", command=self.save_review).grid(row=1, column=2)
        review.columnconfigure(1, weight=1)
        footer = ttk.Frame(self, padding=16)
        footer.pack(side="bottom", fill="x", before=panes)
        buttons = ttk.Frame(footer)
        buttons.pack(fill="x")
        self._lock(ttk.Button(buttons, text="全画像を開始・再開", command=lambda: self.start_batch("all"), style="Accent.TButton")).pack(side="left")
        self._lock(ttk.Button(buttons, text="選択した画像を解析", command=lambda: self.start_batch("selected"))).pack(side="left", padx=6)
        self._lock(ttk.Button(buttons, text="失敗だけ再実行", command=lambda: self.start_batch("errors"))).pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="中断", state="disabled", command=self.stop)
        self.stop_btn.pack(side="left", padx=6)
        self._lock(ttk.Checkbutton(buttons, text="完了済みも再解析（履歴を追加）", variable=self.force)).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.pack(fill="x", pady=(12, 6))
        ttk.Label(footer, textvariable=self.status, foreground=MUTED).pack(anchor="w")

    def update_target_state(self):
        self.target_entry.configure(state="normal" if not self.busy and self.mode.get() != "自由判定" else "disabled")

    def save_settings(self):
        self.store.set_setting("ui", {"endpoint": self.endpoint.get(), "model": self.model.get(), "mode": self.mode.get(), "target": self.target.get(), "edge": self.edge.get(), "timeout": self.timeout.get(), "use_tagger": self.use_tagger.get()})
        save_file_settings({"ollama_profiles": self.profiles, "active_profile": self.profile.get(), "weights": {key: var.get() for key, var in self.weight_vars.items()}})

    def select_profile(self):
        entry = next((p for p in self.profiles if p["name"] == self.profile.get()), None)
        if entry:
            self.endpoint.set(entry["endpoint"])

    def save_profile(self):
        try:
            OllamaClient(self.endpoint.get())
        except Exception as exc:
            messagebox.showerror("接続先", str(exc), parent=self)
            return
        for entry in self.profiles:
            if entry["name"] == self.profile.get():
                entry["endpoint"] = self.endpoint.get().strip()
                self.save_settings()
                self.status.set("接続先をsetting.jsonに保存しました。")
                return

    def add_profile(self):
        name = simpledialog.askstring("接続先を追加", "接続先の名前", parent=self)
        if not name:
            return
        name = name.strip()
        if not name or any(p["name"] == name for p in self.profiles):
            messagebox.showerror("接続先を追加", "空でない重複しない名前を入力してください。", parent=self)
            return
        try:
            OllamaClient(self.endpoint.get())
        except Exception as exc:
            messagebox.showerror("接続先", str(exc), parent=self)
            return
        self.profiles.append({"name": name, "endpoint": self.endpoint.get().strip()})
        self.profile.set(name)
        self._refresh_profiles()
        self.save_settings()

    def _refresh_profiles(self):
        for widget, _state in self.locked_widgets:
            if isinstance(widget, ttk.Combobox) and str(widget.cget("textvariable")) == str(self.profile):
                widget.configure(values=[p["name"] for p in self.profiles])

    def reset_weights(self):
        for key, value in WEIGHTS.items():
            self.weight_vars[key].set(value)
            self.weight_text[key].set(str(value))

    def emit(self, kind, payload):
        self.events.put((kind, payload))

    def start_background(self, action):
        self.background_jobs = [t for t in self.background_jobs if t.is_alive()]
        thread = threading.Thread(target=action, daemon=True)
        self.background_jobs.append(thread)
        thread.start()

    def refresh_models(self):
        if self.busy or self.refreshing:
            return
        self.refreshing = True
        endpoint = self.endpoint.get()
        self.refresh_btn.configure(state="disabled")
        self.connection_text.set("画像対応モデルを確認中…")
        def run():
            try:
                self.emit("models", OllamaClient(endpoint).models())
            except Exception as exc:
                self.emit("models_error", str(exc))
        self.start_background(run)

    def add_files(self):
        paths = filedialog.askopenfilenames(parent=self, title="解析する画像を選択", filetypes=[("画像", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"), ("すべて", "*.*")], initialdir=self.store.get_setting("last_folder", str(ROOT)))
        if paths:
            valid = [p for p in paths if Path(p).suffix.lower() in EXTENSIONS]
            self.store.add(valid)
            self.store.set_setting("last_folder", str(Path(paths[0]).parent))
            self.refresh_rows()
            self.status.set(f"{len(valid)}枚を一覧に追加しました。")

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self, title="画像のあるフォルダーを選択", initialdir=self.store.get_setting("last_folder", str(ROOT)))
        if not folder:
            return
        recursive = self.recursive.get()
        self.store.set_setting("last_folder", folder)
        self.status.set("フォルダー内の画像を探しています…")
        def run():
            try:
                iterator = Path(folder).rglob("*") if recursive else Path(folder).iterdir()
                paths = [p for p in iterator if p.suffix.lower() in EXTENSIONS and p.is_file()]
                self.emit("files", sorted(paths, key=lambda p: str(p).casefold()))
            except Exception as exc:
                self.emit("notification_error", str(exc))
        self.start_background(run)

    def hide_selected(self):
        ids = [int(i) for i in self.tree.selection()]
        if ids:
            self.store.hide(ids)
            self.selected_id = None
            self.selected_run_id = None
            self.refresh_rows()
            self.show_detail(None)
            self.redraw_preview()
            self.status.set("一覧から外しました。元画像と解析履歴は保持しています。画像を追加すると再表示できます。")

    def reset_list(self):
        if self.busy:
            return
        count = len(self.store.assets())
        if count and messagebox.askyesno("一覧を初期化", f"一覧の{count}枚を表示から外しますか？\n元画像と解析履歴は削除されません。", parent=self):
            self.store.hide_all()
            self.selected_id = self.selected_run_id = None
            self.refresh_rows()
            self.show_detail(None)
            self.redraw_preview()
            self.status.set("一覧を初期化しました。元画像と解析履歴は保持しています。")

    def filter_changed(self):
        self.page = 0
        self.refresh_rows()

    @staticmethod
    def top_candidate(row):
        return next(iter((row.get("result") or {}).get("candidates", [])), {})

    def sort_by(self, key):
        self.sort_reverse = not self.sort_reverse if self.sort_key == key else key == "score"
        self.sort_key = key
        self.refresh_rows()

    def refresh_rows(self):
        self.rows = self.store.assets()
        query = self.search.get().casefold().strip()
        chosen = self.filter.get()
        rows = []
        for row in self.rows:
            result = row.get("result") or {}
            haystack = Path(row["path"]).name + " " + " ".join(candidate_label(c) + " " + c.get("work", "") for c in result.get("candidates", [])) + " " + (row.get("review_name") or "")
            if query and query not in haystack.casefold():
                continue
            status = STATUS_LABELS.get(row.get("status"), "未処理")
            if chosen in STATUS_LABELS.values() and chosen != status:
                continue
            if chosen == "候補なし" and not (row.get("status") == "done" and not result.get("candidates")):
                continue
            if chosen == "確認済み" and not row.get("reviewed"):
                continue
            if chosen == "要確認" and (row.get("reviewed") or row.get("status") != "done"):
                continue
            rows.append(row)
        def sort(row):
            if self.sort_key == "name":
                return Path(row["path"]).name.casefold()
            if self.sort_key == "status":
                return row.get("status") or "pending"
            if self.sort_key == "score":
                value = self.top_candidate(row).get("estimated_similarity")
                return -1 if value is None else value
            return row["id"]
        rows.sort(key=sort, reverse=self.sort_reverse)
        self.filtered = rows
        pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page = min(self.page, pages - 1)
        previous = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        self.thumb_epoch += 1
        self.thumb_images = {}
        shown = rows[self.page * PAGE_SIZE:(self.page + 1) * PAGE_SIZE]
        for row in shown:
            top = self.top_candidate(row)
            name = candidate_label(top) if top else ("候補なし" if row.get("status") == "done" else "")
            score = top.get("estimated_similarity")
            value = "—" if score is None else str(score)
            marker = " ✓" if row.get("reviewed") else ""
            label = Path(row["path"]).name + ("\n" + name + marker if name else "")
            status = STATUS_LABELS.get(row.get("status"), "未処理")
            if row.get("run_id") and row.get("crop") != row.get("run_crop"):
                status = "範囲変更"
            self.tree.insert("", "end", iid=str(row["id"]), text=label, values=(status, value), tags=(row.get("status") or "pending",))
        retained = [i for i in previous if self.tree.exists(i)]
        if retained:
            self.tree.selection_set(retained)
        self.pager.set(f"{self.page + 1} / {pages} ページ · {len(rows)}枚")
        done = sum(r.get("status") == "done" for r in self.rows)
        errors = sum(r.get("status") == "error" for r in self.rows)
        self.summary.set(f"全{len(self.rows)}枚　完了{done}　失敗{errors}")
        epoch = self.thumb_epoch
        threading.Thread(target=self.load_thumbs, args=(shown, epoch), daemon=True).start()

    def load_thumbs(self, rows, epoch):
        for row in rows:
            if epoch != self.thumb_epoch or self.closing:
                return
            try:
                stat = Path(row["path"]).stat()
                key = (row["path"], stat.st_mtime_ns, stat.st_size)
                with self.thumb_lock:
                    im = self.thumb_cache.get(key)
                if im is None:
                    im = display_image(row["path"], (36, 36))
                    with self.thumb_lock:
                        self.thumb_cache[key] = im
                        while len(self.thumb_cache) > 500:
                            self.thumb_cache.popitem(last=False)
                if epoch == self.thumb_epoch:
                    self.emit("thumbnail", {"epoch": epoch, "asset_id": row["id"], "image": im})
            except (OSError, ValueError):
                pass

    def turn_page(self, delta):
        self.page = max(0, self.page + delta)
        self.refresh_rows()

    def select_image(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        self.selected_id = int(selected[0])
        self.showing_history = False
        row = next((r for r in self.rows if r["id"] == self.selected_id), None)
        if row:
            self.show_detail(row)
            self.redraw_preview()
            self.history_tree.delete(*self.history_tree.get_children())
            for run in self.store.history(self.selected_id):
                self.history_tree.insert("", "end", iid=str(run["id"]), values=(run["created"][5:19].replace("T", " "), run["config"]["model"], STATUS_LABELS[run["status"]]))

    def select_history(self, _event=None):
        selected = self.history_tree.selection()
        if selected:
            run = self.store.run(int(selected[0]))
            asset = self.store.asset(run["asset_id"])
            self.show_detail({**asset, **run, "run_id": run["id"], "run_crop": run["crop"]})
            self.history_crop = run["crop"]
            self.showing_history = True
            self.redraw_preview()

    def resize_preview(self, _event=None):
        if self.preview_timer:
            self.after_cancel(self.preview_timer)
        self.preview_timer = self.after(120, self.redraw_preview)

    def redraw_preview(self):
        if self.preview_timer:
            self.after_cancel(self.preview_timer)
        self.preview_timer = None
        if self.selected_id is None:
            self.preview_epoch += 1
            self.preview_key = None
            self.preview.delete("all")
            self.preview_item = None
            self.photo = None
            self.preview_caption.set("画像を選択すると、ここに表示します。")
            return
        row = self.store.asset(self.selected_id)
        size = (max(50, self.preview.winfo_width() - 20), max(50, self.preview.winfo_height() - 20))
        crop = self.history_crop if self.showing_history else row["crop"]
        history = self.showing_history
        run = self.store.run(self.selected_run_id) if self.selected_run_id else None
        try:
            stat = Path(row["path"]).stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        key = (row["path"], size, tuple(crop) if crop else None, history, run.get("source_sha") if run else None, stamp)
        if key == self.preview_key:
            return
        self.preview_key = key
        self.preview_epoch += 1
        epoch = self.preview_epoch
        # Keep the old canvas item until its replacement is ready.
        self.preview_caption.set("画像を読み込み中…")
        def run_preview():
            try:
                im = display_image(row["path"], size, crop)
                sha = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                suffix = "\n過去の解析範囲を表示" if history else "\n次の解析対象: " + ("指定範囲" if crop else "画像全体")
                if run and run["source_sha"] and sha != run["source_sha"]:
                    suffix += "\n元画像が解析時から変更されています。再解析してください。"
                self.emit("preview", {"epoch": epoch, "image": im, "caption": Path(row["path"]).name + suffix})
            except Exception as exc:
                self.emit("preview", {"epoch": epoch, "image": None, "caption": f"画像を表示できません: {exc}"})
        threading.Thread(target=run_preview, daemon=True).start()

    def apply_preview(self, payload):
        if payload["epoch"] != self.preview_epoch:
            return
        if payload["image"] is not None:
            photo = ImageTk.PhotoImage(payload["image"], master=self)
            x, y = self.preview.winfo_width() // 2, self.preview.winfo_height() // 2
            if self.preview_item is None:
                self.preview_item = self.preview.create_image(x, y, image=photo)
            else:
                self.preview.itemconfigure(self.preview_item, image=photo)
                self.preview.coords(self.preview_item, x, y)
            self.photo = photo
        else:
            self.preview.delete("all")
            self.preview_item = None
            self.photo = None
            self.preview_key = None
        self.preview_caption.set(payload["caption"])

    def show_detail(self, row):
        self.selected_run_id = row.get("run_id") if row else None
        text = self.detail_text
        obs = self.observation_text
        for widget in (text, obs):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
        if row is None:
            text.insert("end", "画像を選択してください。", "muted")
        else:
            result = row.get("result") or {}
            if not result:
                text.insert("end", STATUS_LABELS.get(row.get("status"), "未処理") + "\n", "title")
                text.insert("end", row.get("error") or "解析すると、候補と特徴別の評価がここに表示されます。")
            else:
                text.insert("end", IDENTIFICATION_LABELS[result["identification"]] + "\n", "title")
                text.insert("end", result["overall_reason"] + "\n")
                for index, candidate in enumerate(result["candidates"], 1):
                    text.insert("end", f"{index}. {candidate_label(candidate)}\n", "title")
                    text.insert("end", f"{candidate['work']} / {candidate['variant']}\n", "muted")
                    if candidate.get("label_source") == "tag":
                        text.insert("end", "日本語名は辞書未登録です。作品名はモデルの推定です。\n", "muted")
                    score = candidate.get("estimated_similarity")
                    text.insert("end", f"推定類似度  {score if score is not None else '評価不能'} / 100\n", "heading")
                    text.insert("end", f"評価できた重み: {candidate['coverage']}%\n確かさ: {CONFIDENCE_LABELS[candidate['confidence']]}\n", "muted")
                    text.insert("end", candidate["rationale"] + "\n")
                    for label, key in (("一致した特徴", "matches"), ("異なる特徴", "differences")):
                        text.insert("end", label + "\n", "heading")
                        text.insert("end", "\n".join("・" + x for x in candidate[key]) + "\n" if candidate[key] else "記載なし\n")
                    text.insert("end", "特徴別の評価\n", "heading")
                    for axis in candidate["axes"]:
                        value = "評価不能" if axis["score"] is None else f"{axis['score']}点"
                        text.insert("end", f"{AXES[axis['axis']]}  {value}\n", "heading")
                        text.insert("end", axis["reason"] + "\n")
                if result.get("limitations"):
                    text.insert("end", "判定の制約\n", "heading")
                    text.insert("end", "\n".join("・" + v for v in result["limitations"]) + "\n", "warning")
                text.insert("end", "\n公式画像との照合なし。点数は暫定評価です。", "muted")
                if result.get("tagger"):
                    text.insert("end", "\n専用モデルの判定\n", "heading")
                    tagged = result["tagger"]
                    for item in tagged["candidates"]:
                        text.insert("end", f"{item['tag']}  分類出力 {item['score']:.3f}\n")
                    if not tagged["candidates"]:
                        text.insert("end", "採用基準を満たす候補なし\n")
                    text.insert("end", tagged["note"], "muted")
            observed = row.get("observation")
            if observed:
                data = observed["data"]
                obs.insert("end", "名前を伏せた外見観察\n", "title")
                obs.insert("end", data["summary"] + "\n")
                obs.insert("end", f"対象数: {data['subject_count'] if data['subject_count'] is not None else '不明'}\n")
                for key, label in AXES.items():
                    obs.insert("end", label + "\n", "heading")
                    obs.insert("end", data["features"][key] + "\n")
            config = row.get("config") or {}
            provenance = result.get("provenance") or {}
            obs.insert("end", "解析条件\n", "title")
            obs.insert("end", f"モデル: {config.get('model', '未設定')}\n判定: {'指定名あり' if config.get('mode') == 'target' else '自由判定'}\n指定名: {config.get('target', '')}\n解析時の範囲: {row.get('run_crop') or '画像全体'}\n処理時間: {provenance.get('seconds', '—')} 秒\n元画像SHA-256: {row.get('source_sha') or '未解析'}\n", "muted")
            obs.insert("end", "\n範囲を変更した場合、過去の解析結果は以前の範囲に対する結果です。次回の解析で更新されます。\nモデル名・プロンプト・画像内容・範囲が同じ場合、保存済みの完了結果を再利用します。", "muted")
        for widget in (text, obs):
            widget.configure(state="disabled")
        review_key = ((row or {}).get("path"), self.selected_run_id)
        # Keep a draft intact when a background result refreshes this same run.
        if review_key != getattr(self, "review_display_key", None):
            self.review_name.set((row or {}).get("review_name") or "")
            self.reviewed.set(bool((row or {}).get("reviewed")))
            self.review_note.delete("1.0", "end")
            self.review_note.insert("1.0", (row or {}).get("review_note") or "")
            self.review_display_key = review_key

    def crop(self):
        if self.selected_id is None:
            messagebox.showinfo("画像を選択", "範囲を指定する画像を選択してください。", parent=self)
            return
        row = self.store.asset(self.selected_id)
        try:
            CropDialog(self, row["path"], self.save_crop)
        except Exception as exc:
            messagebox.showerror("範囲指定", str(exc), parent=self)

    def save_crop(self, crop):
        if self.selected_id is None or self.busy:
            return
        self.store.set_crop(self.selected_id, crop)
        self.refresh_rows()
        self.redraw_preview()
        self.status.set("解析範囲を変更しました。「選択した画像を解析」で新しい範囲を評価できます。")

    def save_review(self):
        if not self.selected_run_id:
            messagebox.showinfo("解析結果なし", "解析結果を選択してから保存してください。", parent=self)
            return
        self.store.review(self.selected_run_id, self.review_name.get(), self.review_note.get("1.0", "end-1c"), self.reviewed.get())
        self.refresh_rows()
        self.status.set("人による確認を保存しました。AIの判定内容は保持しています。")

    def start_batch(self, scope):
        if self.busy or self.refreshing:
            return
        model = next((m for m in self.models if m["name"] == self.model.get()), None)
        if model is None:
            messagebox.showinfo("画像モデルを選択", "接続・モデル確認を実行して、画像対応モデルを選択してください。", parent=self)
            return
        try:
            OllamaClient(self.endpoint.get())
            config = build_config(model["name"], model["digest"], "discover" if self.mode.get() == "自由判定" else "target", self.target.get(), int(self.edge.get()), int(self.timeout.get()), {key: var.get() for key, var in self.weight_vars.items()})
            if self.use_tagger.get():
                ready, reason = tagger_availability()
                if not ready:
                    raise ValueError(reason + "。README.mdの準備手順を確認するか、専用モデルのチェックを外してください。")
                config["tagger"] = tagger_signature()
        except Exception as exc:
            messagebox.showerror("解析条件を確認", str(exc), parent=self)
            return
        rows = self.store.assets()
        if scope == "selected":
            ids = {int(i) for i in self.tree.selection()}
            rows = [r for r in rows if r["id"] in ids]
        elif scope == "errors":
            rows = [r for r in rows if r.get("status") == "error"]
        if not rows:
            messagebox.showinfo("解析する画像なし", "対象の画像を追加・選択してください。", parent=self)
            return
        self.save_settings()
        self.busy = True
        self.started_at = time.monotonic()
        self.completed_batch = set()
        self.current_stage = "モデルを確認"
        self.progress.configure(maximum=len(rows), value=0)
        self.cancel_event = threading.Event()
        self.engine = BatchEngine(self.store, self.endpoint.get(), config, self.emit, self.cancel_event)
        for widget, _ in self.locked_widgets:
            widget.configure(state="disabled")
        self.update_target_state()
        self.stop_btn.configure(state="normal")
        force = self.force.get()
        self.worker = threading.Thread(target=self.engine.run, args=(rows, force), daemon=True)
        self.worker.start()

    def stop(self):
        if self.engine and self.busy:
            self.engine.cancel()
            self.current_stage = "中断処理中… 完了済みの結果を保持しています"
            self.status.set(self.current_stage)
            self.stop_btn.configure(state="disabled")

    def tick(self):
        if self.busy and self.started_at:
            elapsed = int(time.monotonic() - self.started_at)
            self.status.set(f"{self.current_stage}　経過 {elapsed // 60}分{elapsed % 60:02d}秒")
        if not self.closing:
            self.after(1000, self.tick)

    def drain_events(self):
        refresh = False
        selected_changed = False
        # Never let streaming or thumbnail work monopolize Tk's event loop.
        started = time.monotonic()
        for _ in range(40):
            if time.monotonic() - started > 0.015:
                break
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "models":
                self.models = payload
                names = [m["name"] for m in payload]
                self.model_combo.configure(values=names)
                if self.model.get() not in names:
                    self.model.set(next((n for n in names if n in ("qwen3-vl:latest", "qwen3-vl:8b")), names[0] if names else ""))
                self.connection_text.set(f"接続済み · 画像モデル {len(names)}個" if names else "画像対応のローカルモデルがありません")
                self.refreshing = False
                self.refresh_btn.configure(state="normal")
            elif kind == "models_error":
                self.models = []
                self.connection_text.set("接続できません。Ollamaを起動して再確認してください。")
                self.status.set(payload)
                self.refreshing = False
                self.refresh_btn.configure(state="normal")
            elif kind == "files":
                self.store.add(payload)
                refresh = True
                self.status.set(f"{len(payload)}枚を一覧に追加しました。")
            elif kind == "stage":
                self.current_stage = payload["text"]
                if payload.get("index"):
                    self.current_stage = f"{payload['index']} / {payload['total']}枚目 · " + self.current_stage
            elif kind == "stream":
                self.current_stage = payload["text"] + f" · {payload['chars']:,}文字受信"
            elif kind == "item":
                refresh = True
                if payload["asset_id"] == self.selected_id:
                    selected_changed = True
                # Read only once below; the set tracks attempted images for progress.
                self.completed_batch.add(payload["asset_id"])
            elif kind == "thumbnail":
                if payload["epoch"] == self.thumb_epoch and self.tree.exists(str(payload["asset_id"])):
                    im = ImageTk.PhotoImage(payload["image"], master=self)
                    self.thumb_images[payload["asset_id"]] = im
                    self.tree.item(str(payload["asset_id"]), image=im)
            elif kind == "preview":
                self.apply_preview(payload)
            elif kind == "finished":
                self.busy = False
                self.started_at = None
                for widget, state in self.locked_widgets:
                    widget.configure(state=state)
                self.update_target_state()
                self.stop_btn.configure(state="disabled")
                self.status.set(f"{'中断' if self.cancel_event.is_set() else '処理終了'}：新規完了 {payload['done']}枚 / 保存済みを再利用 {payload['cached']}枚 / 失敗 {payload['error']}枚")
                self.progress.configure(value=payload["done"] + payload["cached"] + payload["error"])
                refresh = True
            elif kind == "fatal":
                messagebox.showerror("解析を開始できませんでした", payload, parent=self)
            elif kind == "notification_error":
                messagebox.showerror("処理できませんでした", payload, parent=self)
            elif kind == "exported":
                self.status.set(f"保存しました: {payload}")
        if refresh:
            self.refresh_rows()
            if self.busy:
                completed = sum(r["id"] in self.completed_batch and r.get("status") in ("done", "error") for r in self.rows)
                self.progress.configure(value=completed)
            if selected_changed:
                self.select_image()
        if self.closing and (not self.worker or not self.worker.is_alive()) and not self.refreshing and not any(t.is_alive() for t in self.background_jobs):
            self.store.close()
            self.destroy()
            return
        self.after(80, self.drain_events)

    def export(self, kind):
        rows = list(self.filtered)
        if not rows:
            messagebox.showinfo("画像なし", "書き出す画像がありません。", parent=self)
            return
        extension = "." + kind.lower()
        destination = filedialog.asksaveasfilename(parent=self, title=f"{kind}レポートを保存", defaultextension=extension, filetypes=[(kind, "*" + extension)], initialfile="character_lens_report" + extension)
        if not destination:
            return
        try:
            self.store.validate_destination(destination)
        except ValueError as exc:
            messagebox.showerror("保存先を変更", str(exc), parent=self)
            return
        self.status.set("レポートを保存しています…")
        def run():
            try:
                {"CSV": export_csv, "JSON": export_json, "HTML": export_html}[kind](rows, destination)
                self.emit("exported", destination)
            except Exception as exc:
                self.emit("notification_error", str(exc))
        self.start_background(run)

    def backup(self):
        destination = filedialog.asksaveasfilename(parent=self, title="履歴のバックアップを保存", defaultextension=".sqlite3", filetypes=[("履歴データ", "*.sqlite3")], initialfile="character_lens_backup.sqlite3")
        if destination:
            try:
                self.store.backup(destination)
                self.status.set(f"履歴をバックアップしました: {destination}")
            except Exception as exc:
                messagebox.showerror("バックアップ", str(exc), parent=self)

    def open_setup(self):
        self.save_settings()
        subprocess.Popen([sys.executable, str(ROOT / "app.py"), "--setup-only", "--data-dir", str(self.data_dir)], cwd=ROOT, **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))

    def close(self):
        if self.busy and not messagebox.askyesno("解析中です", "解析を中断して終了しますか？\n完了済みの結果と途中の観察は保存されています。", parent=self):
            return
        self.save_settings()
        self.closing = True
        self.stop()
        if not self.worker or not self.worker.is_alive():
            if not self.refreshing and not any(t.is_alive() for t in self.background_jobs):
                self.store.close()
                self.destroy()

    def report_callback_exception(self, exc, value, tb):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / "errors.log").open("a", encoding="utf-8") as output:
            traceback.print_exception(exc, value, tb, file=output)
        messagebox.showerror("画面処理のエラー", f"{value}\n詳細をdata/errors.logに保存しました。", parent=self)


def instance_lock(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = (data_dir / ".instance.lock").open("a+b")
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            raise RuntimeError("この履歴を使うCharacter Lensは、すでに起動しています。")
    else:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("この履歴を使うCharacter Lensは、すでに起動しています。")
    return handle


def main():
    parser = argparse.ArgumentParser(description="Character Lens")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    lock = None
    try:
        lock = instance_lock(args.data_dir)
        app = CharacterLens(args.data_dir, args.files)
        app.mainloop()
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Character Lensを起動できません", str(exc), parent=root)
        root.destroy()
        raise
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()

