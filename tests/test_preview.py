"""Regression coverage for blanking, redundant reloads and stale preview jobs."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import CharacterLens


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "sample.png"
        self.path.write_bytes(b"sample")
        self.canvas = Mock()
        self.canvas.winfo_width.return_value = 400
        self.canvas.winfo_height.return_value = 500
        self.canvas.create_image.return_value = 17
        self.host = SimpleNamespace(
            preview=self.canvas, preview_timer=None, preview_epoch=0,
            preview_key=None, preview_item=17, photo=object(), selected_id=1,
            selected_run_id=None, showing_history=False, history_crop=None,
            store=Mock(), preview_caption=Mock(), after_cancel=Mock(), emit=Mock())
        self.host.store.asset.return_value = {"path": str(self.path), "crop": None}

    def tearDown(self):
        self.temp.cleanup()

    def test_same_selection_and_size_do_not_restart_load_or_blank_image(self):
        with patch("app.threading.Thread") as thread:
            for _ in range(8):
                CharacterLens.redraw_preview(self.host)
        self.assertEqual(thread.call_count, 1)
        self.canvas.delete.assert_not_called()
        self.assertEqual(self.host.preview_epoch, 1)

    def test_source_crop_and_size_changes_reload_without_blanking(self):
        with patch("app.threading.Thread") as thread:
            CharacterLens.redraw_preview(self.host)
            self.path.write_bytes(b"changed source")
            CharacterLens.redraw_preview(self.host)
            self.host.store.asset.return_value["crop"] = [0, 0, 10, 10]
            CharacterLens.redraw_preview(self.host)
            self.canvas.winfo_width.return_value = 600
            CharacterLens.redraw_preview(self.host)
        self.assertEqual(thread.call_count, 4)
        self.canvas.delete.assert_not_called()

    def test_ready_image_replaces_existing_item_without_deleting_it(self):
        photo = object()
        with patch("app.ImageTk.PhotoImage", return_value=photo):
            CharacterLens.apply_preview(self.host, {"epoch": 0, "image": object(), "caption": "ready"})
        self.canvas.delete.assert_not_called()
        self.canvas.create_image.assert_not_called()
        self.canvas.itemconfigure.assert_called_once_with(17, image=photo)
        self.assertIs(self.host.photo, photo)

    def test_stale_result_cannot_replace_new_selection(self):
        self.host.preview_epoch = 2
        CharacterLens.apply_preview(self.host, {"epoch": 1, "image": None, "caption": "old failure"})
        self.canvas.delete.assert_not_called()
        self.host.preview_caption.set.assert_not_called()

    def test_current_failure_clears_wrong_image_and_allows_retry(self):
        self.host.preview_key = ("pending",)
        CharacterLens.apply_preview(self.host, {"epoch": 0, "image": None, "caption": "failed"})
        self.canvas.delete.assert_called_once_with("all")
        self.assertIsNone(self.host.preview_key)
        self.assertIsNone(self.host.photo)

    def test_direct_refresh_cancels_pending_resize_callback(self):
        self.host.preview_timer = "after#1"
        with patch("app.threading.Thread"):
            CharacterLens.redraw_preview(self.host)
        self.host.after_cancel.assert_called_once_with("after#1")
        self.assertIsNone(self.host.preview_timer)

