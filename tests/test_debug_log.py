from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from simple_api_router import debug_log
from simple_api_router.app import _reapply_debug_log


def _cfg(debug_log_value):
    """Minimal config stand-in exposing .server.debug_log (duck-typed)."""
    return SimpleNamespace(server=SimpleNamespace(debug_log=debug_log_value))


class TestDebugLogConfigure(unittest.TestCase):
    """debug_log.configure() must support enabling, disabling, and re-pathing."""

    def setUp(self):
        self._saved = debug_log._path
        debug_log._path = None

    def tearDown(self):
        debug_log._path = self._saved

    def test_configure_path_enables_and_writes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "dbg.log"   # parent dir does not exist yet
            debug_log.configure(str(p))
            self.assertTrue(debug_log.enabled())
            debug_log.log("req1", "stage1", "hello-body")
            self.assertTrue(p.exists())
            text = p.read_text(encoding="utf-8")
            self.assertIn("hello-body", text)
            self.assertIn("req=req1", text)

    def test_configure_none_disables_and_log_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dbg.log"
            debug_log.configure(str(p))
            self.assertTrue(debug_log.enabled())
            # Disable via None.
            debug_log.configure(None)
            self.assertFalse(debug_log.enabled())
            # log() must be a complete no-op now — nothing written.
            p.unlink(missing_ok=True)
            debug_log.log("req2", "stage1", "should-not-write")
            self.assertFalse(p.exists())

    def test_configure_empty_string_disables(self):
        debug_log.configure("")
        self.assertFalse(debug_log.enabled())


class TestReapplyDebugLogHotReload(unittest.TestCase):
    """_reapply_debug_log() must enable/disable/repath on config hot-reload."""

    def setUp(self):
        self._saved = debug_log._path
        debug_log._path = None
        self.logger = logging.getLogger("test_debug_log")

    def tearDown(self):
        debug_log._path = self._saved

    def test_enable_on_reload(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "dbg.log")
            changed = _reapply_debug_log(_cfg(None), _cfg(p), self.logger)
            self.assertTrue(changed)
            self.assertTrue(debug_log.enabled())

    def test_disable_on_reload(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "dbg.log")
            _reapply_debug_log(None, _cfg(p), self.logger)   # startup enable
            self.assertTrue(debug_log.enabled())
            changed = _reapply_debug_log(_cfg(p), _cfg(None), self.logger)
            self.assertTrue(changed)
            self.assertFalse(debug_log.enabled())

    def test_repath_on_reload(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = str(Path(d) / "a.log")
            p2 = str(Path(d) / "b.log")
            _reapply_debug_log(None, _cfg(p1), self.logger)
            changed = _reapply_debug_log(_cfg(p1), _cfg(p2), self.logger)
            self.assertTrue(changed)
            self.assertTrue(debug_log.enabled())
            self.assertEqual(debug_log._path, Path(p2))

    def test_unchanged_value_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "dbg.log")
            _reapply_debug_log(None, _cfg(p), self.logger)
            changed = _reapply_debug_log(_cfg(p), _cfg(p), self.logger)
            self.assertFalse(changed)
            self.assertTrue(debug_log.enabled())

    def test_startup_none_stays_disabled(self):
        changed = _reapply_debug_log(None, _cfg(None), self.logger)
        self.assertFalse(changed)
        self.assertFalse(debug_log.enabled())


if __name__ == "__main__":
    unittest.main()
