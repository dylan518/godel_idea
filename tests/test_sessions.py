"""Tests for EvoScientist.sessions — thread CRUD, ID generation, helpers."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from EvoScientist.sessions import (
    AGENT_NAME,
    _format_relative_time,
    delete_thread,
    find_similar_threads,
    generate_thread_id,
    get_db_path,
    get_most_recent,
    get_thread_metadata,
    list_threads,
    thread_exists,
)


from tests.conftest import run_async as _run


class TestGenerateThreadId(unittest.TestCase):
    def test_length(self):
        tid = generate_thread_id()
        self.assertEqual(len(tid), 8)

    def test_hex(self):
        tid = generate_thread_id()
        int(tid, 16)  # Should not raise

    def test_uniqueness(self):
        ids = {generate_thread_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestGetDbPath(unittest.TestCase):
    def test_uses_config_dir(self):
        path = get_db_path()
        self.assertTrue(str(path).endswith("sessions.db"))
        self.assertIn(".config", str(path))
        self.assertIn("evoscientist", str(path))


class TestFormatRelativeTime(unittest.TestCase):
    def test_none(self):
        self.assertEqual(_format_relative_time(None), "")

    def test_invalid(self):
        self.assertEqual(_format_relative_time("not-a-date"), "")

    def test_recent(self):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        result = _format_relative_time(now)
        self.assertIn("just now", result)

    def test_minutes(self):
        from datetime import datetime, timedelta, timezone

        ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        result = _format_relative_time(ts)
        self.assertIn("min ago", result)

    def test_hours(self):
        from datetime import datetime, timedelta, timezone

        ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        result = _format_relative_time(ts)
        self.assertIn("hour", result)

    def test_days(self):
        from datetime import datetime, timedelta, timezone

        ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        result = _format_relative_time(ts)
        self.assertIn("day", result)

    def test_months(self):
        from datetime import datetime, timedelta, timezone

        ts = (datetime.now(timezone.utc) - timedelta(days=65)).isoformat()
        result = _format_relative_time(ts)
        self.assertIn("month", result)


class TestThreadFunctions(unittest.TestCase):
    """Tests using a real temporary SQLite database."""

    @classmethod
    def setUpClass(cls):
        """Create a temp DB and populate with test data."""
        cls._tmpdir = tempfile.mkdtemp()
        cls._db_path = os.path.join(cls._tmpdir, "test_sessions.db")

        async def _setup():
            import aiosqlite

            async with aiosqlite.connect(cls._db_path) as conn:
                # Create tables matching LangGraph checkpoint schema
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        blob BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                """)

                # Insert test checkpoints
                for i, tid in enumerate(["abc12345", "abc12399", "def00001"]):
                    meta = json.dumps(
                        {
                            "agent_name": AGENT_NAME,
                            "updated_at": f"2025-01-{15 + i}T10:00:00+00:00",
                            "workspace_dir": f"/tmp/ws_{tid}",
                            "model": "claude-sonnet-4-5",
                        }
                    )
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                        (tid, f"cp_{i}", meta),
                    )

                # Insert a non-EvoScientist checkpoint (should be filtered)
                other_meta = json.dumps(
                    {
                        "agent_name": "OtherAgent",
                        "updated_at": "2025-01-20T10:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    ("zzz99999", "cp_other", other_meta),
                )
                await conn.commit()

        _run(_setup())

        # Patch get_db_path to point to our temp DB
        cls._patcher = patch(
            "EvoScientist.sessions.get_db_path",
            return_value=type(
                "P",
                (),
                {
                    "__str__": lambda s: cls._db_path,
                    "__fspath__": lambda s: cls._db_path,
                },
            )(),
        )
        cls._patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        try:
            os.unlink(cls._db_path)
            os.rmdir(cls._tmpdir)
        except OSError:
            pass

    def test_list_threads(self):
        threads = _run(list_threads(limit=10))
        # Should only contain EvoScientist threads
        self.assertEqual(len(threads), 3)
        # Most recent first
        self.assertEqual(threads[0]["thread_id"], "def00001")

    def test_list_threads_with_message_count(self):
        threads = _run(list_threads(limit=10, include_message_count=True))
        self.assertIn("message_count", threads[0])

    def test_thread_exists_true(self):
        self.assertTrue(_run(thread_exists("abc12345")))

    def test_thread_exists_false(self):
        self.assertFalse(_run(thread_exists("nonexist")))

    def test_find_similar(self):
        similar = _run(find_similar_threads("abc1"))
        self.assertEqual(len(similar), 2)
        self.assertIn("abc12345", similar)
        self.assertIn("abc12399", similar)

    def test_find_similar_no_match(self):
        similar = _run(find_similar_threads("xyz"))
        self.assertEqual(len(similar), 0)

    def test_get_most_recent(self):
        recent = _run(get_most_recent())
        self.assertIsNotNone(recent)
        self.assertEqual(recent, "def00001")

    def test_get_thread_metadata(self):
        meta = _run(get_thread_metadata("abc12345"))
        self.assertIsNotNone(meta)
        self.assertEqual(meta["workspace_dir"], "/tmp/ws_abc12345")
        self.assertEqual(meta["model"], "claude-sonnet-4-5")

    def test_get_thread_metadata_missing(self):
        meta = _run(get_thread_metadata("nonexist"))
        self.assertIsNone(meta)

    def test_delete_thread(self):
        # Insert a thread to delete
        async def _insert():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                meta = json.dumps(
                    {
                        "agent_name": AGENT_NAME,
                        "updated_at": "2025-01-01T00:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    ("todelete", "cp_del", meta),
                )
                await conn.commit()

        _run(_insert())

        self.assertTrue(_run(thread_exists("todelete")))
        self.assertTrue(_run(delete_thread("todelete")))
        self.assertFalse(_run(thread_exists("todelete")))

    def test_delete_nonexistent(self):
        self.assertFalse(_run(delete_thread("nope1234")))

    # -- Agent isolation: OtherAgent data should never be visible --

    def test_thread_exists_ignores_other_agent(self):
        self.assertFalse(_run(thread_exists("zzz99999")))

    def test_find_similar_ignores_other_agent(self):
        similar = _run(find_similar_threads("zzz"))
        self.assertEqual(len(similar), 0)

    def test_get_metadata_ignores_other_agent(self):
        meta = _run(get_thread_metadata("zzz99999"))
        self.assertIsNone(meta)

    def test_delete_ignores_other_agent(self):
        # Should not delete OtherAgent's data
        self.assertFalse(_run(delete_thread("zzz99999")))

    def test_delete_thread_preserves_other_agent_writes(self):
        """Deleting a shared thread_id must only remove writes linked to
        EvoScientist checkpoints, leaving OtherAgent's writes intact."""

        shared_tid = "shared01"

        async def _insert():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                # EvoScientist checkpoint + write
                evo_meta = json.dumps(
                    {
                        "agent_name": AGENT_NAME,
                        "updated_at": "2025-02-01T00:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    (shared_tid, "cp_evo_shared", evo_meta),
                )
                await conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob) "
                    "VALUES (?, '', ?, 't1', 0, 'ch', 'str', X'AA')",
                    (shared_tid, "cp_evo_shared"),
                )

                # OtherAgent checkpoint + write on the SAME thread_id
                other_meta = json.dumps(
                    {
                        "agent_name": "OtherAgent",
                        "updated_at": "2025-02-01T00:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    (shared_tid, "cp_other_shared", other_meta),
                )
                await conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob) "
                    "VALUES (?, '', ?, 't2', 0, 'ch', 'str', X'BB')",
                    (shared_tid, "cp_other_shared"),
                )
                await conn.commit()

        _run(_insert())

        # Delete — should only affect EvoScientist's data
        _run(delete_thread(shared_tid))

        # Verify OtherAgent's writes survive
        async def _check():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT checkpoint_id FROM writes WHERE thread_id = ?",
                    (shared_tid,),
                ) as cur:
                    rows = await cur.fetchall()
                return [r[0] for r in rows]

        remaining = _run(_check())
        self.assertIn("cp_other_shared", remaining)
        self.assertNotIn("cp_evo_shared", remaining)


if __name__ == "__main__":
    unittest.main()
