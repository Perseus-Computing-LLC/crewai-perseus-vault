"""Optional smoke test against a REAL ``mimir`` binary.

Skipped automatically when no ``mimir`` binary is on $PATH. When present, it
starts a real ``mimir serve`` subprocess against a temp DB and round-trips a
remember -> recall through the actual MCP stdio transport.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from crewai_mimir import MimirClient, MimirRecallTool, MimirRememberTool


def _find_mimir() -> str | None:
    """Locate a runnable mimir binary.

    Prefers $MIMIR_BINARY, then PATH. On Windows the released binary may lack a
    ``.exe`` extension, so shutil.which() misses it; also probe ~/bin/mimir.
    """
    env = os.environ.get("MIMIR_BINARY")
    if env and os.path.exists(env):
        return env
    found = shutil.which("mimir")
    if found:
        return found
    candidate = os.path.expanduser("~/bin/mimir")
    if os.path.exists(candidate):
        return candidate
    candidate_exe = os.path.expanduser("~/bin/mimir.exe")
    if os.path.exists(candidate_exe):
        return candidate_exe
    return None


_MIMIR = _find_mimir()

pytestmark = pytest.mark.skipif(
    _MIMIR is None,
    reason="real mimir binary not found (set MIMIR_BINARY or put mimir on PATH)",
)


def test_real_remember_recall(tmp_path):
    db = tmp_path / "mimir.db"
    client = MimirClient(db_path=str(db), mimir_binary=_MIMIR)
    try:
        remember = MimirRememberTool(client=client)
        recall = MimirRecallTool(client=client)

        remember._run(
            content="The capital of crewai-mimir testing is verification.",
            key="smoke-fact",
            category="insight",
            tags=["smoke"],
        )
        out = json.loads(recall._run(query="verification capital", limit=5))
        assert any(
            "verification" in str(r.get("content", "")).lower()
            for r in out["results"]
        ), f"expected memory not recalled: {out}"
    finally:
        client.close()
