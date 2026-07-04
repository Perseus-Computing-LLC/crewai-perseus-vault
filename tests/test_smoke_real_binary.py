"""Optional smoke test against a REAL ``perseus-vault`` binary.

Skipped automatically when no ``perseus-vault`` binary is on $PATH. When
present, it starts a real ``perseus-vault serve`` subprocess against a temp DB
and round-trips a remember -> recall through the actual MCP stdio transport.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from crewai_perseus_vault import (
    PerseusVaultClient,
    PerseusVaultRecallTool,
    PerseusVaultRememberTool,
)


def _find_perseus_vault() -> str | None:
    """Locate a runnable perseus-vault binary.

    Prefers $PERSEUS_VAULT_BINARY (or legacy $MIMIR_BINARY), then PATH for
    ``perseus-vault`` and the ``mimir`` compat name. On Windows the released
    binary may lack a ``.exe`` extension, so shutil.which() misses it; also
    probe ~/bin.
    """
    for env_var in ("PERSEUS_VAULT_BINARY", "MIMIR_BINARY"):
        env = os.environ.get(env_var)
        if env and os.path.exists(env):
            return env
    for name in ("perseus-vault", "mimir"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        "~/bin/perseus-vault",
        "~/bin/perseus-vault.exe",
        "~/bin/mimir",
        "~/bin/mimir.exe",
    ):
        path = os.path.expanduser(candidate)
        if os.path.exists(path):
            return path
    return None


_PERSEUS_VAULT = _find_perseus_vault()

pytestmark = pytest.mark.skipif(
    _PERSEUS_VAULT is None,
    reason=(
        "real perseus-vault binary not found "
        "(set PERSEUS_VAULT_BINARY or put perseus-vault on PATH)"
    ),
)


def test_real_remember_recall(tmp_path):
    db = tmp_path / "perseus-vault.db"
    client = PerseusVaultClient(db_path=str(db), perseus_vault_binary=_PERSEUS_VAULT)
    try:
        remember = PerseusVaultRememberTool(client=client)
        recall = PerseusVaultRecallTool(client=client)

        remember._run(
            content="The capital of crewai-perseus-vault testing is verification.",
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
