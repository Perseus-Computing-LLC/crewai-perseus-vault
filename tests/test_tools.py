"""Tests for crewai-perseus-vault tools against a fake Perseus Vault MCP stdio server.

No real ``mimir`` binary is required: ``subprocess.Popen`` is monkeypatched to
return an in-process fake that speaks JSON-RPC 2.0 over fake stdin/stdout pipes,
so these exercise the real RPC, handshake, and tool ``_run`` code paths.
"""

from __future__ import annotations

import json
import queue

import pytest
from pydantic import ValidationError

import crewai_perseus_vault._client as client_mod
from crewai_perseus_vault import (
    PerseusVaultClient,
    PerseusVaultRecallInput,
    PerseusVaultRecallTool,
    PerseusVaultRememberInput,
    PerseusVaultRememberTool,
    build_perseus_vault_tools,
)


# ── Fake Perseus Vault MCP stdio server ──────────────────────────────────────


class _FakeStdin:
    def __init__(self, on_line):
        self._on_line = on_line

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                self._on_line(line)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeStdout:
    """Blocking, iterable line source fed by the fake server."""

    def __init__(self):
        self._q = queue.Queue()

    def put(self, line):
        self._q.put(line)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item

    def close(self):
        self._q.put(None)


class FakePerseusVault:
    """Minimal Popen-compatible fake of the Perseus Vault MCP stdio server."""

    def __init__(self, cmd=None, **kwargs):
        self.cmd = cmd
        self.store = {}  # (category, key) -> body dict
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(self._handle)
        self._alive = True

    # -- Popen surface used by PerseusVaultClient --
    def terminate(self):
        self._alive = False
        self.stdout.close()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False

    # -- request dispatch --
    def _reply(self, req_id, result):
        self.stdout.put(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}))

    def _handle(self, line):
        req = json.loads(line)
        method = req.get("method")
        req_id = req.get("id")
        if req_id is None:
            return  # notification
        if method == "initialize":
            self._reply(req_id, {"protocolVersion": "2024-11-05"})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            self._reply(req_id, self._call_tool(name, args))
        else:
            self._reply(req_id, {})

    def _call_tool(self, name, args):
        if name == "perseus_vault_remember":
            key = (args.get("category"), args.get("key"))
            body = json.loads(args.get("body_json", "{}"))
            self.store[key] = {**args, "body": body}
            return {"structuredContent": {"stored": True, "key": args.get("key")}}
        if name == "perseus_vault_recall":
            query = (args.get("query") or "").lower()
            cat = args.get("category")
            limit = args.get("limit", 10)
            items = []
            for (category, key), entry in self.store.items():
                if cat and category != cat:
                    continue
                content = str(entry["body"].get("content", "")).lower()
                # OR semantics: any query token present in content.
                if any(tok and tok in content for tok in query.split()):
                    items.append(
                        {"key": key, "category": category, **entry["body"]}
                    )
            return {"structuredContent": {"items": items[:limit]}}
        return {"structuredContent": {}}


@pytest.fixture
def fake_popen(monkeypatch):
    """Patch subprocess.Popen + shutil.which so PerseusVaultClient uses FakePerseusVault."""
    created = {}

    def _fake_popen(cmd, **kwargs):
        fm = FakePerseusVault(cmd=cmd, **kwargs)
        created["proc"] = fm
        return fm

    monkeypatch.setattr(client_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(client_mod.shutil, "which", lambda b: "/fake/perseus-vault")
    return created


# ── args_schema validation (no client needed) ────────────────────────────────


def test_remember_input_requires_content_and_key():
    with pytest.raises(ValidationError):
        PerseusVaultRememberInput(key="k")  # missing content
    with pytest.raises(ValidationError):
        PerseusVaultRememberInput(content="c")  # missing key
    ok = PerseusVaultRememberInput(content="c", key="k")
    assert ok.category == "insight"
    assert ok.importance == 0.5


def test_remember_input_importance_bounds():
    with pytest.raises(ValidationError):
        PerseusVaultRememberInput(content="c", key="k", importance=1.5)
    with pytest.raises(ValidationError):
        PerseusVaultRememberInput(content="c", key="k", importance=-0.1)


def test_recall_input_requires_query_and_limit_bounds():
    with pytest.raises(ValidationError):
        PerseusVaultRecallInput()  # missing query
    with pytest.raises(ValidationError):
        PerseusVaultRecallInput(query="q", limit=0)  # below ge=1
    ok = PerseusVaultRecallInput(query="q")
    assert ok.limit == 5


def test_tool_metadata():
    t = PerseusVaultRememberTool.model_construct()
    assert t.name == "perseus_vault_remember"
    assert t.args_schema is PerseusVaultRememberInput
    r = PerseusVaultRecallTool.model_construct()
    assert r.name == "perseus_vault_recall"
    assert r.args_schema is PerseusVaultRecallInput


# ── _run round-trips against the fake server ──────────────────────────────────


def test_remember_then_recall(fake_popen):
    client = PerseusVaultClient(db_path="./_t/mimir.db")
    remember = PerseusVaultRememberTool(client=client)
    recall = PerseusVaultRecallTool(client=client)

    out = remember._run(
        content="Use PostgreSQL 16 for the main datastore.",
        key="use-postgres-16",
        category="decision",
        tags=["db"],
    )
    parsed = json.loads(out)
    assert parsed["status"] == "remembered"
    assert parsed["key"] == "use-postgres-16"

    out = recall._run(query="postgresql datastore", limit=5)
    parsed = json.loads(out)
    assert parsed["query"] == "postgresql datastore"
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["content"].startswith("Use PostgreSQL 16")
    client.close()


def test_recall_category_filter(fake_popen):
    client = PerseusVaultClient(db_path="./_t/mimir.db")
    remember = PerseusVaultRememberTool(client=client)
    recall = PerseusVaultRecallTool(client=client)

    remember._run(content="alpha fact", key="a", category="insight")
    remember._run(content="alpha decision", key="b", category="decision")

    out = json.loads(recall._run(query="alpha", category="decision"))
    assert len(out["results"]) == 1
    assert out["results"][0]["content"] == "alpha decision"
    client.close()


def test_run_through_crewai_tool_run_wrapper(fake_popen):
    """BaseTool.run(**kwargs) validates kwargs via args_schema then dispatches _run.

    Note: in crewai 1.15.x, run() forwards keyword arguments to _run; passing a
    single positional dict is NOT unpacked, so agents/tooling call with kwargs.
    """
    client = PerseusVaultClient(db_path="./_t/mimir.db")
    remember = PerseusVaultRememberTool(client=client)
    result = remember.run(
        content="wrapped via run()", key="w1", category="insight"
    )
    assert json.loads(result)["status"] == "remembered"
    out = json.loads(PerseusVaultRecallTool(client=client).run(query="wrapped"))
    assert any("wrapped" in r["content"] for r in out["results"])
    client.close()


def test_build_perseus_vault_tools_shares_client(fake_popen):
    tools = build_perseus_vault_tools(db_path="./_t/mimir.db")
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"perseus_vault_remember", "perseus_vault_recall"}
    assert tools[0].client is tools[1].client
    tools[0].client.close()


def test_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(client_mod.shutil, "which", lambda b: None)
    with pytest.raises(RuntimeError, match="perseus-vault binary not found"):
        PerseusVaultClient(
            db_path="./_t/mimir.db", perseus_vault_binary="perseus-vault"
        )
