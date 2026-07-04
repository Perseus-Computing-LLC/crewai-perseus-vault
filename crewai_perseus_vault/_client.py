"""Minimal Perseus Vault MCP stdio client.

Perseus Vault (github.com/Perseus-Computing-LLC/perseus-vault, formerly
"Mimir"/"Mneme") is an open-source (MIT) local-first, encrypted, persistent
memory engine exposing 40+ tools over the Model Context Protocol.  This module
talks to the ``perseus-vault`` binary via JSON-RPC 2.0 over stdin/stdout (the
MCP stdio transport).

The client core (spawn subprocess, background stdout reader, id-correlated RPC
with timeout, MCP initialize handshake) is adapted from the proven
``Perseus-Computing-LLC/adk-mimir-memory`` package.

Requirements:
    A ``perseus-vault`` binary must be on ``$PATH`` or passed explicitly.  Build
    from source or install from
    https://github.com/Perseus-Computing-LLC/perseus-vault.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import subprocess
import threading
import time

__all__ = ["PerseusVaultClient"]


class PerseusVaultClient:
    """Thread-safe JSON-RPC client for a ``perseus-vault serve`` stdio subprocess.

    The client spawns ``perseus-vault serve --db <db_path>`` and performs the MCP
    initialize handshake on construction.  Call :meth:`call_tool` to invoke any
    Perseus Vault MCP tool by name.

    Attributes:
        db_path: Filesystem path to the Perseus Vault SQLite database.
    """

    def __init__(
        self,
        db_path: str = "~/.mimir/data/mimir.db",
        perseus_vault_binary: str = "perseus-vault",
        timeout_s: float = 30.0,
        encryption_key: str | None = None,
    ) -> None:
        """Initializes and starts the Perseus Vault client.

        Args:
            db_path: Path to the Perseus Vault SQLite database.  Created if absent.
            perseus_vault_binary: Name or absolute path of the ``perseus-vault``
                executable.  Defaults to ``perseus-vault`` (a ``mimir`` compat
                symlink may also exist, but is not guaranteed on all installs).
            timeout_s: Per-RPC response timeout, guarding against a hung server.
            encryption_key: Optional path to an AES-256-GCM key file; enables
                encryption at rest.

        Raises:
            RuntimeError: If the ``perseus-vault`` binary cannot be found.
        """
        self.db_path = os.path.expanduser(db_path)
        self._timeout_s = timeout_s

        if os.path.isabs(perseus_vault_binary):
            self._perseus_vault_binary = perseus_vault_binary
        else:
            resolved = shutil.which(perseus_vault_binary)
            if resolved is None:
                raise RuntimeError(
                    "perseus-vault binary not found on $PATH (looked for "
                    f"'{perseus_vault_binary}'). Install Perseus Vault from "
                    "https://github.com/Perseus-Computing-LLC/perseus-vault "
                    "or pass the absolute path via perseus_vault_binary=."
                )
            self._perseus_vault_binary = resolved

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        cmd = [self._perseus_vault_binary, "serve", "--db", self.db_path]
        if encryption_key:
            cmd += ["--encryption-key", os.path.expanduser(encryption_key)]

        # stderr is discarded: nothing drains it, so a chatty server filling the
        # OS pipe buffer would block on its stderr write while we wait on stdout
        # (a two-pipe deadlock).
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._lock = threading.Lock()
        self._request_id = 0

        # Background reader: pump stdout lines into a queue so _rpc can wait with
        # a timeout and correlate responses by id rather than block forever.
        self._recv: queue.Queue = queue.Queue()
        proc_stdout = self._proc.stdout

        def _pump() -> None:
            try:
                for line in proc_stdout:
                    self._recv.put(line)
            except Exception:
                pass
            finally:
                self._recv.put(None)  # EOF sentinel

        self._reader = threading.Thread(target=_pump, daemon=True)
        self._reader.start()

        # MCP handshake: initialize, then the required initialized notification.
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "crewai-perseus-vault", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})

        atexit.register(self.close)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Terminates the Perseus Vault subprocess (idempotent)."""
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "PerseusVaultClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- JSON-RPC core ------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _rpc(self, method: str, params: object) -> dict:
        """Sends a JSON-RPC request and returns its ``result`` dict.

        Holds the lock for the whole request/response exchange so pairs never
        interleave, and honors ``timeout_s`` so a hung server cannot block the
        caller forever.

        Raises:
            RuntimeError: On transport failure, RPC error, or timeout.
        """
        with self._lock:
            req_id = self._next_id()
            req = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            payload = json.dumps(req, default=str)
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(
                    f"Perseus Vault subprocess communication failed: {e}. "
                    "The perseus-vault process may have crashed."
                ) from e

            deadline = time.monotonic() + self._timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Perseus Vault RPC '{method}' timed out after {self._timeout_s}s."
                    )
                try:
                    raw = self._recv.get(timeout=remaining)
                except queue.Empty:
                    raise RuntimeError(
                        f"Perseus Vault RPC '{method}' timed out after {self._timeout_s}s."
                    )
                if raw is None:
                    raise RuntimeError(
                        "Perseus Vault subprocess closed its output (it may have crashed)."
                    )
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    resp = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # non-JSON noise on stdout
                if resp.get("id") != req_id:
                    continue  # notification or a stale/other reply
                if "error" in resp:
                    err = resp["error"]
                    raise RuntimeError(
                        f"Perseus Vault RPC error [{err.get('code')}]: {err.get('message')}"
                    )
                return resp.get("result", {})

    def _notify(self, method: str, params: object) -> None:
        """Sends a JSON-RPC notification (no id, no response expected)."""
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        with self._lock:
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    # -- public API ---------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Calls a Perseus Vault MCP tool and returns its structured result.

        Args:
            name: The Perseus Vault tool name, e.g. ``perseus_vault_remember``
                or ``perseus_vault_recall``.
            arguments: The tool arguments.

        Returns:
            The tool's ``structuredContent`` if present, otherwise the parsed
            text content, otherwise ``{}``.
        """
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        sc = result.get("structuredContent")
        if sc is not None:
            return sc
        content = result.get("content", [])
        if content:
            try:
                return json.loads(content[0].get("text", "{}"))
            except (json.JSONDecodeError, IndexError, KeyError, AttributeError):
                # Surface raw text when it is not JSON.
                try:
                    return {"text": content[0].get("text", "")}
                except (IndexError, KeyError, AttributeError):
                    pass
        return {}
