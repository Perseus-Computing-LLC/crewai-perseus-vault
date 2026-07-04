"""CrewAI tools that wrap the Perseus Vault memory engine.

These are explicit, agent-callable tools (subclasses of ``crewai.tools.BaseTool``)
that let a CrewAI agent deliberately store and retrieve durable memories in
Perseus Vault.  Unlike CrewAI's built-in (implicit) memory or a generic MCP
adapter, these surface ``remember`` and ``recall`` as first-class actions the
agent chooses to invoke, with a typed ``args_schema`` so the LLM sees exactly
what each call needs.

Tools:
    PerseusVaultRememberTool — store a fact/decision/note in Perseus Vault.
    PerseusVaultRecallTool   — search Perseus Vault for stored memories.

Both tools share a single
:class:`~crewai_perseus_vault._client.PerseusVaultClient` so one
``perseus-vault serve`` subprocess backs the whole crew.
"""

from __future__ import annotations

import json
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ._client import PerseusVaultClient

__all__ = [
    "PerseusVaultRememberInput",
    "PerseusVaultRecallInput",
    "PerseusVaultRememberTool",
    "PerseusVaultRecallTool",
    "build_perseus_vault_tools",
]


# ── args schemas ────────────────────────────────────────────────────────────


class PerseusVaultRememberInput(BaseModel):
    """Input schema for :class:`PerseusVaultRememberTool`."""

    content: str = Field(
        ...,
        description="The fact, decision, insight, or note to remember. Stored "
        "verbatim and made searchable.",
    )
    key: str = Field(
        ...,
        description="A short unique identifier for this memory within its "
        "category, e.g. 'use-postgres-16'. Re-using a key updates that memory.",
    )
    category: str = Field(
        default="insight",
        description="Memory category: 'decision', 'architecture', 'convention', "
        "'insight', or a custom label.",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Optional tags for cross-referencing this memory.",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial importance 0.0-1.0; sets the starting decay score.",
    )


class PerseusVaultRecallInput(BaseModel):
    """Input schema for :class:`PerseusVaultRecallTool`."""

    query: str = Field(
        ...,
        description="Search query. Keywords are OR'd together for broad recall.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Maximum number of memories to return.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optionally restrict the search to one category.",
    )


# ── tools ───────────────────────────────────────────────────────────────────


class PerseusVaultRememberTool(BaseTool):
    """Store a durable memory in Perseus Vault.

    Pass a shared :class:`PerseusVaultClient` (recommended, so all tools reuse
    one ``perseus-vault serve`` process), or let the tool lazily start its own
    using ``db_path`` / ``perseus_vault_binary``.
    """

    name: str = "perseus_vault_remember"
    description: str = (
        "Persist a fact, decision, insight, or note to long-term memory "
        "(Perseus Vault) so it survives across sessions. Provide the content "
        "and a short unique key. Use this whenever you learn something worth "
        "remembering later."
    )
    args_schema: Type[BaseModel] = PerseusVaultRememberInput

    # Non-schema configuration (excluded from the LLM-facing args_schema).
    client: Optional[PerseusVaultClient] = None
    db_path: str = "~/.mimir/data/mimir.db"
    perseus_vault_binary: str = "perseus-vault"

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> PerseusVaultClient:
        if self.client is None:
            self.client = PerseusVaultClient(
                db_path=self.db_path,
                perseus_vault_binary=self.perseus_vault_binary,
            )
        return self.client

    def _run(
        self,
        content: str,
        key: str,
        category: str = "insight",
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
    ) -> str:
        client = self._get_client()
        result = client.call_tool(
            "perseus_vault_remember",
            {
                "category": category,
                "key": key,
                "body_json": json.dumps({"content": content}),
                "tags": tags or [],
                "importance": importance,
            },
        )
        return json.dumps(
            {
                "status": "remembered",
                "category": category,
                "key": key,
                "perseus_vault": result,
            }
        )


class PerseusVaultRecallTool(BaseTool):
    """Search Perseus Vault for previously stored memories.

    Pass a shared :class:`PerseusVaultClient` (recommended), or let the tool
    lazily start its own using ``db_path`` / ``perseus_vault_binary``.
    """

    name: str = "perseus_vault_recall"
    description: str = (
        "Search long-term memory (Perseus Vault) for facts, decisions, or notes "
        "stored earlier. Returns the best-matching memories. Use this before "
        "answering to check what you already know."
    )
    args_schema: Type[BaseModel] = PerseusVaultRecallInput

    client: Optional[PerseusVaultClient] = None
    db_path: str = "~/.mimir/data/mimir.db"
    perseus_vault_binary: str = "perseus-vault"

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> PerseusVaultClient:
        if self.client is None:
            self.client = PerseusVaultClient(
                db_path=self.db_path,
                perseus_vault_binary=self.perseus_vault_binary,
            )
        return self.client

    def _run(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> str:
        client = self._get_client()
        arguments: dict = {"query": query, "limit": limit}
        if category:
            arguments["category"] = category
        result = client.call_tool("perseus_vault_recall", arguments)
        items = result.get("items", result) if isinstance(result, dict) else result
        return json.dumps({"query": query, "results": items})


def build_perseus_vault_tools(
    db_path: str = "~/.mimir/data/mimir.db",
    perseus_vault_binary: str = "perseus-vault",
    encryption_key: Optional[str] = None,
) -> list[BaseTool]:
    """Convenience: build remember+recall tools sharing one Perseus Vault process.

    Args:
        db_path: Path to the Perseus Vault SQLite database.
        perseus_vault_binary: Name or absolute path of the ``perseus-vault``
            executable.
        encryption_key: Optional path to an AES-256-GCM key file.

    Returns:
        ``[PerseusVaultRememberTool, PerseusVaultRecallTool]`` backed by a
        single client.
    """
    client = PerseusVaultClient(
        db_path=db_path,
        perseus_vault_binary=perseus_vault_binary,
        encryption_key=encryption_key,
    )
    return [
        PerseusVaultRememberTool(client=client),
        PerseusVaultRecallTool(client=client),
    ]
