"""CrewAI tools that wrap the Mimir memory engine.

These are explicit, agent-callable tools (subclasses of ``crewai.tools.BaseTool``)
that let a CrewAI agent deliberately store and retrieve durable memories in
Mimir.  Unlike CrewAI's built-in (implicit) memory or a generic MCP adapter,
these surface ``remember`` and ``recall`` as first-class actions the agent
chooses to invoke, with a typed ``args_schema`` so the LLM sees exactly what
each call needs.

Tools:
    MimirRememberTool — store a fact/decision/note in Mimir.
    MimirRecallTool   — search Mimir for previously stored memories.

Both tools share a single :class:`~crewai_mimir._client.MimirClient` so one
``mimir serve`` subprocess backs the whole crew.
"""

from __future__ import annotations

import json
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ._client import MimirClient

__all__ = [
    "MimirRememberInput",
    "MimirRecallInput",
    "MimirRememberTool",
    "MimirRecallTool",
    "build_mimir_tools",
]


# ── args schemas ────────────────────────────────────────────────────────────


class MimirRememberInput(BaseModel):
    """Input schema for :class:`MimirRememberTool`."""

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


class MimirRecallInput(BaseModel):
    """Input schema for :class:`MimirRecallTool`."""

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


class MimirRememberTool(BaseTool):
    """Store a durable memory in Mimir.

    Pass a shared :class:`MimirClient` (recommended, so all tools reuse one
    ``mimir serve`` process), or let the tool lazily start its own using
    ``db_path`` / ``mimir_binary``.
    """

    name: str = "mimir_remember"
    description: str = (
        "Persist a fact, decision, insight, or note to long-term memory (Mimir) "
        "so it survives across sessions. Provide the content and a short unique "
        "key. Use this whenever you learn something worth remembering later."
    )
    args_schema: Type[BaseModel] = MimirRememberInput

    # Non-schema configuration (excluded from the LLM-facing args_schema).
    client: Optional[MimirClient] = None
    db_path: str = "~/.mimir/data/mimir.db"
    mimir_binary: str = "mimir"

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> MimirClient:
        if self.client is None:
            self.client = MimirClient(
                db_path=self.db_path, mimir_binary=self.mimir_binary
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
            "mimir_remember",
            {
                "category": category,
                "key": key,
                "body_json": json.dumps({"content": content}),
                "tags": tags or [],
                "importance": importance,
            },
        )
        return json.dumps(
            {"status": "remembered", "category": category, "key": key, "mimir": result}
        )


class MimirRecallTool(BaseTool):
    """Search Mimir for previously stored memories.

    Pass a shared :class:`MimirClient` (recommended), or let the tool lazily
    start its own using ``db_path`` / ``mimir_binary``.
    """

    name: str = "mimir_recall"
    description: str = (
        "Search long-term memory (Mimir) for facts, decisions, or notes stored "
        "earlier. Returns the best-matching memories. Use this before answering "
        "to check what you already know."
    )
    args_schema: Type[BaseModel] = MimirRecallInput

    client: Optional[MimirClient] = None
    db_path: str = "~/.mimir/data/mimir.db"
    mimir_binary: str = "mimir"

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> MimirClient:
        if self.client is None:
            self.client = MimirClient(
                db_path=self.db_path, mimir_binary=self.mimir_binary
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
        result = client.call_tool("mimir_recall", arguments)
        items = result.get("items", result) if isinstance(result, dict) else result
        return json.dumps({"query": query, "results": items})


def build_mimir_tools(
    db_path: str = "~/.mimir/data/mimir.db",
    mimir_binary: str = "mimir",
    encryption_key: Optional[str] = None,
) -> list[BaseTool]:
    """Convenience: build remember+recall tools sharing one Mimir process.

    Args:
        db_path: Path to the Mimir SQLite database.
        mimir_binary: Name or absolute path of the ``mimir`` executable.
        encryption_key: Optional path to an AES-256-GCM key file.

    Returns:
        ``[MimirRememberTool, MimirRecallTool]`` backed by a single client.
    """
    client = MimirClient(
        db_path=db_path, mimir_binary=mimir_binary, encryption_key=encryption_key
    )
    return [
        MimirRememberTool(client=client),
        MimirRecallTool(client=client),
    ]
