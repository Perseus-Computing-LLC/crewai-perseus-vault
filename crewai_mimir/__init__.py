"""crewai-mimir: Mimir long-term memory as CrewAI tools.

Exposes explicit, agent-callable CrewAI tools that store and retrieve durable
memories in Mimir (github.com/Perseus-Computing-LLC/mimir), a local-first,
encrypted, persistent memory engine.

Example::

    from crewai import Agent
    from crewai_mimir import build_mimir_tools

    agent = Agent(role="Researcher", goal="...", backstory="...",
                  tools=build_mimir_tools())
"""

from ._client import MimirClient
from .tools import (
    MimirRecallInput,
    MimirRecallTool,
    MimirRememberInput,
    MimirRememberTool,
    build_mimir_tools,
)

__version__ = "0.1.0"

__all__ = [
    "MimirClient",
    "MimirRememberTool",
    "MimirRecallTool",
    "MimirRememberInput",
    "MimirRecallInput",
    "build_mimir_tools",
    "__version__",
]
