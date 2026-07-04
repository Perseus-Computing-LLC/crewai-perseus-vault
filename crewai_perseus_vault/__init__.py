"""crewai-perseus-vault: Perseus Vault long-term memory as CrewAI tools.

Exposes explicit, agent-callable CrewAI tools that store and retrieve durable
memories in Perseus Vault (github.com/Perseus-Computing-LLC/perseus-vault), a
local-first, encrypted, persistent memory engine.

Example::

    from crewai import Agent
    from crewai_perseus_vault import build_perseus_vault_tools

    agent = Agent(role="Researcher", goal="...", backstory="...",
                  tools=build_perseus_vault_tools())
"""

from ._client import PerseusVaultClient
from .tools import (
    PerseusVaultRecallInput,
    PerseusVaultRecallTool,
    PerseusVaultRememberInput,
    PerseusVaultRememberTool,
    build_perseus_vault_tools,
)

__version__ = "0.1.0"

__all__ = [
    "PerseusVaultClient",
    "PerseusVaultRememberTool",
    "PerseusVaultRecallTool",
    "PerseusVaultRememberInput",
    "PerseusVaultRecallInput",
    "build_perseus_vault_tools",
    "__version__",
]
