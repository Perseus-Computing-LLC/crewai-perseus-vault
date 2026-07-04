# crewai-perseus-vault

**Long-term, local-first, encrypted memory for [CrewAI](https://crewai.com) agents — as explicit, agent-callable tools.**

`crewai-perseus-vault` wraps [Perseus Vault](https://github.com/Perseus-Computing-LLC/perseus-vault) (formerly "Mimir"/"Mneme" — an open-source, MIT-licensed persistent memory engine with 40+ MCP tools, FTS5 + dense hybrid search, and optional AES-256-GCM encryption) as standard CrewAI `BaseTool`s. Your agents get two first-class actions they can deliberately call:

- **`mimir_remember`** — persist a fact, decision, insight, or note that survives across runs.
- **`mimir_recall`** — search what was stored earlier.

### Why tools (and not CrewAI's built-in memory)?

CrewAI ships *implicit* memory (auto-captured short/long-term memory) and a generic MCP adapter. `crewai-perseus-vault` is deliberately different: it exposes **explicit, controllable memory** the agent chooses to invoke, with a typed `args_schema` so the LLM sees exactly what each call needs. Use it when you want the agent to reason about *what* to remember and *when* to recall — backed by a durable, encryptable store you own on disk.

## Prerequisite: the `mimir` binary

The tools talk to a local `mimir` process over JSON-RPC (MCP stdio). You need the `mimir` binary on your `PATH` (or pass an absolute path).

Install it from the [Perseus Vault repository](https://github.com/Perseus-Computing-LLC/perseus-vault) (build from source, or grab a release). Verify:

```bash
mimir --version
```

The tools spawn `mimir serve --db <db_path>` for you — you do **not** start it manually.

## Install

```bash
pip install crewai-perseus-vault
```

(or, from source: `pip install -e ".[test]"`)

## Quickstart

```python
from crewai import Agent, Crew, Task
from crewai_perseus_vault import build_perseus_vault_tools

# One shared mimir process backs both tools.
memory_tools = build_perseus_vault_tools(db_path="~/.mimir/data/crew.db")

researcher = Agent(
    role="Research Analyst",
    goal="Answer questions, remembering durable facts for next time.",
    backstory="You persist key findings to long-term memory and check it before answering.",
    tools=memory_tools,
    verbose=True,
)

remember_task = Task(
    description="Remember that the project deadline is 2026-08-15. Store it under key 'project-deadline'.",
    expected_output="Confirmation the deadline was stored.",
    agent=researcher,
)

recall_task = Task(
    description="What is the project deadline? Check your long-term memory.",
    expected_output="The project deadline date.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[remember_task, recall_task])
result = crew.kickoff()
print(result)
```

### Using the tool classes directly

```python
from crewai_perseus_vault import (
    PerseusVaultRememberTool,
    PerseusVaultRecallTool,
    PerseusVaultClient,
)

client = PerseusVaultClient(db_path="~/.mimir/data/crew.db")   # one shared process
remember = PerseusVaultRememberTool(client=client)
recall = PerseusVaultRecallTool(client=client)

agent = Agent(..., tools=[remember, recall])
```

If you omit `client`, each tool lazily starts its own `mimir serve` on first use
(configurable via `db_path` and `mimir_binary`).

### Encryption at rest

```python
tools = build_perseus_vault_tools(
    db_path="~/.mimir/data/crew.db",
    encryption_key="~/.mimir/key.b64",   # base64-encoded 32-byte AES-256-GCM key
)
```

## Tool reference

| Tool | Required args | Optional args |
|------|---------------|---------------|
| `mimir_remember` | `content`, `key` | `category` (default `insight`), `tags`, `importance` (0.0–1.0) |
| `mimir_recall` | `query` | `limit` (default 5), `category` |

Both return a JSON string. `mimir_recall` returns `{"query": ..., "results": [...]}`.

## How it works

`PerseusVaultClient` spawns `mimir serve --db <path>`, performs the MCP `initialize`
handshake, and issues id-correlated JSON-RPC requests with a per-call timeout
over stdin/stdout. The client core is adapted from the proven
[`adk-mimir-memory`](https://github.com/Perseus-Computing-LLC/adk-mimir-memory)
package.

## Development

```bash
pip install -e ".[test]"
pytest -q
```

Unit tests mock the `mimir` subprocess, so they run with no binary installed.
`tests/test_smoke_real_binary.py` runs an end-to-end round-trip against a real
`mimir` binary when one is found on `PATH` (otherwise it is skipped).

## License

MIT © 2026 Perseus Computing LLC. Perseus Vault (formerly Mimir/Mneme) is MIT-licensed by Perseus Computing LLC.
