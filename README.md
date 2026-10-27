# OpenCode Go Agentic Starter — usage-first, quality-second

This is a project-local starter kit for OpenCode Go. It is intentionally conservative: use a high-throughput model for routine work, invoke stronger coding/review models only when the task warrants it, and avoid wasteful multi-agent fan-out.

## Model routing

| Job | Model in this kit | Why |
|---|---|---|
| Default conversation / ordinary edits | `opencode-go/mimo-v2.6-flash` | High-throughput default for everyday tasks |
| Lightweight scouting and docs | `opencode-go/mimo-v2.6-flash` | Avoid spending specialist quota on reading/summarizing |
| First-pass review and debugging | `opencode-go/deepseek-v4.1-flash` | Cheap first review before escalating |
| Difficult implementation / deep review | `opencode-go/kimi-k2.7-code` | Specialist model; call intentionally |
| Architecture decisions | `opencode-go/deepseek-v4-pro` | Reserve for genuinely consequential design choices |

OpenCode Go model availability and quotas can change. Check `/models` after connecting Go. If any configured model is missing from your account/region, replace its ID with one shown by `/models`. Muse Spark has region limitations, so this kit does not depend on it.

## Install on Windows

1. Extract this ZIP into the **root of one project**. Merge the included `.opencode` folder with any existing `.opencode` folder; do not delete agents or commands you already have.
2. Copy `AGENTS.md` into the project root. If you already have an `AGENTS.md`, merge the sections rather than overwriting your existing rules.
3. If the project has no `opencode.json`, copy `opencode.go.example.jsonc` to `opencode.json`. If it already has one, **do not replace it**: merge only the `model` and `small_model` keys into it, preserving your existing permissions/providers/settings.
4. Run OpenCode from that project and use `/models` to verify that the listed Go models are available.

Example PowerShell commands after extracting the archive (adjust the path):

```powershell
cd "C:\path\to\your\project"
# Start OpenCode after placing the files:
opencode
```

## Commands

- `/triage <task>` — inspect only relevant files and make a compact plan; no code edits.
- `/verify <scope>` — run focused tests/checks and report exact outcomes.
- `/review <scope>` — read-only first-pass review using a high-throughput model.
- `/deep-review <scope>` — stronger read-only review for important/high-risk changes.
- `/deep-fix <task>` — use the specialist coding model for a difficult implementation/fix.
- `/docs <scope>` — update documentation based on verified repository behavior.

You can also invoke agents directly: `@scout`, `@implementer`, `@reviewer`, `@architect`, `@debugger`, or `@docs-writer`.

## Operating policy

- The cheapest suitable model is the default; stronger models are opt-in for difficult work.
- Do not launch multiple agents by default. Use at most one focused scout/reviewer unless genuinely independent tasks justify parallel work.
- Do not make agents read the entire repository for every task. Give them exact paths and load only necessary files.
- Prefer a focused patch over a broad rewrite. Avoid opportunistic refactoring.
- Run the smallest relevant test/check first, then expand verification when warranted.
- A reviewer is read-only. The implementer must not claim tests passed unless they actually ran and passed.
- Ask before destructive operations, dependency upgrades, schema changes, changing public APIs, or deleting user data.
- Treat secrets, `.env` files, credentials, private keys, and tokens as off-limits. Never print them or include them in reports.

## Important

This is a starter kit, not a promise that a specific model will always be available or best on every task. Keep `/models` as the source of truth for your account and review the real diff/test output before accepting substantial changes.
