## Summary
<!-- 1-3 bullet points describing the change. -->

## Motivation
<!-- Why is this change needed? Link related issue(s). -->

## Test plan
- [ ] `pytest tests/unit tests/smoke` green
- [ ] `pytest -m slow tests/integration` green (if touches inference path)
- [ ] `ruff check . && ruff format --check .` clean
- [ ] Manual verification (describe steps below):

## Screenshots / logs
<!-- If UI or transcript output changes. -->

## Checklist
- [ ] Docs updated (README / docs/*.md) if user-facing behavior changed
- [ ] No new large binaries committed (`> 5 MB`)
- [ ] Model IDs unchanged unless intentional
