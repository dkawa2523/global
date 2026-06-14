# Developer Notes

## Principles

- Keep CLI/API behavior small and predictable.
- Keep generated outputs out of git.
- Prefer schema-v2 case files.
- Add backend features behind existing request/result interfaces.
- Avoid adding diagnostics to the core run path unless they are part of normal user output.

## Test Gate

`python -m pytest` runs the lean gate. Optional external dashboards and exploratory benchmark tools should not be required by the default test suite.

## Adding a Backend

1. Implement the backend interface.
2. Register it in `workflows/context.py`.
3. Add a focused unit test and one configuration example if the backend is user-facing.
4. Document model limits in `PHYSICS_NUMERICS.md` if the backend changes interpretation.

