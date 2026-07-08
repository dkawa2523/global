# Developer Notes

## Principles

- Keep CLI/API behavior small and predictable.
- Keep generated run outputs out of git.
- Prefer schema-v2 case files.
- Add backend features behind existing request/result interfaces.
- Keep diagnostics out of the core run path unless they are part of normal user output.

## Test Gate

`py -m pytest` runs the lean gate on Windows. Table-builder helpers under
`tools/` are not part of the default run path.

## Adding a Backend

1. Implement the backend interface.
2. Register it in the matching domain registry, such as `eedf/registry.py`,
   `electrical/registry.py`, or `numerics/registry.py`.
3. Add a focused unit test and one configuration example if the backend is user-facing.
4. Document model limits in `PHYSICS_NUMERICS.md` if the backend changes interpretation.
