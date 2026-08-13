"""Collect explicit, stable test-strength signals from Python source."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from itertools import chain
from typing import NamedTuple

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
SKIP_NAMES = frozenset(
    [
        "pytest.importorskip",
        "pytest.mark.skip",
        "pytest.mark.skipif",
        "pytest.mark.xfail",
        "pytest.skip",
        "pytest.xfail",
        "unittest.expectedFailure",
        "unittest.skip",
        "unittest.skipIf",
        "unittest.skipUnless",
        "self.skipTest",
    ]
)


class FunctionContext(NamedTuple):
    scope: str
    node: FunctionNode


class RaisesSite(NamedTuple):
    scope: str
    exception: str
    has_match: bool
    directly_raises: bool


class AssertionSummary(NamedTuple):
    scope: str
    count: int
    obviously_true: int


class ToleranceSite(NamedTuple):
    scope: str
    ordinal: int
    function: str
    relative: float
    absolute: float
    nan_allowed: bool
    self_fulfilling: bool


class FileSnapshot(NamedTuple):
    tests: tuple[str, ...]
    raises: tuple[RaisesSite, ...]
    assertions: tuple[AssertionSummary, ...]
    tolerances: tuple[ToleranceSite, ...]
    skips: tuple[str, ...]


def _body_nodes(body: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    pending: list[ast.AST] = list(reversed(body))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _qualname(expression: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(expression, ast.Name):
        return aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        owner = _qualname(expression.value, aliases)
        return f"{owner}.{expression.attr}" if owner else None
    return _getattr_name(expression, aliases)


def _getattr_name(expression: ast.expr, aliases: dict[str, str]) -> str | None:
    match expression:
        case ast.Call(
            func=ast.Name(id="getattr"),
            args=[owner, ast.Constant(value=str(attribute))],
        ):
            owner_name = _qualname(owner, aliases)
            return f"{owner_name}.{attribute}" if owner_name else None
    return None


def _apply_import(aliases: dict[str, str], node: ast.AST) -> None:
    if isinstance(node, ast.Import):
        for item in node.names:
            aliases[item.asname or item.name.split(".", maxsplit=1)[0]] = item.name
    elif isinstance(node, ast.ImportFrom) and node.module:
        for item in node.names:
            if item.name != "*":
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"


def _test_contexts(
    body: Sequence[ast.stmt],
    prefix: tuple[str, ...] = (),
) -> list[FunctionContext]:
    result: list[FunctionContext] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            result.extend(_test_contexts(node.body, (*prefix, node.name)))
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test"):
            result.append(FunctionContext(".".join((*prefix, node.name)), node))
    return result


def _literal(expression: ast.expr | None) -> object:
    if expression is None:
        return None
    try:
        return ast.literal_eval(expression)
    except (TypeError, ValueError):
        return _UNKNOWN


_UNKNOWN = object()


def _obviously_true(expression: ast.expr) -> bool:
    value = _literal(expression)
    if value is not _UNKNOWN:
        return bool(value)
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        operand = _literal(expression.operand)
        return operand is not _UNKNOWN and not bool(operand)
    return False


def _pytest_raises_call(
    node: ast.With | ast.AsyncWith, aliases: dict[str, str]
) -> ast.Call | None:
    return next(
        (
            expression
            for item in node.items
            if isinstance(expression := item.context_expr, ast.Call)
            and _qualname(expression.func, aliases) == "pytest.raises"
        ),
        None,
    )


def _raises_site(
    node: ast.With | ast.AsyncWith, scope: str, aliases: dict[str, str]
) -> RaisesSite | None:
    call = _pytest_raises_call(node, aliases)
    if call is None or not call.args:
        return None
    return RaisesSite(
        scope,
        ast.dump(call.args[0], include_attributes=False),
        any(item.arg == "match" for item in call.keywords),
        any(isinstance(item, ast.Raise) for item in _body_nodes(node.body)),
    )


def _raises_sites(
    context: FunctionContext, namespace: str, aliases: dict[str, str]
) -> list[RaisesSite]:
    scope = f"{namespace}::{context.scope}"
    sites = (
        _raises_site(node, scope, aliases)
        for node in _body_nodes(context.node.body)
        if isinstance(node, (ast.With, ast.AsyncWith))
    )
    return [site for site in sites if site is not None]


def _arguments(call: ast.Call, names: Sequence[str]) -> dict[str, ast.expr] | None:
    if any(isinstance(item, ast.Starred) for item in call.args) or len(call.args) > len(
        names
    ):
        return None
    result = dict(zip(names, call.args, strict=False))
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in names or keyword.arg in result:
            return None
        result[keyword.arg] = keyword.value
    return result


def _self_approx(call: ast.Call, context: FunctionContext) -> bool:
    for node in _body_nodes(context.node.body):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        if len(node.test.comparators) != 1:
            continue
        for actual, expected in (
            (node.test.left, node.test.comparators[0]),
            (node.test.comparators[0], node.test.left),
        ):
            if expected is call:
                values = _arguments(call, ("expected", "rel", "abs", "nan_ok"))
                reference = values.get("expected") if values else None
                return reference is not None and ast.dump(actual) == ast.dump(reference)
    return False


def _literal_number(expression: ast.expr | None) -> tuple[bool, float | None]:
    value = _literal(expression)
    if value is None:
        return True, None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False, None
    return True, float(value)


def _literal_settings(
    values: dict[str, ast.expr], names: tuple[str, str, str]
) -> tuple[float | None, float | None, bool | None] | None:
    relative_known, relative = _literal_number(values.get(names[0]))
    absolute_known, absolute = _literal_number(values.get(names[1]))
    nan_value = _literal(values.get(names[2]))
    nan_known = nan_value is None or isinstance(nan_value, bool)
    if not relative_known or not absolute_known or not nan_known:
        return None
    return relative, absolute, nan_value if isinstance(nan_value, bool) else None


def _resolved_settings(
    function: str,
    values: dict[str, ast.expr],
    names: Sequence[str],
    settings: tuple[float | None, float | None, bool | None],
    call: ast.Call,
    context: FunctionContext,
) -> tuple[float, float, bool, bool]:
    relative, absolute, nan_allowed = settings
    if function == "pytest.approx":
        relative = 0.0 if names[-2] in values and relative is None else relative
        defaults = (1e-6, 1e-12, False)
        self_fulfilling = _self_approx(call, context)
    else:
        defaults = (1e-7, 0.0, True)
        self_fulfilling = ast.dump(values["actual"]) == ast.dump(values["desired"])
    return (
        defaults[0] if relative is None else relative,
        defaults[1] if absolute is None else absolute,
        defaults[2] if nan_allowed is None else nan_allowed,
        self_fulfilling,
    )


def _tolerance_site(
    call: ast.Call, function: str, scope: str, ordinal: int, context: FunctionContext
) -> ToleranceSite | None:
    names = (
        ("expected", "rel", "abs", "nan_ok")
        if function == "pytest.approx"
        else ("actual", "desired", "rtol", "atol", "equal_nan")
    )
    values = _arguments(call, names)
    required = {"expected"} if function == "pytest.approx" else {"actual", "desired"}
    if values is None or not required <= values.keys():
        return None
    rel_name, abs_name, nan_name = names[-3:]
    settings = _literal_settings(values, (rel_name, abs_name, nan_name))
    if settings is None:
        return None
    relative, absolute, nan_allowed, self_fulfilling = _resolved_settings(
        function, values, names, settings, call, context
    )
    return ToleranceSite(
        scope,
        ordinal,
        function,
        relative,
        absolute,
        nan_allowed,
        self_fulfilling,
    )


def _tolerances(
    context: FunctionContext, namespace: str, aliases: dict[str, str]
) -> list[ToleranceSite]:
    scope = f"{namespace}::{context.scope}"
    ordinals: dict[str, int] = {}
    result: list[ToleranceSite] = []
    for call in (
        node for node in _body_nodes(context.node.body) if isinstance(node, ast.Call)
    ):
        function = _qualname(call.func, aliases)
        if function not in {"pytest.approx", "numpy.testing.assert_allclose"}:
            continue
        ordinal = ordinals.get(function, 0)
        ordinals[function] = ordinal + 1
        site = _tolerance_site(call, function, scope, ordinal, context)
        if site:
            result.append(site)
    return result


class _SkipScanner:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = dict(aliases or {})
        self.sites: set[tuple[int, str]] = set()

    def _record(self, expression: ast.expr) -> None:
        name = _qualname(expression, self.aliases)
        if name is not None and name in SKIP_NAMES:
            self.sites.add((int(getattr(expression, "lineno", 0)), name))

    def scan(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _apply_import(self.aliases, node)
            return
        if isinstance(node, ast.Assign):
            self.scan(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases.pop(target.id, None)
            return
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            self._definition(node)
            return
        if isinstance(node, ast.Call):
            self._record(node)
            self._record(node.func)
        for child in ast.iter_child_nodes(node):
            self.scan(child)

    def _definition(self, node: ast.ClassDef | FunctionNode) -> None:
        for decorator in node.decorator_list:
            self._record(decorator)
            self.scan(decorator)
        nested = dict(self.aliases)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for parameter in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ):
                nested.pop(parameter.arg, None)
        scanner = _SkipScanner(nested)
        for statement in node.body:
            scanner.scan(statement)
        self.sites.update(scanner.sites)
        self.aliases.pop(node.name, None)


def _assertion_summary(context: FunctionContext, namespace: str) -> AssertionSummary:
    items = [
        node for node in _body_nodes(context.node.body) if isinstance(node, ast.Assert)
    ]
    return AssertionSummary(
        f"{namespace}::{context.scope}",
        len(items),
        sum(_obviously_true(node.test) for node in items),
    )


def _context_sites(
    contexts: Sequence[FunctionContext], namespace: str, aliases: dict[str, str]
) -> tuple[tuple[RaisesSite, ...], tuple[ToleranceSite, ...]]:
    raises = chain.from_iterable(
        _raises_sites(context, namespace, aliases) for context in contexts
    )
    tolerances = chain.from_iterable(
        _tolerances(context, namespace, aliases) for context in contexts
    )
    return tuple(raises), tuple(tolerances)


def _snapshot(source: str, *, namespace: str) -> FileSnapshot:
    tree = ast.parse(source)
    contexts = _test_contexts(tree.body)
    aliases: dict[str, str] = {}
    for node in tree.body:
        _apply_import(aliases, node)
    visitor = _SkipScanner()
    visitor.scan(tree)
    raises, tolerances = _context_sites(contexts, namespace, aliases)
    return FileSnapshot(
        tuple(f"{namespace}::{context.scope}" for context in contexts),
        raises,
        tuple(_assertion_summary(context, namespace) for context in contexts),
        tolerances,
        tuple(
            f"forbidden skip/xfail: {namespace}:{line}: {name}"
            for line, name in sorted(visitor.sites)
        ),
    )
