"""Collect stable test contracts from one AST parse per source file."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass

SKIP_NAMES = frozenset(
    {
        "pytest.importorskip",
        "pytest.mark.skip",
        "pytest.mark.skipif",
        "pytest.mark.xfail",
        "pytest.skip",
        "pytest.xfail",
        "unittest.skip",
        "unittest.skipIf",
        "unittest.skipUnless",
        "unittest.expectedFailure",
        "self.skipTest",
    }
)
REVERSED_COMPARISON = {
    "Lt": "Gt",
    "LtE": "GtE",
    "Gt": "Lt",
    "GtE": "LtE",
}

BindingMap = dict[str, tuple[str, ...]]
ExpressionMap = dict[str, tuple[ast.expr, ...]]
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class FunctionContext:
    """One test function with bindings inherited from its lexical scopes."""

    scope: str
    node: FunctionNode
    bindings: BindingMap
    expressions: ExpressionMap
    aliases: dict[str, str]


@dataclass(frozen=True)
class RaisesSite:
    """One ordered ``pytest.raises`` contract."""

    scope: str
    ordinal: int
    exception: str
    match: str | None
    literal_match: str | None
    guarded_call: str | None
    guarded_target: str | None
    directly_raises: bool


@dataclass(frozen=True)
class ComparisonContract:
    """The stable subject and operator of one plain comparison."""

    subject: str
    operator: str
    bound: float | None
    members: frozenset[str] | None
    membership_fingerprint: str | None


@dataclass(frozen=True)
class AssertionSummary:
    """Plain-assert counts for one function scope."""

    scope: str
    count: int
    obviously_true: int
    comparisons: tuple[ComparisonContract, ...]


@dataclass(frozen=True)
class Tolerance:
    """A tolerance with structural and optional numeric representations."""

    fingerprint: str
    numeric: float | None


@dataclass(frozen=True)
class ToleranceSite:
    """One ordered NumPy assertion or ``pytest.approx`` contract."""

    scope: str
    ordinal: int
    function: str
    subject: str
    reference: str
    relative: Tolerance
    absolute: Tolerance
    nan_fingerprint: str
    nan_allowed: bool | None
    self_fulfilling: bool
    dynamic_arguments: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedCall:
    """Statically known call arguments and unresolved expansions."""

    values: dict[str, ast.expr]
    dynamic: tuple[str, ...]


@dataclass(frozen=True)
class FileSnapshot:
    """Relevant contracts collected from one parsed test file."""

    tests: tuple[str, ...]
    raises: tuple[RaisesSite, ...]
    assertions: tuple[AssertionSummary, ...]
    tolerances: tuple[ToleranceSite, ...]
    skips: tuple[str, ...]


EMPTY_SNAPSHOT = FileSnapshot((), (), (), (), ())


def _nodes(statements: Sequence[ast.stmt]) -> list[ast.AST]:
    pending: list[ast.AST] = list(reversed(statements))
    result: list[ast.AST] = []
    while pending:
        node = pending.pop()
        result.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
    return result


def _assignment(node: ast.AST) -> tuple[list[str], ast.expr] | None:
    if isinstance(node, ast.Assign):
        names = [item.id for item in node.targets if isinstance(item, ast.Name)]
        return (names, node.value) if names else None
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return [node.target.id], node.value
    if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        return [node.target.id], node.value
    return None


def _target_name(target: ast.expr) -> str | None:
    while isinstance(target, (ast.Attribute, ast.Subscript)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _mutated_binding(node: ast.AST) -> tuple[list[str], ast.expr] | None:
    if isinstance(node, ast.Assign):
        names = [
            name
            for target in node.targets
            if not isinstance(target, ast.Name)
            if (name := _target_name(target)) is not None
        ]
        return (names, node.value) if names else None
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        name = _target_name(node.target)
        value = node.value
        if name is not None and not isinstance(node.target, ast.Name) and value:
            return [name], value
    return None


def _binding_update(node: ast.AST) -> tuple[list[str], ast.expr] | None:
    return _assignment(node) or _mutated_binding(node)


def _bindings(statements: Sequence[ast.stmt]) -> BindingMap:
    values: dict[str, list[str]] = {}
    for node in _nodes(statements):
        assigned = _binding_update(node)
        if assigned is None:
            continue
        names, value = assigned
        fingerprint = ast.dump(value, include_attributes=False)
        for name in names:
            values.setdefault(name, []).append(fingerprint)
    return {name: tuple(items) for name, items in values.items()}


def _expression_bindings(statements: Sequence[ast.stmt]) -> ExpressionMap:
    values: dict[str, list[ast.expr]] = {}
    for node in _nodes(statements):
        assigned = _binding_update(node)
        if assigned is None:
            continue
        names, value = assigned
        for name in names:
            values.setdefault(name, []).append(value)
    return {name: tuple(items) for name, items in values.items()}


def _scope_aliases(
    statements: Sequence[ast.stmt], inherited: dict[str, str] | None = None
) -> dict[str, str]:
    aliases = dict(inherited or {})
    for node in _nodes(statements):
        _update_aliases(aliases, node)
    return aliases


def _update_aliases(aliases: dict[str, str], node: ast.AST) -> None:
    assigned = _assignment(node)
    if assigned is not None:
        for name in assigned[0]:
            aliases.pop(name, None)
    _update_import_aliases(aliases, node)


def _update_import_aliases(aliases: dict[str, str], node: ast.AST) -> None:
    if isinstance(node, ast.Import):
        for item in node.names:
            local = item.asname or item.name.split(".", maxsplit=1)[0]
            aliases[local] = item.name
    elif isinstance(node, ast.ImportFrom) and node.module:
        for item in node.names:
            if item.name != "*":
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"


def _decorators(
    node: ast.ClassDef | FunctionNode, bindings: BindingMap
) -> tuple[str, ...]:
    return tuple(_fingerprint(item, bindings) for item in node.decorator_list)


def _function_contexts(
    statements: Sequence[ast.stmt],
    *,
    prefix: tuple[str, ...] = (),
    inherited_bindings: BindingMap | None = None,
    inherited_expressions: ExpressionMap | None = None,
    inherited_aliases: dict[str, str] | None = None,
    inherited_decorators: tuple[str, ...] = (),
) -> list[FunctionContext]:
    bindings = dict(inherited_bindings or {})
    bindings.update(_bindings(statements))
    expressions = dict(inherited_expressions or {})
    expressions.update(_expression_bindings(statements))
    aliases = _scope_aliases(statements, inherited_aliases)
    contexts: list[FunctionContext] = []
    for node in statements:
        if isinstance(node, ast.ClassDef):
            contexts.extend(
                _function_contexts(
                    node.body,
                    prefix=(*prefix, node.name),
                    inherited_bindings=bindings,
                    inherited_expressions=expressions,
                    inherited_aliases=aliases,
                    inherited_decorators=(
                        *inherited_decorators,
                        *_decorators(node, bindings),
                    ),
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith("test")
        ):
            local_bindings = dict(bindings)
            local_bindings.update(_bindings(node.body))
            local_expressions = dict(expressions)
            local_expressions.update(_expression_bindings(node.body))
            local_aliases = dict(aliases)
            decorators = (*inherited_decorators, *_decorators(node, bindings))
            parameters = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            for parameter in parameters:
                local_bindings[parameter.arg] = decorators or ("<parameter>",)
                local_expressions[parameter.arg] = ()
                local_aliases.pop(parameter.arg, None)
            contexts.append(
                FunctionContext(
                    ".".join((*prefix, node.name)),
                    node,
                    local_bindings,
                    local_expressions,
                    _scope_aliases(node.body, local_aliases),
                )
            )
    return contexts


def _qualified_name(expression: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(expression, ast.Name):
        return aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        prefix = _qualified_name(expression.value, aliases)
        return f"{prefix}.{expression.attr}" if prefix else None
    return None


def _fingerprint(expression: ast.expr, bindings: BindingMap) -> str:
    referenced = tuple(
        (name, bindings.get(name, ()))
        for name in sorted(
            {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
        )
    )
    return f"{ast.dump(expression, include_attributes=False)}|{referenced!r}"


def _subject_key(
    expression: ast.expr,
    expressions: ExpressionMap,
    seen: frozenset[str] = frozenset(),
) -> str:
    if isinstance(expression, ast.Name):
        values = expressions.get(expression.id, ())
        if len(values) == 1 and expression.id not in seen:
            value = values[0]
            if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
                return _subject_key(value, expressions, seen | {expression.id})
        return f"name:{expression.id}"
    if isinstance(expression, ast.Attribute):
        return f"attribute:{_subject_key(expression.value, expressions, seen)}"
    if isinstance(expression, ast.Subscript):
        return (
            f"subscript:{_subject_key(expression.value, expressions, seen)}:"
            f"{ast.dump(expression.slice, include_attributes=False)}"
        )
    return ast.dump(expression, include_attributes=False)


def _static_dict(
    expression: ast.expr,
    expressions: ExpressionMap,
    seen: frozenset[str] = frozenset(),
) -> dict[str, ast.expr] | None:
    if isinstance(expression, ast.Name) and expression.id not in seen:
        values = expressions.get(expression.id, ())
        if len(values) == 1:
            return _static_dict(values[0], expressions, seen | {expression.id})
    if not isinstance(expression, ast.Dict):
        return None
    result: dict[str, ast.expr] = {}
    for key, value in zip(expression.keys, expression.values, strict=True):
        if key is None:
            nested = _static_dict(value, expressions, seen)
            if nested is None:
                return None
            result.update(nested)
        elif isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = value
        else:
            return None
    return result


def _resolved_call(
    call: ast.Call,
    parameters: Sequence[str],
    expressions: ExpressionMap,
    bindings: BindingMap,
) -> ResolvedCall:
    values: dict[str, ast.expr] = {}
    dynamic: list[str] = []
    for index, argument in enumerate(call.args):
        if isinstance(argument, ast.Starred):
            dynamic.append(_fingerprint(argument.value, bindings))
        elif index < len(parameters):
            values[parameters[index]] = argument
    for keyword in call.keywords:
        if keyword.arg is not None:
            values[keyword.arg] = keyword.value
            continue
        expanded = _static_dict(keyword.value, expressions)
        if expanded is None:
            dynamic.append(_fingerprint(keyword.value, bindings))
        else:
            values.update(expanded)
    return ResolvedCall(values, tuple(dynamic))


def _location(node: ast.AST) -> tuple[int, int]:
    return getattr(node, "lineno", 0), getattr(node, "col_offset", 0)


def _raises_call(
    statement: ast.With | ast.AsyncWith, aliases: dict[str, str]
) -> ast.Call | None:
    for item in statement.items:
        expression = item.context_expr
        if (
            isinstance(expression, ast.Call)
            and _qualified_name(expression.func, aliases) == "pytest.raises"
        ):
            return expression
    return None


def _outermost_calls(node: ast.AST) -> list[ast.Call]:
    pending = [node]
    calls: list[ast.Call] = []
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Call):
            calls.append(current)
        elif not isinstance(
            current,
            (ast.Raise, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            pending.extend(reversed(list(ast.iter_child_nodes(current))))
    return calls


def _guarded_operation(statement: ast.With | ast.AsyncWith) -> ast.Call | None:
    calls = [call for node in statement.body for call in _outermost_calls(node)]
    return calls[-1] if calls else None


def _raises_sites(context: FunctionContext, namespace: str) -> list[RaisesSite]:
    scope = f"{namespace}::{context.scope}"
    statements = sorted(
        (
            node
            for node in _nodes(context.node.body)
            if isinstance(node, (ast.With, ast.AsyncWith))
        ),
        key=_location,
    )
    sites: list[RaisesSite] = []
    for statement in statements:
        site = _raise_site(
            statement,
            scope,
            len(sites),
            context.bindings,
            context.aliases,
        )
        if site is not None:
            sites.append(site)
    return sites


def _literal_string(expression: ast.expr) -> str | None:
    value = expression.value if isinstance(expression, ast.Constant) else None
    return value if isinstance(value, str) else None


def _raise_site(
    statement: ast.With | ast.AsyncWith,
    scope: str,
    ordinal: int,
    bindings: BindingMap,
    aliases: dict[str, str],
) -> RaisesSite | None:
    call = _raises_call(statement, aliases)
    if call is None or not call.args:
        return None
    match = next((item for item in call.keywords if item.arg == "match"), None)
    guarded = _guarded_operation(statement)
    return RaisesSite(
        scope,
        ordinal,
        _fingerprint(call.args[0], bindings),
        None if match is None else _fingerprint(match.value, bindings),
        None if match is None else _literal_string(match.value),
        None if guarded is None else _qualified_name(guarded.func, aliases),
        None if guarded is None else _fingerprint(guarded, bindings),
        any(isinstance(node, ast.Raise) for node in _nodes(statement.body)),
    )


def _self_comparison(expression: ast.expr) -> bool:
    if not isinstance(expression, ast.Compare) or len(expression.ops) != 1:
        return False
    stable = isinstance(expression.left, (ast.Name, ast.Constant))
    same_value = ast.dump(expression.left) == ast.dump(expression.comparators[0])
    return (
        same_value
        and isinstance(expression.ops[0], (ast.Eq, ast.Is, ast.LtE, ast.GtE))
        and stable
    )


def _boolean_is_obviously_true(expression: ast.BoolOp) -> bool:
    results = [_obviously_true(item) for item in expression.values]
    if isinstance(expression.op, ast.Or):
        return any(results)
    return isinstance(expression.op, ast.And) and all(results)


def _obviously_true(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Constant):
        return bool(expression.value)
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        return isinstance(expression.operand, ast.Constant) and not bool(
            expression.operand.value
        )
    if isinstance(expression, ast.BoolOp):
        return _boolean_is_obviously_true(expression)
    return _self_comparison(expression)


def _and_terms(expression: ast.expr) -> list[ast.expr]:
    if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.And):
        return [term for item in expression.values for term in _and_terms(item)]
    return [expression]


def _comparison_contracts(
    expression: ast.expr, expressions: ExpressionMap, bindings: BindingMap
) -> tuple[ComparisonContract, ...]:
    contracts: list[ComparisonContract] = []
    for term in _and_terms(expression):
        if not isinstance(term, ast.Compare):
            continue
        left = term.left
        for operator, right in zip(term.ops, term.comparators, strict=True):
            contracts.append(
                _comparison_contract(left, operator, right, expressions, bindings)
            )
            left = right
    return tuple(contracts)


def _comparison_contract(
    left: ast.expr,
    comparison: ast.cmpop,
    right: ast.expr,
    expressions: ExpressionMap,
    bindings: BindingMap,
) -> ComparisonContract:
    operator = type(comparison).__name__
    left_bound = _numeric_literal(left)
    right_bound = _numeric_literal(right)
    if left_bound is not None and right_bound is None:
        return ComparisonContract(
            _subject_key(right, expressions),
            REVERSED_COMPARISON.get(operator, operator),
            left_bound,
            None,
            None,
        )
    members = _literal_members(right) if operator in {"In", "NotIn"} else None
    membership_fingerprint = (
        _fingerprint(right, bindings)
        if operator in {"In", "NotIn"} and members is None
        else None
    )
    return ComparisonContract(
        _subject_key(left, expressions),
        operator,
        right_bound,
        members,
        membership_fingerprint,
    )


def _literal_members(expression: ast.expr) -> frozenset[str] | None:
    if not isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return None
    try:
        for item in expression.elts:
            ast.literal_eval(item)
    except (ValueError, TypeError):
        return None
    return frozenset(
        ast.dump(item, include_attributes=False) for item in expression.elts
    )


def _assertion_summary(context: FunctionContext, namespace: str) -> AssertionSummary:
    assertions = [
        node for node in _nodes(context.node.body) if isinstance(node, ast.Assert)
    ]
    return AssertionSummary(
        f"{namespace}::{context.scope}",
        len(assertions),
        sum(_obviously_true(item.test) for item in assertions),
        tuple(
            contract
            for assertion in assertions
            for contract in _comparison_contracts(
                assertion.test, context.expressions, context.bindings
            )
        ),
    )


def _numeric_literal(expression: ast.expr) -> float | None:
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int | float)
        and not isinstance(expression.value, bool)
    ):
        return float(expression.value)
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, (ast.USub, ast.UAdd))
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, int | float)
    ):
        value = float(expression.operand.value)
        return -value if isinstance(expression.op, ast.USub) else value
    return None


def _tolerance(
    arguments: ResolvedCall,
    name: str,
    default: float | None,
    bindings: BindingMap,
) -> Tolerance:
    expression = arguments.values.get(name)
    if expression is None:
        return Tolerance(f"<default:{default!r}>", default)
    return Tolerance(_fingerprint(expression, bindings), _numeric_literal(expression))


def _normalized_tolerance(
    expression: ast.expr | None, default: float, bindings: BindingMap
) -> Tolerance:
    if expression is None or (
        isinstance(expression, ast.Constant) and expression.value is None
    ):
        return Tolerance(f"<default:{default!r}>", default)
    return Tolerance(_fingerprint(expression, bindings), _numeric_literal(expression))


def _approx_tolerances(
    arguments: ResolvedCall, bindings: BindingMap
) -> tuple[Tolerance, Tolerance]:
    absolute_expression = arguments.values.get("abs")
    relative_expression = arguments.values.get("rel")
    absolute = _normalized_tolerance(absolute_expression, 1.0e-12, bindings)
    absolute_was_set = "abs" in arguments.values and not (
        isinstance(absolute_expression, ast.Constant)
        and absolute_expression.value is None
    )
    relative_was_none = "rel" not in arguments.values or (
        isinstance(relative_expression, ast.Constant)
        and relative_expression.value is None
    )
    relative = (
        Tolerance("<disabled>", 0.0)
        if absolute_was_set and relative_was_none
        else _normalized_tolerance(relative_expression, 1.0e-6, bindings)
    )
    return relative, absolute


def _nan_policy(
    arguments: ResolvedCall, name: str, default: bool, bindings: BindingMap
) -> tuple[str, bool | None]:
    value = arguments.values.get(name)
    if value is None:
        return f"<default:{default!r}>", default
    literal = value.value if isinstance(value, ast.Constant) else None
    return _fingerprint(value, bindings), literal if isinstance(literal, bool) else None


def _approx_subjects(
    assertion: ast.Assert,
    context: FunctionContext,
) -> dict[int, tuple[str, bool]]:
    expression = assertion.test
    if not isinstance(expression, ast.Compare) or len(expression.comparators) != 1:
        return {}
    pairs = (
        (expression.left, expression.comparators[0]),
        (expression.comparators[0], expression.left),
    )
    result: dict[int, tuple[str, bool]] = {}
    for actual, expected in pairs:
        if (
            isinstance(expected, ast.Call)
            and _qualified_name(expected.func, context.aliases) == "pytest.approx"
        ):
            arguments = _resolved_call(
                expected,
                ("expected", "rel", "abs", "nan_ok"),
                context.expressions,
                context.bindings,
            )
            reference = arguments.values.get("expected")
            if reference is None:
                continue
            result[id(expected)] = (
                _subject_key(actual, context.expressions),
                ast.dump(actual) == ast.dump(reference),
            )
    return result


def _tolerance_site(
    call: ast.Call,
    *,
    scope: str,
    ordinal: int,
    bindings: BindingMap,
    expressions: ExpressionMap,
    aliases: dict[str, str],
    approx_subjects: dict[int, tuple[str, bool]],
) -> ToleranceSite | None:
    function = _qualified_name(call.func, aliases)
    if function == "pytest.approx":
        arguments = _resolved_call(
            call,
            ("expected", "rel", "abs", "nan_ok"),
            expressions,
            bindings,
        )
        relative, absolute = _approx_tolerances(arguments, bindings)
        nan_fingerprint, nan_allowed = _nan_policy(arguments, "nan_ok", False, bindings)
        subject, self_fulfilling = approx_subjects.get(id(call), ("<approx>", False))
        reference_expression = arguments.values.get("expected")
    elif function is not None and function.startswith("numpy.testing.assert_"):
        arguments = _resolved_call(
            call,
            ("actual", "desired", "rtol", "atol", "equal_nan"),
            expressions,
            bindings,
        )
        default = function == "numpy.testing.assert_allclose"
        relative = _tolerance(arguments, "rtol", 1.0e-7 if default else None, bindings)
        absolute = _tolerance(arguments, "atol", 0.0 if default else None, bindings)
        nan_fingerprint, nan_allowed = _nan_policy(
            arguments, "equal_nan", True, bindings
        )
        actual = arguments.values.get("actual")
        reference_expression = arguments.values.get("desired")
        subject = (
            _subject_key(actual, expressions) if actual is not None else "<missing>"
        )
        self_fulfilling = (
            actual is not None
            and reference_expression is not None
            and ast.dump(actual) == ast.dump(reference_expression)
        )
    else:
        return None
    reference = (
        _fingerprint(reference_expression, bindings)
        if reference_expression is not None
        else "<missing>"
    )
    return ToleranceSite(
        scope,
        ordinal,
        function,
        subject,
        reference,
        relative,
        absolute,
        nan_fingerprint,
        nan_allowed,
        self_fulfilling,
        arguments.dynamic,
    )


def _tolerance_sites(context: FunctionContext, namespace: str) -> list[ToleranceSite]:
    scope = f"{namespace}::{context.scope}"
    nodes = _nodes(context.node.body)
    approx_subjects = {
        call_id: value
        for node in nodes
        if isinstance(node, ast.Assert)
        for call_id, value in _approx_subjects(node, context).items()
    }
    calls = sorted(
        (node for node in nodes if isinstance(node, ast.Call)), key=_location
    )
    sites: list[ToleranceSite] = []
    for call in calls:
        site = _tolerance_site(
            call,
            scope=scope,
            ordinal=len(sites),
            bindings=context.bindings,
            expressions=context.expressions,
            aliases=context.aliases,
            approx_subjects=approx_subjects,
        )
        if site is not None:
            sites.append(site)
    return sites


def _skip_names(
    nodes: Sequence[ast.AST], aliases: dict[str, str]
) -> set[tuple[int, str]]:
    violations: set[tuple[int, str]] = set()
    for node in nodes:
        expression: ast.expr | None = None
        if isinstance(node, ast.Call):
            expression = node.func
        elif isinstance(node, (ast.Attribute, ast.Name)):
            expression = node
        name = _qualified_name(expression, aliases) if expression is not None else None
        if name is not None and name in SKIP_NAMES:
            violations.add((int(getattr(node, "lineno", 0)), name))
        dynamic_name = _getattr_skip_name(node, aliases)
        if dynamic_name is not None:
            violations.add((int(getattr(node, "lineno", 0)), dynamic_name))
    return violations


def _getattr_skip_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return None
    if _qualified_name(node.func, aliases) != "getattr":
        return None
    owner = _qualified_name(node.args[0], aliases)
    attribute = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
    if owner not in {"pytest", "pytest.mark"} or attribute not in {"skip", "xfail"}:
        return None
    return f"{owner}.{attribute}"


def _skip_sites(
    statements: Sequence[ast.stmt],
    inherited_aliases: dict[str, str] | None = None,
) -> set[tuple[int, str]]:
    aliases = dict(inherited_aliases or {})
    violations: set[tuple[int, str]] = set()
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            violations.update(_skip_names(node.decorator_list, aliases))
            nested_aliases = dict(aliases)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                for parameter in parameters:
                    nested_aliases.pop(parameter.arg, None)
            violations.update(_skip_sites(node.body, nested_aliases))
        else:
            nodes = _nodes([node])
            violations.update(_skip_names(nodes, aliases))
            for child in nodes:
                _update_aliases(aliases, child)
    return violations


def _skip_violations(tree: ast.Module, namespace: str) -> tuple[str, ...]:
    violations = _skip_sites(tree.body)
    return tuple(
        f"forbidden skip/xfail: {namespace}:{line}: {name}"
        for line, name in sorted(violations)
    )


def _snapshot(source: str, *, namespace: str) -> FileSnapshot:
    tree = ast.parse(source)
    contexts = _function_contexts(tree.body)
    return FileSnapshot(
        tuple(f"{namespace}::{context.scope}" for context in contexts),
        tuple(
            site for context in contexts for site in _raises_sites(context, namespace)
        ),
        tuple(_assertion_summary(context, namespace) for context in contexts),
        tuple(
            site
            for context in contexts
            for site in _tolerance_sites(context, namespace)
        ),
        _skip_violations(tree, namespace),
    )
