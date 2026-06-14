from __future__ import annotations

import re


_COEFF_RE = re.compile(r'^\s*(?:(\d+(?:\.\d+)?)\s*)?(.*?)\s*$')


def parse_semicolon_map(value: str | None) -> dict[str, float]:
    if value is None:
        return {}
    s = str(value).strip()
    if not s:
        return {}
    out: dict[str, float] = {}
    for part in s.split(';'):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(':', 1)
        out[k.strip()] = float(v.strip())
    return out


def parse_pipe_list(value: str | None) -> list[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split('|') if p.strip()]


def parse_csv_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {'1', 'true', 'yes', 'y'}


def _parse_side(side: str) -> dict[str, float]:
    side = side.strip()
    if not side:
        return {}
    terms = [t.strip() for t in side.split('+') if t.strip()]
    out: dict[str, float] = {}
    for term in terms:
        m = _COEFF_RE.match(term)
        if m is None:
            raise ValueError(f'Could not parse reaction term: {term}')
        coeff_s, species = m.groups()
        coeff = float(coeff_s) if coeff_s else 1.0
        species = species.strip()
        out[species] = out.get(species, 0.0) + coeff
    return out


def parse_equation(equation: str) -> tuple[dict[str, float], dict[str, float]]:
    if '->' not in equation:
        raise ValueError(f'Reaction equation must contain -> : {equation}')
    lhs, rhs = equation.split('->', 1)
    return _parse_side(lhs), _parse_side(rhs)


def balance_delta(reactants: dict[str, float], products: dict[str, float]) -> dict[str, float]:
    keys = set(reactants) | set(products)
    return {k: products.get(k, 0.0) - reactants.get(k, 0.0) for k in keys if products.get(k, 0.0) != reactants.get(k, 0.0)}
