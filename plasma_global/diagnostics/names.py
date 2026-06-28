from __future__ import annotations

import re


def observable_id(value: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', str(value)).strip('_')
