# util.py
import json
from typing import Any, Optional

def _find_first_json_object(s: str) -> Optional[str]:
    """
    Returns the first top-level JSON object substring found in s,
    or None if not found. Handles strings and escapes.
    """
    if not s:
        return None
    in_string = False
    escape = False
    depth = 0
    start = -1

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                # current char is escaped; consume and clear escape
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
            continue

        # not in string
        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return s[start:i+1]
    return None


def try_parse_json(s: str) -> Optional[Any]:
    """
    1) Zkus přímo JSON.parse
    2) Když to nevyjde, vytáhni první JSON objekt pomocí bracket-scanu
    """
    s = (s or "").strip()
    if not s:
        return None
    # 1) přímý parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) bracket-scan
    block = _find_first_json_object(s)
    if block is None:
        return None
    try:
        return json.loads(block)
    except Exception:
        return None


def squeeze(s: str) -> str:
    """Zredukuje vícenásobné prázdné řádky na max 1 a ořízne okraje."""
    if not s:
        return ""
    out = []
    blank = False
    for line in s.splitlines():
        if line.strip() == "":
            if not blank:
                out.append("")
                blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()
