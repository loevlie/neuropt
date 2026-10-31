"""Helpers for pulling structured data out of free-form LLM text."""

import json

_CLOSERS = {"[": list, "{": dict}


def extract_json(text, opener="["):
    """Return the first JSON array (``opener="["``) or object (``"{"``) in ``text``.

    Tries every position where ``opener`` appears and decodes just one value
    from there, so trailing prose (even prose that contains brackets) and
    leading chatter are ignored. Returns None if nothing decodes.
    """
    if not text:
        return None
    expected = _CLOSERS[opener]
    decoder = json.JSONDecoder()
    start = text.find(opener)
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, expected):
                return value
        start = text.find(opener, start + 1)
    return None
