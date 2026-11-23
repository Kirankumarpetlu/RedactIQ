"""Project JSON helpers backed by :mod:`orjson`.

The pipeline stores JSON inside CSV cells and writes JSON/JSONL reports.  Keep
the public helpers text-based so existing file and CSV callers remain
compatible while parsing and serialization use orjson everywhere at runtime.
"""
from __future__ import annotations

from typing import Any, TextIO

import orjson


JSONDecodeError = orjson.JSONDecodeError


def loads(value: str | bytes | bytearray | memoryview) -> Any:
    """Parse a JSON document from text or bytes using orjson."""
    return orjson.loads(value)


def dumps(value: Any, *, ensure_ascii: bool = True, indent: int | None = None,
          sort_keys: bool = False, separators: tuple[str, str] | None = None,
          default: Any = None) -> str:
    """Serialize JSON as UTF-8 text for CSV cells and text file APIs.

    orjson always emits UTF-8 rather than ASCII escape sequences.  The project
    reads and writes UTF-8, so ``ensure_ascii`` is accepted for stdlib-call
    compatibility but does not alter that encoding.  Its compact default is
    equivalent to the only custom separators used by this codebase.
    """
    if indent not in (None, 2):
        raise ValueError("orjson supports indentation of two spaces only")
    if separators not in (None, (",", ":"), (", ", ": ")):
        raise ValueError("unsupported JSON separators")
    options = orjson.OPT_NON_STR_KEYS
    if indent == 2:
        options |= orjson.OPT_INDENT_2
    if sort_keys:
        options |= orjson.OPT_SORT_KEYS
    return orjson.dumps(value, option=options, default=default).decode("utf-8")


def load(stream: TextIO) -> Any:
    """Read and parse a JSON text stream with orjson."""
    return loads(stream.read())


def dump(value: Any, stream: TextIO, **kwargs: Any) -> None:
    """Serialize JSON through orjson into a text stream."""
    stream.write(dumps(value, **kwargs))
