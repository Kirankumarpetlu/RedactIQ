"""Preserve reference row IDs when preparing and evaluating held-out datasets."""


def row_identity(record, fallback_index):
    for key in ("row_index", "row_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def validate_row_identities(records, start=1):
    identities = [row_identity(row, i) for i, row in enumerate(records, start=start)]
    if len(identities) != len(set(identities)):
        raise ValueError("Input contains duplicate row identifiers; held-out rows must have unique IDs")
