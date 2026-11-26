"""Runnable package entry point for the Stage 9 FastAPI service."""
from __future__ import annotations

import os

import uvicorn

from .api_endpoint import app


def main() -> None:
    """Start one production service process using the configured environment."""
    uvicorn.run(
        app,
        host=os.getenv("PII_API_HOST", "0.0.0.0"),
        port=int(os.getenv("PII_API_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
