"""Shared test fixtures.

``fixtures_dir`` points at a committed tiny article directory (PDF + gold ``.nxml``) so
the offline end-to-end and scorer tests need no downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return _FIXTURES


@pytest.fixture
def pilot_dir() -> Path:
    """A dataset directory whose sub-dirs each hold a PDF + gold file."""
    return _FIXTURES
