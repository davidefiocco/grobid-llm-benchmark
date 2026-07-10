"""Unit tests for the extraction models' helpers (notably author-name splitting)."""

import pytest

from grobid_llm_benchmark.models import Author

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("name", "forename", "surname"),
    [
        ("John Smith", "John", "Smith"),
        ("Smith, John", "John", "Smith"),
        ("Cher", "", "Cher"),
        ("van der Waals", "", "van der Waals"),
        ("Johannes van der Waals", "Johannes", "van der Waals"),
        ("Maria dos Santos", "Maria", "dos Santos"),
        ("John Smith Jr.", "John", "Smith Jr."),
        ("Ada Lovelace III", "Ada", "Lovelace III"),
    ],
)
def test_from_full_name(name, forename, surname):
    a = Author.from_full_name(name)
    assert a.forename == forename
    assert a.surname == surname
