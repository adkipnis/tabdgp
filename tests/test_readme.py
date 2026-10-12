"""The README is rendered on PyPI, where relative links break, so every link points to main."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).parents[1] / 'README.md'


def test_readme_links_are_absolute() -> None:
    targets = re.findall(r'\]\(([^)]+)\)', README.read_text())
    relative = [target for target in targets if not target.startswith('https://')]
    assert relative == []


def test_the_readme_quickstart_prints_the_output_it_shows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The first code a user copies must work and print what the README shows.
    A release that changes the dataset a key draws fails here until the README follows.
    """
    quickstart = README.read_text().split('## Quickstart')[1].split('\n## ')[0]
    code = re.findall(r'```python\n(.*?)```', quickstart, re.DOTALL)
    shown = re.findall(r'```text\n(.*?)```', quickstart, re.DOTALL)
    assert len(code) == 1
    assert len(shown) == 1
    exec(code[0], {})
    assert capsys.readouterr().out == shown[0]
