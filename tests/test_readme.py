"""The README is rendered on PyPI, where relative links break, so every link points to main."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).parents[1] / 'README.md'


def test_readme_links_are_absolute() -> None:
    targets = re.findall(r'\]\(([^)]+)\)', README.read_text())
    relative = [target for target in targets if not target.startswith('https://')]
    assert relative == []
