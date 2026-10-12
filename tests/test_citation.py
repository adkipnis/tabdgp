"""CITATION.cff repeats the package version, for GitHub's citation box and Zenodo's record."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_the_citation_names_the_package_version() -> None:
    """A release that bumps pyproject.toml but not CITATION.cff fails here."""
    expected = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    citation = (ROOT / 'CITATION.cff').read_text()
    cited = re.findall(r'^version: (.+)$', citation, re.MULTILINE)
    assert cited == [expected]
