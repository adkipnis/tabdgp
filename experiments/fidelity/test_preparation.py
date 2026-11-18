"""Checks the rules that prepare a real table for its measure, one case per rule.
Run by hand from the repository root:

    uv run pytest experiments/fidelity/test_preparation.py
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# preparation.py is a script's module, not a package's
sys.path.insert(0, str(Path(__file__).parent))

from preparation import Prepared, prepare

# eight rows: a continuous covariate, a categorical one of three levels and a response
X = [0.5, -1.0, 2.0, 0.0, 1.5, 3.0, -2.5, 1.0]
G = ['a', 'a', 'b', 'b', 'c', 'c', 'a', 'b']
SCORE = [1.2, 0.4, 3.1, 2.2, 0.9, 4.0, 1.1, 2.7]
TABLE = pd.DataFrame({'x': X, 'g': G, 'score': SCORE})
KEPT = {'x': 'continuous', 'g': ('categorical', 3)}


def _describe(prepared: Prepared) -> dict:
    """What a prepared table holds: its rows, each covariate's type with its levels, and y's."""
    if prepared.reason is not None:
        return {'reason': prepared.reason}
    columns = {}
    for name, column in prepared.covariates.metadata.items():
        columns[name] = _state(column.type, column.levels)
    y = None
    if prepared.outcome is not None:
        column = prepared.outcome.metadata['y']
        y = _state(column.type, column.levels)
    return {'rows': prepared.covariates.rows, 'columns': columns, 'y': y}


def _state(type: str, levels: int | None) -> str | tuple[str, int]:
    return type if levels is None or type == 'binary' else (type, levels)


def test_a_clean_table_keeps_its_rows_and_columns_and_gives_y_its_values() -> None:
    """The control for the rules below: nothing is dropped, and y leaves the covariates."""
    prepared = prepare(TABLE, 'score', 'continuous')
    assert _describe(prepared) == {'rows': 8, 'columns': KEPT, 'y': 'continuous'}
    assert prepared.outcome.columns['y'].tolist() == np.array(SCORE, dtype=np.float32).tolist()
    assert prepared.covariates.columns['g'].tolist() == [0, 0, 1, 1, 2, 2, 0, 1]


MISSING = [None] * 5 + [1.0, 2.0, 3.0]
DAYS = [datetime.date(2026, 1, day) for day in [1, 1, 2, 2, 3, 3, 1, 2]]
# TABLE with columns added or replaced, then the rows and the covariates that are left
COVARIATES = [
    pytest.param({7: X}, 8, KEPT | {'7': 'continuous'}, id='a name that is no string becomes one'),
    pytest.param(
        {'sparse': MISSING}, 8, KEPT, id='a covariate missing in over half the rows costs no row'
    ),
    pytest.param(
        {'x': [None, *X[1:]]}, 7, KEPT, id='a row with a missing value in a kept column is dropped'
    ),
    pytest.param(
        {'g': ['a', 'a', 'b', 'b', 'c', 'c', ' ', 'b']},
        7,
        KEPT,
        id='a blank label is a missing value, not a level',
    ),
    pytest.param(
        {'g': pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c', '', 'b'])},
        7,
        KEPT,
        id="a factor's blank level is a missing value too",
    ),
    pytest.param({'unit': ['cm'] * 8}, 8, KEPT, id='a constant column is dropped'),
    pytest.param(
        {'x': [None, *X[1:]], 'first': [1, 0, 0, 0, 0, 0, 0, 0]},
        7,
        KEPT,
        id='a column constant over the complete rows is dropped',
    ),
    pytest.param(
        {'weight': [40.0 + 1e-10 * row for row in range(8)]},
        8,
        KEPT,
        id='a column constant in float32, as the covariates store it, is dropped',
    ),
    pytest.param(
        {'name': list('stuvwxyz')}, 8, KEPT, id='a label distinct in every row is dropped'
    ),
    pytest.param(
        {'word': ['p', 'p', 'q', 'r', 's', 't', 'p', 'q']},
        8,
        KEPT,
        id='a label of more levels than half the rows is dropped',
    ),
    pytest.param(
        {'name': [None, *'tuvwxyz']}, 8, KEPT, id="a row label's missing value costs no row"
    ),
    pytest.param(
        {'day': DAYS},
        8,
        KEPT | {'day': ('categorical', 3)},
        id='an object column without strings is read by its labels',
    ),
    pytest.param(
        {'y': X}, 8, KEPT | {'y_covariate': 'continuous'}, id='a covariate named y is renamed'
    ),
    pytest.param(
        {'year': [-3, -2, -1, 0, 1, 2, 3, 4]},
        8,
        KEPT | {'year': 'continuous'},
        id='an integer column is continuous, a negative value included',
    ),
]


@pytest.mark.parametrize(('columns', 'rows', 'kept'), COVARIATES)
def test_a_real_table_loses_the_columns_and_rows_its_measure_cannot_read(
    columns: dict, rows: int, kept: dict
) -> None:
    """Each rule was read off a table of mixedbench's flat sample that `fromPandas` refused
    or that lost most of its rows to a single column.
    """
    table = TABLE.copy()
    for name, values in columns.items():
        table[name] = values  # a name that is no string cannot be a keyword of assign
    prepared = prepare(table, 'score', 'continuous')
    assert _describe(prepared) == {'rows': rows, 'columns': kept, 'y': 'continuous'}


# the response's values, the type mixedbench read and the rows, y's column and its values
OUTCOMES = [
    pytest.param(
        [3, 0, 7, 2, 0, 1, 4, 2], 'count', 8, 'count', [3, 0, 7, 2, 0, 1, 4, 2], id='a count'
    ),
    pytest.param(
        [3.0, 0.0, 7.0, 2.0, 0.0, 1.0, 4.0, 2.0],
        'binomial',
        8,
        'count',
        [3, 0, 7, 2, 0, 1, 4, 2],
        id='the successes of a binomial response are a count, stored as floats too',
    ),
    pytest.param(
        [2, 1, 3, 1, 2, 3, 3, 1],
        'ordinal',
        8,
        ('ordered', 3),
        [1, 0, 2, 0, 1, 2, 2, 0],
        id='an ordinal response is ordered by its sorted values',
    ),
    pytest.param(
        ['r', 'p', 'q', 'p', 'r', 'q', 'q', 'p'],
        'categorical',
        8,
        ('categorical', 3),
        [2, 0, 1, 0, 2, 1, 1, 0],
        id='a categorical response is coded by its sorted labels',
    ),
    pytest.param(
        [1, 0, 0, 1, 0, 1, 1, 0],
        'count',
        8,
        'binary',
        [1, 0, 0, 1, 0, 1, 1, 0],
        id='two values make y binary whatever its type, as for a covariate',
    ),
    pytest.param([5] * 8, 'count', 8, None, None, id='a constant response leaves no y'),
    pytest.param(
        [40.0 + 1e-10 * row for row in range(8)],
        'continuous',
        8,
        None,
        None,
        id='a response constant in float32, as y stores it, leaves no y',
    ),
    pytest.param(
        [None, 0.4, 3.1, 2.2, 0.9, 4.0, 1.1, 2.7],
        'continuous',
        7,
        'continuous',
        np.array(SCORE[1:], dtype=np.float32).tolist(),
        id='a row without its response is dropped',
    ),
]


@pytest.mark.parametrize(('values', 'response_type', 'rows', 'column', 'y'), OUTCOMES)
def test_y_takes_the_type_mixedbench_read_and_its_levels_from_its_values(
    values: list, response_type: str, rows: int, column: str | tuple | None, y: list | None
) -> None:
    """tabdgp has no binomial column, so a binomial response enters as the count of its successes."""
    prepared = prepare(TABLE.assign(score=values), 'score', response_type)
    assert _describe(prepared) == {'rows': rows, 'columns': KEPT, 'y': column}
    if y is not None:
        assert prepared.outcome.columns['y'].tolist() == y


def test_a_count_response_with_a_fraction_fails_naming_the_response() -> None:
    """A count holds whole numbers, so a fraction says the response was read wrongly."""
    fractions = [0.5, 0.0, 7.0, 2.0, 0.0, 1.0, 4.0, 2.0]
    with pytest.raises(ValueError, match='score'):
        prepare(TABLE.assign(score=fractions), 'score', 'binomial')


def test_a_table_without_a_response_is_prepared_without_y() -> None:
    prepared = prepare(TABLE, None, None)
    assert _describe(prepared) == {'rows': 8, 'columns': KEPT | {'score': 'continuous'}, 'y': None}


HALF = [None, 1.0, None, 2.0, None, 3.0, None, 4.0]
OTHER_HALF = [1.0, None, 2.0, None, 3.0, None, 4.0, None]
LEFT_OUT = [
    pytest.param(
        TABLE.assign(x=HALF, g=OTHER_HALF),
        'no complete row',
        id='two columns missing in half the rows each, in turn',
    ),
    pytest.param(
        TABLE.assign(x=1.0, g=list('stuvwxyz')),
        'no covariate',
        id='a constant column and a row label beside the response',
    ),
]


@pytest.mark.parametrize(('table', 'reason'), LEFT_OUT)
def test_a_table_the_rules_empty_is_left_out_with_its_reason(
    table: pd.DataFrame, reason: str
) -> None:
    assert _describe(prepare(table, 'score', 'continuous')) == {'reason': reason}
