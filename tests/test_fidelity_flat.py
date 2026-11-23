"""The fidelity checks of `Scm.flat()` (docs/fidelity.md), on the real flat tables that
experiments/fidelity/flat.py measured.
They redraw from the current code at the committed keys, so a changed default can break them.
The seeds, the draws and the gates are flat.py's, so its report and these tests judge alike.
The named gaps and the controls are this file's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabdgp import Scm, fidelity

# flat.py is a script's module, not a package's
sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments' / 'fidelity'))

import flat

# =====================================================================================
# the named gaps
# =====================================================================================
# the axes the defaults fail today, each with its cause; the list may only shrink

NARROW_SIZES = 'the sizes are a coverage-first guess, narrower than the real tables'
TYPE_MIX = 'the type shares are a guess, and real tables hold more continuous columns'
FEW_CATEGORICAL = 'fewer real tables than drawn ones have a categorical or ordered column'
DISCRETE_NUMBERS = (
    'ties and heaping: real tables store rounded numbers, codes and years, '
    'and the defaults draw nearly every value distinct'
)
OVERDISPERSED_COUNTS = (
    "a real count y's variance exceeds its mean, and the defaults draw a Poisson count"
)
BALANCED_LEVELS = "a real categorical column's levels are more balanced than a drawn one's"
EXACT_DEPENDENCE = (
    'exact dependence: real tables hold continuous columns that are monotone functions '
    'of each other and rank-deficient designs, which the defaults almost never draw'
)
WEAK_CODED_RELATIONS = (
    'real binary and categorical columns relate more weakly to the other columns than drawn ones'
)
STRONG_Y_RELATION = 'a real y relates more strongly to its closest covariate than a drawn one'
BOTH_SIDES = 'real tables lie outside their draws on both sides, for no single known cause'
OUTLIERS = (
    'real tables hold more outliers in their most outlying continuous column, for no known cause'
)

NAMED_GAPS = {
    'schema': {
        'rows < 20': NARROW_SIZES,
        'design_columns > 19': NARROW_SIZES,
        'design_columns > rows / 2': NARROW_SIZES,
        'continuous_share': TYPE_MIX,
        'binary_share': TYPE_MIX,
        'categorical_share': TYPE_MIX,
        'levels_median': FEW_CATEGORICAL,
        'levels_max': FEW_CATEGORICAL,
    },
    'content': {
        'continuous_distinct_share_median': DISCRETE_NUMBERS,
        'continuous_distinct_share_max': DISCRETE_NUMBERS,
        'continuous_mode_share_median': DISCRETE_NUMBERS,
        'y_continuous_distinct_share': DISCRETE_NUMBERS,
        'leverage': DISCRETE_NUMBERS,
        'y_count_dispersion': OVERDISPERSED_COUNTS,
        'y_count_skewness': OVERDISPERSED_COUNTS,
        'y_count_excess_kurtosis': OVERDISPERSED_COUNTS,
        'y_count_outlier_share': OVERDISPERSED_COUNTS,
        'categorical_largest_level_share_median': BALANCED_LEVELS,
        'categorical_largest_level_share_max': BALANCED_LEVELS,
        'categorical_smallest_level_share_median': BALANCED_LEVELS,
        'categorical_smallest_level_share_max': BALANCED_LEVELS,
        'continuous_relation_max': EXACT_DEPENDENCE,
        'condition_index': EXACT_DEPENDENCE,
        'binary_relation_max': WEAK_CODED_RELATIONS,
        'categorical_relation_median': WEAK_CODED_RELATIONS,
        'categorical_relation_max': WEAK_CODED_RELATIONS,
        'y_relation': STRONG_Y_RELATION,
        'continuous_skewness_median': BOTH_SIDES,
        'continuous_excess_kurtosis_median': BOTH_SIDES,
        'continuous_relation_median': BOTH_SIDES,
        'continuous_outlier_share_max': OUTLIERS,
    },
}

# the relations between columns, which the negative control lacks:
# each type's largest relation, the condition index and the leverage
RELATIONS = [
    'continuous_relation_median',
    'continuous_relation_max',
    'count_relation_median',
    'count_relation_max',
    'binary_relation_median',
    'binary_relation_max',
    'categorical_relation_median',
    'categorical_relation_max',
    'ordered_relation_median',
    'ordered_relation_max',
    'condition_index',
    'leverage',
]

# =====================================================================================
# the real tables and their draws
# =====================================================================================

STAND_IN_SEED = flat.SEED + 2  # apart from the paired draws' and the drawn schemas'


@pytest.fixture(scope='module')
def tables() -> pd.DataFrame:
    """The real tables of the certifying part inside the default sizes."""
    real = pd.read_csv(flat.REAL)
    certifying = (real['part'] == 'certifying') & (real['stratum'] == 'inside')
    return real[certifying]


@pytest.fixture(scope='module')
def default_draws(tables: pd.DataFrame) -> np.ndarray:
    return _drawArm(tables, {})


def _drawArm(tables: pd.DataFrame, overrides: dict) -> np.ndarray:
    """flat.DRAWS draws per real table at its schema and keys, under the overrides."""
    draws = []
    for table in tables.to_dict('records'):
        schema = json.loads(table['schema'])
        vectors = flat.drawScm(table['position'], schema | overrides)
        draws.append(np.stack(vectors))
    return np.stack(draws)  # (t, s, p)


def _drawStandIns(tables: pd.DataFrame) -> np.ndarray:
    """A table of the defaults at each real table's schema, measured as the files hold them."""
    stand_ins = []
    for table in tables.to_dict('records'):
        scm = Scm.flat(**json.loads(table['schema']))
        stand_ins.append(flat.measureDraw(scm, STAND_IN_SEED, table['position']))
    return np.stack(stand_ins)  # (t, p)


def _failRecall(recall: fidelity.Recall) -> set[str]:
    """The axes on which more real tables lie outside their draws than the gate allows."""
    gate = flat.gateRecall(recall.tables)
    failing = set()
    for axis, outside, most in zip(flat.AXES, recall.outside, gate):
        if outside > most:
            failing.add(axis)
    return failing


# =====================================================================================
# the schema check
# =====================================================================================


@pytest.mark.slow
def test_the_schema_check_fails_only_on_its_named_gaps() -> None:
    """A size limit fails when more real tables lie beyond it than flat.OUTSIDE_SHARE,
    and an axis of the type mix and the levels when its distance to the drawn schemas
    exceeds flat.SCHEMA_DISTANCE.
    """
    real = pd.read_csv(flat.REAL)
    certifying = real[real['part'] == 'certifying']
    failing = set()
    outside = flat.findOutside(certifying['rows'], certifying['design_columns'])
    for limit, beyond in outside.items():
        if beyond.mean() > flat.OUTSIDE_SHARE:
            failing.add(limit)
    inside = certifying[certifying['stratum'] == 'inside']
    drawn = flat.drawSchemas()
    rng = np.random.default_rng(flat.SEED)
    # the distances alone, without their null
    comparisons = fidelity.distances(
        inside[flat.MIX].to_numpy(), drawn[flat.MIX].to_numpy(), rng, 0
    )
    for axis, comparison in zip(flat.MIX, comparisons):
        if comparison.statistic > flat.SCHEMA_DISTANCE:
            failing.add(axis)
    assert failing == set(NAMED_GAPS['schema'])


# =====================================================================================
# the content check
# =====================================================================================


@pytest.mark.slow
def test_recall_fails_only_on_its_named_gaps(
    tables: pd.DataFrame, default_draws: np.ndarray
) -> None:
    """An axis that fails outside the list is a new gap, and a listed axis that passes
    has closed its gap and leaves the list.
    """
    real = tables[flat.AXES].to_numpy()  # (t, p)
    assert _failRecall(fidelity.recall(real, default_draws)) == set(NAMED_GAPS['content'])


@pytest.mark.slow
def test_precision_stays_above_its_floor(tables: pd.DataFrame, default_draws: np.ndarray) -> None:
    real = tables[flat.AXES].to_numpy()  # (t, p)
    precision = fidelity.precision(real, default_draws, tables['source'].to_numpy())
    assert precision.share >= flat.PRECISION_FLOOR


# =====================================================================================
# the controls
# =====================================================================================


@pytest.mark.slow
def test_independent_covariates_fail_recall_on_the_relations(tables: pd.DataFrame) -> None:
    """The negative control: a check that passes tables without relations between their columns
    is too loose. While the defaults fail these axes too, it shows only that the gates are not
    too wide.
    An axis ranked on no more tables than its gate lets lie outside cannot fail (e.g. a relation
    of ordered columns, which 3 tables show), so the control leaves it out.
    """
    draws = _drawArm(tables, {'dependence': 'none'})
    real = tables[flat.AXES].to_numpy()  # (t, p)
    recall = fidelity.recall(real, draws)
    gate = flat.gateRecall(recall.tables)
    testable = set()
    for axis, ranked, most in zip(flat.AXES, recall.tables, gate):
        if axis in RELATIONS and ranked > most:
            testable.add(axis)
    assert testable - _failRecall(recall) == set()


@pytest.mark.slow
def test_a_drawn_table_in_a_real_tables_place_passes_every_gate(
    tables: pd.DataFrame, default_draws: np.ndarray
) -> None:
    """The positive control: a check that fails the generator's own tables is too strict.
    It also closes every named gap of the content check.
    """
    stand_ins = _drawStandIns(tables)
    assert _failRecall(fidelity.recall(stand_ins, default_draws)) == set()
    precision = fidelity.precision(stand_ins, default_draws, tables['source'].to_numpy())
    assert precision.share >= flat.PRECISION_FLOOR
