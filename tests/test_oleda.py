"""Tests for oleda.

Deliberately light on mocking: the point of most of these is that the report functions run
end to end on awkward-but-realistic frames (missing values, constant columns, single-category
columns, datetime indices) without raising. That is the failure mode an EDA library actually
has - the statistics are usually fine, but a column shape nobody anticipated takes the whole
report down.

The statistical helpers are checked against values computed by hand or by scipy directly, so a
refactor that changes a number fails here rather than silently shipping.
"""
import matplotlib
matplotlib.use('Agg')          # no display in CI; must precede any pyplot import

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import oleda
from oleda.eda import (
    cramers_corrected_stat,
    missing_values_table,
    get_categorical,
)


@pytest.fixture
def frame():
    """A small frame carrying the shapes that break EDA code."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        'num': rng.normal(size=n),
        'num_with_na': np.where(rng.random(size=n) < 0.2, np.nan, rng.normal(size=n)),
        'cat': rng.choice(['a', 'b', 'c'], size=n),
        'cat_with_na': np.where(rng.random(size=n) < 0.3, None,
                                rng.choice(['x', 'y'], size=n)),
        'binary': rng.integers(0, 2, size=n),
        'constant': 1,
        'single_category': 'only',
        'target': rng.integers(0, 2, size=n),
    })


# ── statistical helpers ───────────────────────────────────────────────────────

def test_cramers_v_is_zero_for_independent_table():
    """Perfectly independent 2x2: the bias correction clamps phi2 at 0, so V is exactly 0."""
    table = pd.DataFrame([[50, 50], [50, 50]])
    assert cramers_corrected_stat(table) == 0.0


def test_cramers_v_is_positive_for_dependent_table():
    """A strongly associated table must give V well above 0 and at most 1."""
    table = pd.DataFrame([[90, 10], [10, 90]])
    v = cramers_corrected_stat(table)
    assert 0.5 < v <= 1.0


def test_cramers_v_matches_hand_computation():
    """Pin the formula: V = sqrt(phi2corr / min(kcorr-1, rcorr-1)) on a known table."""
    table = pd.DataFrame([[60, 40], [30, 70]])
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    expected = np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))
    assert cramers_corrected_stat(table) == pytest.approx(expected)


def test_cramers_v_is_symmetric():
    table = pd.DataFrame([[60, 40], [30, 70]])
    assert cramers_corrected_stat(table) == pytest.approx(
        cramers_corrected_stat(table.T))


# ── missing values ────────────────────────────────────────────────────────────

def test_missing_values_table_counts_and_percentages():
    df = pd.DataFrame({'a': [1, None, None, 4], 'b': [1, 2, 3, 4]})
    table = missing_values_table(df)
    assert list(table.index) == ['a'], 'columns with no missing values must be excluded'
    assert table.loc['a', 'Missing'] == 2
    assert table.loc['a', '% of Total'] == pytest.approx(50.0)


def test_missing_values_table_is_empty_when_nothing_is_missing():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert missing_values_table(df).empty


def test_missing_values_table_is_sorted_worst_first():
    df = pd.DataFrame({
        'few': [1, 2, 3, None],
        'many': [None, None, None, 1],
    })
    assert list(missing_values_table(df).index) == ['many', 'few']


# ── categorical selection ─────────────────────────────────────────────────────

def test_get_categorical_picks_up_strings_and_binaries(frame):
    found = get_categorical(frame)
    assert 'cat' in found
    assert 'binary' in found, 'a 0/1 numeric column is categorical when binary=True'
    assert 'num' not in found


def test_get_categorical_drops_constant_columns(frame):
    found = get_categorical(frame)
    assert 'constant' not in found
    assert 'single_category' not in found


def test_get_categorical_can_exclude_binaries(frame):
    assert 'binary' not in get_categorical(frame, binary=False)


def test_get_categorical_drops_high_cardinality_identifiers():
    df = pd.DataFrame({'id': [f'id{i}' for i in range(100)], 'cat': ['a', 'b'] * 50})
    found = get_categorical(df)
    assert 'id' not in found, 'an identifier-like column must not be treated as categorical'
    assert 'cat' in found


# ── the reports run end to end ────────────────────────────────────────────────

def test_report_runs_without_target(frame):
    oleda.report(frame)


def test_report_runs_with_binary_target(frame):
    oleda.report(frame, 'target', nbrmax=3)


def test_report_runs_with_continuous_target(frame):
    frame = frame.copy()
    frame['continuous_target'] = np.linspace(0, 1, len(frame))
    oleda.report(frame, 'continuous_target', nbrmax=3)


def test_report_honours_the_ignore_list(frame):
    oleda.report(frame, 'target', ignore=['cat', 'num'], nbrmax=3)


def test_report_runs_on_a_datetime_index(frame):
    frame = frame.copy()
    frame.index = pd.date_range('2024-01-01', periods=len(frame), freq='h')
    oleda.report(frame)


def test_pairwise_report_runs(frame):
    rng = np.random.default_rng(1)
    other = frame.copy()
    other['num'] = rng.normal(loc=2.0, size=len(other))
    oleda.pairwise_report(frame, other, nbrmax=3)


def test_report_survives_an_all_missing_column(frame):
    frame = frame.copy()
    frame['all_missing'] = np.nan
    oleda.report(frame)


def test_report_survives_a_tiny_frame():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'x'], 'target': [0, 1, 0]})
    oleda.report(df)
