import itertools
import math
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


@pytest.fixture
def metric(monkeypatch):
    # The configurator reads command-line options when the module is imported.
    with monkeypatch.context() as import_context:
        import_context.setattr(sys, "argv", ["sslrec", "--model", "lightgcn", "--device", "cpu"])
        from trainer import metrics

    test_config = {"metrics": ["recall", "precision", "ndcg", "mrr"], "k": [1, 2, 4]}
    monkeypatch.setitem(metrics.configs, "test", test_config)
    monkeypatch.setitem(metrics.configs, "device", "cpu")
    return metrics.Metric(), test_config


def reciprocal_rank(row):
    return next((1 / rank for rank, label in enumerate(row, 1) if label == 1), 0.0)


@pytest.mark.parametrize("row,expected", [
    ([1, 1, 0, 0], 1.0),
    ([0, 1, 1, 0], 0.5),
    ([1, 1, 1, 1], 1.0),
    ([1, 0, 1, 0], 1.0),
    ([1, 0, 0, 0], 1.0),
    ([0, 1, 0, 0], 0.5),
    ([0, 0, 1, 0], 1 / 3),
    ([0, 0, 0, 1], 0.0),
    ([0, 0, 0, 0], 0.0),
])
def test_reciprocal_rank_at_three(metric, row, expected):
    scorer, _ = metric
    assert scorer.mrr(np.array([row]), 3) == pytest.approx(expected)


@pytest.mark.parametrize("shape,k", [((3, 4), 4), ((0, 4), 4), ((3, 0), 0), ((3, 4), 0)])
def test_zero_contribution(metric, shape, k):
    scorer, _ = metric
    assert scorer.mrr(np.zeros(shape), k) == 0.0


def test_later_hits_do_not_increase_the_score(metric):
    scorer, _ = metric
    relevance = np.array([[0, 1, 1, 1]])
    assert [scorer.mrr(relevance, k) for k in (1, 2, 3, 4)] == [0, 0.5, 0.5, 0.5]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_binary_rows_match_first_hit_reference(metric, k):
    scorer, _ = metric
    labels = np.array(list(itertools.product([0, 1], repeat=4)), dtype=float)
    original = labels.copy()
    expected = [reciprocal_rank(row[:k]) for row in labels]
    actual = [scorer.mrr(labels[i:i + 1], k) for i in range(len(labels))]
    assert all(0 <= score <= 1 for score in actual)
    np.testing.assert_allclose(actual, expected)
    assert scorer.mrr(labels, k) == pytest.approx(sum(expected))
    assert scorer.mrr(labels[::-1], k) == pytest.approx(sum(expected))
    assert sum(scorer.mrr(batch, k) for batch in np.array_split(labels, 3)) == pytest.approx(
        sum(expected)
    )
    np.testing.assert_array_equal(labels, original)


GROUND_TRUTH = [[0, 1], [1, 2], [5], [3]]
EXPECTED_MRR_SUMS = [1.0, 1.5, 1.75]


def test_eval_batch_returns_mrr_sums(metric):
    scorer, config = metric
    ranked = torch.arange(4).repeat(4, 1)
    result = scorer.eval_batch((ranked, GROUND_TRUTH), config["k"])
    assert isinstance(result["mrr"], np.ndarray)
    np.testing.assert_allclose(result["mrr"], EXPECTED_MRR_SUMS)


def assert_other_metrics(result, denominator):
    second, third, fourth = (1 / math.log2(rank + 1) for rank in (2, 3, 4))
    expected = {
        "recall": [0.5, 1.5, 3.0],
        "precision": [1.0, 1.5, 1.25],
        "ndcg": [1.0, 1 + second / (1 + second), 1 + (second + third) / (1 + second) + fourth],
    }
    for name, values in expected.items():
        assert isinstance(result[name], np.ndarray)
        np.testing.assert_allclose(result[name], np.array(values) / denominator)


def test_eval_batch_other_metrics(metric):
    scorer, config = metric
    result = scorer.eval_batch((torch.arange(4).repeat(4, 1), GROUND_TRUTH), config["k"])
    assert_other_metrics(result, denominator=1)


class SyntheticUsers:
    def __init__(self):
        self.test_users = list(range(len(GROUND_TRUTH)))
        self.user_pos_lists = GROUND_TRUTH
        self.user_history_lists = [[4] for _ in GROUND_TRUTH]

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, index):
        return self.test_users[index]


class FixedScores:
    def __init__(self):
        # Item 4 has the highest raw score and is masked as user history.
        self.scores = torch.tensor([[4.0, 3.0, 2.0, 1.0, 5.0, 0.0]]).repeat(len(GROUND_TRUTH), 1)

    def full_predict(self, batch):
        return self.scores[batch[0]]

    def generate(self):
        return torch.eye(len(self.scores)), self.scores.T

    def rating(self, users, items):
        return users @ items.T


@pytest.mark.parametrize("one_forward", [False, True], ids=["batch-forward", "one-forward"])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_eval_averages_users_across_batches(metric, one_forward, batch_size):
    scorer, config = metric
    config["eval_at_one_forward"] = one_forward
    dataset = SyntheticUsers()
    loader = DataLoader(dataset, batch_size=batch_size)
    result = scorer.eval(FixedScores(), loader)
    assert_other_metrics(result, denominator=len(dataset))
    np.testing.assert_allclose(result["mrr"], np.array(EXPECTED_MRR_SUMS) / len(dataset))
