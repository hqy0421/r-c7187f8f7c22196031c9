import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import PredictionLedger


class FixedStubClassifier(BaseEstimator, ClassifierMixin):
    """Deterministic binary classifier: `predict` and `predict_proba` return fixed values.

    Used to control exactly what gets written to the ledger, including cases where the hard
    prediction disagrees with `argmax(predict_proba)` and where `classes_` is not sorted.
    """

    def __init__(self, proba=(0.3, 0.7), pred=1, reverse_classes=False):
        self.proba = proba
        self.pred = pred
        self.reverse_classes = reverse_classes

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes[::-1] if self.reverse_classes else classes
        self.n_proba_calls_ = 0
        return self

    def predict_proba(self, X):
        self.n_proba_calls_ += 1
        return np.tile(np.asarray(self.proba, dtype=float), (np.asarray(X).shape[0], 1))

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.pred)


@pytest.fixture
def Xy():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = np.repeat([0, 1], 20)
    return X, y


@parametrize_with_checks([PredictionLedger(LogisticRegression(), check_input=True)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_fit2d_1feature",  # custom message
        "check_sample_weight_equivalence_on_dense_data",  # TODO: come back to this
    }:
        pytest.skip()

    check(estimator)


def test_fit_creates_versions_and_keeps_history_across_refits(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred=1))
    ledger.fit(X, y)
    assert ledger.version_ == 1
    ledger.predict(X[:3], record_ids=["a", "b", "c"])

    # retrain with a differently behaving model: new version, independent classifier
    ledger.set_params(model=FixedStubClassifier(proba=(0.6, 0.4), pred=0))
    ledger.fit(X, y)
    assert ledger.version_ == 2
    ledger.predict(X[:2], record_ids=["a", "d"])

    records = ledger.records()
    assert [row["version"] for row in records] == [1, 1, 1, 2, 2]
    assert [row["record_id"] for row in records] == ["a", "b", "c", "a", "d"]
    # old rows are untouched by the retrain
    assert np.allclose(records[0]["proba"], [0.3, 0.7])
    assert records[0]["y_pred"] == 1
    assert np.allclose(records[3]["proba"], [0.6, 0.4])
    assert records[3]["y_pred"] == 0


def test_records_store_true_predict_not_argmax(Xy):
    X, y = Xy
    # argmax(proba) would be class 1, but the true predict says class 0
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.2, 0.8), pred=0)).fit(X, y)
    out = ledger.predict(X[:2], record_ids=["a", "b"])
    assert list(out) == [0, 0]
    records = ledger.records()
    assert [row["y_pred"] for row in records] == [0, 0]
    assert np.allclose(records[0]["proba"], [0.2, 0.8])

    ledger.add_labels(["a", "b"], [0, 1])
    summary = ledger.evaluation()["overall"]
    # confusion matrix must reflect the true predict (0), not argmax (1)
    assert summary["confusion_matrix"].tolist() == [[1, 0], [1, 0]]


def test_proba_columns_interpreted_via_true_classes(Xy):
    X, y = Xy
    # this estimator reports classes_=[1, 0], so column 0 is P(class 1)=0.9
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.9, 0.1), pred=1, reverse_classes=True)).fit(X, y)
    assert ledger.estimator_.classes_.tolist() == [1, 0]
    ledger.predict(X[:1], record_ids=["a"])
    ledger.add_labels(["a"], [1])
    summary = ledger.evaluation()["overall"]
    # Brier must use the positive-class column located via the true classes_: (0.9 - 1)^2
    assert summary["brier_mean"] == pytest.approx(0.01)
    assert ledger.records()[0]["classes"].tolist() == [1, 0]


def test_metrics_per_version_and_overall(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred=1)).fit(X, y)
    ledger.predict(X[:3], record_ids=["a", "b", "c"])
    ledger.add_labels(["a", "b"], [1, 0])  # "c" stays unlabeled

    ledger.set_params(model=FixedStubClassifier(proba=(0.6, 0.4), pred=0))
    ledger.fit(X, y)
    ledger.predict(X[:2], record_ids=["a", "d"])  # "d" stays unlabeled

    result = ledger.evaluation()

    v1 = result["per_version"][1]
    assert v1["n_predictions"] == 3
    assert v1["n_labeled"] == 2
    assert v1["confusion_matrix"].tolist() == [[0, 1], [0, 1]]
    assert v1["accuracy"] == pytest.approx(0.5)
    # a: (0.7 - 1)^2 = 0.09, b: (0.7 - 0)^2 = 0.49
    assert v1["brier_mean"] == pytest.approx((0.09 + 0.49) / 2)

    v2 = result["per_version"][2]
    assert v2["n_predictions"] == 2
    assert v2["n_labeled"] == 1  # only "a" is labeled; the label follows the record across versions
    assert v2["confusion_matrix"].tolist() == [[0, 0], [1, 0]]
    assert v2["accuracy"] == pytest.approx(0.0)
    assert v2["brier_mean"] == pytest.approx(0.36)  # (0.4 - 1)^2

    overall = result["overall"]
    assert overall["n_predictions"] == 5
    assert overall["n_labeled"] == 3  # unlabeled "c" and "d" enter no denominator
    assert overall["confusion_matrix"].tolist() == [[0, 1], [1, 1]]
    assert overall["accuracy"] == pytest.approx(1 / 3)  # only v1's row "a" is correct
    assert overall["brier_mean"] == pytest.approx((0.09 + 0.49 + 0.36) / 3)


def test_version_without_predictions_is_reported(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier()).fit(X, y)
    ledger.fit(X, y)  # version 2 gets no predictions
    v2 = ledger.evaluation()["per_version"][2]
    assert v2["n_predictions"] == 0
    assert v2["n_labeled"] == 0
    assert v2["confusion_matrix"].tolist() == [[0, 0], [0, 0]]
    assert v2["accuracy"] is None
    assert v2["brier_mean"] is None


def test_correct_and_withdraw_labels(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred=1)).fit(X, y)
    ledger.predict(X[:3], record_ids=["a", "b", "c"])
    ledger.add_labels(["a", "b"], [1, 0])

    ledger.correct_labels(["b"], [1])
    v1 = ledger.evaluation()["per_version"][1]
    assert v1["n_labeled"] == 2
    assert v1["confusion_matrix"].tolist() == [[0, 0], [0, 2]]
    assert v1["accuracy"] == pytest.approx(1.0)
    assert v1["brier_mean"] == pytest.approx(0.09)

    # withdrawing "a" removes the label from every stored prediction of that record
    ledger.set_params(model=FixedStubClassifier(proba=(0.6, 0.4), pred=0))
    ledger.fit(X, y)
    ledger.predict(X[:1], record_ids=["a"])
    ledger.withdraw_labels(["a"])
    result = ledger.evaluation()
    assert result["per_version"][1]["n_labeled"] == 1  # only "b" remains
    assert result["per_version"][2]["n_labeled"] == 0
    assert result["per_version"][2]["accuracy"] is None
    assert result["overall"]["n_labeled"] == 1
    assert ledger.labels() == {"b": 1}


def test_evaluation_uses_history_not_current_model(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred=1)).fit(X, y)
    ledger.predict(X[:2], record_ids=["a", "b"])
    ledger.add_labels(["a", "b"], [1, 0])
    brier_v1_before = ledger.evaluation()["per_version"][1]["brier_mean"]

    # retrain with a model that behaves differently and counts its own predict_proba calls
    ledger.set_params(model=FixedStubClassifier(proba=(0.6, 0.4), pred=0))
    ledger.fit(X, y)
    result = ledger.evaluation()

    assert ledger.estimator_.n_proba_calls_ == 0  # evaluation never asks the current model
    assert result["per_version"][1]["brier_mean"] == pytest.approx(brier_v1_before)


def test_invalid_label_batches_leave_no_partial_results(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier()).fit(X, y)
    ledger.predict(X[:3], record_ids=["a", "b", "c"])
    ledger.add_labels(["a"], [1])

    bad_batches = [
        lambda: ledger.add_labels(["b", "c"], [1, 5]),  # unknown label value
        lambda: ledger.add_labels(["b", "b"], [0, 1]),  # duplicate ids
        lambda: ledger.add_labels(["b"], [0, 1]),  # length mismatch
        lambda: ledger.add_labels(["a"], [0]),  # already labeled: use correct_labels
        lambda: ledger.correct_labels(["b"], [1]),  # not labeled yet: use add_labels
        lambda: ledger.withdraw_labels(["b"]),  # not labeled: nothing to withdraw
        lambda: ledger.add_labels("b", [0]),  # a bare string is not a batch
    ]
    for call in bad_batches:
        with pytest.raises(ValueError):
            call()
        assert ledger.labels() == {"a": 1}  # nothing was applied


def test_invalid_predict_batches_leave_no_partial_results(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier()).fit(X, y)
    ledger.predict(X[:1], record_ids=["a"])

    with pytest.raises(ValueError):
        ledger.predict(X[:2], record_ids=["b", "b"])  # duplicate ids
    with pytest.raises(ValueError):
        ledger.predict(X[:2], record_ids=["b"])  # length mismatch
    with pytest.raises(ValueError):
        ledger.predict(X[:1], record_ids=[["unhashable"]])

    assert [row["record_id"] for row in ledger.records()] == ["a"]


def test_invalid_fit_leaves_no_partial_state(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred=1)).fit(X, y)
    ledger.predict(X[:1], record_ids=["a"])
    estimator_before = ledger.estimator_

    with pytest.raises(ValueError):
        ledger.fit(X[:5], np.zeros(5))  # a single class
    with pytest.raises(ValueError):
        ledger.fit(X, np.resize([0, 1, 2], len(y)))  # three classes
    with pytest.raises(ValueError):
        ledger.fit(X, np.repeat([1, 2], 20))  # binary, but not the fixed classes

    assert ledger.version_ == 1  # no new version was registered
    assert ledger.estimator_ is estimator_before  # current model untouched
    assert [row["record_id"] for row in ledger.records()] == ["a"]


def test_non_probabilistic_model_rejected(Xy):
    X, y = Xy
    with pytest.raises(ValueError):
        PredictionLedger(LinearRegression()).fit(X, y)


def test_returned_views_cannot_corrupt_ledger(Xy):
    X, y = Xy
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred=1)).fit(X, y)
    ledger.predict(X[:2], record_ids=["a", "b"])
    ledger.add_labels(["a", "b"], [1, 0])
    before = ledger.evaluation()

    records = ledger.records()
    records[0]["proba"][:] = 9.9
    records[0]["y_pred"] = 0
    records[0]["classes"][:] = 0
    records.append({"version": 1, "record_id": "fake", "y_pred": 1, "proba": np.ones(2)})

    labels = ledger.labels()
    labels.clear()
    labels["intruder"] = 1

    ev = ledger.evaluation()
    ev["overall"]["confusion_matrix"][:] = 0
    ev["per_version"].clear()

    after = ledger.evaluation()
    assert len(ledger.records()) == 2
    assert ledger.labels() == {"a": 1, "b": 0}
    assert after["overall"]["n_predictions"] == before["overall"]["n_predictions"]
    assert after["overall"]["n_labeled"] == before["overall"]["n_labeled"]
    assert after["overall"]["confusion_matrix"].tolist() == before["overall"]["confusion_matrix"].tolist()
    assert after["overall"]["accuracy"] == pytest.approx(before["overall"]["accuracy"])
    assert after["overall"]["brier_mean"] == pytest.approx(before["overall"]["brier_mean"])
    assert after["per_version"][1]["n_predictions"] == 2


def test_score_keeps_sklearn_semantics_and_does_not_log(Xy):
    X, y = Xy
    ledger = PredictionLedger(LogisticRegression()).fit(X, y)
    reference = LogisticRegression().fit(X, y)
    assert ledger.score(X, y) == pytest.approx(reference.score(X, y))
    assert ledger.records() == []  # scoring did not write to the ledger

    # cross-validation style scoring keeps working on the meta estimator
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(PredictionLedger(LogisticRegression()), X, y, cv=2)
    assert scores.shape == (2,)


def test_predict_without_ids_does_not_log(Xy):
    X, y = Xy
    ledger = PredictionLedger(LogisticRegression()).fit(X, y)
    ledger.predict(X[:3])
    ledger.predict_proba(X[:3])
    assert ledger.records() == []
    assert ledger.evaluation()["overall"]["n_predictions"] == 0


def test_string_classes(Xy):
    X, _ = Xy
    y_str = np.where(np.repeat([0, 1], 20), "pos", "neg")
    ledger = PredictionLedger(FixedStubClassifier(proba=(0.3, 0.7), pred="pos")).fit(X, y_str)
    ledger.predict(X[:2], record_ids=["a", "b"])
    ledger.add_labels(["a", "b"], ["pos", "neg"])
    summary = ledger.evaluation()["overall"]
    # positive class is classes_[1] == "pos" with probability 0.7
    assert summary["brier_mean"] == pytest.approx((0.09 + 0.49) / 2)
    assert summary["confusion_matrix"].tolist() == [[0, 1], [0, 1]]
