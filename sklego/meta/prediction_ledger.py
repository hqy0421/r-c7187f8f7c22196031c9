import logging
from inspect import signature

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn_compat.utils.multiclass import type_of_target
from sklearn_compat.utils.validation import _check_n_features, validate_data

from sklego.base import ProbabilisticClassifier


class PredictionLedger(ClassifierMixin, BaseEstimator):
    """Wrap a binary classifier and keep an append-only ledger of the predictions it makes over time.

    Models get retrained while business labels arrive late, get corrected or get withdrawn. This
    meta-estimator makes that workflow auditable:

    - Every call to `fit` trains a fresh, independent copy of `model` and registers a new version
      (1, 2, 3, ...). Historical ledger rows and labels are kept across retraining.
    - `predict(X, record_ids=...)` stores, per record id, the *actual* `predict` output of the
      model at that moment (never `argmax(predict_proba)) together with the `predict_proba` row,
      tagged with the version that produced them.
    - Labels live apart from predictions and are keyed by record id: `add_labels` (late labels),
      `correct_labels` (fixes) and `withdraw_labels` (retractions).
    - `evaluation` scores the stored historical predictions against the *current* labels, per
      version and overall. It never recomputes anything with the current model: a prediction is
      judged exactly as it was made.

    !!! info
        Only binary classification is supported. The two classes are fixed by the first `fit` and
        every later `fit` must see exactly the same two classes. Probability columns are always
        interpreted through the true `classes_` of the version that produced them.

    Invalid batches (bad labels, unknown record transitions, duplicate ids, ...) raise an error and
    leave no partial results behind. All accessors return copies, so callers cannot corrupt the
    ledger. `score` keeps the usual scikit-learn meaning: accuracy of the *current* model.

    Parameters
    ----------
    model : scikit-learn compatible classifier
        Classifier that will be wrapped. It should implement `predict_proba`.
    check_input : bool, default=False
        Whether or not to check the input data. If False, the checks are delegated to the wrapped
        estimator.

    Attributes
    ----------
    estimator_ : scikit-learn compatible classifier
        The fitted classifier of the current version.
    classes_ : array-like, shape=(2,)
        The two fixed class labels, established at the first `fit`.
    version_ : int
        The current model version. Starts at 1 and increases by 1 on every successful `fit`.

    Example
    -------

    ```py
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklego.meta import PredictionLedger

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 2))
    y = (X[:, 0] > 0).astype(int)

    ledger = PredictionLedger(LogisticRegression())
    ledger.fit(X, y)  # trains version 1
    ledger.predict(X[:3], record_ids=["row-1", "row-2", "row-3"])
    ledger.add_labels(["row-1", "row-2"], [1, 0])  # late labels arrive

    ledger.fit(X, y)  # retrains: version 2, history of version 1 is kept
    ledger.predict(X[:2], record_ids=["row-1", "row-4"])

    result = ledger.evaluation()
    print(result["overall"]["n_predictions"])
    ### 5
    print(result["per_version"][1]["n_labeled"])
    ### 2
    ```
    """

    _required_parameters = ["model"]

    def __init__(self, model, check_input=False):
        self.model = model
        self.check_input = check_input

    def fit(self, X, y, sample_weight=None):
        """Train a fresh copy of the wrapped classifier and register a new model version.

        The ledger (recorded predictions and labels) is never touched by fitting. If anything is
        invalid, an error is raised and no new version is registered.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target values. Must contain exactly the two fixed classes.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample, passed on if the wrapped estimator supports it.

        Returns
        -------
        self : PredictionLedger
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `model` is not a classifier with a `predict_proba` method.
            - If `y` does not contain exactly the two fixed classes.
            - If the fitted estimator does not expose exactly those two classes.
        """
        if not isinstance(self.model, ProbabilisticClassifier):
            raise ValueError("The PredictionLedger meta model only works on classification models with .predict_proba.")

        if self.check_input:
            X, y = validate_data(self, X=X, y=y, ensure_all_finite=False, ensure_min_features=0, reset=True)
        else:
            _check_n_features(self, X, reset=True)

        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(f"Only binary classification is supported. The type of the target is {y_type}.")
        fit_classes = np.unique(y)
        if fit_classes.shape[0] != 2:
            raise ValueError(
                f"PredictionLedger requires exactly two classes in y, "
                f"got {fit_classes.shape[0]} class(es): {fit_classes.tolist()}."
            )
        if getattr(self, "classes_", None) is not None and not np.array_equal(fit_classes, self.classes_):
            raise ValueError(
                f"The classes of this task were fixed at the first fit to {self.classes_.tolist()}, "
                f"but this batch contains {fit_classes.tolist()}."
            )

        estimator = clone(self.model)
        if sample_weight is not None and "sample_weight" in signature(estimator.fit).parameters:
            estimator.fit(X, y, sample_weight=_check_sample_weight(sample_weight, X))
        else:
            if sample_weight is not None:
                logging.warning("Estimator ignores sample_weight.")
            estimator.fit(X, y)

        estimator_classes = np.asarray(estimator.classes_)
        if estimator_classes.shape != (2,) or set(estimator_classes.tolist()) != set(fit_classes.tolist()):
            raise ValueError(
                f"The wrapped estimator was fitted on classes {estimator_classes.tolist()}, "
                f"expected exactly {fit_classes.tolist()}."
            )

        # Everything validated: commit the new version. History and labels are preserved.
        if not hasattr(self, "_records"):
            self._records = []
            self._labels = {}
            self._version_classes = {}
        self.estimator_ = estimator
        self.classes_ = fit_classes
        self.version_ = getattr(self, "version_", 0) + 1
        self._version_classes[self.version_] = estimator_classes.copy()
        return self

    def predict(self, X, record_ids=None):
        """Predict target values for `X` with the current model version.

        When `record_ids` are given, each prediction is appended to the ledger together with the
        true `predict` output and the `predict_proba` row as produced right now. When omitted,
        nothing is written to the ledger.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.
        record_ids : array-like of shape (n_samples,), default=None
            Hashable ids identifying each record. Must be unique within this call. If any id is
            invalid, an error is raised and no partial results are stored.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ["classes_", "estimator_", "version_"])
        if self.check_input:
            X = validate_data(self, X=X, ensure_min_features=0, ensure_all_finite=False, reset=False)
        else:
            _check_n_features(self, X, reset=False)

        if record_ids is None:
            return self.estimator_.predict(X)

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        ids = self._check_record_ids(record_ids, n_expected=n_samples)
        y_pred = np.asarray(self.estimator_.predict(X))
        proba = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        if proba.shape != (n_samples, 2):
            raise ValueError(f"predict_proba returned shape {proba.shape}, expected {(n_samples, 2)}.")

        for record_id, predicted, probabilities in zip(ids, y_pred, proba):
            self._records.append(
                {
                    "version": self.version_,
                    "record_id": record_id,
                    "y_pred": predicted,
                    "proba": probabilities.copy(),
                }
            )
        return y_pred

    def predict_proba(self, X):
        """Alias for `.predict_proba()` of the underlying estimator. Does not write to the ledger."""
        check_is_fitted(self, ["classes_", "estimator_", "version_"])
        if self.check_input:
            X = validate_data(self, X=X, ensure_min_features=0, ensure_all_finite=False, reset=False)
        else:
            _check_n_features(self, X, reset=False)

        return self.estimator_.predict_proba(X)

    def add_labels(self, record_ids, labels):
        """Attach labels to records that have no label yet (late-arriving business labels).

        Labels are keyed by record id and apply to every stored prediction of that record, across
        all versions. The batch is validated before anything is stored: if any id already has a
        label or any label is not one of the two fixed classes, an error is raised and no partial
        results are left behind.

        Parameters
        ----------
        record_ids : array-like of shape (n_records,)
            Hashable ids of the records to label. Must not already have a label.
        labels : array-like of shape (n_records,)
            The labels, each one of the two fixed classes.

        Returns
        -------
        self : PredictionLedger
        """
        check_is_fitted(self, "version_")
        ids, labels = self._check_label_batch(record_ids, labels)
        already_labeled = [record_id for record_id in ids if record_id in self._labels]
        if already_labeled:
            raise ValueError(f"Records {already_labeled} already have a label; use `correct_labels` to change them.")
        for record_id, label in zip(ids, labels):
            self._labels[record_id] = label
        return self

    def correct_labels(self, record_ids, labels):
        """Replace the labels of records that already have one (label corrections).

        The batch is validated before anything is stored: if any id has no label yet or any label
        is not one of the two fixed classes, an error is raised and no partial results are left
        behind.

        Parameters
        ----------
        record_ids : array-like of shape (n_records,)
            Hashable ids of the records to correct. Must already have a label.
        labels : array-like of shape (n_records,)
            The corrected labels, each one of the two fixed classes.

        Returns
        -------
        self : PredictionLedger
        """
        check_is_fitted(self, "version_")
        ids, labels = self._check_label_batch(record_ids, labels)
        missing = [record_id for record_id in ids if record_id not in self._labels]
        if missing:
            raise ValueError(f"Records {missing} have no label to correct; use `add_labels` instead.")
        for record_id, label in zip(ids, labels):
            self._labels[record_id] = label
        return self

    def withdraw_labels(self, record_ids):
        """Remove labels from records (label retractions).

        Withdrawn records become unlabeled again and stop counting towards the denominators in
        `evaluation`. The batch is validated before anything is removed: if any id has no label,
        an error is raised and no partial results are left behind.

        Parameters
        ----------
        record_ids : array-like of shape (n_records,)
            Hashable ids of the records to withdraw. Must currently have a label.

        Returns
        -------
        self : PredictionLedger
        """
        check_is_fitted(self, "version_")
        ids = self._check_record_ids(record_ids)
        missing = [record_id for record_id in ids if record_id not in self._labels]
        if missing:
            raise ValueError(f"Records {missing} have no label to withdraw.")
        for record_id in ids:
            del self._labels[record_id]
        return self

    def evaluation(self):
        """Evaluate the stored historical predictions against the current labels.

        Only rows whose record currently has a label enter the denominators of the confusion
        matrix, the accuracy and the Brier mean; unlabeled predictions are counted in
        `n_predictions` only. The overall view pools the actual stored records across versions
        (it is not an average of the per-version numbers). Nothing is recomputed with the current
        model: each row is judged with the prediction and probabilities stored at the time.

        Returns
        -------
        dict
            A dictionary with keys:

            - `"per_version"`: mapping of version number to a summary dictionary.
            - `"overall"`: one summary dictionary over all stored records.

            Each summary contains `n_predictions` (int), `n_labeled` (int), `confusion_matrix`
            (2x2 array over the fixed classes, true labels as rows), `accuracy` (float or None
            when nothing is labeled) and `brier_mean` (mean binary Brier score
            `(p_positive - y)^2` over labeled rows, float or None). The positive class is
            `classes_[1]`; its probability column is located through the true `classes_` of the
            version that produced each row.
        """
        check_is_fitted(self, "version_")
        per_version = {
            version: self._summarize([row for row in self._records if row["version"] == version])
            for version in range(1, self.version_ + 1)
        }
        return {"per_version": per_version, "overall": self._summarize(list(self._records))}

    def records(self):
        """Return a copy of the ledger rows, one dictionary per stored prediction.

        Each row contains `version`, `record_id`, `y_pred` (the true `predict` output at the
        time), `proba` (the stored probability row) and `classes` (the true `classes_` of the
        version that produced the row, needed to interpret the probability columns). Mutating the
        result does not affect the ledger.
        """
        check_is_fitted(self, "version_")
        return [
            {
                "version": row["version"],
                "record_id": row["record_id"],
                "y_pred": row["y_pred"],
                "proba": row["proba"].copy(),
                "classes": self._version_classes[row["version"]].copy(),
            }
            for row in self._records
        ]

    def labels(self):
        """Return a copy of the current labels, as a mapping of record id to label.

        Mutating the result does not affect the ledger.
        """
        check_is_fitted(self, "version_")
        return dict(self._labels)

    def _summarize(self, rows):
        """Summarize ledger rows against the current labels. Unlabeled rows enter no denominator."""
        labeled = [(self._labels[row["record_id"]], row) for row in rows if row["record_id"] in self._labels]
        summary = {
            "n_predictions": len(rows),
            "n_labeled": len(labeled),
            "confusion_matrix": np.zeros((2, 2), dtype=int),
            "accuracy": None,
            "brier_mean": None,
        }
        if not labeled:
            return summary

        y_true = [label for label, _ in labeled]
        y_pred = [row["y_pred"] for _, row in labeled]
        positive = self.classes_[1]
        brier_terms = []
        for label, row in labeled:
            version_classes = self._version_classes[row["version"]]
            positive_column = int(np.flatnonzero(version_classes == positive)[0])
            brier_terms.append((float(row["proba"][positive_column]) - float(label == positive)) ** 2)

        summary["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=self.classes_)
        summary["accuracy"] = float(accuracy_score(y_true, y_pred))
        summary["brier_mean"] = float(np.mean(brier_terms))
        return summary

    def _check_label_batch(self, record_ids, labels):
        """Validate a (record_ids, labels) batch before anything is stored."""
        ids = self._check_record_ids(record_ids)
        if isinstance(labels, (str, bytes)):
            raise ValueError("labels must be an iterable of class labels, not a single string.")
        try:
            labels = list(labels)
        except TypeError as exc:
            raise ValueError("labels must be an iterable of class labels.") from exc
        if len(labels) != len(ids):
            raise ValueError(f"Got {len(labels)} labels for {len(ids)} record_ids.")
        allowed = set(self.classes_.tolist())
        try:
            invalid = [label for label in labels if label not in allowed]
        except TypeError as exc:
            raise ValueError("labels must be hashable scalars.") from exc
        if invalid:
            raise ValueError(f"Labels {invalid} are not among the fixed classes {self.classes_.tolist()}.")
        return ids, labels

    @staticmethod
    def _check_record_ids(record_ids, n_expected=None):
        """Validate record ids before anything is stored: iterable, hashable, unique, right length."""
        if isinstance(record_ids, (str, bytes)):
            raise ValueError("record_ids must be an iterable of hashable ids, not a single string.")
        try:
            ids = list(record_ids)
        except TypeError as exc:
            raise ValueError("record_ids must be an iterable of hashable ids.") from exc
        if n_expected is not None and len(ids) != n_expected:
            raise ValueError(f"Got {len(ids)} record_ids for {n_expected} samples.")
        try:
            n_unique = len(set(ids))
        except TypeError as exc:
            raise ValueError("record_ids must be hashable.") from exc
        if n_unique != len(ids):
            raise ValueError("record_ids contain duplicates.")
        return ids

    def _more_tags(self):
        return {
            "binary_only": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
