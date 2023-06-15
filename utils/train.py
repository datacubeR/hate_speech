from .utilities import f1_custom

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np


def validation_train(
    df, model, target_labels, stopwords, random_state, n_splits=5, verbose=False
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_score = []
    train_score = []
    precision_train_score = []
    precision_val_score = []
    recall_train_score = []
    recall_val_score = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df[target_labels]), start=1):
        X_train = df.iloc[train_idx]["text"]
        X_val = df.iloc[val_idx]["text"]

        # Labels are transformed to 0 and 1 if Label is greater than 0.

        y_train = df.iloc[train_idx][target_labels].astype(bool).astype(int)
        y_val = df.iloc[val_idx][target_labels].astype(bool).astype(int)

        # Pipeline Definition
        pipe = Pipeline(
            [
                (
                    "featurizer",
                    CountVectorizer(
                        stop_words=list(stopwords),
                        lowercase=True,
                        ngram_range=(1, 1),
                        dtype=np.float32,
                    ),
                ),
                ("clf", MultiOutputClassifier(model)),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred = pipe.predict(X_val)
        sc_train = f1_custom(y_train.values, y_pred_train)
        sc_val = f1_custom(y_val.values, y_pred)
        pr_train = precision_score(y_train.values, y_pred_train, average="macro")
        pr_val = precision_score(y_val.values, y_pred, average="macro")
        rc_train = recall_score(y_train.values, y_pred_train, average="macro")
        rc_val = recall_score(y_val.values, y_pred, average="macro")

        # Reporting Results during Training
        if verbose:
            print(f"Train Score fold {fold}: {sc_train}")
            print(f"Validation Score fold {fold}: {sc_val}")
            print("--------------------------------------------")

        train_score.append(sc_train)
        val_score.append(sc_val)
        precision_train_score.append(pr_train)
        precision_val_score.append(pr_val)
        recall_train_score.append(rc_train)
        recall_val_score.append(rc_val)

    return dict(
        mean_train_score=np.mean(train_score),
        sd_train_score = np.std(train_score),
        mean_val_score=np.mean(val_score),
        sd_val_score = np.std(val_score),
        mean_precision_train_score=np.mean(precision_train_score),
        sd_precision_train_score = np.std(precision_train_score),
        mean_precision_val_score=np.mean(precision_val_score),
        sd_precision_val_score = np.std(precision_val_score),
        mean_recall_train_score=np.mean(recall_train_score),
        sd_recall_train_score = np.std(recall_train_score),
        mean_recall_val_score=np.mean(recall_val_score),
        sd_recall_val_score = np.std(recall_val_score),
    )
