from .utilities import f1_custom

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np


def validation_train(df, model, target_labels, stopwords, random_state, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score = []
    train_score = []
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
        sc = f1_custom(y_val.values, y_pred)

        # Reporting Results during Training

        print(f"Train Score fold {fold}: {sc_train}")
        print(f"Validation Score fold {fold}: {sc}")
        print("--------------------------------------------")

        train_score.append(sc_train)
        score.append(sc)

    return train_score, score
