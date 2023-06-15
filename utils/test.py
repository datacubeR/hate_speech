from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score
from .utilities import f1_custom
import numpy as np


def full_train(df_train, df_test, model, labels, stopwords):
    pipe = Pipeline(
        steps=[
            (
                "cv",
                CountVectorizer(
                    stop_words=list(stopwords),
                    lowercase=True,
                    ngram_range=(1, 1),
                    dtype=np.float32,
                ),
            ),
            ("model", MultiOutputClassifier(model)),
        ]
    )

    pipe.fit(df_train["text"], df_train[labels].astype("bool").astype("int"))
    y_pred = pipe.predict(df_test["text"])

    ground_truth = df_test[labels].astype(bool).astype(int)
    test_metric = f1_custom(ground_truth.values, y_pred)
    test_precision = precision_score(ground_truth.values, y_pred, average="macro")
    test_recall = recall_score(ground_truth.values, y_pred, average="macro")
    return dict(
        test_score=test_metric, test_precision=test_precision, test_recall=test_recall
    )
