import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import f1_score


def import_data():
    df_train = pd.read_csv("./public_data/tweets_train.csv", index_col=0)
    # Labelled Test Data
    df_test = pd.read_csv("./public_data/tweets_test.csv", index_col=0)
    stopwords = np.loadtxt("public_data/stopwords.txt", dtype=str)
    logger.info("Training and Test data succesfully loaded...")
    logger.info(f"Train Shape: {df_train.shape}, Test Shape: {df_test.shape}")
    return df_train, df_test, stopwords


def f1_custom(y, y_pred):
    y_hate = y[:, 0]
    y_pred_hate = y_pred[:, 0]

    y_communities = y[:, 1:]
    y_pred_communities = y_pred[:, 1:]

    f1_hate = f1_score(y_hate, y_pred_hate)
    f1_communities = f1_score(y_communities, y_pred_communities, average="macro")

    return 0.5 * f1_hate + 0.5 * f1_communities
