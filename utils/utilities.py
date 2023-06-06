import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import f1_score


def import_data():
    df_train = pd.read_csv("./public_data/tweets_train.csv", index_col=0)
    # Datos de Test Etiquetados
    df_test = pd.read_csv("./public_data/tweets_test.csv", index_col=0)
    stopwords = np.loadtxt("public_data/stopwords.txt", dtype=str)
    logger.info("Datos de Entrenamiento y Test cargados correctamente...")
    logger.info(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test, stopwords


def f1_custom(y, y_pred):
    y_odio = y[:, 0]
    y_pred_odio = y_pred[:, 0]

    y_comunidades = y[:, 1:]
    y_pred_comunidades = y_pred[:, 1:]

    f1_odio = f1_score(y_odio, y_pred_odio)
    f1_comunidades = f1_score(y_comunidades, y_pred_comunidades, average="macro")

    return 0.5 * f1_odio + 0.5 * f1_comunidades
