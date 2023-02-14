import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def import_data(train_path, test_path):
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    return train, test


def create_folds(df, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    df["kfold"] = -1

    for fold, (_, val_idx) in enumerate(kf.split(df.drop(columns=labels))):
        df.loc[val_idx, "kfold"] = fold

    return df


def f1_custom(y, y_pred):
    y_odio = y[:, 0]
    y_pred_odio = y_pred[:, 0]

    y_comunidades = y[:, 1:]
    y_pred_comunidades = y_pred[:, 1:]

    f1_odio = f1_score(y_odio, y_pred_odio)
    f1_comunidades = f1_score(y_comunidades, y_pred_comunidades, average="macro")

    return 0.5 * f1_odio + 0.5 * f1_comunidades
