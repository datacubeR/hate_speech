# |%%--%%| <k16XHegruH|QLNAyCZNqy>
# Necesario para el desarrollo fluido
%load_ext autoreload
%autoreload 2

#|%%--%%| <QLNAyCZNqy|0jm6RFXlPD>
r"""°°°
# Definición del Modelo
°°°"""
# |%%--%%| <0jm6RFXlPD|gpasnLNd8N>
%%time
from utils import import_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from utils import validation_train, full_train
from loguru import logger
import pandas as pd
import numpy as np


RANDOM_STATE = 42
df_train, df_test, stopwords = import_data()

LABELS = [
    "Odio",
    "Mujeres",
    "Comunidad LGBTQ+",
    "Comunidades Migrantes",
    "Pueblos Originarios",
]
 
#|%%--%%| <gpasnLNd8N|8MY7MMhyK7>
r"""°°°
# Esquema de Validación
°°°"""
#|%%--%%| <8MY7MMhyK7|1fsUEwTqc6>

nb = MultinomialNB()
cb = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, n_estimators=500, thread_count=-1)
et = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)

models_dict = { "Naive Bayes": nb,
                "CatBoost": cb,
                "Extra Trees": et,
                "Random Forest": rf,
                "XGBoost": xgb
               }

validation_results = []
for name, model in models_dict.items():
    train_score, test_score = validation_train(df_train, model, LABELS, stopwords, random_state=RANDOM_STATE)
    mean_train_score = np.mean(train_score)
    mean_test_score = np.mean(test_score)       
    logger.info(f"Mean Training Score: {mean_train_score}")
    logger.info(f"Mean Validation Score: {mean_test_score}")
    dict_results = {}
    dict_results["model_name"] = name
    dict_results["mean_train_score"] = mean_train_score
    dict_results["mean_test_score"] = mean_test_score
    validation_results.append(dict_results)

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv("resultados_cv.csv", index=False)
print(validation_df)
#|%%--%%| <1fsUEwTqc6|IBl3xW3fUG>
r"""°°°
# Full Train
°°°"""
#|%%--%%| <IBl3xW3fUG|vzA1I2eo4i>

training_results = []
for name, model in models_dict.items():
    test_score = full_train(df_train, df_test, model, LABELS, stopwords)
    dict_results = {}
    dict_results["model_name"] = name
    dict_results["test_score"] = test_score
    training_results.append(dict_results)

training_df = pd.DataFrame(training_results)
training_df.to_csv("resultados_training.csv", index=False)
print(training_df)
