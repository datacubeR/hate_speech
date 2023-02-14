import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score

from .datamodule import HateDataModule
from .model import RoberTuito
from .utils import f1_custom


def train_model(fold: list, dm_config, model_config, training_config):

    dm = HateDataModule(**dm_config, fold=fold)
    model = RoberTuito(config=model_config)
    mc = ModelCheckpoint(
        dirpath="checkpoints",
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    mc.CHECKPOINT_NAME_LAST = f"best-checkpoint-latest-{fold}"
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=training_config.patience,
    )

    trainer = pl.Trainer(
        deterministic=True,
        callbacks=[mc, early_stop_callback],
        max_epochs=training_config.max_epochs,
        accelerator="gpu",
        devices=1,
        fast_dev_run=training_config.fast_dev_run,
        overfit_batches=training_config.overfit_batches,
    )

    trainer.fit(model, dm)

    return trainer, model, dm


def predict(trainer, model, dm, validation=True):
    if validation:
        preds = trainer.predict(model, dm.val_dataloader())
    else:
        preds = trainer.predict(model, dm)

    results = []
    for pred in preds:
        results.extend(torch.sigmoid(pred).detach().cpu().numpy())

    return np.array(results)


def evaluate(trainer, model, dm, threshold=0.5, custom=True):
    y = dm.val_dataset.labels
    preds = np.where(predict(trainer, model, dm, validation=True) >= threshold, 1, 0)

    if custom:
        return f1_custom(y, preds)
    else:
        return f1_score(y, preds)