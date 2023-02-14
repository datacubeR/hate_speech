import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel


class RoberTuito(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Linear(self.backbone.config.hidden_size, config.n_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(x.last_hidden_state, dim=1)  # average pooling

        x = self.dropout(pooled_output)
        x = self.classifier(x)

        return x
    
    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def predict_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.w_decay)
        return optimizer
    
    