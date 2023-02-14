import pandas as pd
import torch
import torch.nn as nn
from pysentimiento.preprocessing import preprocess_tweet
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HateDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        tokenizer: str,
        max_token_len: int,
        labels: list,
        text_field="text",
        validation=False,
        fold: list = None,
    ):

        if validation and fold is not None:
            data = dataset.query(f"kfold in {fold}")
            self.tweets = data[text_field].values
            self.labels = self.create_labels(data[labels].values)

        elif fold is not None:
            data = dataset.query(f"kfold not in {fold}")
            self.tweets = data[text_field].values
            self.labels = self.create_labels(data[labels].values)

        else:
            self.tweets = dataset[text_field].values
            self.labels = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_token_len = max_token_len

    @staticmethod
    def create_labels(df: pd.DataFrame):
        return df.astype("bool").astype("int64")

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):

        text = preprocess_tweet(self.tweets[idx])
        text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # adds BOS and EOS tokens
            max_length=self.max_token_len,
            return_tensors="pt",  # returns torch tensors
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

        return dict(
            input_ids=text["input_ids"].flatten(),
            attention_mask=text["attention_mask"].flatten(),
            labels=torch.tensor(self.labels[idx], dtype=torch.float32)
            if self.labels is not None
            else torch.tensor(0, dtype=torch.float32),
        )
