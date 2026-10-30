#! /usr/bin/env python3

from datasets import load_dataset

train_ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", split="train")
valid_ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", split="validation")
test_ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", split="test")

print(f"train_ds: {train_ds}")
print(f"validation_ds: {valid_ds}")
print(f"test_ds: {test_ds}")
