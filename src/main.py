#! /usr/bin/env python3

from datasets import load_dataset
from os import makedirs

def cargar_datasets():
    train_ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", split="train")
    valid_ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", split="validation")
    test_ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", split="test")

    return train_ds, valid_ds, test_ds

def exportar_dataset(dataset, type):
    makedirs("files", exist_ok=True)

    with open("files/" + type + "_ds.txt", "w", encoding="utf-8") as file:
        for item in dataset:
            text = str(item.get("text", item))
            label = str(item.get("label", item))
            file.write(text + "\t" + label + "\n")

train_ds, valid_ds, test_ds = cargar_datasets()


exportar_dataset(train_ds, "train")
exportar_dataset(valid_ds, "valid")
exportar_dataset(test_ds.shuffle(seed=42).select(range(300)), "test") # type: ignore
