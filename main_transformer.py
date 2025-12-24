import os
import re
from collections import Counter

import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from layers.activations import SoftMax
from layers.dense import Dense
from layers.transformer import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerEncoderBlock,
    MeanPooling,
)
from model.model import NNet, cross_entropy
from utils.metrics import accuracy, f1


PAD_IDX = 0
UNK_IDX = 1


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", str(text).lower())


def build_vocab(texts: list[str], max_vocab: int, min_freq: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    tokens.sort(key=lambda t: counter[t], reverse=True)
    if max_vocab:
        tokens = tokens[: max_vocab - 2]

    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for idx, tok in enumerate(tokens, start=2):
        vocab[tok] = idx
    return vocab


def encode_text(text: str, vocab: dict[str, int], seq_len: int) -> list[int]:
    ids = [vocab.get(token, UNK_IDX) for token in tokenize(text)[:seq_len]]
    if len(ids) < seq_len:
        ids += [PAD_IDX] * (seq_len - len(ids))
    return ids


def pick_column(df: pd.DataFrame, options: list[str], kind: str) -> str:
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {kind} column. Available columns: {df.columns.tolist()}")


def load_toxic_comments(
    data_path: str,
    train_ratio: float,
    seq_len: int,
    max_vocab: int,
    min_freq: int = 2,
    max_samples: int | None = None,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found. Download it from Kaggle and place it at this path.")

    df = pd.read_csv(data_path)
    text_col = pick_column(df, ["comment", "comment_text", "text", "content", "message"], "text")
    label_col = pick_column(df, ["toxic", "target", "label", "is_toxic", "toxic_binary"], "label")

    df = df[[text_col, label_col]].dropna()
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    texts = df[text_col].astype(str).tolist()
    labels_np = df[label_col].astype("category").cat.codes.to_numpy(dtype=np.int32)

    vocab = build_vocab(texts, max_vocab=max_vocab, min_freq=min_freq)
    encoded = np.array([encode_text(text, vocab, seq_len) for text in texts], dtype=np.int32)

    num_classes = int(labels_np.max()) + 1
    labels_cp = cp.asarray(labels_np, dtype=cp.int32)
    one_hot = cp.eye(num_classes, dtype=cp.float32)[labels_cp]

    perm = cp.random.permutation(len(encoded))
    split_idx = int(len(encoded) * train_ratio)

    encoded_cp = cp.asarray(encoded, dtype=cp.int32)
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    x_train = encoded_cp[train_idx]
    y_train = one_hot[train_idx]
    x_test = encoded_cp[test_idx]
    y_test = one_hot[test_idx]
    test_labels = labels_cp[test_idx]

    return x_train, y_train, x_test, y_test, test_labels, vocab, num_classes


if __name__ == "__main__":
    DATA_PATH = "data/russian_comments_from_2ch_pikabu.csv"
    TRAIN_RATIO = 0.8
    MAX_VOCAB = 5_000
    SEQ_LEN = 64
    D_MODEL = 64
    NUM_HEADS = 4
    FF_HIDDEN = 128
    EPOCHS = 200
    BATCH_SIZE = 256
    LR = 0.05
    L2 = 1e-4
    MAX_SAMPLES = 40_000

    cp.random.seed(42)
    np.random.seed(42)

    print("Loading toxic comments dataset...")
    (
        X_train,
        y_train,
        X_test,
        y_test,
        test_labels,
        vocab,
        num_classes,
    ) = load_toxic_comments(
        DATA_PATH,
        train_ratio=TRAIN_RATIO,
        seq_len=SEQ_LEN,
        max_vocab=MAX_VOCAB,
        min_freq=2,
        max_samples=MAX_SAMPLES,
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Vocab size: {len(vocab)}")

    model = NNet(
        [
            TokenEmbedding(len(vocab), D_MODEL),
            PositionalEncoding(SEQ_LEN, D_MODEL),
            TransformerEncoderBlock(D_MODEL, NUM_HEADS, FF_HIDDEN),
            TransformerEncoderBlock(D_MODEL, NUM_HEADS, FF_HIDDEN),
            TransformerEncoderBlock(D_MODEL, NUM_HEADS, FF_HIDDEN),
            TransformerEncoderBlock(D_MODEL, NUM_HEADS, FF_HIDDEN),
            MeanPooling(),
            Dense(D_MODEL, num_classes),
            SoftMax(),
        ],
        loss_fn=cross_entropy,
    )

    print("Training transformer classifier...")
    loss_curve = model.fit(
        X_train,
        y_train,
        num_epochs=EPOCHS,
        lr=LR,
        l2=L2,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    probs = model.predict(X_test, batch_size=BATCH_SIZE)
    preds = cp.argmax(probs, axis=1)
    true_labels = cp.argmax(y_test, axis=1)

    print(f"Accuracy: {accuracy(true_labels, preds).item():.4f}")
    print(f"F1 (macro): {f1(true_labels, preds, average='macro').item():.4f}")

    plt.figure()
    plt.plot(range(len(loss_curve)), loss_curve.get())
    plt.title("Transformer test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.grid(True)
    plt.show()
