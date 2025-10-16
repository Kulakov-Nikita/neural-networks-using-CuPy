from typing import List, Tuple

import cupy as cp
import numpy as np
import pandas as pd


def load_mushrooms() -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    clean_data = pd.read_csv('data/Mushrooms.csv') 
    train_df = clean_data.sample(frac=0.7, random_state=42)
    test_df = clean_data.drop(train_df.index)

    train_x = cp.array(train_df.drop('Poisonous', axis=1).to_numpy(dtype=float))
    train_y = cp.array(train_df['Poisonous'].to_numpy(dtype=float))
    test_x = cp.array(test_df.drop('Poisonous', axis=1).to_numpy(dtype=float))
    test_y = cp.array(test_df['Poisonous'].to_numpy(dtype=float))

    train_y = train_y.reshape(-1,1).astype(int)
    test_y = test_y.reshape(-1,1).astype(int)

    return train_x,train_y,test_x,test_y

def load_steel_industry_sequences(
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    train_ratio: float = 0.8,
    data_path: str = "data/Steel_industry_data.csv"
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, List[str]]:

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)

    features = df.drop(columns=["Load_Type"])
    features = features.drop(columns=["date"])
    features = pd.get_dummies(
        features,
        columns=["WeekStatus", "Day_of_week"],
        drop_first=False,
        dtype=np.float32
    )

    features = features.astype(np.float32)
    feature_mean = features.mean()
    feature_std = features.std().replace(0, 1)
    features = (features - feature_mean) / feature_std

    target = df["Load_Type"].astype("category")
    label_names = list(target.cat.categories)
    labels = target.cat.codes.to_numpy(dtype=np.int32)

    feature_values = features.to_numpy()
    sequences = []
    seq_labels = []
    total_steps = len(df) - sequence_length - forecast_horizon + 1
    if total_steps <= 0:
        raise ValueError("sequence_length and forecast_horizon are incompatible with dataset length.")
    for start in range(total_steps):
        end = start + sequence_length
        target_idx = end + forecast_horizon - 1
        sequences.append(feature_values[start:end])
        seq_labels.append(labels[target_idx])

    X_np = np.stack(sequences).astype(np.float32)
    y_idx_np = np.array(seq_labels, dtype=np.int32)

    num_classes = len(label_names)
    split_idx = int(len(X_np) * train_ratio)

    X_train = cp.asarray(X_np[:split_idx])
    X_test = cp.asarray(X_np[split_idx:])

    y_idx = cp.asarray(y_idx_np)
    eye = cp.eye(num_classes, dtype=cp.float32)
    y_one_hot = eye[y_idx]

    y_train = y_one_hot[:split_idx]
    y_test = y_one_hot[split_idx:]

    train_labels = y_idx[:split_idx]
    test_labels = y_idx[split_idx:]

    return X_train, y_train, train_labels, X_test, y_test, test_labels, label_names



def load_dataset(dataset_name: str, train_percent: float, load_percent: float = 1) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    match dataset_name:
        case "Mushrooms":
            train_X, train_y, test_X, test_y = load_mushrooms()
        case "MNIST":
            from keras.datasets import mnist
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
            train_y, test_y = one_hot(train_y), one_hot(test_y)

    X, y = cp.concatenate([cp.asarray(train_X), cp.asarray(test_X)]), cp.concatenate([train_y, test_y])
    X, y = X[:round(len(X)*load_percent)], y[:round(len(X)*load_percent)]
    return X[:round(len(X)*train_percent)], y[:round(len(y)*train_percent)], X[round(len(X)*train_percent):], y[round(len(y)*train_percent):]
