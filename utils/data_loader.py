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
