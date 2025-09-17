import cupy as cp
from typing import Tuple

epsilon = cp.float32(1e-8)

def accuracy(y_true: cp.ndarray, y_pred: cp.ndarray) -> cp.float32:
    return cp.sum(y_true == y_pred) / len(y_pred)

def precision(TP: cp.float32, FP: cp.float32) -> cp.float32:
    return TP/(TP + FP + epsilon)

def recall(TP: cp.float32, FN: cp.float32) -> cp.float32:
    return TP/(TP + FN + epsilon)

def f1(y_true: cp.ndarray, y_pred: cp.ndarray, average: str = "macro") -> cp.float32:
    classes = cp.unique(y_true)
    f1_scores = []

    for c in classes:
        TP = cp.sum((y_pred == c) & (y_true == c))
        FP = cp.sum((y_pred == c) & (y_true != c))
        FN = cp.sum((y_pred != c) & (y_true == c))

        prec = precision(TP, FP)
        rec = recall(TP, FN)
        f1 = 2 * prec * rec / (prec + rec + epsilon)
        f1_scores.append(f1)

    f1_scores = cp.array(f1_scores)

    if average == "macro":
        return f1_scores.mean()
    elif average == "micro":
        TP = cp.sum(y_pred == y_true)
        FP = cp.sum((y_pred != y_true) & (cp.isin(y_pred, classes)))
        FN = cp.sum((y_pred != y_true) & (cp.isin(y_true, classes)))
        return 2 * TP / (2 * TP + FP + FN + epsilon)
    else:
        return f1_scores
    
def roc(y_true: cp.ndarray, y_prob: cp.ndarray, thresholds: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    y_pred = y_prob >= thresholds

    TP = cp.sum(y_pred * y_true[:, :], axis=0)
    FP = cp.sum(y_pred * (1 - y_true[:, :]), axis=0)
    FN = cp.sum((1 - y_pred) * y_true[:, :], axis=0)
    TN = cp.sum((1 - y_pred) * (1 - y_true[:, :]), axis=0)

    TPR = TP / (TP + FN + epsilon)
    FPR = FP / (FP + TN + epsilon)

    return FPR, TPR


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, f1_score
    import matplotlib.pyplot as plt

    cp.random.seed(42)
    n_samples, n_classes = 200, 5

    y_true = cp.random.randint(0, n_classes, size=n_samples)
    y_prob = cp.random.rand(n_samples, n_classes)

    mask = cp.random.rand(n_samples) < 0.7
    y_pred = y_true.copy()
    y_pred[~mask] = cp.random.randint(0, n_classes, size=int((~mask).sum()))


    print(accuracy(y_true, y_pred), accuracy_score(y_true.get(), y_pred.get()))
    print(f1((y_true >= 0.5).astype(int), (y_pred >= 0.5).astype(int)), f1_score((y_true >= 0.5).astype(int).get(), (y_pred >= 0.5).astype(int).get(), average='macro'))

    fpr, tpr = roc(y_true, y_prob, cp.linspace(0, 1, 100))
    for i in range(y_prob.shape[1]):
        plt.plot(fpr[i].get(), tpr[i].get())

    plt.show()