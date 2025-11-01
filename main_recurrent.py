import cupy as cp

from model.model import NNet, cross_entropy
from layers.recurrent.rnn import RNN
from layers.dense import Dense
from layers.activations import SoftMax
from utils.data_loader import load_steel_industry_sequences
from utils.metrics import accuracy, f1


def run_experiment(name, layer_cls, x_train, y_train, x_test, y_test, y_test_labels, hidden_size, batch_size):
    input_size = x_train.shape[2]
    num_classes = y_train.shape[1]

    seq_layer1 = layer_cls(input_size, hidden_size, return_sequences=True)
    seq_layer2 = layer_cls(hidden_size, hidden_size, return_sequences=False)
    classifier = Dense(hidden_size, num_classes)
    softmax = SoftMax()

    model = NNet([seq_layer1, seq_layer2, classifier, softmax], loss_fn=cross_entropy)
    print(f"\nTraining {name} model...")
    loss_curve = model.fit(
        x_train,
        y_train,
        num_epochs=15,
        lr=0.05,
        l2=1e-4,
        batch_size=batch_size,
        shuffle=True,
    )

    probs = model.predict(x_test, batch_size=batch_size)
    preds = cp.argmax(probs, axis=1)

    mse = cp.mean((probs - y_test) ** 2)
    rmse = cp.sqrt(mse)
    y_mean = cp.mean(y_test, axis=0, keepdims=True)
    ss_res = cp.sum((y_test - probs) ** 2)
    ss_tot = cp.sum((y_test - y_mean) ** 2)
    r2 = 1 - ss_res / (ss_tot + cp.float32(1e-8))

    acc = accuracy(y_test_labels, preds).item()
    f1_score = f1(y_test_labels, preds, average="macro").item()
    print(
        f"{name} accuracy: {acc:.4f}, macro F1: {f1_score:.4f}, "
        f"MSE: {mse.item():.4f}, RMSE: {rmse.item():.4f}, R2: {r2.item():.4f}"
    )

    return loss_curve


if __name__ == "__main__":
    SEQ_LEN = 96 
    cp.random.seed(42)

    (
        X_train,
        y_train,
        train_labels,
        X_test,
        y_test,
        test_labels,
        label_names,
    ) = load_steel_industry_sequences(sequence_length=SEQ_LEN, forecast_horizon=1)

    print(f"train_size: {len(X_train)}, test_size: {len(X_test)}")


    hidden_size = 128
    batch_size = 2048

    loss_curve_rnn = run_experiment("RNN", RNN, X_train, y_train, X_test, y_test, test_labels, hidden_size, batch_size)

    import matplotlib.pyplot as plt

    plt.plot(range(len(loss_curve_rnn)), loss_curve_rnn.get(), label="RNN")
    plt.title("Ошибка на тесте")
    plt.xlabel("Loss")
    plt.ylabel("Epoch")
    plt.legend()
    plt.show()
