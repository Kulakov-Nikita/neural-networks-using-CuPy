import cupy as cp


from model.model import NNet, cross_entropy
from layers.conv import Conv2D, Flatten
from layers.dense import Dense
from layers.activations import ReLU, SoftMax
from utils.data_loader import load_dataset
from utils.metrics import accuracy, f1, roc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_dataset('MNIST', 0.7, load_percent=0.1)

    train_X = (train_X.astype(cp.float32) / 255.0).reshape((-1, 1, 28, 28))
    test_X = (test_X.astype(cp.float32) / 255.0).reshape((-1, 1, 28, 28))

    cp.random.seed(42)

    model = NNet([
        Conv2D(1, 32, 5),
        ReLU(),
        Conv2D(32, 16, 3),
        ReLU(),
        Flatten(),
        Dense(16 * 22 * 22, 10),
        SoftMax()
    ], loss_fn=cross_entropy)

    print('Training...')
    loss = model.fit(train_X, train_y, num_epochs=20, lr=0.01, l2=0.0, batch_size=64)

    y_prob = model.predict(test_X, batch_size=256)
    y_pred = cp.argmax(y_prob, axis=1)
    y_true = cp.argmax(test_y, axis=1)

    print('accuracy:', accuracy(y_true, y_pred))
    print('f1 (macro):', f1(y_true, y_pred, average='macro'))
    print(y_true.shape, y_pred.shape, y_prob.shape, y_true.shape)
    print("ROC AUC:", roc_auc_score(y_true.get(), y_prob.get(), multi_class='ovr'))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(loss)), loss.get())
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy')
    plt.grid(True)
    plt.show()

    y_true_np = y_true.get()
    y_prob_np = y_prob.get()
    classes = list(range(y_prob_np.shape[1]))
    y_true_binarized = label_binarize(y_true_np, classes=classes)

    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob_np[:, i])
        plt.plot(fpr, tpr, label=f"class {cls}")

    plt.plot([0, 1], [0, 1], 'k--', label='chance')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.show()
