import cupy as cp


from model.model import NNet, cross_entropy
from layers.dense import Dense
from layers.activations import Sigmoid
from utils.data_loader import load_dataset
from utils.metrics import accuracy, f1, roc, precision, recall


if __name__ == '__main__':
    X_train, y_train, x_test, y_test = load_dataset("Mushrooms", 0.7)
    cp.random.seed(42)
    dense1 = Dense(X_train.shape[1], 50)
    sig1 = Sigmoid()
    dense2 = Dense(50, y_train.shape[1])
    model = NNet([dense1, sig1, dense2], loss_fn=cross_entropy)
    print("train")
    loss = model.fit(X_train, y_train, num_epochs=200, lr=0.01, l2=0.1)
    
    pred = model.forward(x_test)
    
    print("accuracy: ", accuracy(y_test, (pred >= 0.5).astype(int)))
    print("f1: ", f1(y_test, (pred >= 0.5).astype(int)))
    y_pred = (pred >= 0.5).astype(int)
    TP = cp.sum((y_pred == 1) & (y_test == 1))
    FP = cp.sum((y_pred == 1) & (y_test != 1))
    FN = cp.sum((y_pred != 1) & (y_test == 1))
    
    print("precission: ", cp.sum(precision(TP, FP)))
    print("recall: ", cp.sum(recall(TP, FN)))
    
    import matplotlib.pyplot as plt
    plt.plot(range(len(loss)), loss.get())
    plt.show()

    fpr, tpr = roc(y_test, pred, cp.linspace(0, 1, 100))
    plt.plot(fpr.get(), tpr.get())
    idx = cp.argmin(cp.abs(cp.linspace(0, 1, 100) - 0.5)) 
    plt.plot([0,1],[0,1], 'k--') 
    plt.grid(True)
    plt.show()