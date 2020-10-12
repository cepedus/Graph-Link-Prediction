import numpy as np


def load_data():
    with open("hps/X_train_ss.npz", "rb") as f:
        X_train = np.load(f)
    with open("hps/y_train_ss.npz", "rb") as f:
        y_train = np.load(f)
    with open("hps/X_valid.npz", "rb") as f:
        X_valid = np.load(f)
    with open("hps/y_valid.npz", "rb") as f:
        y_valid = np.load(f)

    print("shape X_train SS: ", np.shape(X_train))
    print("shape y_train SS: ", np.shape(y_train))
    print("shape X_valid: ", np.shape(X_valid))
    print("shape y_train: ", np.shape(y_valid))
    return (X_train, y_train), (X_valid, y_valid)


def save_subset():
    with open("hps/X_train.npz", "rb") as f:
        X_train = np.load(f)
    with open("hps/y_train.npz", "rb") as f:
        y_train = np.load(f)

    print("shape X_train OR.: ", np.shape(X_train))
    print("shape y_train OR.: ", np.shape(y_train))

    # SHUFFLE
    size = np.shape(X_train)[0]
    shuf = np.random.permutation(size)
    X_train = X_train[shuf]
    y_train = y_train[shuf]

    prop = 0.1
    sep_index = int(prop * np.shape(X_train)[0])

    X_train = X_train[:sep_index, :]
    y_train = y_train[:sep_index, :]

    with open("hps/X_train_ss.npz", "wb") as f:
        np.save(f, X_train)
    with open("hps/y_train_ss.npz", "wb") as f:
        np.save(f, y_train)


if __name__ == "__main__":
    save_subset()
    load_data()
