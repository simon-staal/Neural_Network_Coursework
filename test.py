import part1_nn_lib as lib
import numpy as np

def test_preprocessor():
    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = lib.Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    print(x_train_pre.mean(axis=0))
    print(x_train_pre.std(axis=0))

    x_train_rev = prep_input.revert(x_train_pre)
    print(x_train_rev.mean(axis=0) - x_train.mean(axis=0))
    print(x_train_rev.std(axis=0) - x_train.std(axis=0))


if __name__ == "__main__":
    test_preprocessor()