import part1_nn_lib as lib
import numpy as np

def test_linear():
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
    x_val_pre = prep_input.apply(x_val)

    layer = lib.LinearLayer(4, 3)
    print(layer._W)
    print(layer._b)
    print(layer._cache_current)
    print(layer._grad_W_current.shape == layer._W.shape)
    print(layer._grad_b_current.shape == layer._b.shape)

    d_in = x_train_pre[:4, :]
    print(d_in.shape)
    out = layer(d_in)
    print(out)
    print(np.isclose(layer._cache_current.transpose(), d_in))


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
    print(x_train_pre[x_train_pre<0])
    print(x_train_pre[x_train_pre>1])

    x_train_rev = prep_input.revert(x_train_pre)
    diff = np.isclose(x_train_rev, x_train)
    print(diff[diff == False])


def test():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    r = np.array([3, 2, 1])
    print(m + r)
    print(r)
    print(m.shape)
    print(m.shape[0])

if __name__ == "__main__":
    test_linear()