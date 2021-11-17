import part1_nn_lib as lib
import numpy as np

def test_activation():
    print("=======Testing Activation Layer========")
    print("=======Testing Sigmoid========")
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

    print("Testing Constructor")
    layer = lib.SigmoidLayer()

    print("Testing Forward")
    d_in = x_train_pre[:4, :]
    print(d_in.shape)
    out = layer(d_in)
    print(out)
    print(out.shape)
    assert(np.isclose(layer._cache_current, d_in).all())
    assert(np.isclose(out, np.reciprocal(np.exp(-d_in) + 1)).all())

    print("Testing Backward")
    grad_z = np.array([[1, -3, 5, 2.6], [0.6, -2.7, 4.3, 2.04], [1.2, -3.2, 4.9, 2.4], [0.8, -2.9, 5.1, 3.1]]) # (4, 4)
    grad_x = layer.backward(grad_z)
    assert(grad_x.shape == grad_z.shape)
    print(grad_x)
    sig = np.reciprocal(np.exp(-layer._cache_current) + 1)
    assert(np.isclose(grad_x, grad_z * sig * (1 - sig)).all())




def test_multi_linear():
    print("=======Testing Multi-Linear Layer========")

def test_linear():
    print("=======Testing Linear Layer========")
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

    print("Testing Constructor")
    layer = lib.LinearLayer(4, 3)
    print(layer._W)
    print(layer._b)
    print(layer._cache_current)
    assert(layer._grad_W_current.shape == layer._W.shape)
    assert(layer._grad_b_current.shape == layer._b.shape)

    print("Testing Forward")
    d_in = x_train_pre[:4, :]
    print(d_in.shape)
    out = layer(d_in)
    print(out)
    assert(np.isclose(layer._cache_current.transpose(), d_in).all())
    assert(np.isclose(out, np.matmul(d_in, layer._W) + layer._b).all())

    print("Testing Backwards")
    grad_z = np.array([[1, -3, 5], [0.6, -2.7, 4.3], [1.2, -3.2, 4.9], [0.8, -2.9, 5.1]]) # (4, 3)
    print("Old grad values")
    print(layer._grad_W_current)
    print(layer._grad_b_current)
    grad_x = layer.backward(grad_z)
    assert(grad_x.shape == (4, 4)) # Should be (4, 4)
    print("New grad values")
    print(layer._grad_W_current)
    print(layer._grad_b_current)
    assert(np.isclose(layer._grad_W_current, np.matmul(layer._cache_current, grad_z)).all())
    assert(np.isclose(layer._grad_b_current, np.ravel(np.matmul(np.ones((1, grad_z.shape[0])), grad_z))).all())
    assert(np.isclose(grad_x, np.matmul(grad_z, layer._W.transpose())).all())

    print("Testing update_params")
    print("Old param values")
    print(layer._W)
    print(layer._b)
    layer.update_params(0.001)
    print(layer._W)
    print(layer._b)
    




def test_preprocessor():
    print("=======Testing Preprocessor========")
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