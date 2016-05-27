def homework(train_X, test_X, train_y):
    import numpy as np
    from sklearn import cross_validation

    # Convert train_y to one hot expression
    def convert_to_one_hot(train_y):
        tmp = np.zeros((train_y.size, 10))
        row_index = np.arange(train_y.size)
        tmp[row_index, train_y] = 1.0
        return tmp

    train_y = convert_to_one_hot(train_y)

    # For the last layer
    def softmax(x):
        maxi = np.max(x)
        exp_x = np.exp(x - maxi + 100)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def deriv_softmax(x):
        return softmax(x) * (1 - softmax(x))

    # For hidden layer
    def tanh(x):
        return np.tanh(x)

    def deriv_tanh(x):
        return 1 - np.tanh(x) ** 2

    class Layer:
        def __init__(self, in_dim, out_dim, function, deriv_function):
            self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype("float32")
            self.b = np.zeros(out_dim).astype("float32")
            self.function = function
            self.deriv_function = deriv_function
            self.u = None
            self.delta = None

        # Forward Propagation
        def f_prop(self, x):
            self.u = np.dot(x, self.W) + self.b
            self.z = self.function(self.u)
            return self.z

        # Back Propagation
        def b_prop(self, delta, W):
            self.delta = np.dot(delta, W.T) * self.deriv_function(self.u)
            return self.delta

    # For overall network
    def f_props(layers, x):
        z = x
        for layer in layers:
            z = layer.f_prop(z)
        return z

    def b_props(layers, delta):
        for i, layer in enumerate(layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.b_prop(delta, _W)
            _W = layer.W

    layers = [Layer(784, 600, tanh, deriv_tanh), Layer(600, 10, softmax, deriv_softmax)]

    def train(X, t, eps=0.01):
        # Forward Propagation
        y = f_props(layers, X)

        # Delta
        delta = y - t

        # Back Propagation
        b_props(layers, delta)

        # Update Parameters
        z = X
        for i, layer in enumerate(layers):
            dW = np.dot(z.T, layer.delta) / 10.0
            db = np.dot(np.ones(len(z)), layer.delta) / 10.0
            layer.W = layer.W - eps * dW
            layer.b = layer.b - eps * db
            z = layer.z

        # Train Cost
        y = f_props(layers, X)

        cost = np.sum(-np.log(y[np.where(t==1.0)]))
        return cost

    #def test(X, t):
    #    #- Test Cost
    #    y = f_props(layers, X)
    #    cost = np.sum(-np.log(y[np.where(t==1.0)]))
    #    return cost, y

    def one_hot_to_class(pred_y_in_one_hot):
        return np.argmax(pred_y_in_one_hot, axis=1)

    for epoch in xrange(10):
        # Mini batch.
        for i in range(0, len(train_X), 10):
            cost = train(train_X[i:i+10], train_y[i:i+10])

    return one_hot_to_class(f_props(layers,test_X))
