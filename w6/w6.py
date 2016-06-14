def homework(train_X, test_X, train_y):
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.datasets import fetch_mldata
    from theano.tensor.shared_randomstreams import RandomStreams
    from collections import OrderedDict

    import matplotlib.pyplot as plt
    import numpy as np
    import theano
    import theano.tensor as T

    train_y = np.eye(10)[train_y].astype('int32')
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

    def sgd(params, g_params, eps=np.float32(0.1)):
        updates = OrderedDict()
        for param, g_param in zip(params, g_params):
            updates[param] = param - eps*g_param
        return updates

    class Autoencoder:
        #- Constructor
        def __init__(self, visible_dim, hidden_dim, W, function):
            self.visible_dim = visible_dim
            self.hidden_dim  = hidden_dim
            self.function    = function
            self.W           = W
            self.a = theano.shared(np.zeros(visible_dim).astype('float32'), name='a')
            self.b = theano.shared(np.zeros(hidden_dim).astype('float32'), name='b')
            self.params = [self.W, self.a, self.b]

        #- Encoder
        def encode(self, x):
            u = T.dot(x, self.W) + self.b
            y = self.function(u)
            return y

        #- Decoder
        def decode(self, x):
            u = T.dot(x, self.W.T) + self.a
            y = self.function(u)
            return y

        #- Forward Propagation
        def f_prop(self, x):
            y = self.encode(x)
            reconst_x = self.decode(y)
            return reconst_x

        #- Reconstruction Error
        def reconst_error(self, x, noise):
            tilde_x = x * noise
            reconst_x = self.f_prop(tilde_x)
            error = T.mean(T.sum((reconst_x - x)**2, axis = 1))
            return error, reconst_x

    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function
            self.W = theano.shared(np.random.normal(loc=0.0, scale=0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
            self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
            self.params = [self.W, self.b]

            self.set_pretraining()

        #- Forward Propagation
        def f_prop(self, x):
            self.u = T.dot(x, self.W) + self.b
            self.z = self.function(self.u)
            return self.z

        #- Set Pretraining
        def set_pretraining(self):
            ae = Autoencoder(self.in_dim, self.out_dim, self.W, self.function)

            x = T.fmatrix(name='x')
            noise = T.fmatrix(name='noise')

            cost, reconst_x = ae.reconst_error(x, noise)
            params = ae.params
            g_params = T.grad(cost=cost, wrt=params)
            updates = sgd(params, g_params)

            self.pretraining = theano.function(inputs=[x, noise], outputs=[cost, reconst_x], updates=updates, allow_input_downcast=True, name='pretraining')

            hidden = ae.encode(x)
            self.encode_function = theano.function(inputs=[x], outputs=hidden, allow_input_downcast=True, name='encode_function')

    layers = [
        Layer(784, 600, T.nnet.relu),
        Layer(600, 10, T.nnet.softmax)
    ]

    X = train_X
    for l, layer in enumerate(layers[:-1]): # Pre-train the network except with the last layer.
        corruption_level = np.float32(0.3)
        batch_size = 10
        n_batches = X.shape[0]

        for epoch in xrange(10):
            X = shuffle(X)
            err_all = []
            for i in xrange(0, n_batches):
                start = i*batch_size
                end = start + batch_size

                noise = np.random.binomial(size=X[start:end].shape, n=1, p=1-corruption_level)
                err, reconst_x = layer.pretraining(X[start:end], noise)
                err_all.append(err)
            print 'Pretraining:: layer:%d, Epoch:%d, Error:%lf' % (l, epoch, np.mean(err))
        X = layer.encode_function(X)

    x = T.fmatrix(name='x')
    t = T.imatrix(name='t')

    params = []
    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.f_prop(x)
        else:
            layer_out = layer.f_prop(layer_out)

    y = layers[-1].z
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))

    g_params = T.grad(cost=cost, wrt=params)
    updates = sgd(params, g_params)

    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function([x], T.argmax(y, axis=1), name='test')

    batch_size = 20
    n_batches = train_X.shape[0]

    for epoch in xrange(30):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            train(train_X[start:end], train_y[start:end])
        if epoch % 10 == 0:
            valid_cost, pred_y = valid(valid_X, valid_y)
            print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), pred_y, average='macro'))
    pred_y = test(test_X)
    return pred_y
