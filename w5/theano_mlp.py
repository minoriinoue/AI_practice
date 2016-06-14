def homework(train_X, test_X, train_y):
    import numpy as np
    from sklearn import cross_validation
    import theano
    import theano.tensor as T
    from collections import OrderedDict
    from sklearn.utils import shuffle

    train_X = train_X[:100]
    train_y = train_y[:100]
    test_X = text_X[:100]

    # Convert train_y to one hot expression
    def convert_to_one_hot(train_y):
        tmp = np.zeros((train_y.size, 10))
        row_index = np.arange(train_y.size)
        tmp[row_index, train_y] = 1.0
        return tmp

    train_y = convert_to_one_hot(train_y)

    #--- Multi Layer Perceptron
    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim   = in_dim
            self.out_dim  = out_dim
            self.function = function
            self.W        = theano.shared(rng.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float32'), name='W')
            self.b        = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
            self.params   = [self.W, self.b]

        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(x, self.W, self.b)
            return self.z

    #--- Stochastic Gradient Descent
    def sgd(params, g_params, eps=np.float32(0.1)):
        updates = OrderedDict()
        for param, g_param in zip(params, g_params):
            updates[param] = param - eps * g_param
        return updates

    z_relu = theano.function(inputs=[x, W, b],
                             outputs=T.nnet.relu(T.dot(x, W) + b),
                             allow_input_downcast=True,
                             name='z_relu')
    z_softmax = theano.function(inputs=[x, W, b],
                             outputs=T.nnet.softmax(T.dot(x, W) + b),
                             allow_input_downcast=True,
                             name='z_relu')
    layers = [
        Layer(784, 600, z_relu),
        Layer(600, 10, z_softmax)
    ]

    x = T.fmatrix('x')
    t = T.imatrix('t')

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
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')

    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    for epoch in xrange(5):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            train(train_X[start:end], train_y[start:end])
        valid_cost, pred_y = valid(valid_X, valid_y)
        print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), pred_y, average='macro'))
