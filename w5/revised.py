def homework(train_X, test_X, train_y):
    import numpy as np
    from sklearn import cross_validation
    import theano
    from sklearn.metrics import f1_score
    import theano.tensor as T
    from collections import OrderedDict
    from sklearn.utils import shuffle
    import time

    # Convert train_y to one hot expression
    def convert_to_one_hot(train_y):
        tmp = np.zeros((train_y.size, 10))
        row_index = np.arange(train_y.size)
        tmp[row_index, train_y] = 1.0
        return tmp

    train_y = convert_to_one_hot(train_y)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

    #--- Multi Layer Perceptron
    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim   = in_dim
            self.out_dim  = out_dim
            self.function = function
            self.W        = theano.shared(np.random.normal(loc=0.0, scale=0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
            self.b        = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
            self.dW       = theano.shared(np.zeros((in_dim, out_dim)).astype('float32'), name='dW')
            self.db       = theano.shared(np.zeros(out_dim).astype('float32'), name='db')
            self.params   = [self.W, self.b]
            self.prev_deriv = [self.dW, self.db]

        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W) + self.b)
            return self.z

    eps = theano.shared(np.float32(0.01))
    divide_rate = theano.shared(np.float32(0.9))
    #--- Stochastic Gradient Descent
    def sgd_momentum(params, g_params, prev_deriv): #sgd_momentum(params, g_params, prev_deriv):
        updates = OrderedDict()
        for param, g_param, pv_drv in zip(params, g_params, prev_deriv):
            vel = divide_rate * pv_drv - (1.0 - divide_rate) * eps * g_param
            print "vel.eval = ",
            print vel.eval
            print "param.eval = ",
            print param.eval
            print "g_param.eval = ",
            print g_param.eval
            updates[param] = param + vel
            updates[pv_drv] = vel
        return updates

    layers = [
        Layer(784, 600, T.nnet.relu),
        Layer(600, 600, T.nnet.relu),
        Layer(600, 10, T.nnet.softmax)
    ]

    x = T.fmatrix('x')
    t = T.imatrix('t')

    params = []
    prev_deriv = []
    for i, layer in enumerate(layers):
        params += layer.params
        prev_deriv += layer.prev_deriv
        if i == 0:
            layer_out = layer.f_prop(x)
        else:
            layer_out = layer.f_prop(layer_out)

    y = layers[-1].z
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))

    g_params = T.grad(cost=cost, wrt=params)
    updates = sgd_momentum(params, g_params, prev_deriv)

    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')

    batch_size = 10
    n_batches = train_X.shape[0]
    limit_second = 60 * 5
    start = time.time()
    for epoch in xrange(10):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            train(train_X[start:end], train_y[start:end])
        eps = eps - 0.0001
        valid_cost, pred_y = valid(valid_X, valid_y)
        print 'Validation cost: %.3f, Validation F1: %.3f' % (valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), pred_y, average='macro'))

    pred_y = test(test_X)
    return pred_y
